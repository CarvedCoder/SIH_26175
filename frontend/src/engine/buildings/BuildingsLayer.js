/**
 * DepthWizard — BuildingsLayer
 *
 * Geometry-aware 3D building reconstruction overlay for the terrain
 * viewer. Consumes the backend's buildings3d.json (footprint polygons in
 * prediction-grid pixel coordinates + heights taken from the predicted
 * metric DSM + per-building confidence) and extrudes each footprint into
 * a watertight prism via THREE.ExtrudeGeometry.
 *
 * ALIGNMENT CONTRACT (must mirror the terrain shader exactly):
 *   - Footprint pixels map through geoRef.pixelToLocal (same convention
 *     as every terrain chunk).
 *   - Y placement reproduces TerrainMaterial's exaggeration formula
 *     y' = uMinElevation + max(0, y - uMinElevation) * exaggeration by
 *     storing vertex Y RELATIVE to minElevation and scaling the group:
 *     group.position.y = minElevation, group.scale.y = exaggeration.
 *   - 1 world unit = 1 metre.
 *
 * Each building is ONE mesh (its parts/levels are sub-geometries merged
 * into it) so a composite structure reads as a single object, plus crisp
 * roof-edge lines. Confidence drives color: green >= 0.7, cyan >= 0.45,
 * amber below.
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const CONF_COLORS = {
  high: 0x34d399,   // >= 0.7
  mid:  0x38bdf8,   // >= 0.45
  low:  0xfbbf24,
};

// Disaster damage classes (matches depthwizard.disaster DAMAGE_COLORS_HEX
// and the backend preview) — override confidence colors when the damage
// model classified the scene.
const DAMAGE_COLORS = {
  'no-damage':    0x4caf50,
  'minor-damage': 0xffc107,
  'major-damage': 0xff5722,
  'destroyed':    0xd32f2f,
};

function colorForBuilding(building) {
  const dc = building?.damage_class;
  if (dc && DAMAGE_COLORS[dc]) return DAMAGE_COLORS[dc];
  const confidence = building?.confidence ?? 0.5;
  if (confidence >= 0.7) return CONF_COLORS.high;
  if (confidence >= 0.45) return CONF_COLORS.mid;
  return CONF_COLORS.low;
}

/** Polygon ring (pixel coords) -> THREE.Shape in the local world plane. */
function ringToShapePath(ringPx, geoRef) {
  const points = ringPx.map(([px, py]) => {
    const { x, z } = geoRef.pixelToLocal(px, py, 0);
    return new THREE.Vector2(x, -z); // world +Z is South; shape Y = -Z so the
    // extruded geometry (XY plane) maps back onto XZ with correct winding
  });
  return points;
}

/** Build one extruded prism geometry for a footprint ring (+ optional levels).
 *
 * Y spans [base - minElevation, base - minElevation + height] so the group
 * (position.y = minElevation, scale.y = exaggeration) reproduces the
 * terrain shader's exaggeration formula exactly. Levels are prisms from
 * the SAME base up to their own DSM height (podium + tower geometry). */
function buildingGeometries(building, geoRef, minElevation) {
  const geoms = [];
  const base = (building.base_elevation_m ?? minElevation) - minElevation;

  const addPrism = (ringPx, heightM) => {
    if (!ringPx || ringPx.length < 3 || !(heightM > 0)) return;
    const shape = new THREE.Shape(ringToShapePath(ringPx, geoRef));
    const geom = new THREE.ExtrudeGeometry(shape, {
      depth: heightM,
      bevelEnabled: false,
      curveSegments: 1, // rings are already tessellated polylines
    });
    // shape plane is XY, extrusion +Z; we want the footprint on the XZ
    // ground plane and extrusion +Y (up): rotateX(-90°) maps +Z -> +Y
    // and +Y -> -Z (world +Z is South — matches GeoReference).
    geom.rotateX(-Math.PI / 2);
    if (base !== 0) geom.translate(0, base, 0);
    geoms.push(geom);
  };

  addPrism(building.footprint_px, building.height_m);
  for (const level of building.levels ?? []) {
    addPrism(level.polygon_px, level.height_m);
  }
  return geoms;
}

/** Canvas sprite carrying a building's estimated height (metres). */
function makeHeightLabel(text, worldSpan) {
  const c = document.createElement('canvas');
  c.width = 160; c.height = 56;
  const ctx = c.getContext('2d');
  ctx.font = '600 34px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(10,14,23,0.75)';
  const w = ctx.measureText(text).width + 24;
  ctx.fillRect((160 - w) / 2, 6, w, 44);
  ctx.fillStyle = '#eaf6ff';
  ctx.fillText(text, 80, 30);
  const tex = new THREE.CanvasTexture(c);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false,
  }));
  spr.scale.set(worldSpan * 0.02, worldSpan * 0.007, 1);
  spr.renderOrder = 60;
  spr.userData.disposeMap = true;
  return spr;
}

/** Low-poly tree: trunk cylinder + two-tier crown cones. */
function buildTreeGeometry(heightM, crownRadius) {
  const geoms = [];
  const trunkH = Math.max(0.6, heightM * 0.22);
  const crownH = Math.max(0.8, heightM * 0.78);
  const trunk = new THREE.CylinderGeometry(
    crownRadius * 0.12, crownRadius * 0.16, trunkH, 6,
  );
  trunk.translate(0, trunkH / 2, 0);
  geoms.push(trunk);
  const crown1 = new THREE.ConeGeometry(crownRadius, crownH * 0.62, 7);
  crown1.translate(0, trunkH + crownH * 0.31, 0);
  geoms.push(crown1);
  const crown2 = new THREE.ConeGeometry(crownRadius * 0.72, crownH * 0.5, 7);
  crown2.translate(0, trunkH + crownH * 0.7, 0);
  geoms.push(crown2);
  const merged = mergeGeometries(geoms, false) ?? geoms[0];
  for (const g of geoms) if (g !== merged) g.dispose();
  return merged;
}

export class BuildingsLayer {
  /**
   * @param {THREE.Scene} scene
   * @param {import('../geo/GeoReference.js').GeoReference} geoRef
   */
  constructor(scene, geoRef) {
    this.scene = scene;
    this.geoRef = geoRef;
    this.group = new THREE.Group();
    this.group.name = 'Buildings3D';
    this.group.visible = false;
    scene.add(this.group);

    this.buildings = [];
    this._labels = [];
    this._trees = [];
    this._lights = null;
    this.loaded = false;
    this.disposed = false;
  }

  /**
   * Build meshes from the backend reconstruction payload.
   * @param {Object} data - buildings3d.json content
   */
  load(data) {
    if (this.disposed || !data?.available || !Array.isArray(data.buildings)) return;
    this.clear();

    const minElevation = this.geoRef.minElevation ?? 0;

    for (const building of data.buildings) {
      const geoms = buildingGeometries(building, this.geoRef, minElevation);
      if (!geoms.length) continue;

      const merged = mergeGeometries(geoms, false) ?? geoms[0];
      for (const g of geoms) if (g !== merged) g.dispose();

      // vertex Y is already relative to minElevation (base offset applied
      // per-prism), so group.position.y = minElevation and group.scale.y
      // = exaggeration reproduce the terrain shader exactly
      merged.computeVertexNormals();

      const color = colorForBuilding(building);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.55,
        metalness: 0.05,
        flatShading: true,
        transparent: true,
        opacity: 0.92,
      });
      const mesh = new THREE.Mesh(merged, material);
      mesh.userData.building = building;
      mesh.name = `building_${building.id}_${building.damage_class ?? 'ok'}`;
      mesh.renderOrder = 2;
      this.group.add(mesh);

      // crisp roof outline — reads as a clean reconstructed structure
      const edges = new THREE.EdgesGeometry(merged, 25);
      const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({
          color: 0x0a0e17,
          transparent: true,
          opacity: 0.55,
        }),
      );
      line.renderOrder = 3;
      this.group.add(line);

      this.buildings.push({ mesh, line, building });
    }

    // The terrain material is a custom shader — the scene carries no
    // scene-wide lights, so the buildings bring their own (scoped to this
    // group, disposed with it).
    const hemi = new THREE.HemisphereLight(0xdfeaff, 0x35404f, 1.5);
    const dir = new THREE.DirectionalLight(0xffffff, 2.2);
    dir.position.set(0.4, 1.0, 0.35).normalize();
    const dir2 = new THREE.DirectionalLight(0xcfe4ff, 0.8);
    dir2.position.set(-0.6, 0.7, -0.5).normalize();
    this.group.add(hemi, dir, dir2);
    this._lights = [hemi, dir, dir2];

    // Per-building estimated-height labels (model-derived, from the DSM)
    const worldSpan = Math.max(this.geoRef.worldWidth, this.geoRef.worldDepth);
    for (const entry of this.buildings) {
      const b = entry.building;
      const label = makeHeightLabel(`${b.height_m.toFixed(1)} m`, worldSpan);
      label.position.copy(entry.mesh.position);
      // place above the building's tallest point
      entry.mesh.geometry.computeBoundingBox();
      const box = entry.mesh.geometry.boundingBox;
      label.position.y += (box?.max.y ?? 0) + worldSpan * 0.012;
      this.group.add(label);
      this._labels.push(label);
    }

    // Custom tree objects from vegetation candidates (green + elevated)
    const treeMat = new THREE.MeshStandardMaterial({
      color: 0x2fbf71, roughness: 0.85, flatShading: true,
    });
    const trunkMat = new THREE.MeshStandardMaterial({
      color: 0x6b4a2f, roughness: 0.9, flatShading: true,
    });
    const treeR = Math.max(1.6, worldSpan * 0.004);
    for (const tree of data.trees ?? []) {
      if (!(tree.height_m > 0.5)) continue;
      const { x, z } = this.geoRef.pixelToLocal(tree.x_px, tree.y_px, 0);
      const geom = buildTreeGeometry(tree.height_m, treeR);
      const mesh = new THREE.Mesh(geom, [trunkMat, treeMat]);
      // geometry is Y-up with base at 0; place relative to min elevation,
      // standing on the GROUND surface (buildings removed under trees too)
      mesh.position.set(x, (tree.ground_elevation_m ?? minElevation) - minElevation, z);
      this.group.add(mesh);
      this._trees.push(mesh);
    }

    this.group.position.y = minElevation;
    this.group.scale.y = 1.0; // exaggeration applied via setExaggeration
    this.loaded = this.buildings.length > 0;
    this.group.visible = this.loaded;
  }

  setExaggeration(factor) {
    if (this.disposed) return;
    this.group.scale.y = Math.max(0.1, Number(factor) || 1.0);
  }

  setVisible(visible) {
    if (this.disposed) return;
    this.group.visible = !!visible && this.loaded;
  }

  get visible() {
    return this.group.visible;
  }

  get count() {
    return this.buildings.length;
  }

  clear() {
    for (const light of this._lights ?? []) {
      this.group.remove(light);
      light.dispose?.();
    }
    this._lights = null;
    for (const l of this._labels) {
      l.material.map?.dispose();
      l.material.dispose();
      this.group.remove(l);
    }
    this._labels = [];
    for (const t of this._trees) {
      t.geometry.dispose();
      (Array.isArray(t.material) ? t.material : [t.material]).forEach(m => m.dispose());
      this.group.remove(t);
    }
    this._trees = [];
    for (const entry of this.buildings) {
      entry.mesh.geometry.dispose();
      entry.mesh.material.dispose();
      entry.line.geometry.dispose();
      entry.line.material.dispose();
      this.group.remove(entry.mesh);
      this.group.remove(entry.line);
    }
    this.buildings = [];
    this.loaded = false;
  }

  dispose() {
    this.disposed = true;
    this.clear();
    if (this.group.parent) {
      this.group.parent.remove(this.group);
    }
  }
}
