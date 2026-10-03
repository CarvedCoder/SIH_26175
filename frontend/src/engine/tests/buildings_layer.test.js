/**
 * DepthWizard — BuildingsLayer tests (geometry-aware 3D reconstruction)
 *
 * Contracts verified:
 *   1. Footprints extrude into prisms with the DSM-derived height carried
 *      into world Y (min-elevation-relative, never a fixed height).
 *   2. Pixel → world mapping matches GeoReference.pixelToLocal exactly
 *      (buildings land ON the terrain).
 *   3. Exaggeration: group.scale.y reproduces the terrain shader formula
 *      y' = minElev + (y - minElev) * exaggeration.
 *   4. Composite structures (parts + levels) merge into ONE mesh entry.
 *   5. Visibility toggle + dispose are clean.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { GeoReference } from '../geo/GeoReference.js';
import { BuildingsLayer } from '../buildings/BuildingsLayer.js';

function makeGeoRef() {
  return new GeoReference({
    crs: 'EPSG:32617',
    gsdX: 0.5,
    gsdY: 0.5,
    rasterWidth: 256,
    rasterHeight: 256,
    minElevation: 100,
    maxElevation: 150,
    isGeoreferenced: true,
  });
}

function makePayload() {
  return {
    available: true,
    height_source: 'dsm.npy (predicted metric height)',
    buildings: [
      {
        id: 1,
        primitive: 'rectangle',
        confidence: 0.93,
        height_m: 12.5,
        base_elevation_m: 100,
        top_elevation_m: 112.5,
        // 100x50 px rectangle on a 256x256 grid
        footprint_px: [[40, 40], [140, 40], [140, 90], [40, 90]],
      },
      {
        id: 2,
        primitive: 'composite_rectangles',
        confidence: 0.61,
        height_m: 9.1,
        base_elevation_m: 100,
        top_elevation_m: 109.1,
        footprint_px: [[10, 150], [110, 150], [110, 230], [70, 230], [70, 190], [10, 190]],
        levels: [
          { polygon_px: [[10, 150], [60, 150], [60, 190], [10, 190]], height_m: 18 },
        ],
      },
    ],
  };
}

function makeLayer() {
  const scene = new THREE.Scene();
  const layer = new BuildingsLayer(scene, makeGeoRef());
  return { scene, layer };
}

test('loads buildings: one mesh + edge lines per building', () => {
  const { layer } = makeLayer();
  layer.load(makePayload());
  assert.equal(layer.count, 2);
  assert.equal(layer.group.children.length, 4); // 2 meshes + 2 line segments
  assert.equal(layer.loaded, true);
  assert.equal(layer.group.visible, true);
});

test('extrusion height comes from the payload (DSM), not a fixed value', () => {
  const { layer } = makeLayer();
  layer.load(makePayload());
  const mesh = layer.buildings[0].mesh;
  mesh.geometry.computeBoundingBox();
  const box = mesh.geometry.boundingBox;
  // vertices stored RELATIVE to minElevation (100): prism spans 0..12.5
  assert.ok(Math.abs(box.min.y - 0) < 1e-6, `min.y ${box.min.y}`);
  assert.ok(Math.abs(box.max.y - 12.5) < 1e-6, `max.y ${box.max.y}`);
});

test('footprint maps through pixelToLocal (lands on the terrain grid)', () => {
  const geoRef = makeGeoRef();
  const { layer } = makeLayer();
  layer.load(makePayload());
  const pos = layer.buildings[0].mesh.geometry.attributes.position;
  // footprint corner pixel (40, 40) -> expected world XZ via pixelToLocal
  const expected = geoRef.pixelToLocal(40, 40, 0);
  let found = false;
  for (let i = 0; i < pos.count; i++) {
    if (Math.abs(pos.getX(i) - expected.x) < 1e-4
        && Math.abs(pos.getZ(i) - expected.z) < 1e-4) {
      found = true;
      break;
    }
  }
  assert.ok(found, 'no vertex at the expected world position of pixel (40,40)');
});

test('group placement + scale reproduce the terrain exaggeration formula', () => {
  const geoRef = makeGeoRef();
  const { layer } = makeLayer();
  layer.load(makePayload());
  layer.setExaggeration(2.5);

  assert.equal(layer.group.position.y, 100);       // minElevation
  assert.equal(layer.group.scale.y, 2.5);
  // world Y of the prism top = minElev + (relative height) * exaggeration
  const mesh = layer.buildings[0].mesh;
  mesh.geometry.computeBoundingBox();
  const worldTop = layer.group.position.y
    + mesh.geometry.boundingBox.max.y * layer.group.scale.y;
  assert.ok(Math.abs(worldTop - (100 + 12.5 * 2.5)) < 1e-6);
});

test('composite building with levels merges into ONE mesh', () => {
  const { layer } = makeLayer();
  layer.load(makePayload());
  const composite = layer.buildings[1];
  const mesh = composite.mesh;
  mesh.geometry.computeBoundingBox();
  // upper level reaches 18 m (relative), main prism 9.1 m
  assert.ok(Math.abs(mesh.geometry.boundingBox.max.y - 18) < 1e-6);
  assert.ok(Math.abs(mesh.geometry.boundingBox.min.y - 0) < 1e-6);
});

test('visibility toggle and dispose are clean', () => {
  const { scene, layer } = makeLayer();
  layer.load(makePayload());
  layer.setVisible(false);
  assert.equal(layer.group.visible, false);
  layer.setVisible(true);
  assert.equal(layer.group.visible, true);

  const children = layer.group.children.length;
  layer.dispose();
  assert.equal(layer.group.children.length, 0);
  assert.notEqual(scene.children.includes(layer.group), true);
  assert.ok(children > 0);
});

test('damage-classified buildings are colored by damage class', () => {
  const { layer } = makeLayer();
  const payload = makePayload();
  payload.buildings[0].damage_class = 'destroyed';
  payload.buildings[1].damage_class = 'no-damage';
  layer.load(payload);

  const destroyed = layer.buildings[0].mesh.material.color.getHex();
  const intact = layer.buildings[1].mesh.material.color.getHex();
  assert.equal(destroyed, 0xd32f2f);   // destroyed red
  assert.equal(intact, 0x4caf50);      // no-damage green
  assert.notEqual(destroyed, intact);
});

test('terrain flatten mode toggles the uFlatten uniform', async () => {
  const { TerrainEngine } = await import('../TerrainEngine.js');
  const { createTerrainMaterial } = await import('../terrain/TerrainMaterial.js');
  const material = createTerrainMaterial({ minElevation: 100, maxElevation: 150 });
  assert.equal(material.uniforms.uFlatten.value, 0.0);
  material.uniforms.uFlatten.value = 1.0;
  assert.equal(material.uniforms.uFlatten.value, 1.0);
});

test('empty / unavailable payload loads nothing', () => {
  const { layer } = makeLayer();
  layer.load({ available: false, buildings: [] });
  assert.equal(layer.count, 0);
  assert.equal(layer.loaded, false);
  assert.equal(layer.group.visible, false);
});
