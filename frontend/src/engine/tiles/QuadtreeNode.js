/**
 * DepthWizard Geospatial Engine — QuadtreeNode
 *
 * Hierarchical quadtree spatial partitioning node for continuous distance-
 * and screen-space-driven Level of Detail (LOD) and frustum culling.
 */

import { TerrainTile } from './TerrainTile.js';

export class QuadtreeNode {
  /**
   * @param {Object} options
   * @param {string} options.id - e.g. "0", "0_0", "0_1"
   * @param {number} options.level - LOD level (0 = root)
   * @param {number} options.maxLod - Maximum allowable LOD depth
   * @param {number} [options.tx=0] - Quadtree column index at this level
   * @param {number} [options.ty=0] - Quadtree row index at this level (row 0 = north)
   * @param {QuadtreeNode} [options.parent] - Parent node (null for root)
   * @param {number} options.worldMinX - West bounds in meters
   * @param {number} options.worldMaxX - East bounds in meters
   * @param {number} options.worldMinZ - North bounds in meters
   * @param {number} options.worldMaxZ - South bounds in meters
   * @param {number} options.uMin - UV west [0, 1]
   * @param {number} options.uMax - UV east [0, 1]
   * @param {number} options.vMin - UV north [0, 1]
   * @param {number} options.vMax - UV south [0, 1]
   * @param {number} options.segments - Geometry subdivisions
   * @param {Function} options.sampleHeight - (u, v) => elevation in meters
   * @param {THREE.Material} options.material - Unified terrain material
   */
  constructor(options) {
    this.id = options.id;
    this.level = options.level;
    this.maxLod = options.maxLod || 3;
    this.tx = options.tx ?? 0;
    this.ty = options.ty ?? 0;
    /** @type {QuadtreeNode|null} */
    this.parent = options.parent ?? null;

    this.worldMinX = options.worldMinX;
    this.worldMaxX = options.worldMaxX;
    this.worldMinZ = options.worldMinZ;
    this.worldMaxZ = options.worldMaxZ;

    this.uMin = options.uMin;
    this.uMax = options.uMax;
    this.vMin = options.vMin;
    this.vMax = options.vMax;

    this.segments = options.segments || 32;
    this.sampleHeight = options.sampleHeight;
    this.material = options.material;

    // Diagonal extent in meters
    this.diagonal = Math.hypot(this.worldMaxX - this.worldMinX, this.worldMaxZ - this.worldMinZ);

    // Associated terrain tile
    this.tile = new TerrainTile({
      id: this.id,
      level: this.level,
      tx: this.tx,
      ty: this.ty,
      worldMinX: this.worldMinX,
      worldMaxX: this.worldMaxX,
      worldMinZ: this.worldMinZ,
      worldMaxZ: this.worldMaxZ,
      uMin: this.uMin,
      uMax: this.uMax,
      vMin: this.vMin,
      vMax: this.vMax,
      segments: this.segments,
      sampleHeight: this.sampleHeight,
      material: this.material,
    });
    // The tile's parentNode is the PARENT node (whose .tile is the ready
    // coarse ancestor used for fallback rendering); null for the root.
    this.tile.parentNode = this.parent;

    /** @type {QuadtreeNode[]|null} */
    this.children = null;
    this.isSplit = false;
  }

  /**
   * Subdivides this node into 4 children (NW, NE, SW, SE).
   */
  split() {
    if (this.isSplit || this.level >= this.maxLod) return;

    const midX = (this.worldMinX + this.worldMaxX) * 0.5;
    const midZ = (this.worldMinZ + this.worldMaxZ) * 0.5;
    const midU = (this.uMin + this.uMax) * 0.5;
    const midV = (this.vMin + this.vMax) * 0.5;
    const nextLevel = this.level + 1;

    // Quadtree indices: row 0 = north (matches backend tile convention and
    // v = 0 north). NW/NE are the north half (ty*2), SW/SE the south (ty*2+1).
    const common = (col, row) => ({
      id: `${this.id}_${['NW', 'NE', 'SW', 'SE'][(row * 2) + col]}`,
      level: nextLevel,
      maxLod: this.maxLod,
      tx: this.tx * 2 + col,
      ty: this.ty * 2 + row,
      parent: this,
      segments: this.segments,
      sampleHeight: this.sampleHeight,
      material: this.material,
    });

    // NW: [minX, midX] x [minZ, midZ]
    const nw = new QuadtreeNode({
      ...common(0, 0),
      worldMinX: this.worldMinX, worldMaxX: midX,
      worldMinZ: this.worldMinZ, worldMaxZ: midZ,
      uMin: this.uMin, uMax: midU,
      vMin: this.vMin, vMax: midV,
    });

    // NE: [midX, maxX] x [minZ, midZ]
    const ne = new QuadtreeNode({
      ...common(1, 0),
      worldMinX: midX, worldMaxX: this.worldMaxX,
      worldMinZ: this.worldMinZ, worldMaxZ: midZ,
      uMin: midU, uMax: this.uMax,
      vMin: this.vMin, vMax: midV,
    });

    // SW: [minX, midX] x [midZ, maxZ]
    const sw = new QuadtreeNode({
      ...common(0, 1),
      worldMinX: this.worldMinX, worldMaxX: midX,
      worldMinZ: midZ, worldMaxZ: this.worldMaxZ,
      uMin: this.uMin, uMax: midU,
      vMin: midV, vMax: this.vMax,
    });

    // SE: [midX, maxX] x [midZ, maxZ]
    const se = new QuadtreeNode({
      ...common(1, 1),
      worldMinX: midX, worldMaxX: this.worldMaxX,
      worldMinZ: midZ, worldMaxZ: this.worldMaxZ,
      uMin: midU, uMax: this.uMax,
      vMin: midV, vMax: this.vMax,
    });

    this.children = [nw, ne, sw, se];
    this.isSplit = true;
  }

  /**
   * Merges child nodes and frees their resources.
   */
  merge() {
    if (!this.isSplit || !this.children) return;
    for (const child of this.children) {
      child.dispose();
    }
    this.children = null;
    this.isSplit = false;
  }

  /**
   * Evaluates LOD and frustum visibility recursively.
   *
   * @param {THREE.Camera} camera
   * @param {THREE.Frustum} frustum
   * @param {number} splitThreshold - Ratio of chunk diagonal to camera distance
   * @param {TerrainTile[]} visibleTiles - Array to collect active visible tiles
   */
  evaluate(camera, frustum, splitThreshold, visibleTiles) {
    // 1. Frustum Culling: If chunk bounding box is outside frustum, skip
    if (!frustum.intersectsBox(this.tile.boundingBox)) {
      return;
    }

    // 2. Metric Distance to Camera in meters
    const dist = Math.max(1.0, this.tile.distanceToCamera(camera.position));

    // Screen-space metric: ratio of physical diagonal to camera distance
    const metric = this.diagonal / dist;

    // 3. Subdivide or Leaf
    if (this.level < this.maxLod && metric > splitThreshold) {
      if (!this.isSplit) {
        this.split();
      }
      if (this.children) {
        for (const child of this.children) {
          child.evaluate(camera, frustum, splitThreshold, visibleTiles);
        }
      }
    } else {
      // Coarser LOD or leaf level: merge any obsolete children and render this tile
      if (this.isSplit) {
        this.merge();
      }
      visibleTiles.push(this.tile);
    }
  }

  /**
   * Recursive disposal.
   */
  dispose() {
    this.merge();
    this.tile.dispose();
  }
}
