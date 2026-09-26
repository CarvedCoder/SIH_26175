/**
 * DepthWizard Geospatial Engine — TerrainTile
 *
 * Represents an individual renderable terrain chunk with:
 *   - Local metric world bounds (meters)
 *   - LOD level (0 = highest detail, N = coarse)
 *   - Geometry with perimeter skirts
 *   - Bounding box and sphere for frustum culling
 */

import * as THREE from 'three';
import { buildChunkGeometry } from '../terrain/ChunkGeometry.js';

export const TileState = {
  UNLOADED: 'UNLOADED',
  LOADING: 'LOADING',
  LOADED: 'LOADED',
  DISPOSED: 'DISPOSED',
};

export class TerrainTile {
  /**
   * @param {Object} options
   * @param {string} options.id - e.g. "L1_X0_Y1"
   * @param {number} options.level - LOD level (0 = max detail)
   * @param {number} options.tx - Quadtree column index at this level
   * @param {number} options.ty - Quadtree row index at this level (row 0 = north)
   * @param {number} options.worldMinX - West bounds (meters)
   * @param {number} options.worldMaxX - East bounds (meters)
   * @param {number} options.worldMinZ - North bounds (meters)
   * @param {number} options.worldMaxZ - South bounds (meters)
   * @param {number} options.uMin - UV west [0, 1]
   * @param {number} options.uMax - UV east [0, 1]
   * @param {number} options.vMin - UV north [0, 1]
   * @param {number} options.vMax - UV south [0, 1]
   * @param {number} [options.segments=32] - Grid subdivision segments
   * @param {Function} options.sampleHeight - (u, v) => elevation in meters
   * @param {THREE.Material} options.material - Chunk material
   */
  constructor(options) {
    this.id = options.id;
    this.level = options.level;
    this.tx = options.tx ?? 0;
    this.ty = options.ty ?? 0;

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

    this.state = TileState.UNLOADED;
    this.mesh = null;
    this.geometry = null;
    /** @type {import('./QuadtreeNode.js').QuadtreeNode|null} set by the quadtree */
    this.parentNode = null;
    /** True once this tile's own streamed height patch has arrived */
    this.patchReady = false;

    this.boundingBox = new THREE.Box3(
      new THREE.Vector3(this.worldMinX, -100, this.worldMinZ),
      new THREE.Vector3(this.worldMaxX, 1000, this.worldMaxZ)
    );

    this.center = new THREE.Vector3(
      (this.worldMinX + this.worldMaxX) * 0.5,
      0,
      (this.worldMinZ + this.worldMaxZ) * 0.5
    );
  }

  /**
   * Builds the GPU BufferGeometry and THREE.Mesh for this tile.
   *
   * @param {THREE.Material} [material] - Per-tile material (falls back to constructor material)
   * @returns {THREE.Mesh}
   */
  buildMesh(material) {
    if (this.state === TileState.LOADED && this.mesh) return this.mesh;

    const mat = material || this.material;
    // Skirt depth must cover the worst T-junction gap against a coarser
    // neighbour (which scales with the tile's cell size) — a fixed 1.2 m
    // skirt opened visible tears at LOD boundaries on steep urban DSMs.
    // The exaggeration displacement in the vertex shader stretches the
    // skirt proportionally, so this stays valid at any exaggeration.
    const tileWidth = this.worldMaxX - this.worldMinX;
    const skirtDepth = Math.max(2.5, tileWidth * 0.03);
    this.geometry = buildChunkGeometry({
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
      addSkirt: true,
      skirtDepth,
    });

    this.boundingBox.copy(this.geometry.boundingBox);
    this.boundingBox.getCenter(this.center);

    this.mesh = new THREE.Mesh(this.geometry, mat);
    this.mesh.frustumCulled = true;
    this.mesh.name = `TerrainChunk_${this.id}`;
    this.mesh.userData = { tile: this };

    this.state = TileState.LOADED;
    return this.mesh;
  }

  /**
   * A tile is renderable once its mesh exists (geometry data has arrived).
   *
   * @returns {boolean}
   */
  isReady() {
    return this.state === TileState.LOADED && this.mesh !== null;
  }

  /**
   * Distance from camera position to the chunk's bounding box.
   *
   * @param {THREE.Vector3} cameraPosition
   * @returns {number} Distance in meters
   */
  distanceToCamera(cameraPosition) {
    return this.boundingBox.distanceToPoint(cameraPosition);
  }

  /**
   * Disposes WebGL buffers and cleans up memory.
   */
  dispose() {
    this.state = TileState.DISPOSED;
    if (this.geometry) {
      this.geometry.dispose();
      this.geometry = null;
    }
    if (this.mesh) {
      if (this.mesh.parent) {
        this.mesh.parent.remove(this.mesh);
      }
      this.mesh = null;
    }
  }
}
