/**
 * DepthWizard Geospatial Engine — LODManager
 *
 * Coordinates quadtree updates, frustum culling, and asynchronous height
 * streaming in the 3D scene.
 *
 * ARCHITECTURE (simplified):
 *   - GEOMETRY: quadtree LOD over the scene; tiles share ONE terrain
 *     material whose UVs are GLOBAL raster UVs (baked into the geometry),
 *     so a single full-resolution drape texture serves every tile at any
 *     LOD. No per-tile material clones, no per-tile textures, no UV-offset
 *     juggling — that subsystem was the source of recurring close-range
 *     rendering corruption (disposed materials, wrong texture quadrants,
 *     LOD seam mismatches) and was removed by design.
 *   - HEIGHTS: the full-resolution detail streams per tile from the
 *     backend quadtree endpoints (ancestors render while tiles load — no
 *     holes), so close-range geometry keeps urban-scale structure.
 *
 * Textures: ONE global drape texture (set via TerrainEngine.setTexture)
 * with max anisotropy; value-encoded layers (depth/DSM/slope) are set the
 * same way with NoColorSpace data textures.
 */

import * as THREE from 'three';
import { QuadtreeNode } from '../tiles/QuadtreeNode.js';
import { PatchHeightfield } from '../streaming/PatchHeightfield.js';

export class LODManager {
  /**
   * @param {Object} options
   * @param {import('../geo/GeoReference.js').GeoReference} options.geoRef
   * @param {import('../terrain/TerrainDataset.js').TerrainDataset} options.dataset
   * @param {THREE.Material} options.material - THE shared terrain material
   * @param {THREE.Scene} options.scene
   * @param {number} [options.maxLod=3]
   * @param {number} [options.splitThreshold=0.85]
   * @param {number} [options.segments=32]
   * @param {Object|null} [options.heightTiles=null] - { streamer: TileStreamer }
   */
  constructor(options) {
    this.geoRef = options.geoRef;
    this.dataset = options.dataset;
    this.material = options.material;
    this.scene = options.scene;

    this.maxLod = options.maxLod || 3;
    this.splitThreshold = options.splitThreshold || 0.85;
    this.segments = options.segments || 32;

    /** @type {{streamer: import('../streaming/TileStreamer.js').TileStreamer}|null} */
    this.heightTiles = options.heightTiles || null;

    this.frustum = new THREE.Frustum();
    this.projScreenMatrix = new THREE.Matrix4();

    // Set of active tiles currently added to scene
    this.activeTiles = new Map(); // tileId -> TerrainTile
    /** @type {Set<string>} tile ids with an in-flight height request */
    this.pendingBuilds = new Set();
    /** @type {Set<string>} tile ids wanted by the most recent update() */
    this._wantedIds = new Set();

    // Root Quadtree node covering full footprint
    const halfW = this.geoRef.worldWidth * 0.5;
    const halfD = this.geoRef.worldDepth * 0.5;

    this.root = new QuadtreeNode({
      id: 'root',
      level: 0,
      tx: 0,
      ty: 0,
      maxLod: this.maxLod,
      worldMinX: -halfW, worldMaxX: halfW,
      worldMinZ: -halfD, worldMaxZ: halfD,
      uMin: 0.0, uMax: 1.0,
      vMin: 0.0, vMax: 1.0,
      segments: this.segments,
      sampleHeight: (u, v) => this.dataset.sampleElevation(u, v),
      material: this.material,
    });

    // In streamed mode the quadtree's dataset-backed sampler must NOT build
    // geometry: each tile renders only from its own streamed patch. Null out
    // the root sampler and prefetch the root tile so the world appears as
    // soon as the first (coarsest) patch arrives.
    if (this.heightTiles) {
      this.root.tile.sampleHeight = null;
      this._requestHeightBuild(this.root.tile, 0);
    }

    // Statistics for telemetry/debug
    this.stats = {
      visibleTileCount: 0,
      activeTileCount: 0,
      pendingTileCount: 0,
      triangleCount: 0,
      lodDistribution: {},
    };
  }

  /**
   * Evaluates LOD and updates the Three.js scene geometry.
   *
   * @param {THREE.Camera} camera
   */
  update(camera) {
    if (!camera || !this.scene) return;

    // 1. Update Camera Frustum
    this.projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

    // 2. Evaluate Quadtree visibility & LOD
    /** @type {import('../tiles/TerrainTile.js').TerrainTile[]} */
    const neededTiles = [];
    this.root.evaluate(camera, this.frustum, this.splitThreshold, neededTiles);

    // 3. Resolve the render set (streamed mode substitutes ready ancestors
    //    for tiles whose data has not arrived yet — no holes, no popping).
    let renderTiles;
    if (this.heightTiles) {
      this._wantedIds = new Set(neededTiles.map((t) => t.id));
      const camPos = camera.position;
      for (const tile of neededTiles) {
        const dist = tile.distanceToCamera(camPos);
        this._requestHeightBuild(tile, 1 / (1 + Math.max(0, dist)));
        // A tile whose patch already arrived but was not yet built (e.g. it
        // re-entered the wanted set) gets its mesh built here.
        if (tile.patchReady && !tile.mesh) {
          tile.buildMesh(this.material);
        }
      }
      const readySet = new Set();
      for (const tile of neededTiles) {
        let t = tile;
        while (t && !t.isReady()) {
          t = t.parentNode ? t.parentNode.tile : null;
        }
        if (t) readySet.add(t);
      }
      renderTiles = [...readySet];
    } else {
      renderTiles = neededTiles;
    }

    const neededSet = new Set(renderTiles.map((t) => t.id));

    // 4. Remove tiles no longer needed. Meshes stay attached to their
    //    TerrainTile (sharing the single material) so returning tiles are
    //    reused as-is — nothing to dispose or rebuild.
    for (const [id, tile] of this.activeTiles.entries()) {
      if (!neededSet.has(id)) {
        if (tile.mesh && tile.mesh.parent) {
          tile.mesh.parent.remove(tile.mesh);
        }
        this.activeTiles.delete(id);
      }
    }

    // 5. Add newly needed tiles
    let triCount = 0;
    const lodCounts = {};

    for (const tile of renderTiles) {
      if (!this.activeTiles.has(tile.id)) {
        const mesh = tile.mesh ?? tile.buildMesh(this.material);
        this.scene.add(mesh);
        this.activeTiles.set(tile.id, tile);
      }

      triCount += (tile.geometry?.index ? tile.geometry.index.count / 3 : 0);
      lodCounts[tile.level] = (lodCounts[tile.level] || 0) + 1;
    }

    // Update telemetry stats
    this.stats.visibleTileCount = renderTiles.length;
    this.stats.activeTileCount = this.activeTiles.size;
    this.stats.pendingTileCount = this.pendingBuilds.size;
    this.stats.triangleCount = Math.round(triCount);
    this.stats.lodDistribution = lodCounts;
  }

  /**
   * Async height-tile request for one tile (streamed mode only).
   *
   * @param {import('../tiles/TerrainTile.js').TerrainTile} tile
   * @param {number} priority - Higher = more urgent (1/(1+distance))
   */
  _requestHeightBuild(tile, priority) {
    if (!this.heightTiles) return;
    if (tile.isReady() || tile.patchReady || tile._buildPending) return;

    tile._buildPending = true;
    this.pendingBuilds.add(tile.id);

    this.heightTiles.streamer
      .request(tile.level, tile.tx, tile.ty, this.heightTiles.tileSize ?? 256, { priority })
      .then((patch) => {
        tile._buildPending = false;
        this.pendingBuilds.delete(tile.id);
        // Tile may have been merged away while the request was in flight
        if (tile.state === 'DISPOSED') return;
        if (!this._wantedIds.has(tile.id) && tile.level > 0) return;

        const pf = new PatchHeightfield({
          data: patch.data,
          size: patch.width,
          z: tile.level,
          tx: tile.tx,
          ty: tile.ty,
          rasterWidth: this.geoRef.rasterWidth,
          rasterHeight: this.geoRef.rasterHeight,
          minElevation: this.geoRef.minElevation,
          maxElevation: this.geoRef.maxElevation,
        });
        tile.sampleHeight = (u, v) => pf.sampleGlobal(u, v);
        tile.patchReady = true;
      })
      .catch((err) => {
        tile._buildPending = false;
        this.pendingBuilds.delete(tile.id);
        // Aborted requests are expected on scene switches; log the rest and
        // allow a later update() to retry the tile.
        if (!/aborted/i.test(String(err?.message || err))) {
          console.warn(`[lod] height tile ${tile.id} failed:`, err);
        }
      });
  }

  /**
   * Rebuild every tile from the CURRENT dataset — used by the
   * Buildings-3D mode to swap the heightfield (building-removed ground
   * DSM) without tearing down the engine. Streaming is reset with the
   * tiles; the caller re-enables it when restoring the original field.
   */
  resetTiles() {
    for (const [, tile] of this.activeTiles.entries()) {
      if (tile.mesh && tile.mesh.parent) {
        tile.mesh.parent.remove(tile.mesh);
      }
      tile.dispose();
    }
    this.activeTiles.clear();
    this.pendingBuilds.clear();
    this._wantedIds.clear();
    this.root.dispose();

    const halfW = this.geoRef.worldWidth * 0.5;
    const halfD = this.geoRef.worldDepth * 0.5;
    this.root = new QuadtreeNode({
      id: 'root',
      level: 0,
      tx: 0,
      ty: 0,
      maxLod: this.maxLod,
      worldMinX: -halfW, worldMaxX: halfW,
      worldMinZ: -halfD, worldMaxZ: halfD,
      uMin: 0.0, uMax: 1.0,
      vMin: 0.0, vMax: 1.0,
      segments: this.segments,
      sampleHeight: (u, v) => this.dataset.sampleElevation(u, v),
      material: this.material,
    });
    if (this.heightTiles) {
      this.root.tile.sampleHeight = null;
      this._requestHeightBuild(this.root.tile, 0);
    }
  }

  /**
   * Clean up all tiles, materials, and geometries.
   */
  dispose() {
    for (const [, tile] of this.activeTiles.entries()) {
      if (tile.mesh && tile.mesh.parent) {
        tile.mesh.parent.remove(tile.mesh);
      }
      tile.dispose();
    }
    this.activeTiles.clear();
    this.pendingBuilds.clear();
    this.root.dispose();
  }
}
