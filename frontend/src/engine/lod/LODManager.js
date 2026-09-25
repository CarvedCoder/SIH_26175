/**
 * DepthWizard Geospatial Engine — LODManager
 *
 * Coordinates quadtree updates, frustum culling, per-tile materials, and
 * asynchronous terrain streaming in the 3D scene.
 *
 * Two height modes:
 *   - SYNC (default): the full heightmap is resident in TerrainDataset; chunk
 *     geometry builds instantly with exact shared-edge heights (zero seams).
 *   - STREAMED: heights arrive as backend quadtree tiles on demand. Chunk
 *     builds are asynchronous; while a tile loads, its nearest ready ancestor
 *     is rendered instead (no holes), and tiles are prioritized by camera
 *     distance. Chosen by TerrainCanvas for very large rasters (spec §13/§29).
 *
 * Textures: each tile gets its own material clone and may receive its own
 * streamed texture (tile-local UVs); texture LOD is independent of geometry
 * LOD (spec §17/§19). The global texture remains the fallback until a tile's
 * texture arrives.
 */

import * as THREE from 'three';
import { QuadtreeNode } from '../tiles/QuadtreeNode.js';
import { PatchHeightfield } from '../streaming/PatchHeightfield.js';

export class LODManager {
  /**
   * @param {Object} options
   * @param {import('../geo/GeoReference.js').GeoReference} options.geoRef
   * @param {import('../terrain/TerrainDataset.js').TerrainDataset} options.dataset
   * @param {THREE.Material} options.material - Base material (template for per-tile clones)
   * @param {THREE.Scene} options.scene
   * @param {number} [options.maxLod=3]
   * @param {number} [options.splitThreshold=0.85]
   * @param {number} [options.segments=32]
   * @param {Object|null} [options.heightTiles=null] - { streamer: TileStreamer }
   * @param {Object|null} [options.textureTiles=null] - { streamer: TileStreamer, maxLevel, anisotropy }
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
    /** @type {{streamer: import('../streaming/TileStreamer.js').TileStreamer, maxLevel: number, anisotropy?: number}|null} */
    this.textureTiles = options.textureTiles || null;

    this.frustum = new THREE.Frustum();
    this.projScreenMatrix = new THREE.Matrix4();

    // Set of active tiles currently added to scene
    this.activeTiles = new Map(); // tileId -> TerrainTile
    /** @type {Map<string, THREE.Material>} tileId -> per-tile material clone */
    this.tileMaterials = new Map();
    /** @type {Set<string>} tile ids with an in-flight height request */
    this.pendingBuilds = new Set();
    /** @type {Set<string>} tile ids wanted by the most recent update() */
    this._wantedIds = new Set();
    /** @type {Map<string, THREE.Texture>} decoded tile textures, keyed by tile key */
    this._tileTextureObjects = new Map();

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
          tile.buildMesh(this._materialFor(tile));
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

    // 4. Remove tiles no longer needed (dispose their per-tile materials)
    for (const [id, tile] of this.activeTiles.entries()) {
      if (!neededSet.has(id)) {
        if (tile.mesh && tile.mesh.parent) {
          tile.mesh.parent.remove(tile.mesh);
        }
        this.activeTiles.delete(id);
      }
    }
    for (const id of [...this.tileMaterials.keys()]) {
      if (!neededSet.has(id)) {
        const m = this.tileMaterials.get(id);
        // Dispose the clone, never the shared textures it references
        m.dispose();
        this.tileMaterials.delete(id);
      }
    }

    // 5. Add newly needed tiles
    let triCount = 0;
    const lodCounts = {};

    for (const tile of renderTiles) {
      if (!this.activeTiles.has(tile.id)) {
        const mesh = tile.mesh ?? tile.buildMesh(this._materialFor(tile));
        this.scene.add(mesh);
        this.activeTiles.set(tile.id, tile);
        this._requestTileTexture(tile, camera);
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
   * Per-tile material: clone the base material and map this tile's tile-local
   * UVs onto the global raster frame. Geometry carries UVs in [0, 1] within
   * the tile, so uUvOffset/uUvScale restore global UVs for semantic overlays,
   * colormaps, and contour lines.
   *
   * @param {import('../tiles/TerrainTile.js').TerrainTile} tile
   * @returns {THREE.Material}
   */
  _materialFor(tile) {
    if (this.tileMaterials.has(tile.id)) {
      return this.tileMaterials.get(tile.id);
    }
    // Non-engine materials (plain MeshBasicMaterial etc.) carry no tile UV
    // mapping — share the base material untouched.
    if (!this.material.uniforms?.uUvOffset) {
      return this.material;
    }
    const m = this.material.clone();
    const uSpan = tile.uMax - tile.uMin;
    const vSpan = tile.vMax - tile.vMin;
    m.uniforms.uUvOffset.value.set(tile.uMin, 1.0 - tile.vMax);
    m.uniforms.uUvScale.value.set(uSpan, vSpan);
    this.tileMaterials.set(tile.id, m);
    return m;
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
   * Request this tile's own streamed texture. Until it arrives (or if
   * streaming is off), the tile renders with the global texture.
   *
   * @param {import('../tiles/TerrainTile.js').TerrainTile} tile
   * @param {THREE.Camera} camera
   */
  _requestTileTexture(tile, camera) {
    if (!this.textureTiles || !tile.mesh) return;
    // Texture LOD is independent of geometry LOD: cap at the provider's max
    // level; deeper geometry levels simply reuse the finest texture level.
    const texLevel = Math.min(tile.level, this.textureTiles.maxLevel ?? tile.level);

    const dist = tile.distanceToCamera(camera.position);
    const priority = 1 / (1 + Math.max(0, dist));
    const key = `${texLevel}/${tile.tx}/${tile.ty}`;

    this.textureTiles.streamer
      .request(texLevel, tile.tx, tile.ty, this.textureTiles.tileSize ?? 256, { priority })
      .then((bitmap) => {
        if (!this.activeTiles.has(tile.id)) return;
        const u = tile.mesh?.material?.uniforms;
        if (!u) return;
        // A user-selected layer (colormap mode != 0) always samples the
        // global texture; per-tile imagery only applies in RGB mode.
        if (u.uColormapMode.value !== 0) return;

        let tex = this._tileTextureObjects.get(key);
        if (!tex) {
          tex = new THREE.CanvasTexture(bitmap);
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.minFilter = THREE.LinearMipmapLinearFilter;
          tex.magFilter = THREE.LinearFilter;
          tex.wrapS = THREE.ClampToEdgeWrapping;
          tex.wrapT = THREE.ClampToEdgeWrapping;
          tex.generateMipmaps = true;
          tex.flipY = true;
          tex.anisotropy = this.textureTiles.anisotropy ?? 4;
          this._tileTextureObjects.set(key, tex);
        }
        u.uTexture.value = tex;
        u.uPerTileTexture.value = 1.0;
      })
      .catch((err) => {
        if (!/aborted|disposed/i.test(String(err?.message || err))) {
          console.warn(`[lod] texture tile ${key} failed:`, err);
        }
      });
  }

  /**
   * Re-apply per-tile textures after a global texture/colormap change resets
   * them. Tiles whose colormap mode is back to RGB will re-request lazily.
   */
  invalidateTileTextures() {
    for (const [, material] of this.tileMaterials.entries()) {
      material.uniforms.uPerTileTexture.value = 0.0;
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
    for (const [, m] of this.tileMaterials.entries()) {
      m.dispose();
    }
    this.tileMaterials.clear();
    for (const [, tex] of this._tileTextureObjects.entries()) {
      tex.dispose();
    }
    this._tileTextureObjects.clear();
    this.pendingBuilds.clear();
    this.root.dispose();
  }
}
