/**
 * DepthWizard Geospatial Engine — TerrainEngine
 *
 * Primary coordinator uniting GeoReference, TerrainDataset, ChunkGeometry,
 * LODManager, TerrainCollision, and SpatialModel into a cohesive metric terrain engine.
 *
 * Streaming (optional, enabled by the caller via heightTiles/textureTiles):
 *   - heightTiles:  { url: (z,x,y,size) => string, tileSize } — async quadtree
 *     height streaming for very large rasters; while tiles load, ready
 *     ancestors are rendered (no holes).
 *   - textureTiles: { url: (z,x,y,size) => string, tileSize, maxLevel } —
 *     per-tile imagery with a texture LOD independent of geometry LOD.
 * All global visual settings (texture, colormap, contours, fog, semantic
 * overlay, exaggeration, wireframe) propagate to every tile material.
 */

import * as THREE from 'three';
import { GeoReference } from './geo/GeoReference.js';
import { TerrainDataset } from './terrain/TerrainDataset.js';
import { createTerrainMaterial } from './terrain/TerrainMaterial.js';
import { LODManager } from './lod/LODManager.js';
import { TerrainCollision } from './camera/TerrainCollision.js';
import { SpatialModel } from './spatial/SpatialModel.js';
import { TileStreamer } from './streaming/TileStreamer.js';
import { decodeHeightPng16, decodeHeightViaCanvasBytes, despikeHeightfield } from './streaming/heightDecode.js';

export class TerrainEngine {
  /**
   * @param {Object} options
   * @param {THREE.Scene} options.scene
   * @param {THREE.Camera} options.camera
   * @param {Object} options.terrainMeta - Backend terrain API response
   * @param {Float32Array} options.heightData - 16-bit decoded or metric float height array
   * @param {number} options.hmWidth - Height array width
   * @param {number} options.hmHeight - Height array height
   * @param {THREE.Texture} [options.diffuseTexture] - Initial RGB texture
   * @param {Object|null} [options.heightTiles=null] - Height tile streaming config
   * @param {Object|null} [options.textureTiles=null] - Texture tile streaming config
   * @param {Object|Function} [options.tileFetchHeaders] - Auth headers (or
   *   async provider) attached to every tile fetch. Tile URLs
   *   (heightTiles.url / textureTiles.url) always point at authenticated
   *   FastAPI quadtree endpoints — never at object storage — so this is
   *   the one place in the streaming path where attaching the Supabase
   *   JWT is correct. It must never be reused for a storage/asset fetch;
   *   see api/client.js::assetFetch for those.
   */
  constructor(options) {
    this.scene = options.scene;
    this.camera = options.camera;

    // 1. Establish Geospatial Reference
    this.geoRef = new GeoReference({
      crs: options.terrainMeta.crs,
      projectedCrs: options.terrainMeta.projected_crs,
      affineTransform: options.terrainMeta.affine_transform,
      gsdX: options.terrainMeta.gsd_x,
      gsdY: options.terrainMeta.gsd_y,
      rasterWidth: options.terrainMeta.raster_width || options.hmWidth,
      rasterHeight: options.terrainMeta.raster_height || options.hmHeight,
      worldWidth: options.terrainMeta.world_width_m,
      worldDepth: options.terrainMeta.world_depth_m,
      isGeoreferenced: options.terrainMeta.is_georeferenced_scale,
      localOrigin: options.terrainMeta.local_origin,
      minElevation: options.terrainMeta.min_elevation,
      maxElevation: options.terrainMeta.max_elevation,
      elevationMode: options.terrainMeta.terrain?.elevation_mode || 'relative',
    });

    // 2. Instantiate Terrain Dataset
    this.dataset = new TerrainDataset({
      heightData: options.heightData,
      width: options.hmWidth,
      height: options.hmHeight,
      minElevation: this.geoRef.minElevation,
      maxElevation: this.geoRef.maxElevation,
      isNormalized: true,
      geoRef: this.geoRef,
    });

    // 3. Create THE shared terrain material — every tile uses this same
    //    instance (UVs are global; one full-resolution drape texture), so
    //    there are no per-tile clones to track or dispose.
    this.segments = 32;
    const maxLod = Math.min(
      4,
      Math.max(
        2,
        Math.ceil(Math.log2(Math.max(this.geoRef.rasterWidth, this.geoRef.rasterHeight) / 256)),
      ),
    );
    // Wireframe metric-grid spacing matches the finest quadtree cell.
    const finestTile = Math.max(this.geoRef.worldWidth, this.geoRef.worldDepth) / (2 ** maxLod);
    this.material = createTerrainMaterial({
      texture: options.diffuseTexture,
      minElevation: this.geoRef.minElevation,
      maxElevation: this.geoRef.maxElevation,
      exaggeration: 1.0,
      colormapMode: 0.0,
      meshEnabled: false,
      meshSpacing: Math.max(1, finestTile / this.segments),
    });

    // 4. Group for holding chunk meshes in scene
    this.terrainGroup = new THREE.Group();
    this.terrainGroup.name = 'TerrainEngine_Chunks';
    this.scene.add(this.terrainGroup);

    // 5. Tile streamers (transport + LRU cache), created only when the
    //    caller supplies tile URL builders.
    this._tileFetchHeaders = options.tileFetchHeaders ?? null;
    this.heightStreamer = this._createHeightStreamer(options.heightTiles);

    // 6. Initialize Quadtree LOD Manager. maxLod follows the raster: the
    //    finest quadtree level tiles the raster into ~256-px cells, so a
    //    2540-px raster refines 4 levels deep (level-4 tiles are ~10 m
    //    across at GSD 1 m with 32 segments ≈ 0.3 m mesh resolution).
    this.lodManager = new LODManager({
      geoRef: this.geoRef,
      dataset: this.dataset,
      material: this.material,
      scene: this.terrainGroup,
      maxLod,
      splitThreshold: 0.85,
      segments: 32,
      heightTiles: this.heightStreamer
        ? { streamer: this.heightStreamer, tileSize: options.heightTiles.tileSize ?? 256 }
        : null,
    });

    // 7. Camera Collision Engine
    this.collision = new TerrainCollision({
      geoRef: this.geoRef,
      dataset: this.dataset,
      eyeHeight: 1.7,
      minClearance: 1.5,
    });

    // 8. Spatial Query Model
    this.spatial = new SpatialModel({
      geoRef: this.geoRef,
      dataset: this.dataset,
      collision: this.collision,
    });

    this.exaggeration = 1.0;
    // True once an RGB/layer texture has been applied (drives hybrid view)
    this.textureReady = false;
    this.disposed = false;

    // Initial LOD evaluation
    if (this.camera) {
      this.lodManager.update(this.camera);
    }
  }

  _createHeightStreamer(config) {
    if (!config?.url) return null;
    const streamer = new TileStreamer({
      fetchTile: async ({ z, x, y, size }, signal) => {
        // Authenticated backend endpoint (never object storage) — the JWT
        // belongs here.
        const headers = typeof this._tileFetchHeaders === 'function'
          ? await this._tileFetchHeaders() : (this._tileFetchHeaders || {});
        const res = await fetch(config.url(z, x, y, size), { signal, headers });
        if (!res.ok) throw new Error(`height tile fetch failed: ${res.status}`);
        const buf = await res.arrayBuffer();
        const decoded = (await decodeHeightPng16(buf)) || (await decodeHeightViaCanvasBytes(buf));
        // Rendering-quality safety net: kill isolated 1-px height needles
        // in the streamed patch (mirrors the full-heightmap decode path).
        despikeHeightfield(decoded.data, decoded.width, decoded.height);
        return decoded;
      },
      maxCacheEntries: 96,
    });
    return streamer;
  }

  /**
   * Per-frame update loop.
   *
   * @param {THREE.Camera} camera
   */
  update(camera) {
    if (this.disposed) return;
    const cam = camera || this.camera;
    if (cam) {
      this.lodManager.update(cam);
    }
  }

  /**
   * Update vertical exaggeration.
   *
   * @param {number} factor - 1.0 = true metric scale
   */
  setExaggeration(factor) {
    this.exaggeration = Math.max(0.1, Number(factor) || 1.0);
    this.material.uniforms.uExaggeration.value = this.exaggeration;
  }

  /**
   * Toggle wireframe display.
   *
   * SINGLE wireframe system: the shader's per-tile mesh-grid overlay
   * (uMeshEnabled). Brute-force material.wireframe is never used — it
   * draws every LOD triangle including skirt walls as giant stretched
   * lines, which is exactly the visual failure this replaces.
   */
  setWireframe(enabled) {
    this.material.uniforms.uMeshEnabled.value = enabled ? 1.0 : 0.0;
  }

  /**
   * Solid relief view: no imagery, no colormap — flat shaded geometry.
   *
   * @param {boolean} [wireframe] - Keep the current wireframe overlay state
   */
  setSolidView(wireframe = false) {
    const u = this.material.uniforms;
    u.uTextureReady.value = 0.0;
    u.uColormapMode.value = 0.0;
    u.uReliefStrength.value = 0.0;
    u.uMeshEnabled.value = wireframe ? 1.0 : 0.0;
  }

  /**
   * Hybrid view: photographic RGB imagery + enhanced terrain relief.
   * Requires an RGB texture to be set (uTextureReady = 1); combines the
   * source imagery with stronger sun shading, slope shadowing and a
   * gentle elevation modulation so relief reads without destroying the
   * imagery's colors.
   */
  setHybridView(wireframe = false) {
    const u = this.material.uniforms;
    u.uTextureReady.value = this.textureReady ? 1.0 : u.uTextureReady.value;
    u.uColormapMode.value = 0.0;
    u.uReliefStrength.value = 1.0;
    u.uMeshEnabled.value = wireframe ? 1.0 : 0.0;
  }

  /**
   * Update base diffuse texture.
   *
   * @param {THREE.Texture} texture
   * @param {Object} [opts]
   * @param {boolean} [opts.override] - True when a user-selected layer replaces
   *   base imagery; disables per-tile streamed textures until invalidated.
   */
  setTexture(texture) {
    if (!texture) return;
    this.textureReady = true;
    const u = this.material.uniforms;
    u.uTexture.value = texture;
    u.uTextureReady.value = 1.0;
    this.material.needsUpdate = true;
  }

  /**
   * Update colormap mode (0=rgb, 1=greyscale, 2=viridis, 3=diverging).
   */
  setColormapMode(mode) {
    this.material.uniforms.uColormapMode.value = Number(mode) || 0.0;
  }

  /**
   * Configure elevation contour lines.
   */
  setContours(enabled, interval = 5.0) {
    const u = this.material.uniforms;
    u.uContoursEnabled.value = enabled ? 1.0 : 0.0;
    if (interval > 0) {
      u.uContourInterval.value = interval;
    }
  }

  /**
   * Configure atmospheric depth fog.
   */
  setFog(enabled, near = null, far = null, color = null) {
    const u = this.material.uniforms;
    u.uFogEnabled.value = enabled ? 1.0 : 0.0;
    const diag = Math.hypot(this.geoRef.worldWidth, this.geoRef.worldDepth);
    u.uFogNear.value = near ?? 0.6 * diag;
    u.uFogFar.value = far ?? 3.0 * diag;
    if (color) {
      u.uFogColor.value.copy(color);
    }
  }

  /**
   * Update semantic mask texture and styling.
   *
   * @param {Object} options
   * @param {THREE.DataTexture} [options.texture] - RedFormat uint8 class-ID mask
   * @param {boolean} [options.enabled]
   * @param {number} [options.opacity]
   * @param {number} [options.highlightClass] - -1 disables class isolation
   * @param {THREE.DataTexture} [options.confidenceTexture] - RedFormat [0,1]
   *   confidence mask; when enabled, low-confidence pixels attenuate the
   *   overlay back toward the base layer.
   * @param {boolean} [options.confidenceEnabled]
   */
  setSemanticOverlay(options = {}) {
    const {
      texture, enabled, opacity, highlightClass,
      confidenceTexture, confidenceEnabled,
    } = options;
    const u = this.material.uniforms;
    if (texture) {
      u.uSemanticTex.value = texture;
    }
    if (confidenceTexture) {
      u.uSemanticConfTex.value = confidenceTexture;
    }
    if (typeof confidenceEnabled === 'boolean') {
      u.uSemanticConfEnabled.value = confidenceEnabled ? 1.0 : 0.0;
    }
    if (typeof enabled === 'boolean') {
      u.uSemanticEnabled.value = enabled ? 1.0 : 0.0;
    }
    if (typeof opacity === 'number') {
      u.uSemanticOpacity.value = Math.max(0, Math.min(1, opacity));
    }
    if (typeof highlightClass === 'number') {
      u.uSemanticHighlightClass.value = highlightClass;
    }
    this.material.needsUpdate = true;
  }

  /**
   * Raycast against all currently active chunk meshes.
   *
   * @param {THREE.Raycaster} raycaster
   * @returns {Object|null} Intersection info with elevation, slope, normal, and world coordinates
   */
  raycast(raycaster) {
    const activeMeshes = [];
    for (const [, tile] of this.lodManager.activeTiles.entries()) {
      if (tile.mesh) activeMeshes.push(tile.mesh);
    }

    const hits = raycaster.intersectObjects(activeMeshes, false);
    if (!hits || hits.length === 0) return null;

    const hit = hits[0];
    const wx = hit.point.x;
    const wz = hit.point.z;
    const { u, v } = this.geoRef.localToUv(wx, wz);

    const elevation = this.spatial.sampleElevation(wx, wz);
    const slope = this.spatial.sampleSlope(wx, wz);
    const normal = this.spatial.sampleNormal(wx, wz);

    return {
      point: hit.point,
      worldX: wx,
      worldZ: wz,
      elevation,
      slope,
      normal,
      u,
      v,
      tile: hit.object.userData?.tile || null,
    };
  }

  /**
   * Query performance and memory telemetry.
   */
  getTelemetry() {
    return {
      ...this.lodManager.stats,
      worldWidth_m: this.geoRef.worldWidth,
      worldDepth_m: this.geoRef.worldDepth,
      gsd_x: this.geoRef.gsdX,
      gsd_y: this.geoRef.gsdY,
      isGeoreferenced: this.geoRef.isGeoreferenced,
      crs: this.geoRef.crs,
      heightStreaming: !!this.heightStreamer,
      heightStreamCache: this.heightStreamer?.stats ?? null,
    };
  }

  /**
   * Clean up all GPU and CPU resources.
   */
  dispose() {
    this.disposed = true;
    this.lodManager.dispose();
    this.heightStreamer?.dispose();
    this.material.dispose();
    if (this.terrainGroup.parent) {
      this.terrainGroup.parent.remove(this.terrainGroup);
    }
  }
}
