p = 'frontend/src/engine/TerrainEngine.js'
s = open(p, encoding='utf-8').read()

# 1. constructor: meshSpacing + remove textureTiles pass (moved maxLod up)
old = '''    // 3. Create Terrain Material (base/template — every tile gets a clone).
    //    uMeshDensity is seeded with the tile segment count so the wireframe
    //    grid overlay traces each chunk's real quad grid.
    this.segments = 32;
    this.material = createTerrainMaterial({
      texture: options.diffuseTexture,
      minElevation: this.geoRef.minElevation,
      maxElevation: this.geoRef.maxElevation,
      exaggeration: 1.0,
      colormapMode: 0.0,
      meshEnabled: false,
      segments: this.segments,
    });'''
new = '''    // 3. Create THE shared terrain material — every tile uses this same
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
    });'''
assert old in s, "ctor material block"
s = s.replace(old, new)

# 2. LODManager construction: drop textureTiles + duplicate maxLod calc
old = '''    // 6. Initialize Quadtree LOD Manager. maxLod follows the raster: the
    //    finest quadtree level tiles the raster into ~256-px cells, so a
    //    2540-px raster refines 4 levels deep (level-4 tiles are ~10 m
    //    across at GSD 1 m with 32 segments ≈ 0.3 m mesh resolution).
    const maxLod = Math.min(
      4,
      Math.max(
        2,
        Math.ceil(Math.log2(Math.max(this.geoRef.rasterWidth, this.geoRef.rasterHeight) / 256)),
      ),
    );
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
      textureTiles: this.textureStreamer
        ? {
            streamer: this.textureStreamer,
            tileSize: options.textureTiles.tileSize ?? 256,
            maxLevel: options.textureTiles.maxLevel ?? 3,
            anisotropy: options.textureTiles.anisotropy,
          }
        : null,
    });'''
new = '''    // 6. Initialize Quadtree LOD Manager. maxLod follows the raster: the
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
    });'''
assert old in s, "lodmanager ctor"
s = s.replace(old, new)

# 3. streamers
old = '''    this._tileFetchHeaders = options.tileFetchHeaders ?? null;
    this.heightStreamer = this._createHeightStreamer(options.heightTiles);
    this.textureStreamer = this._createTextureStreamer(options.textureTiles);'''
new = '''    this._tileFetchHeaders = options.tileFetchHeaders ?? null;
    this.heightStreamer = this._createHeightStreamer(options.heightTiles);'''
assert old in s, "streamers"
s = s.replace(old, new)

old = '''  _createTextureStreamer(config) {
    if (!config?.url) return null;
    const streamer = new TileStreamer({
      fetchTile: async ({ z, x, y, size }, signal) => {
        // Authenticated backend endpoint (never object storage) — the JWT
        // belongs here.
        const headers = typeof this._tileFetchHeaders === 'function'
          ? await this._tileFetchHeaders() : (this._tileFetchHeaders || {});
        const res = await fetch(config.url(z, x, y, size), { signal, headers });
        if (!res.ok) throw new Error(`texture tile fetch failed: ${res.status}`);
        const blob = await res.blob();
        return createImageBitmap(blob);
      },
      maxCacheEntries: 96,
    });
    return streamer;
  }

'''
assert old in s, "texture streamer fn"
s = s.replace(old, '')

# 4. setSolidView / setHybridView
old = '''  setSolidView(wireframe = false) {
    this._applyToMaterials((u) => {
      u.uTextureReady.value = 0.0;
      u.uColormapMode.value = 0.0;
      u.uReliefStrength.value = 0.0;
      u.uMeshEnabled.value = wireframe ? 1.0 : 0.0;
      u.uPerTileTexture.value = 0.0;
    });
  }'''
new = '''  setSolidView(wireframe = false) {
    const u = this.material.uniforms;
    u.uTextureReady.value = 0.0;
    u.uColormapMode.value = 0.0;
    u.uReliefStrength.value = 0.0;
    u.uMeshEnabled.value = wireframe ? 1.0 : 0.0;
  }'''
assert old in s, "setSolidView"
s = s.replace(old, new)

old = '''  setHybridView(wireframe = false) {
    this._applyToMaterials((u) => {
      u.uTextureReady.value = this.textureReady ? 1.0 : u.uTextureReady.value;
      u.uColormapMode.value = 0.0;
      u.uReliefStrength.value = 1.0;
      u.uMeshEnabled.value = wireframe ? 1.0 : 0.0;
      u.uPerTileTexture.value = 0.0;
    });
  }'''
new = '''  setHybridView(wireframe = false) {
    const u = this.material.uniforms;
    u.uTextureReady.value = this.textureReady ? 1.0 : u.uTextureReady.value;
    u.uColormapMode.value = 0.0;
    u.uReliefStrength.value = 1.0;
    u.uMeshEnabled.value = wireframe ? 1.0 : 0.0;
  }'''
assert old in s, "setHybridView"
s = s.replace(old, new)

# 5. setTexture + enablePerTileTextures
old = '''  setTexture(texture, opts = {}) {
    if (!texture) return;
    if (opts.override) {
      this.perTileTexturesEnabled = false;
    }
    this.textureReady = true;
    this._applyToMaterials((u) => {
      u.uTexture.value = texture;
      u.uTextureReady.value = 1.0;
      u.uPerTileTexture.value = 0.0;
    });
    this.material.needsUpdate = true;
  }

  /**
   * Re-enable per-tile streamed imagery (e.g. after a user layer is closed).
   */
  enablePerTileTextures() {
    this.perTileTexturesEnabled = true;
    this._applyToMaterials((u) => {
      u.uPerTileTexture.value = 0.0;
    });
    // Active tiles re-fetch lazily on their next add cycle
    this.lodManager.invalidateTileTextures();
  }'''
new = '''  setTexture(texture) {
    if (!texture) return;
    this.textureReady = true;
    const u = this.material.uniforms;
    u.uTexture.value = texture;
    u.uTextureReady.value = 1.0;
    this.material.needsUpdate = true;
  }'''
assert old in s, "setTexture"
s = s.replace(old, new)

# 6. setColormapMode
old = '''  setColormapMode(mode) {
    const m = Number(mode) || 0.0;
    this._applyToMaterials((u) => {
      u.uColormapMode.value = m;
      // Colormaps sample the global texture; leave uPerTileTexture reset to
      // the global path whenever a colormap is active.
      if (m !== 0) {
        u.uPerTileTexture.value = 0.0;
      }
    });
  }'''
new = '''  setColormapMode(mode) {
    this.material.uniforms.uColormapMode.value = Number(mode) || 0.0;
  }'''
assert old in s, "setColormapMode"
s = s.replace(old, new)

# 7. setters
old = '''  setExaggeration(factor) {
    this.exaggeration = Math.max(0.1, Number(factor) || 1.0);
    this._applyToMaterials((u) => {
      if (u.uExaggeration) u.uExaggeration.value = this.exaggeration;
    });
  }'''
new = '''  setExaggeration(factor) {
    this.exaggeration = Math.max(0.1, Number(factor) || 1.0);
    this.material.uniforms.uExaggeration.value = this.exaggeration;
  }'''
assert old in s, "setExaggeration"
s = s.replace(old, new)

old = '''  setWireframe(enabled) {
    const on = !!enabled;
    this._applyToMaterials((u) => {
      if (u.uMeshEnabled) u.uMeshEnabled.value = on ? 1.0 : 0.0;
    });
  }'''
new = '''  setWireframe(enabled) {
    this.material.uniforms.uMeshEnabled.value = enabled ? 1.0 : 0.0;
  }'''
assert old in s, "setWireframe"
s = s.replace(old, new)

old = '''  setContours(enabled, interval = 5.0) {
    this._applyToMaterials((u) => {
      u.uContoursEnabled.value = enabled ? 1.0 : 0.0;
      if (interval > 0) {
        u.uContourInterval.value = interval;
      }
    });
  }'''
new = '''  setContours(enabled, interval = 5.0) {
    const u = this.material.uniforms;
    u.uContoursEnabled.value = enabled ? 1.0 : 0.0;
    if (interval > 0) {
      u.uContourInterval.value = interval;
    }
  }'''
assert old in s, "setContours"
s = s.replace(old, new)

old = '''  setFog(enabled, near = null, far = null, color = null) {
    this._applyToMaterials((u) => {
      u.uFogEnabled.value = enabled ? 1.0 : 0.0;
      const diag = Math.hypot(this.geoRef.worldWidth, this.geoRef.worldDepth);
      u.uFogNear.value = near ?? 0.6 * diag;
      u.uFogFar.value = far ?? 3.0 * diag;
      if (color) {
        u.uFogColor.value.copy(color);
      }
    });
  }'''
new = '''  setFog(enabled, near = null, far = null, color = null) {
    const u = this.material.uniforms;
    u.uFogEnabled.value = enabled ? 1.0 : 0.0;
    const diag = Math.hypot(this.geoRef.worldWidth, this.geoRef.worldDepth);
    u.uFogNear.value = near ?? 0.6 * diag;
    u.uFogFar.value = far ?? 3.0 * diag;
    if (color) {
      u.uFogColor.value.copy(color);
    }
  }'''
assert old in s, "setFog"
s = s.replace(old, new)

# 8. setSemanticOverlay
old = '''    this._applyToMaterials((u) => {
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
    });
    this.material.needsUpdate = true;
  }'''
new = '''    const u = this.material.uniforms;
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
  }'''
assert old in s, "setSemanticOverlay"
s = s.replace(old, new)

# 9. telemetry
old = '''      heightStreaming: !!this.heightStreamer,
      textureStreaming: !!this.textureStreamer && this.perTileTexturesEnabled,
      heightStreamCache: this.heightStreamer?.stats ?? null,
      textureStreamCache: this.textureStreamer?.stats ?? null,'''
new = '''      heightStreaming: !!this.heightStreamer,
      heightStreamCache: this.heightStreamer?.stats ?? null,'''
assert old in s, "telemetry"
s = s.replace(old, new)

# 10. dispose
old = '''    this.disposed = true;
    this.lodManager.dispose();
    this.heightStreamer?.dispose();
    this.textureStreamer?.dispose();
    this.material.dispose();'''
new = '''    this.disposed = true;
    this.lodManager.dispose();
    this.heightStreamer?.dispose();
    this.material.dispose();'''
assert old in s, "dispose"
s = s.replace(old, new)

# 11. _applyToMaterials
old = '''  /**
   * Apply a uniform mutation to the base material and every live tile clone.
   *
   * @param {Function} fn - (uniforms) => void
   */
  _applyToMaterials(fn) {
    if (this.material?.uniforms) fn(this.material.uniforms);
    if (this.lodManager) {
      for (const [, m] of this.lodManager.tileMaterials.entries()) {
        if (m.uniforms) fn(m.uniforms);
      }
    }
  }

'''
assert old in s, "_applyToMaterials"
s = s.replace(old, '')

# 12. perTileTexturesEnabled init
old = '''    this.exaggeration = 1.0;
    // True once an RGB/layer texture has been applied (drives hybrid view)
    this.textureReady = false;
    // Per-tile streamed imagery applies only while no user-selected layer
    // (colormap) overrides the base RGB texture.
    this.perTileTexturesEnabled = true;
    this.disposed = false;'''
new = '''    this.exaggeration = 1.0;
    // True once an RGB/layer texture has been applied (drives hybrid view)
    this.textureReady = false;
    this.disposed = false;'''
assert old in s, "perTile init"
s = s.replace(old, new)

open(p, 'w', encoding='utf-8').write(s)
print("TerrainEngine simplified OK")
