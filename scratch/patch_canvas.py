p = 'frontend/src/components/TerrainViewer/TerrainCanvas.jsx'
s = open(p, encoding='utf-8').read()

# 1. thresholds comment/constants (already edited HEIGHT gate; fix streamTextures var usage)
old = '''    const streamHeights = !!(tileUrls && rasterMaxPx > HEIGHT_STREAM_THRESHOLD_PX);
    const streamTextures = !!(tileUrls && rasterMaxPx > TEXTURE_STREAM_THRESHOLD_PX);'''
new = '''    const streamHeights = !!(tileUrls && rasterMaxPx > HEIGHT_STREAM_THRESHOLD_PX);'''
assert old in s, "stream vars"
s = s.replace(old, new)

# 2. loadAuthTexture doc mentions textures; keep as-is. Engine config: drop textureTiles
old = '''    const engine = new TerrainEngine({
      scene,
      camera: g.camera,
      terrainMeta,
      heightData: data,
      hmWidth: width,
      hmHeight: height,
      heightTiles: streamHeights && tileUrls
        ? { url: tileUrls.height, tileSize: tileConfig.tile_size || 256 }
        : null,
      textureTiles: streamTextures && tileUrls
        ? {
            url: tileUrls.texture,
            tileSize: tileConfig.tile_size || 256,
            maxLevel: tileConfig.max_lod ?? 3,
          }
        : null,'''
new = '''    const engine = new TerrainEngine({
      scene,
      camera: g.camera,
      terrainMeta,
      heightData: data,
      hmWidth: width,
      hmHeight: height,
      heightTiles: streamHeights && tileUrls
        ? { url: tileUrls.height, tileSize: tileConfig.tile_size || 256 }
        : null,'''
assert old in s, "engine cfg"
s = s.replace(old, new)

# 3. setTexture calls: drop override opts
old = '''          // Selecting the RGB layer itself should refresh the resident RGB
          // drape used by hybrid view (and dispose the stale one).
          const isRgbLayer =
            g.rgbTextureUrl && resolveAssetUrl(url) === resolveAssetUrl(g.rgbTextureUrl);
          if (isRgbLayer) {
            if (g.rgbTexture && g.rgbTexture !== tex) g.rgbTexture.dispose();
            g.rgbTexture = tex;
          }
          // A user-selected layer overrides base imagery: disable per-tile
          // streamed textures so the layer samples global UVs correctly.
          g.engine.setTexture(tex, { override: true });
          // The previous override texture is no longer referenced by any
          // material — free its GPU memory (never dispose the base RGB
          // drape or engine-managed streamed textures).
          if (g.overrideTexture && g.overrideTexture !== tex) {
            g.overrideTexture.dispose();
          }
          g.overrideTexture = tex;
          g.engine.setWireframe(g.wireframe);'''
new = '''          // Selecting the RGB layer itself should refresh the resident RGB
          // drape used by hybrid view (and dispose the stale one).
          const isRgbLayer =
            g.rgbTextureUrl && resolveAssetUrl(url) === resolveAssetUrl(g.rgbTextureUrl);
          if (isRgbLayer) {
            if (g.rgbTexture && g.rgbTexture !== tex) g.rgbTexture.dispose();
            g.rgbTexture = tex;
          }
          // The previous override texture is no longer referenced by the
          // shared material — free its GPU memory.
          if (g.overrideTexture && g.overrideTexture !== tex) {
            g.overrideTexture.dispose();
          }
          g.overrideTexture = tex;
          g.engine.setTexture(tex);
          g.engine.setWireframe(g.wireframe);'''
assert old in s, "setLayerTexture"
s = s.replace(old, new)

# 4. hybrid: setTexture without override
old = '''      if (g.rgbTexture) {
        g.engine.setTexture(g.rgbTexture, { override: false });
        g.overrideTexture = null;
        applyRelief();
      } else if (g.rgbTextureUrl) {
        loadAuthTexture(resolveAssetUrl(g.rgbTextureUrl))
          .then((tex) => {
            if (g.disposed) { tex.dispose(); return; }
            const maxAniso = g.renderer?.capabilities?.getMaxAnisotropy?.() || 8;
            tex.anisotropy = Math.min(16, maxAniso);
            g.rgbTexture = tex;
            g.engine.setTexture(tex, { override: false });
            applyRelief();
          })'''
new = '''      if (g.rgbTexture) {
        g.engine.setTexture(g.rgbTexture);
        g.overrideTexture = null;
        applyRelief();
      } else if (g.rgbTextureUrl) {
        loadAuthTexture(resolveAssetUrl(g.rgbTextureUrl))
          .then((tex) => {
            if (g.disposed) { tex.dispose(); return; }
            const maxAniso = g.renderer?.capabilities?.getMaxAnisotropy?.() || 8;
            tex.anisotropy = Math.min(16, maxAniso);
            g.rgbTexture = tex;
            g.engine.setTexture(tex);
            applyRelief();
          })'''
assert old in s, "hybrid"
s = s.replace(old, new)

# 5. initial diffuse load: drop aniso dup is fine; keep
# 6. getTelemetry: textureStreaming field gone
old = '''      return {
        ...base,
        drawCalls: info?.render?.calls ?? 0,
        textureCount: info?.memory?.textures ?? 0,
        geometries: info?.memory?.geometries ?? 0,
        programs: info?.programs?.length ?? 0,
        fps: g.lastFps ?? null,
      };'''
new = '''      return {
        ...base,
        drawCalls: info?.render?.calls ?? 0,
        textureCount: info?.memory?.textures ?? 0,
        geometries: info?.memory?.geometries ?? 0,
        programs: info?.programs?.length ?? 0,
      };'''
assert old in s, "telemetry"
s = s.replace(old, new)

# 7. scene-change disposal: keep overrideTexture/rgbTexture; semanticTextures fine
open(p, 'w', encoding='utf-8').write(s)
print("TerrainCanvas OK")
