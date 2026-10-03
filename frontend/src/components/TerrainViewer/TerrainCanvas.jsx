/**
 * DepthWizard — TerrainCanvas (Geospatial Metric Terrain Engine)
 *
 * Full-bleed 3D viewport powered by the Geospatial Terrain Engine:
 *   - 1 Three.js world unit = 1 meter
 *   - Continuous Quadtree LOD & distance-based chunk refinement
 *   - Perimeter skirt geometry sealing all inter-tile seams and cracks
 *   - Metric camera navigation with ground collision & terrain following
 *   - Real-world meters for elevation, slope, distances, and routing
 *   - Dynamic atmospheric fog, metric contour lines, and colormaps
 *   - Independent semantic mask layer
 *   - Real-time developer & mission-control telemetry HUD
 */

import { useEffect, useRef, useState, useImperativeHandle, forwardRef, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useApp } from '../../store/appStore.jsx';
import { getTerrain } from '../../api/terrain.js';
import { resolveAssetUrl, authHeaders, assetFetch } from '../../api/client.js';
import { TerrainEngine } from '../../engine/TerrainEngine.js';
import TerrainDebugHUD from '../../engine/debug/TerrainDebugHUD.jsx';
import { decodeHeightPng16 } from '../../engine/streaming/heightDecode.js';
import { decodeHeightmap } from '../../engine/streaming/heightFetch.js';
import { BuildingsLayer } from '../../engine/buildings/BuildingsLayer.js';
import { getBuildings3D } from '../../api/buildings3d.js';
import { overviewToHeightfield } from '../../engine/streaming/PatchHeightfield.js';

/* ─── Streaming thresholds ─────────────────────────────────────────────────
 * Inline heightmap: decoded once into a Float32Array (4 bytes/sample). Fine
 * up to ~2048×2048 (16 MB); beyond that heights stream from the backend's
 * quadtree tile endpoints with a coarse overview for global queries (§29).
 * The 2048 gate matches the texture-streaming threshold: a 0.5 m urban DSM
 * downsampled to 1024 px loses building-scale geometry entirely, which read
 * as smooth "melted" blobs the moment the camera flew close. Per-tile
 * streaming keeps full source resolution at every LOD.
 * Per-tile textures are streamed when the raster is large enough for the
 * single global texture to be a memory/quality concern (§17). */
const HEIGHT_STREAM_THRESHOLD_PX = 4096;
const OVERVIEW_SIZE_PX = 1024;

/** Build tile URL callbacks from the backend tile_config (fractional-quadtree
 * convention: z = level, 0 <= x,y < 2^z, row 0 = north). */
function buildTileUrlFns(tileConfig) {
  if (!tileConfig?.height_tile_url) return null;
  return {
    height: (z, x, y, size) =>
      resolveAssetUrl(`${tileConfig.height_tile_url}?x=${x}&y=${y}&z=${z}&size=${size}`),
  };
}

/** Fetch the z=0 overview tile (whole raster, downsampled) and convert it to
 * a rectangular normalized heightfield for global spatial queries. This is
 * always an authenticated FastAPI quadtree endpoint (tileConfig.height_tile_url
 * is built server-side as a backend route, never a presigned storage URL),
 * so attaching the JWT here is correct — unlike the storage-asset fetches
 * below (loadAuthTexture, decodeHeightmap), which must not. */
async function fetchOverviewHeightfield(urlFn, rasterWidth, rasterHeight) {
  const res = await fetch(urlFn(0, 0, 0, OVERVIEW_SIZE_PX), { headers: await authHeaders() });
  if (!res.ok) throw new Error(`overview tile fetch failed: ${res.status}`);
  const buf = await res.arrayBuffer();
  const decoded = await decodeHeightPng16(buf);
  if (!decoded) throw new Error('overview tile is not a 16-bit grayscale PNG');
  return overviewToHeightfield({
    data: decoded.data,
    size: decoded.width,
    rasterWidth,
    rasterHeight,
    maxEdge: OVERVIEW_SIZE_PX,
  });
}

/** Load a result-texture URL into a THREE texture via fetch (TextureLoader
 * cannot send headers anyway). `url` is a storage-asset URL — it may
 * resolve to a presigned S3/RustFS URL or the legacy backend file route — so
 * this goes through assetFetch(), which decides per-request whether the
 * JWT belongs on it, rather than always attaching it.
 *
 * `colormapMode` 0 = photographic RGB drape (sRGB-encoded: GPU decodes to
 * linear, the shader re-encodes after lighting). Any other mode = a RAW
 * DATA layer (greyscale depth/DSM/slope/error): the shader colormaps the
 * exact stored value, so the texture must carry NoColorSpace — an sRGB
 * decode here would silently skew every value layer's normalization. */
async function loadAuthTexture(url, { rawData = false } = {}) {
  const res = await assetFetch(url);
  if (!res.ok) throw new Error(`texture fetch failed: ${res.status}`);
  const bitmap = await createImageBitmap(await res.blob());
  const tex = new THREE.CanvasTexture(bitmap);
  tex.colorSpace = rawData ? THREE.NoColorSpace : THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/* ─── R3F Inner Bridge Component ────────────────────────────────────────── */

function SceneBridge({ canvasRef, glRef, orbitControlsRef, sceneState, actions, onEngineReady }) {
  const { gl, scene, camera } = useThree();

  useEffect(() => {
    canvasRef.current = gl.domElement;
    const g = glRef.current;
    g.renderer = gl;
    g.scene = scene;
    g.camera = camera;
    g.orbit = orbitControlsRef.current;
    g.controls = orbitControlsRef.current;
  }, [gl, scene, camera, canvasRef, glRef, orbitControlsRef]);

  // Handle terrain loading whenever scene_id changes
  useEffect(() => {
    const sceneId = sceneState?.scene?.scene_id;
    if (!sceneId) return;

    const g = glRef.current;
    g.disposed = false;
    loadTerrainData(g, scene, sceneId, actions, onEngineReady);

    return () => {
      g.disposed = true;
      if (g.engine) {
        g.engine.dispose();
        g.engine = null;
      }
      if (g.measureGroup) {
        scene.remove(g.measureGroup);
        g.measureGroup = null;
      }
      if (g.routeGroup) {
        scene.remove(g.routeGroup);
        g.routeGroup = null;
      }
      if (g.buildingsLayer) {
        g.buildingsLayer.dispose();
        g.buildingsLayer = null;
      }
    };
  }, [sceneState?.scene?.scene_id, scene, actions, glRef, onEngineReady]);

  // Frame tick animation loop
  useFrame((_, delta) => {
    const g = glRef.current;
    if (g.disposed) return;
    const dt = Math.min(delta, 0.1);

    if (!g.orbit && orbitControlsRef.current) {
      g.orbit = orbitControlsRef.current;
      g.controls = orbitControlsRef.current;
    }

    // 1. Update Geospatial Terrain Engine (Quadtree LOD & Frustum Culling)
    if (g.engine) {
      g.engine.update(camera, dt);
      // Synchronize g.tiles array with currently active chunk meshes for backward compatibility
      const activeMeshes = [];
      for (const [, tile] of g.engine.lodManager.activeTiles.entries()) {
        if (tile.mesh) activeMeshes.push(tile.mesh);
      }
      g.tiles = activeMeshes;
      g.mesh = activeMeshes[0] || null;
    }

    // 2. Animate Route Assist landing pin / pulse
    const pulse = g.routeGroup?.userData?.pulse;
    if (pulse?.pin) {
      pulse.t += dt * 3.2;
      const s = 1.0 + 0.25 * Math.sin(pulse.t);
      pulse.pin.scale.setScalar(pulse.base * s);
      if (pulse.disc) {
        pulse.disc.material.opacity = 0.2 + 0.14 * (0.5 + 0.5 * Math.sin(pulse.t));
        pulse.ring.material.opacity = 0.7 + 0.3 * (0.5 + 0.5 * Math.sin(pulse.t));
        const rs = 1.0 + 0.12 * Math.sin(pulse.t);
        pulse.ring.scale.setScalar(rs);
        pulse.disc.scale.setScalar(rs);
      }
    }

    // 3. Walkthrough camera tick with ground collision
    if (g.cameraMode === 'first-person' && typeof g.fpTick === 'function') {
      g.fpTick(dt);
    }
  });

  return null;
}

/* ─── Main TerrainCanvas Component ──────────────────────────────────────── */

const TerrainCanvas = forwardRef(function TerrainCanvas({ onReady }, ref) {
  const { state, actions } = useApp();
  const canvasRef = useRef(null);
  const orbitControlsRef = useRef(null);
  const [activeEngine, setActiveEngine] = useState(null);
  const [showDebugHud, setShowDebugHud] = useState(false);

  // Shared state ref for imperative handles & parent consumers
  const glRef = useRef({
    renderer: null,
    camera: null,
    scene: null,
    orbit: null,
    controls: null,
    mesh: null,
    tiles: [],
    material: null,
    engine: null,
    heightData: null,
    hmWidth: 0,
    hmHeight: 0,
    heightScale: 1.0,
    visualHeightScale: 1.0,
    exaggeration: 1.0,
    cameraMode: 'orbit',
    fpTick: null,
    walkMode: false,
    disposed: false,
    wireframe: false,
    fogEnabled: false,
    contoursEnabled: false,
    contourInterval: 5.0,
    minElevation: 0.0,
    maxElevation: 100.0,
    elevationSpan: 100.0,
    worldWidth: 1024,
    worldDepth: 1024,
    isGeoreferencedScale: false,
    measureGroup: null,
    routeGroup: null,
    buildingsLayer: null,
    buildings3dEnabled: true,
  });

  // Toggle debug HUD with backtick or 'H'
  useEffect(() => {
    function onKeyDown(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '`' || e.key === '~' || e.key === 'h' || e.key === 'H') {
        setShowDebugHud(prev => !prev);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const handleEngineReady = useCallback((engine) => {
    setActiveEngine(engine);
    onReady?.(engine);
  }, [onReady]);

  /* ── Expose imperative handle to parent ── */
  useImperativeHandle(ref, () => ({
    getEngine() {
      return glRef.current.engine;
    },
    toggleDebugHUD() {
      setShowDebugHud(prev => !prev);
    },
    setSemanticEnabled(enabled) {
      glRef.current.engine?.setSemanticOverlay({ enabled: !!enabled });
    },
    setSemanticOpacity(val) {
      glRef.current.engine?.setSemanticOverlay({ opacity: val });
    },
    setSemanticHighlightClass(classId) {
      glRef.current.engine?.setSemanticOverlay({ highlightClass: classId });
    },
    updateSemanticTexture(labelsArray, width, height, confidenceArray = null) {
      const g = glRef.current;
      if (!g?.engine || !labelsArray || !width || !height) return;

      const tex = new THREE.DataTexture(
        labelsArray,
        width,
        height,
        THREE.RedFormat,
        THREE.UnsignedByteType
      );
      tex.minFilter = THREE.NearestFilter;
      tex.magFilter = THREE.NearestFilter;
      tex.generateMipmaps = false;
      // The shader samples the mask with the GLOBAL raster UV convention
      // (vUv.y = 1 - rasterRow). Regular textures flip on upload; a
      // DataTexture does NOT flip by default, so without this the mask
      // renders upside-down relative to the terrain.
      tex.flipY = true;
      tex.needsUpdate = true;

      let confTex = null;
      if (confidenceArray) {
        confTex = new THREE.DataTexture(
          confidenceArray instanceof Float32Array
            ? confidenceArray
            : Float32Array.from(confidenceArray),
          width,
          height,
          THREE.RedFormat,
          THREE.FloatType
        );
        confTex.minFilter = THREE.LinearFilter;
        confTex.magFilter = THREE.LinearFilter;
        confTex.generateMipmaps = false;
        confTex.flipY = false;
        confTex.needsUpdate = true;
      }

      g.engine.setSemanticOverlay({
        texture: tex,
        confidenceTexture: confTex,
        confidenceEnabled: !!confTex,
        // The overlay is loaded ready but stays hidden until the Semantics
        // layer is selected (setSemanticOverlay({enabled}) from the layer
        // switch) — highlight probing still works via uSemanticHighlightClass.
        enabled: g.semanticLayerActive === true,
      });

      // Free the previous masks' GPU memory.
      if (g.semanticTextures) {
        for (const t of g.semanticTextures) t.dispose();
      }
      g.semanticTextures = confTex ? [tex, confTex] : [tex];
    },
    setBuildings3DEnabled(enabled) {
      const g = glRef.current;
      g.buildings3dEnabled = !!enabled;
      g.buildingsLayer?.setVisible(!!enabled);
      // Buildings-3D ON  -> the terrain renders from the building-removed
      // ground field (mountains keep their elevation; only structures
      // leave the relief), falling back to the flat datum when no ground
      // field exists. OFF -> the original DSM returns with every detection
      // overlay (semantics, footprints, damage) draping it as before.
      const swapped = g.engine?.setBuildings3DTerrain(!!enabled);
      if (!swapped) {
        g.engine?.setElevationVisible(!!enabled);
      }
    },
    setSemanticLayerActive(active) {
      const g = glRef.current;
      g.semanticLayerActive = !!active;
      g.engine?.setSemanticOverlay({ enabled: !!active });
    },
    setExaggeration(v) {
      const g = glRef.current;
      const prev = g.exaggeration ?? 1;
      g.exaggeration = v;
      g.engine?.setExaggeration(v);
      g.buildingsLayer?.setExaggeration(v);
      // Raising exaggeration raises the rendered surface — an orbit camera
      // framed at the old scale can end up UNDER the mesh, viewing mirrored
      // backface shards (the "melted/shredded terrain" failure). Lift the
      // camera (and the orbit target with it) so it stays above the surface.
      if (g.engine?.collision && g.camera && prev !== v) {
        const minE = g.engine.geoRef.minElevation;
        const ground = g.engine.collision.getTerrainHeight(
          g.camera.position.x, g.camera.position.z,
        );
        const visual = minE + (ground - minE) * v;
        const margin = Math.max(5, (g.maxElevation ?? 0) * 0.25 * v);
        if (g.camera.position.y < visual + margin) {
          const raise = visual + margin - g.camera.position.y;
          g.camera.position.y += raise;
          if (g.orbit?.target) {
            g.orbit.target.y += raise;
            g.orbit.update?.();
          }
        }
      }
    },
    setWireframe(v) {
      const g = glRef.current;
      g.wireframe = !!v;
      g.engine?.setWireframe(g.wireframe);
    },
    setSolidView() {
      const g = glRef.current;
      g.engine?.setSolidView(g.wireframe);
      g.hybridView = false;
    },
    /** Hybrid view: RGB imagery + enhanced relief (never a solid mesh). */
    setHybridView() {
      const g = glRef.current;
      if (!g.engine) return;
      // Hybrid always drapes the RGB SOURCE IMAGERY — never the previous
      // override layer (a stale greyscale depth/DSM texture here is what
      // made hybrid look like a washed-out depth map). Reuse the resident
      // RGB texture when we have one; otherwise load it once and keep it.
      const applyRelief = () => {
        if (g.disposed) return;
        g.engine.setHybridView(g.wireframe);
        g.hybridView = true;
      };
      if (g.rgbTexture) {
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
          })
          .catch(() => {
            // No imagery available: hybrid degrades to solid relief, not white
            g.engine.setSolidView(g.wireframe);
          });
      } else {
        g.engine.setSolidView(g.wireframe);
      }
    },
    setFog(v) {
      const g = glRef.current;
      g.fogEnabled = !!v;
      g.engine?.setFog(g.fogEnabled);
    },
    setContours(enabled, interval) {
      const g = glRef.current;
      g.contoursEnabled = !!enabled;
      if (typeof interval === 'number' && interval > 0) {
        g.contourInterval = interval;
      }
      g.engine?.setContours(g.contoursEnabled, g.contourInterval);
    },
    resetCamera() {
      const g = glRef.current;
      if (g.camera) {
        const d = Math.max(g.worldWidth || 1024, g.worldDepth || 1024);
        const avgElev = ((g.minElevation || 0) + (g.maxElevation || 100)) * 0.5;
        g.camera.position.set(0, avgElev + d * 0.65, d * 1.15);
        g.camera.lookAt(0, avgElev, 0);
        if (g.orbit) {
          g.orbit.target?.set(0, avgElev, 0);
          g.orbit.update?.();
        }
      }
      g.cameraMode = 'orbit';
      if (g.orbit) g.orbit.enabled = true;
      g.fpTick = null;
    },
    setFpTick(fn) {
      glRef.current.fpTick = fn;
    },
    setCameraMode(newMode) {
      glRef.current.cameraMode = newMode;
      if (glRef.current.orbit) {
        glRef.current.orbit.enabled = (newMode === 'orbit');
      }
    },
    setWalkMode(enabled) {
      glRef.current.walkMode = !!enabled;
    },
    getRef() {
      return glRef;
    },
    getCanvas() {
      return canvasRef;
    },
    /** Live renderer + engine telemetry for the status strip / HUD. */
    getTelemetry() {
      const g = glRef.current;
      const base = g.engine?.getTelemetry?.() ?? {};
      const info = g.renderer?.info;
      return {
        ...base,
        drawCalls: info?.render?.calls ?? 0,
        textureCount: info?.memory?.textures ?? 0,
        geometries: info?.memory?.geometries ?? 0,
        programs: info?.programs?.length ?? 0,
      };
    },
    captureSnapshot() {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const g = glRef.current;
      if (g.renderer && g.scene && g.camera) {
        g.renderer.render(g.scene, g.camera);
      }
      try {
        return canvas.toDataURL('image/png');
      } catch (err) {
        console.error('[TerrainCanvas] captureSnapshot error', err);
        return null;
      }
    },
    setLayerTexture(url, colormapMode = 0) {
      const g = glRef.current;
      if (!g.engine) return;

      g.engine.setColormapMode(colormapMode);
      g.engine.setSemanticOverlay({ enabled: false });
      g.hybridView = false;

      loadAuthTexture(resolveAssetUrl(url), { rawData: colormapMode !== 0 })
        .then((tex) => {
          if (g.disposed) {
            tex.dispose();
            return;
          }
          const maxAniso = g.renderer?.capabilities?.getMaxAnisotropy?.() || 8;
          tex.anisotropy = Math.min(16, maxAniso);
          // Selecting the RGB layer itself should refresh the resident RGB
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
          g.engine.setWireframe(g.wireframe);
        })
        .catch((err) => console.warn('[terrain] layer texture load failed:', err));
    },
    sampleElevation(nx, nz) {
      const g = glRef.current;
      if (g.engine?.spatial) {
        // If coordinate is in normalized UV [0, 1]
        if (nx >= 0 && nx <= 1 && nz >= 0 && nz <= 1) {
          const pt = g.engine.geoRef.uvToLocal(nx, nz);
          return g.engine.spatial.sampleElevation(pt.x, pt.z);
        }
        // Otherwise treat as local world coordinate (meters)
        return g.engine.spatial.sampleElevation(nx, nz);
      }
      return null;
    },
    setMeasurePoints(a, b, label) {
      const g = glRef.current;
      if (!g.scene) return;

      const disposeGroup = () => {
        if (!g.measureGroup) return;
        g.scene.remove(g.measureGroup);
        g.measureGroup.traverse((o) => {
          o.geometry?.dispose?.();
          const m = o.material;
          if (Array.isArray(m)) m.forEach(x => { x.map?.dispose?.(); x.dispose?.(); });
          else { m?.map?.dispose?.(); m?.dispose?.(); }
        });
        g.measureGroup = null;
      };

      if (!a || !b) {
        disposeGroup();
        return;
      }

      const worldR = Math.max(g.worldWidth || 1024, g.worldDepth || 1024);
      const markerR = Math.max(1.0, worldR * 0.006);

      // Visual Y elevation
      const visualY = (elev) => {
        const minE = g.minElevation || 0;
        return minE + (elev - minE) * (g.exaggeration || 1.0);
      };

      const vA = new THREE.Vector3(a.x, visualY(a.elevation), a.z);
      const vB = new THREE.Vector3(b.x, visualY(b.elevation), b.z);

      if (!g.measureGroup) {
        const grp = new THREE.Group();
        const markerMat = new THREE.MeshBasicMaterial({
          color: 0x4f8cff, depthTest: false, transparent: true, opacity: 0.95,
        });
        const sphereGeo = new THREE.SphereGeometry(1, 20, 14);
        const mA = new THREE.Mesh(sphereGeo, markerMat);
        const mB = new THREE.Mesh(sphereGeo, markerMat);
        const lineMat = new THREE.LineDashedMaterial({
          color: 0x4f8cff, depthTest: false, transparent: true, opacity: 0.9,
        });
        const lineGeo = new THREE.BufferGeometry().setFromPoints([vA, vB]);
        const line = new THREE.Line(lineGeo, lineMat);
        const labelCanvas = document.createElement('canvas');
        const labelTex = new THREE.CanvasTexture(labelCanvas);
        labelTex.minFilter = THREE.LinearFilter;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
          map: labelTex, depthTest: false, transparent: true,
        }));
        [mA, mB, line, sprite].forEach(o => { o.renderOrder = 999; });
        grp.add(mA, mB, line, sprite);
        g.scene.add(grp);
        g.measureGroup = grp;
      }

      const grp = g.measureGroup;
      const [mA, mB, line, sprite] = grp.children;
      mA.position.copy(vA);
      mB.position.copy(vB);
      mA.scale.setScalar(markerR);
      mB.scale.setScalar(markerR);
      line.geometry.setFromPoints([vA, vB]);
      line.computeLineDistances();
      const lineMat = line.material;
      lineMat.dashSize = markerR * 2.2;
      lineMat.gapSize = markerR * 1.1;

      if (label) {
        const canvas = sprite.material.map.image;
        canvas.width = 256;
        canvas.height = 72;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.roundRect(4, 4, canvas.width - 8, canvas.height - 8, 18);
        ctx.fillStyle = 'rgba(8, 12, 20, 0.92)';
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#4f8cff';
        ctx.stroke();
        ctx.font = '600 32px ui-monospace, "SF Mono", Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#eaf0ff';
        ctx.fillText(label, canvas.width / 2, canvas.height / 2 + 2);
        sprite.material.map.needsUpdate = true;
        sprite.visible = true;
        const mid = vA.clone().add(vB).multiplyScalar(0.5);
        sprite.position.set(mid.x, mid.y + markerR * 7, mid.z);
        sprite.scale.set(markerR * 16, markerR * 16 * (canvas.height / canvas.width), 1);
      } else {
        sprite.visible = false;
      }
    },
    setRoutePath(pathPixels, verdictColor = '#2ecc71', opts = {}) {
      const g = glRef.current;
      if (!g.scene || !g.engine) return;

      const disposeGroup = () => {
        if (!g.routeGroup) return;
        g.scene.remove(g.routeGroup);
        g.routeGroup.traverse((o) => {
          o.geometry?.dispose?.();
          o.material?.dispose?.();
          if (o.userData?.disposeMap) o.material?.map?.dispose?.();
        });
        g.routeGroup = null;
      };

      if (!pathPixels || pathPixels.length < 2) {
        disposeGroup();
        return;
      }

      const toWorld = (px, py) => {
        const pt = g.engine.spatial.pixelToWorld(px, py);
        const minE = g.minElevation || 0;
        const visualY = minE + (pt.y - minE) * (g.exaggeration || 1.0);
        return new THREE.Vector3(pt.x, visualY, pt.z);
      };

      const points = pathPixels.map(p => toWorld(p.x, p.y));
      const worldR = Math.max(g.worldWidth || 1024, g.worldDepth || 1024);
      const markerR = Math.max(1.0, worldR * 0.008);
      const beaconH = worldR * 0.05;

      const makeLabel = (text, cssColor) => {
        const c = document.createElement('canvas');
        c.width = 256; c.height = 64;
        const ctx = c.getContext('2d');
        ctx.font = 'bold 40px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = cssColor;
        ctx.shadowColor = 'rgba(0,0,0,0.9)';
        ctx.shadowBlur = 8;
        ctx.fillText(text, 128, 32);
        const tex = new THREE.CanvasTexture(c);
        const spr = new THREE.Sprite(new THREE.SpriteMaterial({
          map: tex, depthTest: false, transparent: true,
        }));
        spr.userData.disposeMap = true;
        spr.scale.set(markerR * 10, markerR * 2.5, 1);
        spr.renderOrder = 1000;
        return spr;
      };

      const marker = (colorHex, label) => {
        const pin = new THREE.Mesh(
          new THREE.SphereGeometry(1, 16, 12),
          new THREE.MeshBasicMaterial({ color: colorHex, depthTest: false, transparent: true, opacity: 0.95 }),
        );
        pin.scale.setScalar(markerR);
        pin.renderOrder = 999;
        const beacon = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, beaconH, 0),
          ]),
          new THREE.LineBasicMaterial({ color: colorHex, depthTest: false, transparent: true, opacity: 0.6 }),
        );
        beacon.renderOrder = 998;
        const grp = new THREE.Group();
        grp.add(pin, beacon);
        if (label) {
          const spr = makeLabel(label.text, label.color);
          spr.position.set(0, beaconH + markerR * 2, 0);
          grp.add(spr);
        }
        return { grp, pin };
      };

      disposeGroup();
      const grp = new THREE.Group();
      const color = new THREE.Color(verdictColor);

      const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.0);
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(
          curve,
          Math.min(points.length * 4, 1200),
          markerR * 0.6,
          8,
          false
        ),
        new THREE.MeshBasicMaterial({
          color, depthTest: false, transparent: true, opacity: 0.9,
        })
      );
      tube.renderOrder = 997;
      grp.add(tube);

      const glow = new THREE.Mesh(
        new THREE.TubeGeometry(
          curve,
          Math.min(points.length * 4, 1200),
          markerR * 1.15,
          8,
          false
        ),
        new THREE.MeshBasicMaterial({
          color, depthTest: false, transparent: true, opacity: 0.22,
        })
      );
      glow.renderOrder = 996;
      grp.add(glow);

      const dashMat = new THREE.LineDashedMaterial({
        color: 0xffffff, depthTest: false, transparent: true,
        opacity: 0.7, dashSize: markerR * 2.2, gapSize: markerR * 1.6,
        scale: 1,
      });
      const dashLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points), dashMat
      );
      dashLine.computeLineDistances();
      dashLine.renderOrder = 999;
      grp.add(dashLine);

      const endLabel = opts.endLabel ?? 'END';
      const startM = marker(0x4f8cff, { text: 'START', color: '#9fc4ff' });
      startM.grp.position.copy(points[0]);
      const endM = marker(color, { text: endLabel, color: `#${color.getHexString()}` });
      endM.grp.position.copy(points[points.length - 1]);
      endM.pin.scale.setScalar(markerR * 1.3);
      grp.add(startM.grp, endM.grp);

      const endPt = points[points.length - 1];
      const lzY = endPt.y + markerR * 0.35;
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(markerR * 7, 48),
        new THREE.MeshBasicMaterial({
          color, depthTest: false, transparent: true, opacity: 0.3,
          side: THREE.DoubleSide,
        })
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.set(endPt.x, lzY, endPt.z);
      disc.renderOrder = 995;
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(markerR * 6.1, markerR * 7, 48),
        new THREE.MeshBasicMaterial({
          color, depthTest: false, transparent: true, opacity: 0.95,
          side: THREE.DoubleSide,
        })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(endPt.x, lzY + markerR * 0.05, endPt.z);
      ring.renderOrder = 996;
      grp.add(disc, ring);

      grp.userData.pulse = {
        pin: endM.pin, base: markerR * 1.3, t: 0,
        disc, ring, ringR: markerR * 7,
      };

      g.scene.add(grp);
      g.routeGroup = grp;
    },
    getTerrainPointFromEvent(event) {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const clientX = event.clientX - rect.left;
      const clientY = event.clientY - rect.top;
      if (clientX < 0 || clientX > rect.width || clientY < 0 || clientY > rect.height) return null;
      const ndcX = (clientX / rect.width) * 2 - 1;
      const ndcY = -(clientY / rect.height) * 2 + 1;

      const g = glRef.current;
      if (g.camera && g.engine) {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2(ndcX, ndcY);
        raycaster.setFromCamera(mouse, g.camera);
        const hit = g.engine.raycast(raycaster);
        if (hit) {
          return {
            x: hit.worldX,
            z: hit.worldZ,
            elevation: hit.elevation,
            u: hit.u,
            v: hit.v,
            slope: hit.slope,
            normal: hit.normal,
          };
        }
      }

      // Mathematical approximation fallback
      const halfW = (g.worldWidth || 1024) / 2;
      const halfD = (g.worldDepth || 1024) / 2;
      let wx = ndcX * halfW;
      let wz = -ndcY * halfD;
      if (g.orbit?.target && g.camera) {
        const dist = g.camera.position.distanceTo ? g.camera.position.distanceTo(g.orbit.target) : halfD * 1.25;
        wx = (g.orbit.target.x ?? 0) + ndcX * dist * 0.45;
        wz = (g.orbit.target.z ?? 0) - ndcY * dist * 0.45;
      }
      const nx = Math.max(0, Math.min(1, wx / (2 * halfW) + 0.5));
      const nz = Math.max(0, Math.min(1, wz / (2 * halfD) + 0.5));
      const elevation = this.sampleElevation(nx, nz) ?? 0;
      return { x: wx, z: wz, elevation, u: nx, v: nz };
    },
  }));

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        gl={{
          // preserveDrawingBuffer stays OFF: it forces the browser to keep
          // every frame's buffer alive and measurably hurts throughput.
          // captureSnapshot() renders synchronously and reads the canvas
          // within the same task, which does not need the flag.
          preserveDrawingBuffer: false,
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          // Metric scene spans ~1 m (walkthrough eye) to tens of km (camera
          // far plane) — a linear depth buffer z-fights at distance
          // (floating terrain fragments). Log depth fixes the precision
          // cliff across that range.
          logarithmicDepthBuffer: true,
        }}
        dpr={[1, 2]}
        camera={{
          position: [0, 600, 1100],
          fov: 45,
          near: 0.5,
          far: 25000,
        }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color(0.028, 0.035, 0.055), 1);
        }}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          background: 'var(--dw-void)',
          outline: 'none',
        }}
        tabIndex={0}
        aria-label="3D geospatial terrain engine"
      >
        <SceneBridge
          canvasRef={canvasRef}
          glRef={glRef}
          orbitControlsRef={orbitControlsRef}
          sceneState={state}
          actions={actions}
          onEngineReady={handleEngineReady}
        />
        <OrbitControls
          ref={orbitControlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.08}
          enablePan
          panSpeed={0.8}
          minDistance={1.0}
          maxDistance={30000}
          minPolarAngle={0.01}
          maxPolarAngle={Math.PI * 0.49}
        />
      </Canvas>

      {/* Telemetry / Mission-Control Debug HUD */}
      <TerrainDebugHUD
        engine={activeEngine}
        camera={glRef.current.camera}
        visible={showDebugHud}
      />
    </div>
  );
});

export default TerrainCanvas;

/* ─── Async terrain data loader ────────────────────────────────────────── */

async function loadTerrainData(g, scene, sceneId, actions, onEngineReady) {
  try {
    console.info('[terrain-engine] loading terrain metadata for', sceneId);
    const terrainMeta = await getTerrain(sceneId);
    if (g.disposed) return;

    const {
      heightmap_url,
      texture_url,
      height_scale,
      min_elevation,
      max_elevation,
      world_width_m,
      world_depth_m,
      is_georeferenced_scale,
    } = terrainMeta;

    const worldWidthKnown = typeof world_width_m === 'number' && world_width_m > 0;
    const worldDepthKnown = typeof world_depth_m === 'number' && world_depth_m > 0;
    g.worldWidth = worldWidthKnown ? world_width_m : (terrainMeta.raster_width || 1024);
    g.worldDepth = worldDepthKnown ? world_depth_m : (terrainMeta.raster_height || 1024);
    g.isGeoreferencedScale = is_georeferenced_scale === true;

    const hs = typeof height_scale === 'number' && height_scale > 0 ? height_scale : 1.0;
    g.heightScale = hs;
    g.minElevation = typeof min_elevation === 'number' ? min_elevation : 0.0;
    g.maxElevation = typeof max_elevation === 'number' ? max_elevation : (g.minElevation + hs * 100.0);
    g.elevationSpan = Math.max(0.001, g.maxElevation - g.minElevation);

    // Height source: inline 16-bit heightmap decode for normal rasters, or
    // backend quadtree tile streaming for very large ones (spec §13/§29).
    const tileConfig = terrainMeta.tile_config || null;
    const tileUrls = tileConfig ? buildTileUrlFns(tileConfig) : null;
    const rasterMaxPx = Math.max(
      terrainMeta.raster_width || 0,
      terrainMeta.raster_height || 0
    );
    const streamHeights = !!(tileUrls && rasterMaxPx > HEIGHT_STREAM_THRESHOLD_PX);

    let data, width, height;
    if (streamHeights) {
      console.info(
        `[terrain-engine] raster ${terrainMeta.raster_width}x${terrainMeta.raster_height} exceeds ` +
        `${HEIGHT_STREAM_THRESHOLD_PX}px — streaming heights via quadtree tiles`
      );
      const overview = await fetchOverviewHeightfield(
        tileUrls.height,
        terrainMeta.raster_width,
        terrainMeta.raster_height
      );
      if (g.disposed) return;
      data = overview.data;
      width = overview.width;
      height = overview.height;
    } else {
      console.info('[terrain-engine] decoding heightmap:', heightmap_url);
      const decoded = await decodeHeightmap(resolveAssetUrl(heightmap_url));
      if (g.disposed) return;
      ({ width, height, data } = decoded);
    }

    g.heightData = data;
    g.hmWidth = width;
    g.hmHeight = height;
    // Metres of world-Y per unit of the normalized [0,1] heightfield — the
    // visual scale the mesh actually renders (see useCameraController's
    // fallback ground sampling). Kept in sync with the elevation span so
    // walkthrough collision can never use a stale/fallback constant.
    g.visualHeightScale = Math.max(0.001, g.maxElevation - g.minElevation);

    // Dispose scene-scoped GPU resources from the previous scene.
    if (g.overrideTexture) {
      g.overrideTexture.dispose();
      g.overrideTexture = null;
    }
    if (g.rgbTexture) {
      g.rgbTexture.dispose();
      g.rgbTexture = null;
    }
    if (g.semanticTextures) {
      for (const t of g.semanticTextures) t.dispose();
      g.semanticTextures = null;
    }
    g.semanticLayerActive = false;
    g.hybridView = false;

    // Dispose old engine instance if switching scenes
    if (g.engine) {
      g.engine.dispose();
      g.engine = null;
    }

    // Instantiate Geospatial Terrain Engine
    console.info('[terrain-engine] initializing Quadtree & Metric Engine');
    const engine = new TerrainEngine({
      scene,
      camera: g.camera,
      terrainMeta,
      heightData: data,
      hmWidth: width,
      hmHeight: height,
      heightTiles: streamHeights && tileUrls
        ? { url: tileUrls.height, tileSize: tileConfig.tile_size || 256 }
        : null,
      // Height/texture *tile* endpoints are always authenticated FastAPI
      // routes (see buildTileUrlFns) — never object storage — so the JWT
      // belongs here. Contrast with loadAuthTexture/decodeHeightmap above,
      // which fetch storage-asset URLs through assetFetch() instead.
      tileFetchHeaders: authHeaders,
    });
    g.engine = engine;
    g.material = engine.material;
    // Debug hook for live inspection (telemetry HUD / automated checks).
    if (typeof window !== 'undefined') window.__dwEngine = engine;

    // Set camera frustum & standoff based on real metric footprint
    const sceneD = Math.max(g.worldWidth, g.worldDepth);
    const diag = Math.hypot(g.worldWidth, g.worldDepth);
    const avgElev = (g.minElevation + g.maxElevation) * 0.5;

    if (g.camera) {
      g.camera.position.set(sceneD * 0.38, avgElev + sceneD * 0.42, sceneD * 0.82);
      g.camera.lookAt(0, avgElev, 0);
      g.camera.near = 0.5;
      g.camera.far = Math.max(25000, diag * 6);
      g.camera.updateProjectionMatrix();
    }
    if (g.orbit) {
      g.orbit.target?.set(0, avgElev, 0);
      g.orbit.maxDistance = Math.max(30000, sceneD * 6);
      g.orbit.minDistance = 1.0;
      g.orbit.update?.();
    }

    onEngineReady?.(engine);

    // Mark terrain ready in store
    actions.terrainReady(terrainMeta);

    // Geometry-aware 3D building reconstruction overlay (footprints +
    // DSM-derived heights from the backend's buildings3d.json). Purely
    // additive: a missing/failed payload leaves the terrain untouched.
    loadBuildings3D(g, scene, sceneId);

    // Load diffuse texture if available (authenticated fetch — TextureLoader
    // cannot send the Authorization header and result files are auth-gated)
    if (texture_url) {
      g.rgbTextureUrl = texture_url;
      loadAuthTexture(resolveAssetUrl(texture_url))
        .then((tex) => {
          if (g.disposed) {
            tex.dispose();
            return;
          }
          const maxAniso = g.renderer?.capabilities?.getMaxAnisotropy?.() || 8;
          tex.anisotropy = Math.min(16, maxAniso);
          g.rgbTexture = tex;
          engine.setTexture(tex);
          engine.setWireframe(g.wireframe);
        })
        .catch((err) => console.warn('[terrain] diffuse texture load failed:', err));
    }
  } catch (err) {
    console.error('[terrain-engine] failed to initialize terrain engine:', err);
    if (!g.disposed) {
      actions.terrainFail({
        code: err.code ?? 'TERRAIN_LOAD_ERROR',
        message: err.message ?? 'Failed to load terrain engine.',
        recoverable: true,
      });
    }
  }
}

/** Fetch and mount the 3D building reconstruction layer for a scene.
 * Availability follows the artifact on disk — scenes processed before
 * this feature (or with no building candidates) simply render without it. */
async function loadBuildings3D(g, scene, sceneId) {
  try {
    const data = await getBuildings3D(sceneId);
    if (g.disposed || !g.engine) return;
    if (!data?.available || !data.buildings?.length) return;

    // dispose a previous scene's layer
    if (g.buildingsLayer) {
      g.buildingsLayer.dispose();
      g.buildingsLayer = null;
    }

    const layer = new BuildingsLayer(scene, g.engine.geoRef);
    layer.load(data);
    layer.setExaggeration(g.exaggeration ?? 1.0);
    layer.setVisible(g.buildings3dEnabled !== false);

    // Ground heightfield: the DSM with building regions replaced by their
    // surrounding ground. Buildings-3D mode swaps THIS in — mountains and
    // hills keep their elevation; only the structures leave the relief.
    // Scenes without one fall back to the flat datum (uFlatten).
    if (data.has_ground) {
      try {
        const ground = await decodeHeightmap(
          resolveAssetUrl(`/api/v1/scenes/${sceneId}/results/ground-heightmap`)
        );
        if (ground.width === g.hmWidth && ground.height === g.hmHeight) {
          g.engine.setGroundHeightField(ground.data, ground.width, ground.height);
        }
      } catch (err) {
        console.warn('[buildings3d] ground heightfield unavailable:', err?.message ?? err);
      }
    }

    if (layer.visible) {
      // auto-activated on load: swap to the ground field (mountains stay),
      // falling back to the flat datum when no ground field exists
      if (!g.engine.setBuildings3DTerrain(true)) {
        g.engine.setElevationVisible(false);
      }
    }
    g.buildingsLayer = layer;
    console.info(
      `[buildings3d] ${layer.count} structure(s) reconstructed ` +
      `(heights from ${data.height_source ?? 'predicted DSM'})`
    );
  } catch (err) {
    if (typeof window !== 'undefined') {
      window.__b3dError = String(err?.stack ?? err?.message ?? err);
    }
    console.warn('[buildings3d] reconstruction layer unavailable:', err?.message ?? err);
  }
}
