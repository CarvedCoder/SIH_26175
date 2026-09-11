/**
 * DepthWizard — TerrainCanvas (React Three Fiber Migration + Main Merge)
 *
 * React Three Fiber (R3F) + Three.js terrain renderer. Full-bleed canvas that:
 *   - Renders 3D terrain via R3F <Canvas> and Three.js ShaderMaterial
 *   - Decodes /terrain/heightmap PNG and applies vertex displacement
 *   - Fetches /terrain/texture (RGB / Depth / DSM / Slope) as diffuse map
 *   - Directional (sun) + ambient lighting; geospatial mission-control look
 *   - Terrain exaggeration via uniform float uExaggeration
 *   - Wireframe toggle via ShaderMaterial.wireframe
 *   - Solid view via setSolidView()
 *   - Progressive reveal: low-res mesh first, swaps to high-res when ready
 *   - Integrates Drei <OrbitControls> with smooth damping
 *   - Full backward compatibility with TerrainWorkspace, CameraHUD, and Minimap
 *
 * DECISIONS.md §D06 — React Three Fiber renderer choice.
 * DESIGN.md — dark, instrument-panel; no neon, no bloom.
 */

import { useEffect, useRef, useImperativeHandle, forwardRef, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useApp } from '../../store/appStore.jsx';
import { getTerrain } from '../../api/terrain.js';
import { resolveAssetUrl } from '../../api/client.js';

/* ─── Shader source ─────────────────────────────────────────────────────── */

/**
 * Vertex shader:
 * - Reads heightmap Y from uHeightmap texture at (uv.x, uv.y)
 * - Displaces Y by height × uExaggeration × uHeightScale
 * - Lerp from flat to displaced using uProgress (progressive reveal)
 * - Passes vNormal (world-space) for lighting
 */
const VERT = /* glsl */ `
  uniform sampler2D uHeightmap;
  uniform float uExaggeration;
  uniform float uHeightScale;
  uniform float uProgress;       // 0 -> 1, progressive reveal

  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vHeight;         // normalised [0,1] for colormap fallback
  varying float vViewDist;       // distance from camera for depth fog

  void main() {
    vUv = uv;

    // Sample heightmap (red channel = normalised elevation)
    float raw = texture2D(uHeightmap, uv).r;
    vHeight = raw;

    // Displace Y in world space
    float displaced = raw * uHeightScale * uExaggeration;
    vec3 pos = position;
    pos.y = mix(0.0, displaced, uProgress);

    vec4 viewPos = modelViewMatrix * vec4(pos, 1.0);
    vViewDist = -viewPos.z;

    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * viewPos;
  }
`;

/**
 * Fragment shader:
 * - Diffuse texture (RGB/Depth/DSM) as base colour
 * - Simple directional (sun) + ambient lighting
 * - Colormap blending via uColormapMode:
 *     0 = RGB (texture as-is)
 *     1 = Greyscale (depth)
 *     2 = Viridis (DSM, slope)
 *     3 = Diverging red-blue (error map)
 * - Atmospheric depth fog
 */
const FRAG = /* glsl */ `
  #ifdef GL_OES_standard_derivatives
  #extension GL_OES_standard_derivatives : enable
  #endif

  uniform sampler2D uTexture;
  uniform vec3 uSunDir;          // normalised
  uniform vec3 uSunColor;
  uniform float uAmbient;
  uniform float uTextureReady;   // 0 before texture loaded, 1 after
  uniform float uColormapMode;   // 0=rgb, 1=greyscale, 2=viridis, 3=diverging
  uniform float uContoursEnabled; // 0=off, 1=on
  uniform float uContourInterval;
  uniform float uElevationSpan;
  uniform float uMinElevation;
  uniform float uFogEnabled;     // 0=off, 1=on
  uniform vec3 uFogColor;        // matches background [0.028, 0.035, 0.055]

  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vHeight;
  varying float vViewDist;

  // Viridis colormap polynomial approximation
  vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.274, 0.004, 0.329);
    vec3 c1 = vec3(0.263, 0.388, 0.683);
    vec3 c2 = vec3(-0.196, 0.490, 0.098);
    vec3 c3 = vec3(-0.094, -0.550, 0.447);
    vec3 c4 = vec3(0.006, 0.276, -0.588);
    vec3 c5 = vec3(0.010, 0.052, 0.244);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*c5))));
  }

  // Diverging red-blue (error map): blue=negative, white=zero, red=positive
  vec3 diverging(float t) {
    if (t < 0.5) {
      return mix(vec3(0.145, 0.396, 0.933), vec3(0.87, 0.89, 0.93), t * 2.0);
    } else {
      return mix(vec3(0.87, 0.89, 0.93), vec3(0.933, 0.145, 0.145), (t - 0.5) * 2.0);
    }
  }

  void main() {
    // Height-based fallback colour until texture loads
    vec3 heightColor = mix(
      vec3(0.08, 0.12, 0.16),
      vec3(0.55, 0.62, 0.70),
      vHeight
    );

    vec4 texSample = texture2D(uTexture, vUv);
    vec3 texColor  = texSample.rgb;

    // Apply colormap
    vec3 baseColor;
    int mode = int(uColormapMode + 0.5);
    if (mode == 1) {
      float lum = texSample.r;
      baseColor = vec3(lum);
    } else if (mode == 2) {
      baseColor = viridis(texSample.r);
    } else if (mode == 3) {
      baseColor = diverging(texSample.r);
    } else {
      baseColor = texColor;
    }

    // Blend: fallback -> mapped colour based on texture readiness
    baseColor = mix(heightColor, baseColor, uTextureReady);

    // Diffuse lighting
    float diff = max(dot(vNormal, uSunDir), 0.0);
    vec3 lit = baseColor * (uAmbient + diff * (1.0 - uAmbient));

    // Slight tonal grading toward cool shadow
    vec3 shadow = mix(vec3(0.04, 0.07, 0.12), lit, clamp(diff + uAmbient, 0.0, 1.0));

    // Optional contour lines
    if (uContoursEnabled > 0.5 && uContourInterval > 0.001) {
      float elev = vHeight * uElevationSpan + uMinElevation;
      float lineDist = abs(fract(elev / uContourInterval - 0.5) - 0.5) * uContourInterval;
      float fw = max(fwidth(elev), 0.0001);
      float contour = 1.0 - smoothstep(0.0, fw * 1.5, lineDist);

      float majorInterval = uContourInterval * 5.0;
      float majorDist = abs(fract(elev / majorInterval - 0.5) - 0.5) * majorInterval;
      float majorContour = 1.0 - smoothstep(0.0, fw * 2.2, majorDist);

      vec3 contourColor = vec3(0.06, 0.09, 0.14);
      shadow = mix(shadow, contourColor, clamp(contour * 0.45 + majorContour * 0.35, 0.0, 0.85));
    }

    // Optional atmospheric depth fog
    if (uFogEnabled > 0.5) {
      float fogFactor = clamp((vViewDist - 1.2) / 6.0, 0.0, 0.85);
      shadow = mix(shadow, uFogColor, fogFactor);
    }

    gl_FragColor = vec4(shadow, 1.0);
  }
`;

/* ─── Constants ─────────────────────────────────────────────────────────── */

const LO_SEGS = 64;
const HI_SEGS = 256;
const DEFAULT_CAMERA_POS = [0, 1.2, 2.5];

/**
 * Visual vertical scale used for the mesh's Y-displacement, decoupled from
 * the backend's `height_scale` (which is a physical/real-world elevation
 * factor used for measurements — see ElevationProbe.jsx, StructureInspector.jsx).
 * The ground plane spans a fixed footprint of [-1, 1] (2 world units wide)
 * regardless of scene mode, so a real-world `height_scale` (which can be in
 * the hundreds for absolute/meters-based scenes) produces wildly
 * disproportionate spikes if used directly as the mesh multiplier. This
 * constant keeps rendered terrain proportional across both relative and
 * absolute scenes; the `exaggeration` slider (1x-5x) still scales on top of it.
 */
const BASE_VISUAL_HEIGHT_SCALE = 0.22;

/* ─── Helpers ───────────────────────────────────────────────────────────── */

/**
 * Decode a PNG image URL into a Float32Array of normalised [0,1] red-channel values.
 */
async function decodeHeightmap(url) {
  const resolved = resolveAssetUrl(url);
  const res = await fetch(resolved);
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);

  const { width, height } = bitmap;
  let ctx;

  if (typeof OffscreenCanvas !== 'undefined') {
    const oc = new OffscreenCanvas(width, height);
    ctx = oc.getContext('2d');
  } else {
    const c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    ctx = c.getContext('2d');
  }

  ctx.drawImage(bitmap, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height).data;

  const data = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    data[i] = pixels[i * 4] / 255;
  }

  bitmap.close();
  return { data, width, height };
}

/**
 * Box-blur a heightmap in place-safe fashion (returns a new Float32Array).
 * Reduces per-pixel noise so individual vertices don't spike into thin
 * needles once displaced and exaggerated. `radius` in pixels; radius=1
 * means a 3x3 average, radius=2 means 5x5, etc.
 */
function smoothHeightData(data, width, height, radius = 2) {
  if (radius <= 0) return data;
  const out = new Float32Array(data.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const sy = y + dy;
        if (sy < 0 || sy >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const sx = x + dx;
          if (sx < 0 || sx >= width) continue;
          sum += data[sy * width + sx];
          count++;
        }
      }
      out[y * width + x] = count > 0 ? sum / count : data[y * width + x];
    }
  }

  return out;
}

/**
 * Build a PlaneGeometry tile representing a subsection [uMin, uMax] x [vMin, vMax]
 */
function buildTerrainTileGeometry(heightData, hmWidth, hmHeight, segs, heightScale, exaggeration, tx, ty, numTiles = 2) {
  const uMin = tx / numTiles;
  const uMax = (tx + 1) / numTiles;
  const vMin = ty / numTiles;
  const vMax = (ty + 1) / numTiles;

  const wSegs = segs;
  const hSegs = segs;
  const num = (wSegs + 1) * (hSegs + 1);
  const numIndices = wSegs * hSegs * 6;

  const position = new Float32Array(num * 3);
  const normal = new Float32Array(num * 3);
  const uv = new Float32Array(num * 2);
  const index = numIndices > 65536 ? new Uint32Array(numIndices) : new Uint16Array(numIndices);

  let i = 0;
  for (let iy = 0; iy <= hSegs; iy++) {
    const fracY = iy / hSegs;
    const v = vMin + fracY * (vMax - vMin);
    for (let ix = 0; ix <= wSegs; ix++, i++) {
      const fracX = ix / wSegs;
      const u = uMin + fracX * (uMax - uMin);

      const x = (u - 0.5) * 2; // [-1, 1]
      const z = (v - 0.5) * 2; // [-1, 1]

      let h = 0;
      if (heightData && hmWidth && hmHeight) {
        const px = Math.min(Math.max(Math.round(u * (hmWidth - 1)), 0), hmWidth - 1);
        const py = Math.min(Math.max(Math.round(v * (hmHeight - 1)), 0), hmHeight - 1);
        h = heightData[py * hmWidth + px] * heightScale * exaggeration;
      }

      position[i * 3] = x;
      position[i * 3 + 1] = h;
      position[i * 3 + 2] = z;

      normal[i * 3] = 0;
      normal[i * 3 + 1] = 1;
      normal[i * 3 + 2] = 0;

      uv[i * 2] = u;
      uv[i * 2 + 1] = 1 - v;
    }
  }

  let ii = 0;
  for (let iy = 0; iy < hSegs; iy++) {
    for (let ix = 0; ix < wSegs; ix++) {
      const a = ix + iy * (wSegs + 1);
      const b = ix + (iy + 1) * (wSegs + 1);
      const c = ix + (iy + 1) * (wSegs + 1) + 1;
      const d = ix + iy * (wSegs + 1) + 1;
      index[ii * 6] = a;
      index[ii * 6 + 1] = b;
      index[ii * 6 + 2] = d;
      index[ii * 6 + 3] = b;
      index[ii * 6 + 4] = c;
      index[ii * 6 + 5] = d;
      ii++;
    }
  }

  computeNormals(position, index, normal, wSegs, hSegs);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.computeBoundingSphere();
  geo.computeBoundingBox();
  return geo;
}

function computeNormals(position, index, normal, wSegs, hSegs) {
  const totalVerts = (wSegs + 1) * (hSegs + 1);
  const count = new Float32Array(totalVerts);

  for (let i = 0; i < index.length; i += 3) {
    const ia = index[i], ib = index[i + 1], ic = index[i + 2];

    const ax = position[ib * 3] - position[ia * 3];
    const ay = position[ib * 3 + 1] - position[ia * 3 + 1];
    const az = position[ib * 3 + 2] - position[ia * 3 + 2];
    const bx = position[ic * 3] - position[ia * 3];
    const by = position[ic * 3 + 1] - position[ia * 3 + 1];
    const bz = position[ic * 3 + 2] - position[ia * 3 + 2];

    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;

    for (const vi of [ia, ib, ic]) {
      normal[vi * 3] += nx;
      normal[vi * 3 + 1] += ny;
      normal[vi * 3 + 2] += nz;
      count[vi]++;
    }
  }

  for (let i = 0; i < totalVerts; i++) {
    if (count[i] === 0) continue;
    const len = Math.hypot(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2]);
    if (len > 0) {
      normal[i * 3] /= len;
      normal[i * 3 + 1] /= len;
      normal[i * 3 + 2] /= len;
    }
  }
}

function createEmptyTexture() {
  const pixel = new Uint8Array([0, 0, 0, 255]);
  const tex = new THREE.DataTexture(pixel, 1, 1, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return tex;
}

/* ─── R3F Inner Bridge Component ────────────────────────────────────────── */

function SceneBridge({ canvasRef, glRef, orbitControlsRef, materialRef, sceneState, actions }) {
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
    const mat = materialRef.current;
    if (!mat) return;

    g.disposed = false;
    loadTerrainData(g, mat, scene, sceneId, actions);

    return () => {
      g.disposed = true;
      if (g.tiles) {
        g.tiles.forEach(m => {
          m.geometry?.dispose?.();
          scene.remove(m);
        });
        g.tiles = [];
      }
      if (g.activeTextures) {
        g.activeTextures.forEach(t => t.dispose?.());
        g.activeTextures.clear();
      }
    };
  }, [sceneState?.scene?.scene_id, scene, actions, glRef, materialRef]);

  // Frame tick animation loop
  useFrame((_, delta) => {
    const g = glRef.current;
    if (g.disposed) return;
    const dt = Math.min(delta, 0.1);

    const mat = materialRef.current;
    if (mat && mat.uniforms.uProgress) {
      const prefersReducedMotion = typeof window !== 'undefined' &&
        window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

      if (prefersReducedMotion) {
        g.progressCurrent = g.progress;
        mat.uniforms.uProgress.value = g.progress;
      } else if (g.progressCurrent < g.progress) {
        g.progressCurrent = Math.min(g.progressCurrent + dt * (1 / 0.6), g.progress);
        mat.uniforms.uProgress.value = g.progressCurrent;
      }
    }

    // Walkthrough mode (internal id 'first-person') — tick is registered
    // from TerrainWorkspace via setFpTick; free-flight movement, no clamp.
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
    heightData: null,
    hmWidth: 0,
    hmHeight: 0,
    heightScale: 1.0,
    visualHeightScale: BASE_VISUAL_HEIGHT_SCALE,
    exaggeration: 1.5,
    progress: 0.0,
    progressCurrent: 0.0,
    textureReady: 0,
    cameraMode: 'orbit',
    fpTick: null,
    disposed: false,
    activeTextures: new Set(),
    wireframe: false,
    fogEnabled: false,
    contoursEnabled: false,
    contourInterval: 5.0,
    minElevation: 0.0,
    maxElevation: 100.0,
    elevationSpan: 100.0,
  });

  // Create initial Three.js ShaderMaterial
  const sunDir = [0.6, 0.9, 0.4].map(v => v / Math.hypot(0.6, 0.9, 0.4));
  const materialRef = useRef(null);

  if (!materialRef.current) {
    const emptyHm = createEmptyTexture();
    const emptyRgb = createEmptyTexture();
    glRef.current.activeTextures.add(emptyHm);
    glRef.current.activeTextures.add(emptyRgb);

    materialRef.current = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uHeightmap: { value: emptyHm },
        uTexture: { value: emptyRgb },
        uExaggeration: { value: 1.5 },
        uHeightScale: { value: BASE_VISUAL_HEIGHT_SCALE },
        uProgress: { value: 0.0 },
        uSunDir: { value: new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]) },
        uSunColor: { value: new THREE.Color(1.0, 0.95, 0.85) },
        uAmbient: { value: 0.32 },
        uTextureReady: { value: 0.0 },
        uColormapMode: { value: 0.0 },
        uContoursEnabled: { value: 0.0 },
        uContourInterval: { value: 5.0 },
        uElevationSpan: { value: 100.0 },
        uMinElevation: { value: 0.0 },
        uFogEnabled: { value: 0.0 },
        uFogColor: { value: new THREE.Color(0.028, 0.035, 0.055) },
      },
      wireframe: false,
      transparent: false,
      depthTest: true,
      depthWrite: true,
    });
    glRef.current.material = materialRef.current;
  }

  /* ── Expose imperative handle to parent ── */
  useImperativeHandle(ref, () => ({
    setExaggeration(v) {
      const g = glRef.current;
      g.exaggeration = v;
      const mat = materialRef.current;
      if (mat?.uniforms?.uExaggeration) {
        mat.uniforms.uExaggeration.value = v;
      }
    },
    setWireframe(v) {
      const g = glRef.current;
      g.wireframe = v;
      const mat = materialRef.current;
      if (mat) {
        mat.wireframe = !!v;
      }
    },
    setSolidView() {
      const g = glRef.current;
      g.textureReady = 0;
      const mat = materialRef.current;
      if (mat?.uniforms?.uTextureReady) {
        mat.uniforms.uTextureReady.value = 0.0;
      }
      if (mat?.uniforms?.uColormapMode) {
        mat.uniforms.uColormapMode.value = 0.0;
      }
    },
    setFog(v) {
      const g = glRef.current;
      g.fogEnabled = !!v;
      const mat = materialRef.current;
      if (mat?.uniforms?.uFogEnabled) {
        mat.uniforms.uFogEnabled.value = v ? 1.0 : 0.0;
      }
    },
    resetCamera() {
      const g = glRef.current;
      if (g.camera && g.orbit) {
        g.camera.position.set(...DEFAULT_CAMERA_POS);
        g.orbit.target.set(0, 0, 0);
        g.orbit.update?.();
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
    getRef() {
      return glRef;
    },
    getCanvas() {
      return canvasRef;
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
      const mat = materialRef.current;
      if (!mat) return;

      if (mat.uniforms.uTextureReady) {
        mat.uniforms.uTextureReady.value = 0.0;
        g.textureReady = 0;
      }
      if (mat.uniforms.uColormapMode) {
        mat.uniforms.uColormapMode.value = colormapMode;
      }

      const resolved = resolveAssetUrl(url);
      const loader = new THREE.TextureLoader();
      loader.setCrossOrigin('anonymous');
      loader.load(resolved, (tex) => {
        if (g.disposed) {
          tex.dispose();
          return;
        }
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.generateMipmaps = true;
        tex.flipY = true;
        tex.needsUpdate = true;

        const oldTex = mat.uniforms.uTexture?.value;
        mat.uniforms.uTexture.value = tex;
        g.activeTextures.add(tex);

        if (oldTex && oldTex.dispose) {
          oldTex.dispose();
          g.activeTextures.delete(oldTex);
        }

        const prefersReducedMotion = typeof window !== 'undefined' &&
          window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

        if (prefersReducedMotion) {
          mat.uniforms.uTextureReady.value = 1.0;
          g.textureReady = 1;
          return;
        }

        const start = performance.now();
        function fadeIn() {
          if (g.disposed) return;
          const t = Math.min((performance.now() - start) / 250, 1);
          if (mat.uniforms.uTextureReady) {
            mat.uniforms.uTextureReady.value = t;
          }
          if (t < 1) requestAnimationFrame(fadeIn);
          else {
            g.textureReady = 1;
          }
        }
        requestAnimationFrame(fadeIn);
      });
    },
    setContours(enabled, interval) {
      const g = glRef.current;
      g.contoursEnabled = !!enabled;
      if (typeof interval === 'number' && interval > 0) {
        g.contourInterval = interval;
      }
      const mat = materialRef.current;
      if (mat) {
        if (mat.uniforms.uContoursEnabled) {
          mat.uniforms.uContoursEnabled.value = enabled ? 1.0 : 0.0;
        }
        if (mat.uniforms.uContourInterval && typeof interval === 'number' && interval > 0) {
          mat.uniforms.uContourInterval.value = interval;
        }
      }
    },
    sampleElevation(nx, nz) {
      const g = glRef.current;
      if (!g.heightData || !g.hmWidth || !g.hmHeight) return null;
      const px = Math.min(Math.max(Math.round(nx * (g.hmWidth - 1)), 0), g.hmWidth - 1);
      const pz = Math.min(Math.max(Math.round(nz * (g.hmHeight - 1)), 0), g.hmHeight - 1);
      const raw = g.heightData[pz * g.hmWidth + px];
      if (typeof g.minElevation === 'number' && typeof g.elevationSpan === 'number') {
        return g.minElevation + raw * g.elevationSpan;
      }
      return raw * 100.0 * (g.heightScale ?? 1.0);
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
      // Precise Raycasting with Three.js
      if (g.camera && g.tiles && g.tiles.length > 0) {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2(ndcX, ndcY);
        raycaster.setFromCamera(mouse, g.camera);
        const hits = raycaster.intersectObjects(g.tiles, false);
        if (hits.length > 0) {
          const hit = hits[0];
          const wx = hit.point.x;
          const wz = hit.point.z;
          const nx = Math.max(0, Math.min(1, (wx + 1) / 2));
          const nz = Math.max(0, Math.min(1, (wz + 1) / 2));
          const elevation = this.sampleElevation(nx, nz) ?? hit.point.y;
          return { x: wx, z: wz, elevation };
        }
      }

      // Mathematical approximation fallback
      let wx = ndcX;
      let wz = -ndcY;
      if (g.orbit?.target && g.camera) {
        const dist = g.camera.position.distanceTo ? g.camera.position.distanceTo(g.orbit.target) : 2.5;
        wx = (g.orbit.target.x ?? 0) + ndcX * dist * 0.45;
        wz = (g.orbit.target.z ?? 0) - ndcY * dist * 0.45;
      }
      const nx = Math.max(0, Math.min(1, (wx + 1) / 2));
      const nz = Math.max(0, Math.min(1, (wz + 1) / 2));
      const elevation = this.sampleElevation(nx, nz) ?? 0;
      return { x: nx * 2 - 1, z: nz * 2 - 1, elevation };
    },
  }));

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        gl={{
          preserveDrawingBuffer: true,
          antialias: true,
          alpha: false,
        }}
        camera={{
          position: DEFAULT_CAMERA_POS,
          fov: 45,
          near: 0.01,
          far: 100,
        }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color(0.028, 0.035, 0.055), 1);
        }}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          background: '#07090e',
          outline: 'none',
        }}
        tabIndex={0}
        aria-label="3D terrain viewer"
      >
        <SceneBridge
          canvasRef={canvasRef}
          glRef={glRef}
          orbitControlsRef={orbitControlsRef}
          materialRef={materialRef}
          sceneState={state}
          actions={actions}
        />
        <OrbitControls
          ref={orbitControlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.08}
          enablePan
          panSpeed={0.5}
          minDistance={0.3}
          maxDistance={8}
          minPolarAngle={0.05}
          maxPolarAngle={Math.PI * 0.48}
        />
      </Canvas>
    </div>
  );
});

export default TerrainCanvas;

/* ─── Async terrain data loader ────────────────────────────────────────── */

async function loadTerrainData(g, material, scene, sceneId, actions) {
  try {
    console.info('[terrain] loading terrain for', sceneId);
    const terrainMeta = await getTerrain(sceneId);
    if (g.disposed) return;

    console.info('[terrain] metadata loaded:', {
      heightmap_url: terrainMeta?.heightmap_url,
      texture_url: terrainMeta?.texture_url,
      height_scale: terrainMeta?.height_scale,
      min_elevation: terrainMeta?.min_elevation,
      max_elevation: terrainMeta?.max_elevation,
    });

    const { heightmap_url, texture_url, height_scale, min_elevation, max_elevation } = terrainMeta;

    const hs = typeof height_scale === 'number' && height_scale > 0 ? height_scale : 1.0;
    // Physical height_scale — kept as-is for real-world elevation math
    // (ElevationProbe, StructureInspector, CameraHUD all read g.heightScale).
    g.heightScale = hs;
    // Visual height_scale — used only for mesh Y-displacement so the terrain
    // stays proportional to its fixed [-1,1] footprint regardless of the
    // backend's physical units. See BASE_VISUAL_HEIGHT_SCALE comment above.
    g.visualHeightScale = BASE_VISUAL_HEIGHT_SCALE;
    if (material.uniforms.uHeightScale) {
      material.uniforms.uHeightScale.value = g.visualHeightScale;
    }

    const minElev = typeof min_elevation === 'number' ? min_elevation : 0.0;
    const maxElev = typeof max_elevation === 'number' ? max_elevation : (minElev + hs * 100.0);
    const span = Math.max(0.001, maxElev - minElev);
    g.minElevation = minElev;
    g.maxElevation = maxElev;
    g.elevationSpan = span;
    if (material.uniforms.uMinElevation) material.uniforms.uMinElevation.value = minElev;
    if (material.uniforms.uElevationSpan) material.uniforms.uElevationSpan.value = span;

    // Remove existing placeholder tiles if present
    if (g.tiles && g.tiles.length > 0) {
      g.tiles.forEach(m => {
        m.geometry?.dispose?.();
        scene.remove(m);
      });
      g.tiles = [];
    }

    // Decode heightmap
    const decoded = await decodeHeightmap(heightmap_url);
    if (g.disposed) return;
    const { width, height } = decoded;
    // Smooth to remove per-pixel noise that would otherwise spike into
    // thin vertical needles once displaced and exaggerated.
    const data = smoothHeightData(decoded.data, width, height, 2);

    g.heightData = data;
    g.hmWidth = width;
    g.hmHeight = height;

    // Create 32-bit float heightmap texture for Three.js
    const uint8 = new Uint8Array(data.length * 4);
    for (let i = 0; i < data.length; i++) {
      const val = Math.round(data[i] * 255);
      uint8[i * 4] = val;
      uint8[i * 4 + 1] = val;
      uint8[i * 4 + 2] = val;
      uint8[i * 4 + 3] = 255;
    }
    const hmTex = new THREE.DataTexture(uint8, width, height, THREE.RGBAFormat);
    hmTex.minFilter = THREE.LinearFilter;
    hmTex.magFilter = THREE.LinearFilter;
    hmTex.wrapS = THREE.ClampToEdgeWrapping;
    hmTex.wrapT = THREE.ClampToEdgeWrapping;
    hmTex.flipY = false;
    hmTex.needsUpdate = true;

    const oldHmTex = material.uniforms.uHeightmap?.value;
    material.uniforms.uHeightmap.value = hmTex;
    g.activeTextures.add(hmTex);

    if (oldHmTex && oldHmTex.dispose) {
      oldHmTex.dispose();
      g.activeTextures.delete(oldHmTex);
    }

    // Create 2x2 low-res tile meshes
    const tiles = [];
    for (let idx = 0; idx < 4; idx++) {
      const tx = idx % 2;
      const ty = Math.floor(idx / 2);
      const loGeo = buildTerrainTileGeometry(data, width, height, LO_SEGS / 2, g.visualHeightScale, g.exaggeration, tx, ty, 2);
      const tileMesh = new THREE.Mesh(loGeo, material);
      tileMesh.frustumCulled = true;
      scene.add(tileMesh);
      tiles.push(tileMesh);
    }
    g.tiles = tiles;
    g.mesh = tiles[0];
    g.progress = 1.0;

    // Mark terrain ready in state machine
    actions.terrainReady(terrainMeta);

    // Swap to high-res after short timeout
    setTimeout(() => {
      if (g.disposed) return;
      g.tiles.forEach((m, idx) => {
        const tx = idx % 2;
        const ty = Math.floor(idx / 2);
        const hiGeo = buildTerrainTileGeometry(data, width, height, HI_SEGS / 2, g.visualHeightScale, g.exaggeration, tx, ty, 2);
        const oldGeo = m.geometry;
        m.geometry = hiGeo;
        oldGeo?.dispose?.();
      });
      g.segs = HI_SEGS;
    }, 100);

    // Load diffuse texture if provided
    if (texture_url) {
      const resolved = resolveAssetUrl(texture_url);
      const loader = new THREE.TextureLoader();
      loader.setCrossOrigin('anonymous');
      loader.load(resolved, (tex) => {
        if (g.disposed) {
          tex.dispose();
          return;
        }
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.generateMipmaps = true;
        tex.flipY = true;
        tex.needsUpdate = true;

        const oldTex = material.uniforms.uTexture?.value;
        material.uniforms.uTexture.value = tex;
        material.uniforms.uTextureReady.value = 1.0;
        g.textureReady = 1;
        g.activeTextures.add(tex);

        if (oldTex && oldTex.dispose) {
          oldTex.dispose();
          g.activeTextures.delete(oldTex);
        }
      });
    }
  } catch (err) {
    console.error('[terrain] failed to load terrain data:', err);
    if (!g.disposed) {
      actions.terrainFail({
        code: err.code ?? 'TERRAIN_LOAD_ERROR',
        message: err.message ?? 'Failed to load terrain data.',
        recoverable: true,
      });
    }
  }
}