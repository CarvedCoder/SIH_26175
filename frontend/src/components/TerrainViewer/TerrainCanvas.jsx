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
  uniform vec2 uHeightmapSize;   // (width, height) in texels
  uniform vec2 uWorldSize;       // (worldWidth, worldDepth) in world units

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

    // Analytic normal from heightmap central differences, in WORLD scale —
    // always consistent with the current uHeightScale/uExaggeration and the
    // GPU's own texture sampling (CPU-baked normals went stale whenever the
    // exaggeration slider moved). ClampToEdge wrapping makes the one-sided
    // difference at the borders automatic.
    vec2 texel = 1.0 / uHeightmapSize;
    float hL = texture2D(uHeightmap, uv - vec2(texel.x, 0.0)).r;
    float hR = texture2D(uHeightmap, uv + vec2(texel.x, 0.0)).r;
    float hD = texture2D(uHeightmap, uv - vec2(0.0, texel.y)).r;
    float hU = texture2D(uHeightmap, uv + vec2(0.0, texel.y)).r;
    float vScale = uHeightScale * uExaggeration * uProgress;
    float stepX = max(uWorldSize.x * texel.x, 1e-6);
    float stepZ = max(uWorldSize.y * texel.y, 1e-6);
    // uv.y = 1 - v and world z grows with v, hence the hD/hU order here
    float dhdx = (hR - hL) * vScale / (2.0 * stepX);
    float dhdz = (hD - hU) * vScale / (2.0 * stepZ);
    vNormal = normalize(vec3(-dhdx, 1.0, -dhdz));

    vec4 viewPos = modelViewMatrix * vec4(pos, 1.0);
    vViewDist = -viewPos.z;

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
  uniform float uFogNear;        // fog start distance (world units, scene-scaled)
  uniform float uFogFar;         // fog full distance (world units, scene-scaled)
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

    // Optional atmospheric depth fog (near/far scale with the physical
    // scene size — see loadTerrainData's uFogNear/uFogFar computation)
    if (uFogEnabled > 0.5) {
      float fogFactor = clamp((vViewDist - uFogNear) / max(uFogFar - uFogNear, 0.001), 0.0, 0.85);
      shadow = mix(shadow, uFogColor, fogFactor);
    }

    gl_FragColor = vec4(shadow, 1.0);
  }
`;

/* ─── Constants ─────────────────────────────────────────────────────────── */

const LO_SEGS = 64;
const HI_SEGS = 256;
const DEFAULT_CAMERA_POS = [0, 0.8, 1.6];

/**
 * Visual vertical scale for meshes whose footprint the backend could not
 * describe physically (missing world_width_m/world_depth_m — a legacy API
 * response). The ground plane then spans the legacy fixed footprint of
 * [-1, 1] (2 world units wide), so a real-world `height_scale` (which can
 * be in the hundreds) would produce wildly disproportionate spikes if used
 * directly; this constant keeps rendered terrain proportional in that
 * fallback case.
 *
 * When the backend DOES provide the physical footprint (the normal path),
 * the plane is sized to world_width_m × world_depth_m metres and the
 * vertical axis uses the scene's real elevation span in metres
 * (see loadTerrainData) — vertical and horizontal are then both true
 * metric proportion, and the exaggeration slider (1x-5x) scales on top.
 */
const BASE_VISUAL_HEIGHT_SCALE = 0.22;

/** Legacy fallback footprint (world units) when the API response carries
 * no physical world size. Matches the pre-metric [-1,1] plane. */
const LEGACY_WORLD_SIZE = 2.0;

/* ─── Helpers ───────────────────────────────────────────────────────────── */

/**
 * Decode a heightmap PNG into a Float32Array of normalised [0,1] values.
 *
 * The backend writes 16-BIT grayscale PNGs (65536 elevation levels). The
 * browser canvas API silently quantises those to 8 bits, which re-introduces
 * the terracing the 16-bit encoding exists to avoid — so 16-bit files are
 * parsed directly here (PNG chunk walk + native DecompressionStream inflate
 * + row-filter reconstruction). Legacy 8-bit files fall back to the canvas
 * path.
 */
async function decodeHeightmap(url) {
  const resolved = resolveAssetUrl(url);
  const res = await fetch(resolved);
  if (!res.ok) throw new Error(`heightmap fetch failed: ${res.status}`);

  const buf = await res.arrayBuffer();
  try {
    const parsed = await decodePngGray16(buf);
    if (parsed) return parsed;
  } catch (err) {
    console.warn('[terrain] 16-bit heightmap decode failed, falling back to 8-bit canvas decode', err);
  }
  return decodeHeightmapViaCanvas(resolved);
}

/** Parse a non-interlaced 16-bit grayscale PNG (color type 0, bit depth 16).
 *  Returns { data: Float32Array, width, height } or null for other variants. */
async function decodePngGray16(buf) {
  const bytes = new Uint8Array(buf);
  const SIG = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) {
    if (bytes[i] !== SIG[i]) throw new Error('not a PNG');
  }
  const dv = new DataView(buf);
  let p = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat = [];
  while (p + 8 <= bytes.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(bytes[p + 4], bytes[p + 5], bytes[p + 6], bytes[p + 7]);
    const data = bytes.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(p + 8);
      height = dv.getUint32(p + 12);
      bitDepth = bytes[p + 16];
      colorType = bytes[p + 17];
      interlace = bytes[p + 20];
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    p += 12 + len;
  }
  if (colorType !== 0 || bitDepth !== 16 || interlace !== 0) return null;

  // Inflate the image stream with the native DecompressionStream.
  const stream = new Blob(idat).stream().pipeThrough(new DecompressionStream('deflate'));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());

  // Reconstruct filtered scanlines (bpp = 2 bytes per 16-bit gray sample).
  const bpp = 2;
  const stride = width * bpp;
  const out = new Uint8Array(height * stride);
  let src = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[src++];
    const row = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? row[x - bpp] : 0;              // left
      const b = prev ? prev[x] : 0;                        // up
      const c = (prev && x >= bpp) ? prev[x - bpp] : 0;    // upper-left
      let v = raw[src + x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      row[x] = v & 0xff;
    }
    src += stride;
  }

  // Big-endian 16-bit samples → normalised float
  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i++) {
    data[i] = ((out[i * 2] << 8) | out[i * 2 + 1]) / 65535;
  }
  return { data, width, height };
}

/** Legacy 8-bit path: canvas decode of the red channel (256 levels). */
async function decodeHeightmapViaCanvas(resolvedUrl) {
  const res = await fetch(resolvedUrl);
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
  for (let i = 0; i < data.length; i++) data[i] = pixels[i * 4] / 255;
  bitmap.close();
  return { data, width, height };
}

/** Bilinear heightmap sample at normalised [0,1] coords — matches the GPU's
 *  LinearFilter sampling so CPU geometry agrees with shader displacement. */
function sampleHeightBilinear(data, w, h, u, v) {
  const fx = Math.min(Math.max(u, 0), 1) * (w - 1);
  const fy = Math.min(Math.max(v, 0), 1) * (h - 1);
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const tx = fx - x0, ty = fy - y0;
  const a = data[y0 * w + x0], b = data[y0 * w + x1];
  const c = data[y1 * w + x0], d = data[y1 * w + x1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

/**
 * Build a PlaneGeometry tile representing a subsection [uMin, uMax] x [vMin, vMax]
 * of a ground plane spanning worldWidth x worldDepth world units (metres on
 * the physical-scale path; the legacy 2x2 footprint in the fallback).
 */
function buildTerrainTileGeometry(heightData, hmWidth, hmHeight, segs, heightScale, exaggeration, tx, ty, numTiles = 2, worldWidth = LEGACY_WORLD_SIZE, worldDepth = LEGACY_WORLD_SIZE) {
  const uMin = tx / numTiles;
  const uMax = (tx + 1) / numTiles;
  const vMin = ty / numTiles;
  const vMax = (ty + 1) / numTiles;

  const wSegs = segs;
  const hSegs = segs;
  const num = (wSegs + 1) * (hSegs + 1);
  const numIndices = wSegs * hSegs * 6;

  const position = new Float32Array(num * 3);
  const uv = new Float32Array(num * 2);
  const index = numIndices > 65536 ? new Uint32Array(numIndices) : new Uint16Array(numIndices);

  let i = 0;
  for (let iy = 0; iy <= hSegs; iy++) {
    const fracY = iy / hSegs;
    const v = vMin + fracY * (vMax - vMin);
    for (let ix = 0; ix <= wSegs; ix++, i++) {
      const fracX = ix / wSegs;
      const u = uMin + fracX * (uMax - uMin);

      const x = (u - 0.5) * worldWidth;
      const z = (v - 0.5) * worldDepth;

      let h = 0;
      if (heightData && hmWidth && hmHeight) {
        // Bilinear — matches the shader's LinearFilter texture sampling
        h = sampleHeightBilinear(heightData, hmWidth, hmHeight, u, v) * heightScale * exaggeration;
      }

      position[i * 3] = x;
      position[i * 3 + 1] = h;
      position[i * 3 + 2] = z;

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

  // Normals are computed analytically in the vertex shader from the
  // heightmap (see VERT) — no CPU normal attribute is needed. Three.js
  // ShaderMaterial without a 'normal' attribute works because the shader
  // never reads the built-in `normal` varying input.
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.computeBoundingSphere();
  geo.computeBoundingBox();
  return geo;
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
    exaggeration: 3.0,
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
    // Physical world footprint in world units (metres when the backend
    // reports a scale; LEGACY_WORLD_SIZE fallback for legacy responses).
    worldWidth: LEGACY_WORLD_SIZE,
    worldDepth: LEGACY_WORLD_SIZE,
    isGeoreferencedScale: false,
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
        uExaggeration: { value: 3.0 },
        uHeightScale: { value: BASE_VISUAL_HEIGHT_SCALE },
        uHeightmapSize: { value: new THREE.Vector2(1, 1) },
        uWorldSize: { value: new THREE.Vector2(LEGACY_WORLD_SIZE, LEGACY_WORLD_SIZE) },
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
        uFogNear: { value: 1.7 },
        uFogFar: { value: 8.5 },
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
          // world → normalised [0,1] using the plane's real footprint
          const nx = Math.max(0, Math.min(1, wx / (g.worldWidth ?? LEGACY_WORLD_SIZE) + 0.5));
          const nz = Math.max(0, Math.min(1, wz / (g.worldDepth ?? LEGACY_WORLD_SIZE) + 0.5));
          const elevation = this.sampleElevation(nx, nz) ?? hit.point.y;
          return { x: wx, z: wz, elevation };
        }
      }

      // Mathematical approximation fallback
      const halfW = (g.worldWidth ?? LEGACY_WORLD_SIZE) / 2;
      const halfD = (g.worldDepth ?? LEGACY_WORLD_SIZE) / 2;
      let wx = ndcX * halfW;
      let wz = -ndcY * halfD;
      if (g.orbit?.target && g.camera) {
        const dist = g.camera.position.distanceTo ? g.camera.position.distanceTo(g.orbit.target) : halfD * 1.25;
        wx = (g.orbit.target.x ?? 0) + ndcX * dist * 0.45;
        wz = (g.orbit.target.z ?? 0) - ndcY * dist * 0.45;
      }
      const nx = Math.max(0, Math.min(1, wx / (g.worldWidth ?? LEGACY_WORLD_SIZE) + 0.5));
      const nz = Math.max(0, Math.min(1, wz / (g.worldDepth ?? LEGACY_WORLD_SIZE) + 0.5));
      const elevation = this.sampleElevation(nx, nz) ?? 0;
      return { x: wx, z: wz, elevation };
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
          fov: 60,
          near: 0.005,
          far: 1000,
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
          maxDistance={5}
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

    // Physical world footprint (real-world terrain scale). The backend
    // computes raster dims × GSD from the CRS+transform; for non-
    // georeferenced scenes it applies a documented 1 m/pixel fallback and
    // flags it via is_georeferenced_scale=false. Missing fields (legacy
    // API) fall back to the legacy 2x2 normalized footprint.
    const worldWidthKnown = typeof terrainMeta.world_width_m === 'number' && terrainMeta.world_width_m > 0;
    const worldDepthKnown = typeof terrainMeta.world_depth_m === 'number' && terrainMeta.world_depth_m > 0;
    g.worldWidth = worldWidthKnown ? terrainMeta.world_width_m : LEGACY_WORLD_SIZE;
    g.worldDepth = worldDepthKnown ? terrainMeta.world_depth_m : LEGACY_WORLD_SIZE;
    g.isGeoreferencedScale = terrainMeta.is_georeferenced_scale === true;
    console.info('[terrain] world scale:', {
      width_m: g.worldWidth, depth_m: g.worldDepth,
      georeferenced_scale: g.isGeoreferencedScale,
    });

    const hs = typeof height_scale === 'number' && height_scale > 0 ? height_scale : 1.0;
    // Physical height_scale — kept as-is for real-world elevation math
    // (ElevationProbe, StructureInspector, CameraHUD all read g.heightScale).
    g.heightScale = hs;

    const minElev = typeof min_elevation === 'number' ? min_elevation : 0.0;
    const maxElev = typeof max_elevation === 'number' ? max_elevation : (minElev + hs * 100.0);
    const span = Math.max(0.001, maxElev - minElev);
    g.minElevation = minElev;
    g.maxElevation = maxElev;
    g.elevationSpan = span;

    // Visual height scale: with a physical footprint the vertical axis uses
    // the scene's REAL elevation span in metres — the mesh is then in true
    // metric proportion on both axes. The legacy constant only applies when
    // the world size is unknown (see BASE_VISUAL_HEIGHT_SCALE doc).
    g.visualHeightScale = (worldWidthKnown || worldDepthKnown) ? span : BASE_VISUAL_HEIGHT_SCALE;
    if (material.uniforms.uHeightScale) {
      material.uniforms.uHeightScale.value = g.visualHeightScale;
    }
    if (material.uniforms.uMinElevation) material.uniforms.uMinElevation.value = minElev;
    if (material.uniforms.uElevationSpan) material.uniforms.uElevationSpan.value = span;

    // Fog range scales with the physical scene diagonal. Tighter fog
    // creates stronger atmospheric perspective, making the terrain feel
    // more expansive — objects fading into distance imply vast scale.
    const diag = Math.hypot(g.worldWidth, g.worldDepth);
    if (material.uniforms.uFogNear) material.uniforms.uFogNear.value = 0.4 * diag;
    if (material.uniforms.uFogFar) material.uniforms.uFogFar.value = 2.2 * diag;

    // Frame the whole scene: orbit distances and the default camera pose
    // must scale with the physical footprint (a 204.8 m terrain needs a
    // stand-off distance proportional to its size). Camera is placed close
    // enough that the terrain fills the viewport — a tighter framing
    // produces a stronger sense of scale and immersion.
    const sceneD = Math.max(g.worldWidth, g.worldDepth, LEGACY_WORLD_SIZE);
    if (g.camera) {
      g.camera.position.set(0, sceneD * 0.35, sceneD * 0.7);
      g.camera.lookAt(0, 0, 0);
      if (g.camera.far < diag * 10) {
        g.camera.far = diag * 10;
        g.camera.updateProjectionMatrix();
      }
    }
    if (g.orbit) {
      g.orbit.target?.set(0, 0, 0);
      g.orbit.maxDistance = sceneD * 2.5;
      g.orbit.minDistance = Math.min(0.3, sceneD * 0.02);
      g.orbit.update?.();
    }

    // Remove existing placeholder tiles if present
    if (g.tiles && g.tiles.length > 0) {
      g.tiles.forEach(m => {
        m.geometry?.dispose?.();
        scene.remove(m);
      });
      g.tiles = [];
    }

    // Decode heightmap (16-bit path — full 65536-level precision, no blur:
    // the backend already low-pass filtered before downsampling, and a
    // frontend box blur destroyed building edges).
    const decoded = await decodeHeightmap(heightmap_url);
    if (g.disposed) return;
    const { width, height } = decoded;
    const data = decoded.data;

    g.heightData = data;
    g.hmWidth = width;
    g.hmHeight = height;

    // Upload as a single-channel FLOAT texture — the previous 8-bit RGBA
    // upload re-quantised the decoded heights to 256 levels, reintroducing
    // on the GPU the exact terracing the 16-bit PNG pipeline removes.
    const hmTex = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.FloatType);
    hmTex.minFilter = THREE.LinearFilter;
    hmTex.magFilter = THREE.LinearFilter;
    hmTex.wrapS = THREE.ClampToEdgeWrapping;
    hmTex.wrapT = THREE.ClampToEdgeWrapping;
    hmTex.flipY = true;
    hmTex.needsUpdate = true;

    const oldHmTex = material.uniforms.uHeightmap?.value;
    material.uniforms.uHeightmap.value = hmTex;
    g.activeTextures.add(hmTex);

    // Shader-normal uniforms: texture resolution (for central-difference
    // steps) and the world footprint (to convert height gradient into
    // world-space slope). Geometry tiles must know both too.
    if (material.uniforms.uHeightmapSize) {
      material.uniforms.uHeightmapSize.value.set(width, height);
    }
    if (material.uniforms.uWorldSize) {
      material.uniforms.uWorldSize.value.set(g.worldWidth, g.worldDepth);
    }

    if (oldHmTex && oldHmTex.dispose) {
      oldHmTex.dispose();
      g.activeTextures.delete(oldHmTex);
    }

    // Create 2x2 low-res tile meshes
    const tiles = [];
    for (let idx = 0; idx < 4; idx++) {
      const tx = idx % 2;
      const ty = Math.floor(idx / 2);
      const loGeo = buildTerrainTileGeometry(data, width, height, LO_SEGS / 2, g.visualHeightScale, g.exaggeration, tx, ty, 2, g.worldWidth, g.worldDepth);
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
        const hiGeo = buildTerrainTileGeometry(data, width, height, HI_SEGS / 2, g.visualHeightScale, g.exaggeration, tx, ty, 2, g.worldWidth, g.worldDepth);
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