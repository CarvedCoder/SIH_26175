/**
 * DepthWizard — TerrainCanvas (Phase 4)
 *
 * OGL WebGL terrain renderer. Full-bleed canvas that:
 *   4.1 — initialises Renderer, Camera, Scene, GL context
 *   4.2 — fetches /terrain/heightmap PNG; decodes via OffscreenCanvas;
 *          builds PlaneGeometry with vertex Y displacement; transitions TERRAIN_READY
 *   4.3 — fetches /terrain/texture; applies as diffuse map
 *   4.4 — directional (sun) + ambient lighting; professional geospatial look
 *   4.5 — terrain exaggeration via uniform float uExaggeration (vertex shader only)
 *   4.6 — wireframe toggle via mesh.mode
 *   4.7 — progressive: low-res mesh first, swap to full-res when ready
 *
 * Exposes a ref-forwarded handle: { setExaggeration(v), setWireframe(v), resetCamera() }
 * Parent (TerrainWorkspace) calls these from the controls UI.
 *
 * DECISIONS.md §D06 — OGL renderer choices.
 * DESIGN.md — dark, instrument-panel; no neon, no bloom.
 * Spec §7, §20, §32, §75.
 */
import { useEffect, useRef, useImperativeHandle, forwardRef, useCallback } from 'react';
import {
  Renderer, Camera, Transform, Geometry, Program, Mesh, Texture,
} from 'ogl';
import { Orbit } from 'ogl/src/extras/Orbit.js';
import { useApp } from '../../store/appStore.jsx';
import { getTerrain } from '../../api/terrain.js';

/* ─── Shader source ─────────────────────────────────────────────────────── */

/**
 * Vertex shader:
 * - Reads heightmap Y from uHeightmap texture at (uv.x, uv.y)
 * - Displaces Y by height × uExaggeration × uHeightScale
 * - Lerp from flat to displaced using uProgress (progressive reveal)
 * - Passes vNormal (world-space approximation) for lighting
 */
const VERT = /* glsl */ `
  precision highp float;

  attribute vec3 position;
  attribute vec2 uv;
  attribute vec3 normal;

  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform mat3 normalMatrix;

  uniform sampler2D uHeightmap;
  uniform float uExaggeration;
  uniform float uHeightScale;
  uniform float uProgress;       // 0 → 1, progressive reveal

  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vHeight;         // normalised [0,1] for colormap fallback

  void main() {
    vUv = uv;

    // Sample heightmap (red channel = normalised elevation)
    float raw = texture2D(uHeightmap, uv).r;
    vHeight = raw;

    // Displace Y in world space
    float displaced = raw * uHeightScale * uExaggeration;
    vec3 pos = position;
    pos.y = mix(0.0, displaced, uProgress);

    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
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
 * - No neon, no bloom (DESIGN.md anti-patterns)
 */
const FRAG = /* glsl */ `
  precision highp float;

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

  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vHeight;

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
    // t in [0,1]; centre=0.5
    if (t < 0.5) {
      return mix(vec3(0.145, 0.396, 0.933), vec3(0.87, 0.89, 0.93), t * 2.0);
    } else {
      return mix(vec3(0.87, 0.89, 0.93), vec3(0.933, 0.145, 0.145), (t - 0.5) * 2.0);
    }
  }

  void main() {
    // Height-based fallback colour (greyscale elevation tint) until texture loads
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
      // Greyscale: use red channel (luminance from depth PNG)
      float lum = texSample.r;
      baseColor = vec3(lum);
    } else if (mode == 2) {
      // Viridis: red channel as normalised value
      baseColor = viridis(texSample.r);
    } else if (mode == 3) {
      // Diverging: red channel as position [0,1]
      baseColor = diverging(texSample.r);
    } else {
      // RGB (default)
      baseColor = texColor;
    }

    // Blend: fallback → mapped colour based on texture readiness
    baseColor = mix(heightColor, baseColor, uTextureReady);

    // Diffuse lighting
    float diff = max(dot(vNormal, uSunDir), 0.0);
    vec3 lit = baseColor * (uAmbient + diff * (1.0 - uAmbient));

    // Slight tonal grading toward cool shadow — geospatial look
    vec3 shadow = mix(vec3(0.04, 0.07, 0.12), lit, clamp(diff + uAmbient, 0.0, 1.0));

    // Optional contour lines (§21)
    if (uContoursEnabled > 0.5 && uContourInterval > 0.001) {
      float elev = vHeight * uElevationSpan + uMinElevation;
      float lineDist = abs(fract(elev / uContourInterval - 0.5) - 0.5) * uContourInterval;
      float fw = max(fwidth(elev), 0.0001);
      float contour = 1.0 - smoothstep(0.0, fw * 1.5, lineDist);

      // Major index contour every 5 intervals
      float majorInterval = uContourInterval * 5.0;
      float majorDist = abs(fract(elev / majorInterval - 0.5) - 0.5) * majorInterval;
      float majorContour = 1.0 - smoothstep(0.0, fw * 2.2, majorDist);

      // Subtle crisp cartographic line
      vec3 contourColor = vec3(0.06, 0.09, 0.14);
      shadow = mix(shadow, contourColor, clamp(contour * 0.45 + majorContour * 0.35, 0.0, 0.85));
    }

    gl_FragColor = vec4(shadow, 1.0);
  }
`;

/* ─── Constants ─────────────────────────────────────────────────────────── */

/** Low-res: 64×64 segments; high-res: 256×256 */
const LO_SEGS = 64;
const HI_SEGS = 256;

/** Default camera position (orbit target = origin) */
const DEFAULT_CAMERA_POS = [0, 1.2, 2.5];

/* ─── Helpers ───────────────────────────────────────────────────────────── */

/**
 * Decode a PNG image URL into a Float32Array of normalised [0,1] red-channel values.
 * Uses OffscreenCanvas; falls back to ImageBitmap for Safari.
 * @param {string} url
 * @returns {Promise<{ data: Float32Array, width: number, height: number }>}
 */
async function decodeHeightmap(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);

  const { width, height } = bitmap;
  let ctx;

  if (typeof OffscreenCanvas !== 'undefined') {
    const oc = new OffscreenCanvas(width, height);
    ctx = oc.getContext('2d');
  } else {
    // Fallback: regular canvas (never shown)
    const c = document.createElement('canvas');
    c.width = width; c.height = height;
    ctx = c.getContext('2d');
  }

  ctx.drawImage(bitmap, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height).data; // Uint8ClampedArray, RGBA

  // Extract red channel, normalise to [0, 1]
  const data = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    data[i] = pixels[i * 4] / 255;
  }

  bitmap.close();
  return { data, width, height };
}

/**
 * Build a PlaneGeometry with Y displacement baked into position buffer.
 * Uses a custom Geometry (not Plane class) so we control the segment count.
 * @param {WebGLRenderingContext} gl
 * @param {Float32Array} heightData
 * @param {number} hmWidth   - heightmap pixel width
 * @param {number} hmHeight  - heightmap pixel height
 * @param {number} segs      - grid segments (LO_SEGS or HI_SEGS)
 * @param {number} heightScale
 * @param {number} exaggeration
 * @returns {Geometry}
 */
function buildTerrainGeometry(gl, heightData, hmWidth, hmHeight, segs, heightScale, exaggeration) {
  const wSegs = segs;
  const hSegs = segs;
  const num = (wSegs + 1) * (hSegs + 1);
  const numIndices = wSegs * hSegs * 6;

  const position = new Float32Array(num * 3);
  const normal   = new Float32Array(num * 3);
  const uv       = new Float32Array(num * 2);
  const index    = numIndices > 65536 ? new Uint32Array(numIndices) : new Uint16Array(numIndices);

  let i = 0;
  for (let iy = 0; iy <= hSegs; iy++) {
    const v = iy / hSegs;
    for (let ix = 0; ix <= wSegs; ix++, i++) {
      const u = ix / wSegs;
      const x = (u - 0.5) * 2; // [-1, 1]
      const z = (v - 0.5) * 2; // [-1, 1]

      // Sample heightmap at (u, v) — nearest neighbour
      const px = Math.min(Math.round(u * (hmWidth - 1)), hmWidth - 1);
      const py = Math.min(Math.round(v * (hmHeight - 1)), hmHeight - 1);
      const h  = heightData[py * hmWidth + px] * heightScale * exaggeration;

      position[i * 3]     = x;
      position[i * 3 + 1] = h;
      position[i * 3 + 2] = z;

      // Up normal (will be updated by computeNormals below, but init here)
      normal[i * 3]     = 0;
      normal[i * 3 + 1] = 1;
      normal[i * 3 + 2] = 0;

      uv[i * 2]     = u;
      uv[i * 2 + 1] = 1 - v; // flip V for WebGL (origin bottom-left)
    }
  }

  // Build index buffer (two triangles per quad)
  let ii = 0;
  for (let iy = 0; iy < hSegs; iy++) {
    for (let ix = 0; ix < wSegs; ix++) {
      const a = ix + iy * (wSegs + 1);
      const b = ix + (iy + 1) * (wSegs + 1);
      const c = ix + (iy + 1) * (wSegs + 1) + 1;
      const d = ix + iy * (wSegs + 1) + 1;
      index[ii * 6]     = a;
      index[ii * 6 + 1] = b;
      index[ii * 6 + 2] = d;
      index[ii * 6 + 3] = b;
      index[ii * 6 + 4] = c;
      index[ii * 6 + 5] = d;
      ii++;
    }
  }

  // Compute smooth normals from cross-products of surrounding triangles
  computeNormals(position, index, normal, wSegs, hSegs);

  return new Geometry(gl, {
    position: { size: 3, data: position },
    normal:   { size: 3, data: normal },
    uv:       { size: 2, data: uv },
    index:    { data: index },
  });
}

/**
 * Very simple smooth normal computation via face normal accumulation.
 */
function computeNormals(position, index, normal, wSegs, hSegs) {
  const totalVerts = (wSegs + 1) * (hSegs + 1);
  const count = new Float32Array(totalVerts);

  for (let i = 0; i < index.length; i += 3) {
    const ia = index[i], ib = index[i + 1], ic = index[i + 2];

    // Edge vectors
    const ax = position[ib * 3]     - position[ia * 3];
    const ay = position[ib * 3 + 1] - position[ia * 3 + 1];
    const az = position[ib * 3 + 2] - position[ia * 3 + 2];
    const bx = position[ic * 3]     - position[ia * 3];
    const by = position[ic * 3 + 1] - position[ia * 3 + 1];
    const bz = position[ic * 3 + 2] - position[ia * 3 + 2];

    // Cross product
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;

    for (const vi of [ia, ib, ic]) {
      normal[vi * 3]     += nx;
      normal[vi * 3 + 1] += ny;
      normal[vi * 3 + 2] += nz;
      count[vi]++;
    }
  }

  for (let i = 0; i < totalVerts; i++) {
    if (count[i] === 0) continue;
    const len = Math.hypot(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2]);
    if (len > 0) {
      normal[i * 3]     /= len;
      normal[i * 3 + 1] /= len;
      normal[i * 3 + 2] /= len;
    }
  }
}

/* ─── Component ──────────────────────────────────────────────────────────── */

/**
 * @param {{ onReady?: () => void }} props
 * @param {React.Ref} ref
 */
const TerrainCanvas = forwardRef(function TerrainCanvas({ onReady }, ref) {
  const { state, actions } = useApp();
  const canvasRef  = useRef(null);

  // Mutable GL state (not React state — lives across renders without re-render)
  const glRef = useRef({
    renderer: null,
    camera: null,
    scene: null,
    orbit: null,
    program: null,
    mesh: null,
    rafId: null,
    exaggeration: 1.5,
    wireframe: false,
    progress: 0,         // progressive reveal lerp target
    progressCurrent: 0,  // current interpolated value
    heightData: null,
    hmWidth: 0,
    hmHeight: 0,
    heightScale: 1,
    segs: LO_SEGS,
    textureReady: 0,
    disposed: false,
    cameraMode: 'orbit',        // 'orbit' | 'first-person' | 'top'
    fpTick: null,               // callback injected from useCameraController
    contoursEnabled: false,
    contourInterval: 5.0,
    minElevation: 0.0,
    maxElevation: 100.0,
    elevationSpan: 100.0,
  });

  /* ── Expose handle to parent ── */
  useImperativeHandle(ref, () => ({
    setExaggeration(v) {
      const g = glRef.current;
      g.exaggeration = v;
      if (g.program) {
        g.program.uniforms.uExaggeration.value = v;
      }
    },
    setWireframe(v) {
      const g = glRef.current;
      g.wireframe = v;
      if (g.mesh) {
        const gl = g.renderer?.gl;
        if (gl) {
          g.mesh.mode = v ? gl.LINES : gl.TRIANGLES;
        }
      }
    },
    resetCamera() {
      const g = glRef.current;
      if (g.camera && g.orbit) {
        g.camera.position.set(...DEFAULT_CAMERA_POS);
        g.orbit.target.set(0, 0, 0);
      }
      g.cameraMode = 'orbit';
      if (g.orbit) g.orbit.enabled = true;
      g.fpTick = null;
    },
    /** Register first-person tick callback — called every frame when mode is fp */
    setFpTick(fn) {
      glRef.current.fpTick = fn;
    },
    /** Set camera mode so render loop knows which controller is active */
    setCameraMode(newMode) {
      glRef.current.cameraMode = newMode;
      if (glRef.current.orbit) {
        glRef.current.orbit.enabled = (newMode === 'orbit');
      }
    },
    /** Expose raw glRef so useCameraController can read camera and heightData */
    getRef() { return glRef; },
    /** Expose canvas ref for pointer-lock in first-person mode */
    getCanvas() { return canvasRef; },
    /**
     * Capture a snapshot of current WebGL scene as PNG data URL
     * @returns {string|null}
     */
    captureSnapshot() {
      const g = glRef.current;
      if (g.renderer && g.scene && g.camera) {
        g.renderer.render({ scene: g.scene, camera: g.camera });
      }
      const canvas = canvasRef.current;
      if (!canvas) return null;
      try {
        return canvas.toDataURL('image/png');
      } catch (err) {
        console.error('[TerrainCanvas] captureSnapshot error', err);
        return null;
      }
    },
    /**
     * Swap the terrain texture to a new image URL (layer switch, task 8.2).
     * Camera and minimap state are preserved — only the texture changes.
     * @param {string} url - new texture URL
     * @param {number} colormapMode - 0=rgb, 1=greyscale, 2=viridis, 3=diverging
     */
    setLayerTexture(url, colormapMode = 0) {
      const g = glRef.current;
      if (!g.renderer || !g.program) return;
      const gl = g.renderer.gl;
      // Reset texture-ready flag for cross-fade
      if (g.program.uniforms.uTextureReady) {
        g.program.uniforms.uTextureReady.value = 0.0;
        g.textureReady = 0;
      }
      if (g.program.uniforms.uColormapMode) {
        g.program.uniforms.uColormapMode.value = colormapMode;
      }
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        if (g.disposed) return;
        const tex = new Texture(gl, {
          image: img,
          minFilter: gl.LINEAR_MIPMAP_LINEAR,
          magFilter: gl.LINEAR,
          wrapS: gl.CLAMP_TO_EDGE,
          wrapT: gl.CLAMP_TO_EDGE,
          generateMipmaps: true,
          flipY: true,
        });
        tex.needsUpdate = true;
        if (g.program.uniforms.uTexture) {
          g.program.uniforms.uTexture.value = tex;
        }
        // Animate cross-fade: ramp textureReady from 0→1 over 250ms
        const start = performance.now();
        function fadeIn() {
          if (g.disposed) return;
          const t = Math.min((performance.now() - start) / 250, 1);
          if (g.program.uniforms.uTextureReady) {
            g.program.uniforms.uTextureReady.value = t;
          }
          if (t < 1) requestAnimationFrame(fadeIn);
          else {
            g.textureReady = 1;
          }
        }
        requestAnimationFrame(fadeIn);
      };
      img.src = url;
    },
    /**
     * Toggle contour lines and set elevation interval (§21, task 8.4).
     * @param {boolean} enabled
     * @param {number} [interval]
     */
    setContours(enabled, interval) {
      const g = glRef.current;
      g.contoursEnabled = !!enabled;
      if (typeof interval === 'number' && interval > 0) {
        g.contourInterval = interval;
      }
      if (g.program) {
        if (g.program.uniforms.uContoursEnabled) {
          g.program.uniforms.uContoursEnabled.value = enabled ? 1.0 : 0.0;
        }
        if (g.program.uniforms.uContourInterval && typeof interval === 'number' && interval > 0) {
          g.program.uniforms.uContourInterval.value = interval;
        }
      }
    },
    /** Sample elevation from heightmap at normalised coordinates [0, 1] */
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
    /** Convert a client mouse event to terrain coordinates { x, z, elevation } */
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
      let wx = ndcX;
      let wz = -ndcY;
      if (g.orbit?.target && g.camera) {
        const dist = g.camera.position.distance ? g.camera.position.distance(g.orbit.target) : 2.5;
        wx = (g.orbit.target.x ?? 0) + ndcX * dist * 0.45;
        wz = (g.orbit.target.z ?? 0) - ndcY * dist * 0.45;
      }
      const nx = Math.max(0, Math.min(1, (wx + 1) / 2));
      const nz = Math.max(0, Math.min(1, (wz + 1) / 2));
      const elevation = this.sampleElevation(nx, nz) ?? 0;
      return { x: nx * 2 - 1, z: nz * 2 - 1, elevation };
    },
  }));

  /* ── Resize handler ── */
  const handleResize = useCallback(() => {
    const g = glRef.current;
    if (!g.renderer || !g.camera || !canvasRef.current) return;
    const { clientWidth: w, clientHeight: h } = canvasRef.current.parentElement;
    g.renderer.setSize(w, h);
    g.camera.perspective({ aspect: w / h });
  }, []);

  /* ── Main init effect ── */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state.scene?.scene_id) return;

    const g = glRef.current;
    g.disposed = false;

    // ── 4.1: Init Renderer, Camera, Scene ──
    const container = canvas.parentElement;
    const w = container.clientWidth;
    const h = container.clientHeight;

    const renderer = new Renderer({
      canvas,
      width: w,
      height: h,
      dpr: Math.min(window.devicePixelRatio, 2),
      antialias: true,
      preserveDrawingBuffer: true,
      alpha: false,
    });
    const gl = renderer.gl;
    gl.clearColor(0.028, 0.035, 0.055, 1); // near --dw-void
    g.renderer = renderer;

    const camera = new Camera(gl, { fov: 45, near: 0.01, far: 100 });
    camera.position.set(...DEFAULT_CAMERA_POS);
    g.camera = camera;

    const scene = new Transform();
    g.scene = scene;

    // Orbit controller (task 5.1 lives here, basic setup; full camera system in Phase 5)
    const orbit = new Orbit(camera, {
      element: canvas,
      target: { x: 0, y: 0, z: 0 },
      ease: 0.08,
      inertia: 0.7,
      enablePan: true,
      panSpeed: 0.5,
      minDistance: 0.5,
      maxDistance: 8,
      minPolarAngle: 0.1,
      maxPolarAngle: Math.PI * 0.48,
    });
    g.orbit = orbit;

    // ── 4.4: Lighting uniforms (baked into program below) ──
    // Sun direction: upper left at ~45° elevation (geospatial look §20)
    const sunDir = [0.6, 0.9, 0.4].map(v => v / Math.hypot(0.6, 0.9, 0.4));

    // ── 4.2: Create placeholder program + 1×1 black heightmap ──
    // We create a 1px black texture so the program compiles immediately
    const blackPixel = new Uint8Array([0, 0, 0, 255]);
    const hmTex = new Texture(gl, {
      image: { data: blackPixel, width: 1, height: 1 },
      width: 1, height: 1,
      minFilter: gl.LINEAR,
      magFilter: gl.LINEAR,
      wrapS: gl.CLAMP_TO_EDGE,
      wrapT: gl.CLAMP_TO_EDGE,
      flipY: false,
    });

    const rgbTex = new Texture(gl, {
      image: { data: blackPixel, width: 1, height: 1 },
      width: 1, height: 1,
      minFilter: gl.LINEAR_MIPMAP_LINEAR,
      magFilter: gl.LINEAR,
      wrapS: gl.CLAMP_TO_EDGE,
      wrapT: gl.CLAMP_TO_EDGE,
      generateMipmaps: true,
    });

    const program = new Program(gl, {
      vertex: VERT,
      fragment: FRAG,
      uniforms: {
        uHeightmap:    { value: hmTex },
        uTexture:      { value: rgbTex },
        uExaggeration: { value: g.exaggeration },
        uHeightScale:  { value: 1.0 },
        uProgress:     { value: 0.0 },
        uSunDir:       { value: sunDir },
        uSunColor:     { value: [1.0, 0.95, 0.85] },
        uAmbient:      { value: 0.32 },
        uTextureReady: { value: 0.0 },
        uColormapMode:    { value: 0.0 },  // 0=rgb, 1=greyscale, 2=viridis, 3=diverging
        uContoursEnabled: { value: g.contoursEnabled ? 1.0 : 0.0 },
        uContourInterval: { value: g.contourInterval },
        uElevationSpan:   { value: g.elevationSpan },
        uMinElevation:    { value: g.minElevation },
      },
      transparent: false,
      depthTest: true,
      depthWrite: true,
    });
    g.program = program;

    // ── 4.7: Low-resolution flat plane as placeholder ──
    const loGeo = new Geometry(gl, {
      position: { size: 3, data: new Float32Array([(LO_SEGS + 1) ** 2 * 3]) },
      normal:   { size: 3, data: new Float32Array([(LO_SEGS + 1) ** 2 * 3]) },
      uv:       { size: 2, data: new Float32Array([(LO_SEGS + 1) ** 2 * 2]) },
      index:    { data: new Uint16Array([LO_SEGS ** 2 * 6]) },
    });

    // Build a flat LO_SEGS×LO_SEGS plane (no height) immediately
    const flatGeo = buildFlatPlane(gl, LO_SEGS);
    const mesh = new Mesh(gl, { geometry: flatGeo, program });
    mesh.setParent(scene);
    g.mesh = mesh;
    g.segs = LO_SEGS;

    // ── Render loop ──
    let lastTime = 0;
    function render(time) {
      if (g.disposed) return;
      g.rafId = requestAnimationFrame(render);

      const dt = Math.min((time - lastTime) / 1000, 0.1);
      lastTime = time;

      // Smooth progress reveal (task 4.7 — 600ms lerp)
      if (g.progressCurrent < g.progress) {
        g.progressCurrent = Math.min(g.progressCurrent + dt * (1 / 0.6), g.progress);
        if (program.uniforms.uProgress) {
          program.uniforms.uProgress.value = g.progressCurrent;
        }
      }

      orbit.update();
      // First-person tick: called every frame if mode is fp and callback is registered
      if (g.cameraMode === 'first-person' && typeof g.fpTick === 'function') {
        g.fpTick(dt);
      }
      renderer.render({ scene, camera });
    }
    g.rafId = requestAnimationFrame(render);

    // ── Resize ──
    const ro = new ResizeObserver(handleResize);
    ro.observe(container);

    // ── 4.2 + 4.3: Fetch terrain data from API ──
    const sceneId = state.scene.scene_id;
    loadTerrainData(gl, g, program, mesh, scene, sceneId, actions);

    return () => {
      g.disposed = true;
      cancelAnimationFrame(g.rafId);
      ro.disconnect();
      // GPU disposal (§D06)
      flatGeo.remove?.();
      program.remove?.();
      // Don't call renderer dispose — canvas still in DOM during unmount
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.scene?.scene_id]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: 'block',
        width: '100%',
        height: '100%',
        background: '#07090e',
        outline: 'none',
      }}
      tabIndex={0}
      aria-label="3D terrain viewer"
    />
  );
});

export default TerrainCanvas;

/* ─── Async terrain data loader ────────────────────────────────────────── */

async function loadTerrainData(gl, g, program, mesh, scene, sceneId, actions) {
  try {
    // Fetch terrain metadata
    const terrainMeta = await getTerrain(sceneId);
    if (g.disposed) return;

    const { heightmap_url, texture_url, height_scale, min_elevation, max_elevation } = terrainMeta;

    // Normalise height scale: if backend gives absolute, we store it; else default to 1
    const hs = typeof height_scale === 'number' && height_scale > 0 ? height_scale : 1.0;
    g.heightScale = hs;
    if (program.uniforms.uHeightScale) {
      program.uniforms.uHeightScale.value = hs;
    }

    const minElev = typeof min_elevation === 'number' ? min_elevation : 0.0;
    const maxElev = typeof max_elevation === 'number' ? max_elevation : (minElev + hs * 100.0);
    const span = Math.max(0.001, maxElev - minElev);
    g.minElevation = minElev;
    g.maxElevation = maxElev;
    g.elevationSpan = span;
    if (program.uniforms.uMinElevation) program.uniforms.uMinElevation.value = minElev;
    if (program.uniforms.uElevationSpan) program.uniforms.uElevationSpan.value = span;

    // ── 4.2: Decode heightmap ──
    const { data, width, height } = await decodeHeightmap(heightmap_url);
    if (g.disposed) return;

    g.heightData  = data;
    g.hmWidth     = width;
    g.hmHeight    = height;

    // Upload heightmap as GL texture (full resolution)
    const hmTex = new Texture(gl, {
      image: { data: new Uint8Array(data.length * 4), width, height }, // placeholder size
      width,
      height,
      internalFormat: gl.R8 ?? gl.LUMINANCE,
      format: gl.RED ?? gl.LUMINANCE,
      type: gl.UNSIGNED_BYTE,
      minFilter: gl.LINEAR,
      magFilter: gl.LINEAR,
      wrapS: gl.CLAMP_TO_EDGE,
      wrapT: gl.CLAMP_TO_EDGE,
      flipY: false,
      generateMipmaps: false,
    });

    // Write the actual 8-bit data
    const uint8 = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i++) uint8[i] = Math.round(data[i] * 255);
    hmTex.image = { data: uint8, width, height };
    hmTex.needsUpdate = true;
    program.uniforms.uHeightmap.value = hmTex;

    // ── 4.7 — Low-res mesh with actual height data ──
    const loGeo = buildTerrainGeometry(gl, data, width, height, LO_SEGS, hs, g.exaggeration);
    const oldGeo = mesh.geometry;
    mesh.geometry = loGeo;
    oldGeo?.remove?.();
    g.progress = 1.0; // start progressive reveal

    // Mark terrain ready in state machine
    actions.terrainReady(terrainMeta);

    // ── 4.7 — High-res swap (async, after state machine transitions) ──
    setTimeout(async () => {
      if (g.disposed) return;
      const hiGeo = buildTerrainGeometry(gl, data, width, height, HI_SEGS, hs, g.exaggeration);
      const lo = mesh.geometry;
      mesh.geometry = hiGeo;
      lo?.remove?.();
      g.segs = HI_SEGS;
    }, 100);

    // ── 4.3: Load RGB/depth texture ──
    if (texture_url) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        if (g.disposed) return;
        const tex = new Texture(gl, {
          image: img,
          minFilter: gl.LINEAR_MIPMAP_LINEAR,
          magFilter: gl.LINEAR,
          wrapS: gl.CLAMP_TO_EDGE,
          wrapT: gl.CLAMP_TO_EDGE,
          generateMipmaps: true,
          flipY: true,
        });
        tex.needsUpdate = true;
        program.uniforms.uTexture.value = tex;
        program.uniforms.uTextureReady.value = 1.0;
        g.textureReady = 1;
      };
      img.src = texture_url;
    }
  } catch (err) {
    if (!g.disposed) {
      actions.terrainFail({
        code: err.code ?? 'TERRAIN_LOAD_ERROR',
        message: err.message ?? 'Failed to load terrain data.',
        recoverable: true,
      });
    }
  }
}

/* ─── Flat plane builder (4.7 placeholder) ─────────────────────────────── */

function buildFlatPlane(gl, segs) {
  const num = (segs + 1) ** 2;
  const numIdx = segs * segs * 6;

  const position = new Float32Array(num * 3);
  const normal   = new Float32Array(num * 3);
  const uv       = new Float32Array(num * 2);
  const index    = numIdx > 65536 ? new Uint32Array(numIdx) : new Uint16Array(numIdx);

  let i = 0;
  for (let iy = 0; iy <= segs; iy++) {
    for (let ix = 0; ix <= segs; ix++, i++) {
      const u = ix / segs;
      const v = iy / segs;
      position[i * 3]     = (u - 0.5) * 2;
      position[i * 3 + 1] = 0;
      position[i * 3 + 2] = (v - 0.5) * 2;
      normal[i * 3 + 1] = 1;
      uv[i * 2]     = u;
      uv[i * 2 + 1] = 1 - v;
    }
  }

  let ii = 0;
  for (let iy = 0; iy < segs; iy++) {
    for (let ix = 0; ix < segs; ix++) {
      const a = ix + iy * (segs + 1);
      const b = ix + (iy + 1) * (segs + 1);
      const c = ix + (iy + 1) * (segs + 1) + 1;
      const d = ix + iy * (segs + 1) + 1;
      index[ii * 6] = a; index[ii * 6 + 1] = b; index[ii * 6 + 2] = d;
      index[ii * 6 + 3] = b; index[ii * 6 + 4] = c; index[ii * 6 + 5] = d;
      ii++;
    }
  }

  return new Geometry(gl, {
    position: { size: 3, data: position },
    normal:   { size: 3, data: normal },
    uv:       { size: 2, data: uv },
    index:    { data: index },
  });
}
