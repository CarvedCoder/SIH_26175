/**
 * DepthWizard Geospatial Engine — TerrainMaterial
 *
 * Unified ShaderMaterial for chunked terrain rendering:
 *   - Directional sun lighting with tonal shadow grading
 *   - Texture tiling support with per-chunk UV offset/scale
 *   - Real-world metric elevation contours (e.g. 5m minor, 25m major intervals)
 *   - Colormap blending (RGB, Greyscale, Viridis, Diverging)
 *   - Hybrid mode: photographic RGB drape + enhanced relief shading
 *   - Dynamic atmospheric fog scaled to scene diagonal
 *   - Independent semantic class overlays with optional confidence
 *     attenuation, without rebuilding geometry
 *   - Wireframe = per-tile mesh-grid overlay drawn on tile-local UVs, so
 *     it exactly matches each chunk's real triangulation grid at every
 *     LOD and never renders skirt walls as giant stretched triangles
 *
 * COLOR-SPACE CONTRACT (single convention, do not scatter):
 *   - RGB drape textures (mode 0) are sRGB-encoded: the GPU decodes them
 *     to linear on sample, lighting happens in linear, and the fragment
 *     converts back to sRGB before output.
 *   - Value-encoded layer textures (greyscale depth/DSM/slope/error, the
 *     semantic label texture) are RAW DATA: they must be loaded with
 *     NoColorSpace so texSample.r is the exact stored value. Colormap
 *     palettes (viridis/diverging/grey) output display-space colors and
 *     skip the linear->sRGB conversion.
 */

import * as THREE from 'three';

const TERRAIN_VERT = /* glsl */ `
  uniform float uExaggeration;
  uniform float uMinElevation;
  uniform vec2 uUvOffset;
  uniform vec2 uUvScale;

  varying vec2 vUv;       // GLOBAL raster UV [0, 1] — semantic masks, colormaps, mesh overlay
  varying vec2 vTileUv;   // Tile-LOCAL UV [0, 1] — streamed per-tile textures + mesh grid
  varying vec3 vNormal;
  varying float vElevation;
  varying float vViewDist;

  void main() {
    vTileUv = uv;
    vUv = uUvOffset + uv * uUvScale;

    // Apply visualization-only vertical exaggeration around the min elevation datum
    vec3 displacedPos = position;
    if (abs(uExaggeration - 1.0) > 0.001) {
      float baseH = max(0.0, position.y - uMinElevation);
      displacedPos.y = uMinElevation + baseH * uExaggeration;
    }

    vElevation = position.y;
    // Calculate world normal from model matrix
    vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);

    vec4 viewPos = modelViewMatrix * vec4(displacedPos, 1.0);
    vViewDist = -viewPos.z;

    gl_Position = projectionMatrix * viewPos;
  }
`;

const TERRAIN_FRAG = /* glsl */ `
  uniform sampler2D uTexture;
  uniform float uTextureReady;
  uniform float uPerTileTexture;  // 1 = uTexture is this tile's own streamed texture (sample with vTileUv)
  uniform vec3 uSunDir;
  uniform vec3 uSunColor;
  uniform float uAmbient;
  uniform float uReliefStrength; // 0 = photographic drape, 1 = hybrid relief emphasis
  uniform float uColormapMode;   // 0=rgb/solid, 1=greyscale, 2=viridis, 3=diverging
  uniform float uMinElevation;
  uniform float uMaxElevation;

  uniform float uContoursEnabled;
  uniform float uContourInterval; // interval in meters

  uniform float uMeshEnabled;
  uniform vec3 uMeshColor;
  uniform float uMeshDensity;    // per-tile grid density = the tile's segment count

  uniform float uFogEnabled;
  uniform float uFogNear;
  uniform float uFogFar;
  uniform vec3 uFogColor;

  // Semantic overlay and class isolation
  uniform sampler2D uSemanticTex;
  uniform float uSemanticEnabled;
  uniform float uSemanticOpacity;
  uniform float uSemanticHighlightClass;
  uniform sampler2D uSemanticConfTex;
  uniform float uSemanticConfEnabled;

  varying vec2 vUv;
  varying vec2 vTileUv;
  varying vec3 vNormal;
  varying float vElevation;
  varying float vViewDist;

  // uTexture is either the global raster image (sampled with global UV) or a
  // per-tile streamed texture (sampled with tile-local UV). Terrain LOD and
  // texture LOD are decoupled: the tile's texture level is chosen
  // independently of its geometry subdivision (spec §19).
  vec2 textureUv() {
    return uPerTileTexture > 0.5 ? vTileUv : vUv;
  }

  // Canonical semantic palette
  vec3 getSemanticColor(float classId) {
    int c = int(classId + 0.5);
    if (c == 0) return vec3(0.906, 0.298, 0.235); // building (#E74C3C)
    if (c == 1) return vec3(0.180, 0.800, 0.443); // vegetation (#2ECC71)
    if (c == 2) return vec3(0.608, 0.608, 0.608); // road (#9B9B9B)
    if (c == 3) return vec3(0.204, 0.596, 0.859); // water (#3498DB)
    if (c == 4) return vec3(0.824, 0.706, 0.549); // ground (#D2B48C)
    return vec3(0.584, 0.510, 0.659);             // other (#9582A8)
  }

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

  vec3 diverging(float t) {
    if (t < 0.5) {
      return mix(vec3(0.145, 0.396, 0.933), vec3(0.87, 0.89, 0.93), t * 2.0);
    } else {
      return mix(vec3(0.87, 0.89, 0.93), vec3(0.933, 0.145, 0.145), (t - 0.5) * 2.0);
    }
  }

  // linear -> sRGB (legacy GLSL 1.0: no built-in, use the standard power curve)
  vec3 linearToSrgb(vec3 c) {
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
  }

  void main() {
    // Default warm architectural/geospatial relief tone
    vec3 baseColor = vec3(0.68, 0.65, 0.58);

    int mode = int(uColormapMode + 0.5);

    if (mode > 0) {
      // Analytical colormap mode (1=greyscale, 2=viridis, 3=diverging).
      // Layer textures are RAW DATA (NoColorSpace): texSample.r is the
      // exact normalized value — no hidden sRGB decode must touch it.
      float t = 0.5;
      if (uTextureReady > 0.5) {
        vec4 texSample = texture2D(uTexture, textureUv());
        t = texSample.r;
      } else {
        float span = max(0.001, uMaxElevation - uMinElevation);
        t = clamp((vElevation - uMinElevation) / span, 0.0, 1.0);
      }

      if (mode == 1) {
        baseColor = vec3(t);
      } else if (mode == 2) {
        baseColor = viridis(t);
      } else if (mode == 3) {
        baseColor = diverging(t);
      }
    } else {
      // RGB Satellite/Aerial or Solid relief mode
      if (uTextureReady > 0.5) {
        vec4 texSample = texture2D(uTexture, textureUv());
        // Nodata pixels (alpha=0 — source-raster nodata frame) fall back
        // to a dim neutral relief tone instead of literal black plates.
        vec3 noDataTone = vec3(0.16, 0.18, 0.21);
        baseColor = mix(noDataTone, texSample.rgb, texSample.a);
      }
    }

    // Optional semantic overlay
    if (uSemanticEnabled > 0.5) {
      vec4 semSample = texture2D(uSemanticTex, vUv);
      float rawClass = floor(semSample.r * 255.0 + 0.5);
      vec3 semColor = getSemanticColor(rawClass);

      // Confidence-aware attenuation: uSemanticConfTex carries the
      // per-pixel segmentation confidence (raw [0,1] data). Low-confidence
      // pixels fade back to the base layer instead of pretending every
      // mask pixel is equally reliable.
      float opacity = uSemanticOpacity;
      if (uSemanticConfEnabled > 0.5) {
        opacity *= clamp(texture2D(uSemanticConfTex, vUv).r, 0.0, 1.0);
      }

      if (uSemanticHighlightClass >= -0.5) {
        float targetClass = floor(uSemanticHighlightClass + 0.5);
        if (abs(rawClass - targetClass) < 0.2) {
          baseColor = mix(baseColor, semColor, 0.88);
        } else {
          baseColor = mix(baseColor, semColor, 0.15) * 0.38;
        }
      } else {
        baseColor = mix(baseColor, semColor, opacity);
      }
    }

    // Outdoor Geospatial Illumination: Hemispheric Sky/Ground Ambient + Direct Sun
    // Value-encoded colormaps (depth greyscale, DSM/slope viridis, error
    // diverging) are VALUE-ENCODED — full sun shading double-darkens them
    // into unreadable black, so they render near-unlit with a faint
    // slope-relief cue. Backfaces flip the normal (terrain material is
    // DoubleSide).
    vec3 nrm = gl_FrontFacing ? vNormal : -vNormal;
    float hemi = clamp(nrm.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyAmbient = vec3(0.55, 0.63, 0.74) * uAmbient;
    vec3 groundAmbient = vec3(0.26, 0.28, 0.32) * uAmbient;
    vec3 ambientLight = mix(groundAmbient, skyAmbient, hemi);

    float diff = max(dot(nrm, uSunDir), 0.0);

    float elevSpan = max(0.001, uMaxElevation - uMinElevation);
    float elevT = clamp((vElevation - uMinElevation) / elevSpan, 0.0, 1.0);

    vec3 shaded;
    if (mode == 0) {
      // Lighting math in LINEAR (the sRGB drape texture was decoded by
      // the GPU on sample).
      float sunBoost = 1.0 + uReliefStrength * 0.85;   // hybrid: stronger relief
      float shadeAo = 1.0 - uReliefStrength * 0.22 * (1.0 - diff); // hybrid: shadowed slopes deepen
      vec3 sunLight = uSunColor * (diff * 0.85 * sunBoost);
      // Hybrid elevation modulation: a gentle cool->warm high-light lift
      // keeps the imagery recognizable while tying brightness to altitude.
      vec3 elevMod = mix(vec3(1.0), vec3(0.97, 1.0, 1.05), (elevT - 0.5) * uReliefStrength);
      vec3 lit = baseColor * (ambientLight + sunLight) * shadeAo * elevMod;
      // The drape texture is sRGB-encoded: convert the linear lit result
      // back to display space. This is the ONLY colorspace conversion —
      // colormap modes below output display-space colors directly.
      shaded = linearToSrgb(lit);
    } else {
      // Colormaps are display-space values: near-unlit, faint relief cue.
      shaded = baseColor * (0.88 + 0.12 * diff);
    }

    // Metric contour lines (meters)
    #ifdef GL_OES_standard_derivatives
    if (uContoursEnabled > 0.5 && uContourInterval > 0.1) {
      float elev = vElevation;
      float lineDist = abs(fract(elev / uContourInterval - 0.5) - 0.5) * uContourInterval;
      float fw = max(fwidth(elev), 0.01);
      float contour = 1.0 - smoothstep(0.0, fw * 1.5, lineDist);

      float majorInterval = uContourInterval * 5.0;
      float majorDist = abs(fract(elev / majorInterval - 0.5) - 0.5) * majorInterval;
      float majorContour = 1.0 - smoothstep(0.0, fw * 2.2, majorDist);

      vec3 contourColor = vec3(0.06, 0.09, 0.14);
      shaded = mix(shaded, contourColor, clamp(contour * 0.45 + majorContour * 0.35, 0.0, 0.85));
    }
    #endif

    // Wireframe: per-tile MESH-GRID overlay on tile-local UVs. This is the
    // single wireframe system (brute-force material.wireframe is never
    // used): the grid density equals the chunk's segment count, so the
    // drawn lines ARE the chunk's real quad grid — aligned at every LOD,
    // no giant stretched triangles, no z-fighting (same-surface mix), and
    // skirt walls show only their vertical columns.
    if (uMeshEnabled > 0.5) {
      vec2 g = vTileUv * uMeshDensity;
      vec2 f = fract(g);
      vec2 d = min(f, 1.0 - f);
      vec2 fw2 = fwidth(g) * 1.1;
      float line = 1.0 - min(smoothstep(0.0, fw2.x, d.x), smoothstep(0.0, fw2.y, d.y));
      shaded = mix(shaded, uMeshColor, line * 0.55);
    }

    // Atmospheric depth fog
    if (uFogEnabled > 0.5) {
      float fogFactor = clamp((vViewDist - uFogNear) / max(uFogFar - uFogNear, 0.1), 0.0, 0.85);
      shaded = mix(shaded, uFogColor, fogFactor);
    }

    gl_FragColor = vec4(shaded, 1.0);
  }
`;

/**
 * Creates a configured Terrain ShaderMaterial.
 */
export function createTerrainMaterial(options = {}) {
  const sunDir = [0.45, 0.78, 0.42].map(v => v / Math.hypot(0.45, 0.78, 0.42));

  const emptyTex = createEmptyDataTexture();

  const uniforms = {
    uTexture: { value: options.texture || emptyTex },
    uPerTileTexture: { value: 0.0 },
    uTextureReady: { value: options.texture ? 1.0 : 0.0 },
    uSunDir: { value: new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]) },
    uSunColor: { value: new THREE.Color(1.0, 0.96, 0.88) },
    uAmbient: { value: 0.42 },
    uReliefStrength: { value: 0.0 },
    uExaggeration: { value: options.exaggeration || 1.0 },
    uMinElevation: { value: options.minElevation || 0.0 },
    uMaxElevation: { value: options.maxElevation || 100.0 },
    uUvOffset: { value: new THREE.Vector2(0, 0) },
    uUvScale: { value: new THREE.Vector2(1, 1) },
    uColormapMode: { value: options.colormapMode || 0.0 },
    uContoursEnabled: { value: 0.0 },
    uContourInterval: { value: 5.0 },
    uMeshEnabled: { value: options.meshEnabled ? 1.0 : 0.0 },
    uMeshColor: { value: new THREE.Color(0.12, 0.16, 0.20) },
    // Per-tile grid density — matches the chunk segment count so wireframe
    // lines trace the real geometry grid (set per clone by LODManager).
    uMeshDensity: { value: options.segments || 32.0 },
    uFogEnabled: { value: 0.0 },
    uFogNear: { value: 200.0 },
    uFogFar: { value: 8000.0 },
    uFogColor: { value: new THREE.Color(0.028, 0.035, 0.055) },
    uSemanticTex: { value: emptyTex },
    uSemanticEnabled: { value: 0.0 },
    uSemanticOpacity: { value: 0.65 },
    uSemanticHighlightClass: { value: -1.0 },
    uSemanticConfTex: { value: emptyTex },
    uSemanticConfEnabled: { value: 0.0 },
  };

  const material = new THREE.ShaderMaterial({
    vertexShader: TERRAIN_VERT,
    fragmentShader: TERRAIN_FRAG,
    uniforms,
    extensions: {
      derivatives: true,
    },
    // NOTE: brute-force material.wireframe is deliberately never enabled —
    // the wireframe view is the uMeshEnabled per-tile grid overlay above.
    wireframe: false,
    side: THREE.DoubleSide,
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });

  return material;
}

function createEmptyDataTexture() {
  const pixel = new Uint8Array([128, 128, 128, 255]);
  const tex = new THREE.DataTexture(pixel, 1, 1, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return tex;
}
