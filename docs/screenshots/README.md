# Frontend visualization evidence

Rendered proof that the 3D terrain pipeline works end-to-end, captured from
the real React/Vite frontend (`frontend/`, Vite dev server) driven through a
full user flow: landing → login → demo scene (upload → staged processing →
results dashboard) → 3D terrain workspace.

| file | what it proves |
|---|---|
| `01_orbit_rgb_drape.png` | Orbit mode: GPU-shader heightmap with the RGB texture projected — no visible tile seams, the white summit and brown ridge in the source image (top-left minimap) land exactly on the mesh (RGB-to-terrain alignment). |
| `02_layer_depth.png` | Depth layer: grayscale value encoding is legible (before the shader fix this rendered near-black — see `bug_depth_layer_before_fix.png`). |
| `03_layer_metric_dsm.png` | Metric DSM layer: viridis colormap, value-encoded, correctly lit. |
| `04_walkthrough_first_person.png` | First-person walkthrough: pointer-lock mode at 1.7 m eye height, HUD (altitude/heading/slope/position), movement joystick, minimap camera marker, RGB-draped ridge ahead. |
| `terramesh_demo.webm` | Screen recording (~10 s): layer switching RGB → Depth → Metric DSM in walkthrough, back to orbit, another layer sweep, orbit rotation. |
| `bug_depth_layer_before_fix.png` | The Depth layer BEFORE the shader fix — kept as the honest before/after pair for the colormap-lighting fix below. |

## Rendering bugs found and fixed during this evidence pass

All fixes are in `frontend/src/components/TerrainViewer/TerrainCanvas.jsx`
and `frontend/src/hooks/useCameraController.js`:

1. **Depth/DSM/slope layers rendered near-black.** The sun-diffuse +
   cool-shadow grading was applied to value-encoded colormaps, double-darkening
   them. Scientific layers (colormap modes ≠ RGB) now render near-unlit;
   full lighting stays on the RGB drape.
2. **Walkthrough camera spawned below the rendered surface** (seeing the
   mesh underside as giant smooth sheets, with real peaks poking through as
   "floating fragments"). The tick now clamps the camera to
   `ground + eye height` every frame — self-healing even when the vertical
   exaggeration changes under an already-spawned camera.
3. **CPU/GPU height sampling made explicitly consistent** (`DataTexture`
   uploaded with `flipY: false`, vertex shader samples `(u, 1-v)`), so the
   collision/spawn field can never mirror against the rendered surface.
4. **Backface holes on near-vertical exaggerated ridges**: terrain material
   is now `DoubleSide` with the normal flipped for backfaces in the fragment
   shader.
5. **Logarithmic depth buffer** enabled: the scene spans ~1 cm (eye height)
   to ~40 km (camera far plane); the linear depth buffer quantized at
   distance.

## Reproduce

```bash
# terminal 1: mock backend (demo scenes; no GPU needed)
python tools/mock_backend_for_camera_test.py          # :8000
# terminal 2: frontend
cd frontend && npm install && npm run dev             # :5173
```

Open http://localhost:5173 → Try Prototype → (local login: any email +
8-char password) → View Demo (Scene_042) → Enter 3D Terrain → switch layers,
switch to Fly, click the terrain to capture the cursor (WASD / Space / Ctrl).
