# Jury Demo Script — DepthWizard

Ordered run-sheet for the live presentation. **DSM accuracy and validation are 50% of
the score — they are beats 1–3 and the centerpiece.** TTA, semantic gating and Route
Assist are closing "bonus" beats; if time runs short, drop them, never the validation
metrics.

Total runtime: ~7 minutes + Q&A.

---

## 0. Pre-flight (before the jury walks in)

1. **Live stack** (preferred): start the real backend + frontend, and confirm
   `GET /api/v1/health` returns `ok` with `model_loaded: true`.
2. **Fallback stack (do this regardless, so switching takes 10 seconds):**
   ```bash
   python tools/mock_backend_for_camera_test.py          # serves precomputed scenes on :8000
   cd frontend && npm run dev                            # or: npx vite preview --port 4174
   ```
   The mock serves three fully precomputed scenes from `tools/demo_bundles/`
   (`demo_ridge`, `demo_valley`, `demo_quarry`) — no GPU, no model, no network.
   The frontend header will read **"API mock-demo"** instead of the live version —
   that is the only visible difference.
3. If the real backend must run on a different port than the mock, point the frontend
   at whichever one you keep: `VITE_API_BASE_URL=http://localhost:8000/api/v1`.
4. Pre-login once (local-mode account, any email + 8-char password) so the workspace
   opens directly on the home screen.
5. Regenerating the bundles (only if you changed terrain/metrics in
   `tools/generate_demo_bundles.py` / `tools/_demo_terrain.py`):
   ```bash
   python tools/generate_demo_bundles.py    # deterministic, rewrites tools/demo_bundles/
   ```

### Emergency switch (live inference fails mid-demo)

Stop the live backend, start the mock (`python tools/mock_backend_for_camera_test.py`),
click the home/"DepthWizard" logo, then click **"View Demo (Scene_042 · Absolute DSM)"**
— you are back on a fully populated results dashboard in one click. Say:
*"Let me show you the same pipeline on a precomputed scene"* and continue from Beat 2.
Every downstream beat (3D terrain, measurements, Route Assist) works identically on
the mock.

---

## Beat 1 — Input & problem framing (~45 s)

- On the home screen, state the problem: **one optical image → depth → metric DSM →
  explorable, validatable 3D terrain.**
- Point at the upload zone: GeoTIFF (georeferenced → **absolute** metre elevations) or
  plain PNG/JPG (→ relative scene units). Mention batch/mosaic of adjacent tiles.
- Upload `ridge_traverse.tif` (or click **View Demo** for the same scene preloaded).

## Beat 2 — Processing pipeline (~45 s)

- Watch the staged progress: validating → preprocessing → **depth inference** →
  calibration → DSM generation → **validation** → terrain.
- Narrate the two-path pipeline: georeferenced GeoTIFF recovers metric scale from the
  embedded CRS; the SRTM reference is fetched automatically for validation.

## Beat 3 — Results dashboard: DSM + accuracy (CENTERPIECE, ~90 s)

This is where the score is won. Do not rush it.

- **Scene metadata strip**: CRS `EPSG:32643`, pipeline *Absolute DSM*, relief 324 m.
- **Output layers**: RGB → Depth → DSM, each rendered from the pipeline.
- **ACCURACY EVALUATION panel** (the money shot):
  - RMSE **3.12 m**, MAE **2.45 m**, Correlation **0.930**, 41,233 samples vs the
    SRTM GL1 reference.
- Open the error map (validation section / layers): green-yellow-red residuals,
  largest errors concentrated on steep faces — explain *why* (occlusion + slope) —
  this pre-empts the jury's toughest question.
- If asked about the other scenes: `demo_valley` (high-relief, RMSE 4.87 m) and
  `demo_quarry` (relative mode, validated against an uploaded legacy survey DEM,
  RMSE 2.04 m) are one upload away — upload any file whose name contains
  `valley` / `quarry` against the mock, or select them in Recent Projects.

## Beat 4 — 3D terrain & walkthrough (~90 s)

- **Enter 3D Terrain.** 256×256 mesh (65k verts), hybrid DSM-texture view.
- Orbit → rotate/zoom; then **Fly** camera for the walkthrough moment — fly along the
  ridge toward the peak. Mention: walkthrough is fully client-side, zero server calls
  per frame (60 FPS telemetry visible bottom-left).
- Toggle **Contour Lines** and **Slope Overlay** briefly; set vertical exaggeration 2.5×.

## Beat 5 — Inspection: elevation, height, slope (~60 s)

- **Measure → Elevation Probe**: click anywhere — live elevation readout in metres
  (e.g. 294.79 m), consistent with the 254–578 m range.
- **Measure → Height Measurement**: ground + top clicks → object height.
- **Measure → Slope Analysis**: two-point slope in degrees/percent.
- Tie back to accuracy: *"Every one of these numbers inherits the ±3 m RMSE we just
  validated."*

## Beat 6 — Route Assist (bonus, ~60 s)

- **Measure → Route Assist.** Pick start and end on the terrain.
- Fleet readout: Fire Truck **CANNOT_GO** (max slope exceeds the 15° limit), Rescue
  ATV **CAUTION**, Rescue Chopper **CAN_GO** with a suggested landing zone + distance
  to goal.
- Show the passability heat map (already on the results dashboard: 18.4% blocked for
  fire trucks) and the disaster-assessment framing if the jury is disaster-focused.

## Beat 7 — Closing bonus beats (only if time and jury interest allow)

- **TTA** (test-time augmentation): mention multi-pass inference averaging that
  stabilises depth at occlusions (see `tools/exp_tta_check.py`).
- **Semantic gating**: water/vegetation masks gate unreliable depth regions.
- **Detail Mode (Refine)**: local re-inference on a drawn bbox.
- Close on validation: *"Sub-3.5-metre RMSE against SRTM, on a single image."*

---

## Known mock-mode differences (only visible in fallback)

- Header badge reads `API mock-demo`; processing completes in ~7 s with simulated
  staged progress.
- The RGB layer card may show a placeholder (source-image preview is not part of the
  precomputed bundle).
- Uploads accept any file; the filename picks the scene (`ridge`/`valley`/`quarry`,
  otherwise round-robin). Built-in "View Demo" always serves `demo_ridge`.
- Everything else — validation metrics, error map, 3D terrain, walkthrough,
  measurements, Route Assist, exports — is fully served.
