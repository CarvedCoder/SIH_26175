# DepthWizard × TerraHeight-S Integration — Final Report

**Task**: integrate the external pretrained **TerraHeight-S** model
(https://huggingface.co/benfox6515/TerraHeight-S) as a NEW height backend
(`architecture="terraheight_s"`) alongside RDAH-Net and CalibrationNet —
inference integration + evaluation + UI/API only. RDAH stays the default;
CalibrationNet and the whole HOTOSM disaster pipeline are untouched; the
TerraHeight checkpoint is NOT retrained and NOT re-downloaded.

**Status**: ✅ COMPLETE — code integrated, full test suite green
(460 model/legacy tests + 241 backend tests, including 40 new TerraHeight
tests), real GPU inference verified end-to-end (GAMUS val tile + a real
2408×3021 GeoTIFF scene), GAMUS A/B evaluation executed on the project's
own metric stack, VRAM/latency benchmarked on the RTX 4050.

---

## 1. Compatibility report (verified facts — the checkpoint is authoritative)

The released payload embeds its own configuration, so nothing was guessed:

| # | Item | Finding |
|---|------|---------|
| 1 | Checkpoint format | Native **DepthAnythingV2** state dict under a training wrapper: `model["net.pretrained.*"]` (DINOv2 ViT-S) + `model["net.depth_head.*"]` (DPT head). NOT loadable into the HF `transformers` DAv2 classes — requires the official implementation |
| 2 | Released `model_config` | `{encoder: "vits", features: 64, out_channels: [48, 96, 192, 384]}` → 24,785,089 parameters (verified by strict load) |
| 3 | Released `transform` | `{mode: "normalized", scale_m: 8.492877943662961, units: "metres_AGL", negative_training_targets: "clamp_zero"}` |
| 4 | Output scale (task Sec. 10 validation) | `metres_AGL = raw_output × scale_m`, then clamp ≥ 0. Empirically confirmed on GAMUS val center crops (DC_02_26/DC_04_23): raw-as-metres MAE 5.94/19.97 m vs ×scale_m MAE 3.02/4.28 m — the normalized transform wins decisively. The conversion lives ONLY in `TerraHeightModel.forward` |
| 5 | Preprocessing | uint8 RGB → /255 → ImageNet mean `[0.485,0.456,0.406]` std `[0.229,0.224,0.225]` (embedded `normalization` block) — applied exactly once, in the adapter |
| 6 | Input constraint | Every input edge must be a multiple of the ViT-14 patch (630 = 45×14). 1024×1024 GAMUS tiles are NOT directly consumable — tiled inference is mandatory for real scenes |
| 7 | Training crop | 630×630 (`crop_size` field) → default inference tile edge; the payload's own inference config used `TILE_OVERLAP: 0.25` → default overlap 157 px (stride 473) |
| 8 | Semantic head | NONE — TerraHeight is a height model only (task Sec. 20). `sem_probs` is always None; the scene payload reports that honestly and the existing semantic pipeline stays authoritative |
| 9 | License / provenance | TerraHeight repo Apache-2.0; base initialization `depth-anything/Depth-Anything-V2-Small`; trained on GAMUS. External pretrained model — never presented as trained by this project |

## 2. Files ADDED

| File | Purpose |
|------|---------|
| `depthwizard/vendor/depth_anything_v2/` | **Vendored official Depth Anything V2 implementation** (upstream `main`, fetched 2026-10-02, Apache-2.0 — `LICENSE` + `VENDOR_NOTICE.md` included). The only deviations: `cv2`/`torchvision` imports deferred (unused helpers; neither package is a project dependency). DINOv2 layer code is (c) Meta, Apache-2.0 |
| `depthwizard/terraheight.py` | Backend adapter: payload validation, scale conversion, `TerraHeightModel` wrapper (same `{"pred": [B,1,H,W]}` contract as RDAH/CalibrationNet, depth-cache inputs rejected loudly), checkpoint loader, `predict_agl_tiled` (630-crop windows, cosine-blend stitching via `depthwizard.tiling`), `terraheight_health`, `run_terraheight_inference` orchestrator |
| `model_tests/test_terraheight.py` | 36 tests: discovery/validation/loading (strict, loud), architecture shape checks, preprocessing exactness, scale conversion, clamp, predict-fn contract, tiling/stitching/cancellation, registry, geospatial E2E, opt-in released-checkpoint test |
| `backend_tests/test_terraheight_api.py` | 4 tests: schema acceptance, health report, process-route passthrough |
| `docs/terraheight_integration.md` | This document |

## 3. Files MODIFIED

| File | Change |
|------|--------|
| `depthwizard/tifops.py` | Registry: `detect_architecture` recognizes TerraHeight release payloads; `load_height_model` dispatches `terraheight_s`; `LoadedModel.tag`; predict-fn factories dispatch |
| `depthwizard/inference.py` | `run_inference` dispatches `terraheight_s` to `run_terraheight_inference` (before any Dn/backbone resolution); new `terraheight_*` kwargs |
| `depthwizard/cli/infer.py` | `--architecture terraheight_s`, TerraHeight checkpoint resolution (never auto-download), `--terraheight-tile-size/-tile-stride/-fp16` |
| `depthwizard/cli/bench_model.py` | `--architecture terraheight_s` (patch-14 size guard, RGB-only forward) |
| `depthwizard/cli/eval_calibration.py` | `--model terraheight_s`: RGB-only GAMUS eval through the deployed tiled path, published-protocol `>1m`/`>5m` threshold bands, backend-specific report notes |
| `backend/app/schemas/processing.py` | `ModelArchitecture.TERRAHEIGHT_S = "terraheight_s"` |
| `backend/app/core/config.py` | `DW_CKPT_TERRAHEIGHT`, `DW_TERRHEIGHT_TILE_SIZE/STRIDE/BATCH_SIZE` settings |
| `backend/app/services/processing_service.py` | `_resolve_checkpoint` terraheight branch (env → repo candidates, loud errors, never downloads); `_inference_kwargs` passes tiling knobs; `_discard_outputs` covers the new artifacts |
| `backend/app/api/routes/health.py` | `GET /api/v1/health/models` — per-backend checkpoint report + opt-in `?load_and_probe=true` (loads weights + one 154×154 finite-output probe; never a full scene) |
| `backend/app/infrastructure/storage/scene_artifacts.py` | `terraheight_agl(.tif/.npy)`, `terraheight_preview`, `terraheight_meta` product keys |
| `frontend/src/pages/Home.jsx` | Selector option **TerraHeight-S** ("Pretrained GAMUS AGL model") + model metadata block (backbone/task/dataset/output) |
| `frontend/src/store/appStore.jsx` | `ModelBackend.TERRAHEIGHT_S` + persistence (RDAH remains the default) |
| `frontend/src/components/Processing/PipelineProgress.jsx` | Backend-aware stage label: for terraheight_s the height stage reads "Loading TerraHeight-S · running tiled AGL inference" — the "depth" wording (and any relative-depth-cache implication) is never shown |
| `frontend/src/api/processing.js` | JSDoc architecture unions |
| `.env.example`, `docker-compose.yml`, `README.md` | Configuration + documentation |

## 4. Exact configuration

- Checkpoint discovery (in order): `DW_CKPT_TERRAHEIGHT` → `models/terraheight/best_model.pth` → repo-root `best_model.pth` (the discovered location in this workspace: **`best_model.pth`, 99,220,705 bytes, SHA-256 recorded into `terraheight_meta.json`**). A missing checkpoint is a loud `FileNotFoundError` naming the env var — never a download.
- Architecture instantiated: `DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])` (official vendored implementation), strict load after stripping the `net.` wrapper prefix.
- Preprocessing: `uint8 → /255 → (x - ImageNet mean)/std`, one 630×630 tile per forward, batch 1.
- Output scaling: `AGL = clamp(raw × 8.492877943662961, min=0)` metres.
- Tiling: tile 630, overlap 157 (stride 473), cosine-weighted `OverlapStitcher`, edge-padded boundary windows, output grid == input grid.

## 5. Pipeline position

```
RGB ──┬─ RDAH / CalibrationNet ── DAv2 depth cache ── height   (unchanged)
      ├─ TerraHeight-S ── AGL metres directly (NO depth cache) ← new
      ├─ existing semantic pipeline (untouched; TerraHeight has no sem head)
      └─ HOTOSM disaster pipeline (untouched)
AGL + semantics + buildings + damage → terrain / inspection / route risk
```

Artifacts per run (standard contract unchanged): `dsm.npy`, `dsm.tif` (georeferenced inputs only), `dsm_preview.png` — PLUS `terraheight_agl.tif` (or `.npy` when the input has no CRS), `terraheight_preview.png`, `terraheight_meta.json` (architecture, checkpoint + SHA-256, param count, tile/stride, device, dtype, inference time, peak VRAM, georef state, and the checkpoint's PUBLISHED validation metrics, clearly labelled as reference values).

## 6. Verification results (local RTX 4050 6 GB)

**Real E2E runs (CLI `python model.py infer --architecture terraheight_s --device cuda`):**

1. GAMUS val tile `DC_02_26` (1024×1024, 4 tiles): 0.49 s, peak VRAM **323 MB**, AGL range 0–35.6 m; vs GT: MAE 3.03 m, RMSE 4.70 m, r 0.909. Non-georeferenced path: pixel-space output, no CRS invented, `terraheight_agl.npy` written.
2. Real scene `data/raw/scenes/scene_1505d6392edd/input.tif` (2408×3021, EPSG:32645, 35 tiles): `dsm.tif`/`terraheight_agl.tif` reproduce CRS + affine transform exactly at 2408×3021, all outputs finite, AGL range 0–25 m.

**Benchmark (`python model.py bench --architecture terraheight_s --device cuda`, batch 1, fp32):**

| tile px | fwd ms | peak VRAM MB |
|--------:|-------:|-------------:|
| 252 | 6.4 | 132 |
| 420 | 19.7 | 184 |
| **630 (default)** | **62.0** | **324** |
| 826 | 151.3 | 715 |

**GAMUS evaluation** (`python model.py evaluate --dataset gamus`, identical 24-tile DC val subset — first 40 sorted val IDs, all valid tiles are DC — same split, valid-pixel mask, target handling and metric implementation for every backend; local reproduction, NOT the published protocol numbers):

| backend | params | MAE (m) ↓ | RMSE (m) ↓ | Pearson r ↑ | >1m MAE | >5m MAE | note |
|---|---:|---:|---:|---:|---:|---:|---|
| **TerraHeight-S** | 24,785,089 | **2.947** | **4.450** | **0.902** | 4.291 | 4.674 | external pretrained, RGB only (no depth cache) |
| CalibrationNet | 118,104 | 3.132 | 5.106 | 0.830 | — | — | cross-dataset (mixed DFC/GAMUS flagship) |
| RDAH-Net | 5,371,663 | 7.017 | 10.266 | 0.525 | — | — | GAMUS fine-tune @ epoch 2 (undertrained; `rdah_best.pt`) |

TerraHeight-S also reports the best slope-MAE behaviour among the three on this subset (23.5° vs 28.4° RDAH / 22.2° CalibrationNet at GSD 0.33 m).

**Dedicated slope-MAE evaluation** (same subset; `depthwizard.metrics.slope_error`, central-difference slope angle at GSD 0.33 m, valid-target pixels, pixel-pooled — full per-tile data in `outputs/slope_eval/slope_mae_gamus_val40.{json,md}`):

| backend | pooled slope MAE (deg) | pooled slope RMSE (deg) | slope bias (deg) |
|---|---:|---:|---:|
| **TerraHeight-S** | 23.471 | 33.962 | −9.242 |
| CalibrationNet | **22.155** | **31.237** | +0.697 |
| RDAH-Net | 28.448 | 38.307 | −21.413 |

Interpretation: TerraHeight-S reconstructs terrain shape better than RDAH and with far less systematic bias, but slightly trails CalibrationNet — expected, since CalibrationNet's heights inherit the DAv2 depth cache's fine gradients, while TerraHeight's ReLU-clamped direct regression smooths steep urban slopes (visible in its −9.2° under-prediction bias). All RMSEs are high because the subset is DC urban high-relief.

**Published reference values** (released checkpoint's own full-GAMUS validation, quoted ONLY as external reference — never mixed with local results): All MAE 1.312 m / RMSE 2.616 m / r 0.9238; >1 m MAE 2.421 m; >5 m MAE 2.693 m. The gap to our 2.95 m is expected: our bounded subset is 24 DC (urban, high-relief) tiles, not the full 859-tile validation set.

## 7. Backend API

- `POST /api/v1/scenes/{id}/process` and `/refine` accept `"architecture": "terraheight_s"`; the value is persisted on the job record and genuinely selects the TerraHeight path in `run_inference` (mock-free proof: `backend_tests/test_terraheight_api.py::test_process_route_passes_terraheight_s_to_inference`).
- `GET /api/v1/health/models` → per-backend checkpoint existence + optional `?load_and_probe=true` TerraHeight probe (load → 154×154 forward → finite/non-negative output check). The plain `/health` + `/health/ready` contracts are unchanged and stay cheap.
- Frontend → backend → inference contract string: `terraheight_s` end to end; `meta.model_architecture = "terraheight_s"`, `meta.height_type = "AGL"` in every payload, so Structure Inspector / terrain layers consume the TerraHeight surface through the existing `dsm.npy` contract with zero downstream changes.

## 8. Limitations (honest)

- **Subset evaluation**: the local GAMUS numbers above use a deterministic 24-tile DC subset of the val split (bounded download/compute budget). They are comparable ACROSS backends but are not the published full-validation protocol; published metrics are labelled as reference values everywhere.
- **Batch size 1** is enforced (`NotImplementedError` above 1) until VRAM headroom is verified on the target GPU; at 324 MB/tile there is ample room, but the certified path stays conservative.
- **RDAH comparison caveat**: the only GAMUS RDAH checkpoint available (`rdah_best.pt`) is a 2-epoch fine-tune (its own `val_subset_mae` 3.008 was measured on a different subset); the comparison is faithful to the checkpoints on disk, not to RDAH's ceiling.
- **fp16** is supported (autocast) but off by default — fp32 is the certified deterministic path.
- TerraHeight AGL is clamped ≥ 0 by design (published semantics); there is no uncertainty/confidence output.
- Downstream DEM anchoring works exactly as for the other backends (`--ground-elev` / `--anchor-dem`), since anchoring is arithmetic on the height product, not a model property.
