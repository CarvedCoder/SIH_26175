# DepthWizard (SIH26175) — Modular Monorepo

**Single-view aerial RGB → LiDAR-derived height (AGL) estimation + 3D flythrough.**

One repository, two connected halves, **one entry point each**:

| Half | Entry point | What it does |
|---|---|---|
| Backend (Python) | `python model.py <command>` | dataset audit → splits → depth cache → baselines → training → **citable evaluation** → inference → FastAPI service (`backend.app.main:app`) |
| Frontend (React/Vite) | `frontend/` → `npm run dev` | upload → process → 3D terrain viewer: orbit, RGB/Depth/DSM/Slope layers, first-person walkthrough |

They meet at **one contract**: the `/api/v1` scene/job API served by
`backend.app.main:app` — upload, processing jobs, results, validation and
terrain tiles all funnel through the same certified inference path
(`depthwizard/inference.py :: run_inference`), so the viewer can never
drift from the certified pipeline.

> **Governance (frozen):** FINAL / citable numbers come ONLY from
> `python model.py evaluate`. Everything else is exploration, demonstration,
> or bookkeeping. Unknowns print `UNKNOWN` — never guessed.

---

## 1. Layout

```
model.py                    THE backend entry point (lazy subcommand dispatch)
main.py                     GeoTIFF DEM preprocessing pipeline (separate world,
                            NOT wired to depthwizard — kept, never merged)
depthwizard/               core library — frozen contracts, import never rewrite
  normalize.py               Dn min-max + AGL cleaning   [single source of truth]
  metrics.py                 masked MAE/RMSE/MedAE/bias/r, pooled + per-tile
                             + building/slope/scene-type extensions (Phase 5)
  splits.py                  scene-level tile/block splits + overlap guards
  geo.py                     rasterio I/O, tile pairing, quicklooks,
                             namespaced depth-cache paths (dataset level)
  dataset.py                 DFC2019Dataset (joint crop/aug, ImageNet RGB) —
                             frozen contract, + dataset/sample_id meta
  datasets/                  MULTI-DATASET ADAPTER PACKAGE (GAMUS integration)
    transforms.py              joint crop/flip/rot — single source (shared)
    semantics.py               VERIFIED legends -> 6 project classes one-hot
                               + explicit ignore mask (anti-fabrication)
    base.py                    BaseDepthDataset shared pipeline + None-safe
                               collate (torch>=2.13 compatible)
    dfc2019.py                 DFC2019Adapter (thin subclass, byte-equal)
    gamus.py                   GAMUSDataset: raw HDF5 (hf lazy | local dir),
                               official splits, dtype casts, honesty meta
    mixed.py                   MixedDataset — GATED on verified stats
    factory.py                 build_dataset(s) from the dataset: YAML schema
  calibration_net.py         CalibrationNet  H = clamp(a·Dn+b, 0)
                             [Dn | RGB | SEM | DEM] channels + derive_in_ch
  losses.py                  loss registry: l1/huber + w_grad/w_smooth/w_sem
                             (ALL weights default 0 = exact frozen behavior)
  scene_types.py             urban/sparse/forest/hilly/flat tile classifier
                             (documented heuristics, stratification only)
  tifops.py                  checkpoint loading + predict-fn factory
                             (use_sem/sem_classes fields, old-ckpt compatible)
  backbone.py                LIVE Depth-Anything-V2-Base (ViT-B, aligned with
                             the cache-building variant)
  inference.py               image → AGL/DSM + scene payload (CLI & service)
  anchoring.py               AGL + ground/DTM → absolute DSM  [ANCHORED, not learned]
  demprior.py                SYNTHETIC-DEM-PROXY (tagged, honest)
  streaming.py               memory-light pooled metric accumulators
  cli/                      one module per pipeline command (no numbered scripts)
                             + dataset_stats.py (the mixing GATE)
service/api.py             LEGACY FastAPI bridge (superseded by backend/app;
                           kept for the e2e bridge check only)
configs/                   phase1/2 · infer · gamus.yaml · gamus_experiments
                           · exp4_sem.yaml · exp5_sem_dem.yaml
model_tests/               depthwizard tests (frozen-path regression pins +
                           GAMUS fixtures, network-free)
tests/                     src/ GeoTIFF pipeline tests (separate world)
backend/                   THE canonical FastAPI backend (backend.app.main:app,
                           /api/v1 scene/job contract — see readme.md)
backend_tests/             backend API test suite (mocked inference)
frontend/                  React/Vite frontend (Vite dev server :5173;
                           its own README/design docs live inside)
tools/make_fake_dataset.py synthetic DFC2019 mini-dataset (smoke tests)
tools/make_fake_gamus.py   synthetic GAMUS-shaped HDF5 mini-dataset
tools/e2e_bridge_check.py  CLI + service contract check (all-green)
docker-compose.yml         rustfs + backend + frontend, pre-wired
docs/                      design docs + rendered frontend evidence
                           (docs/screenshots/)
```

### The GAMUS integration in one paragraph

GAMUS (8,724 HDF5 tiles, DC/NYC/PHL, official splits, nDSM heights with
undocumented units — assumed metres and flagged as such, no CRS, GSD
0.33 m, CC-BY-4.0) is ingested through the SAME sample contract as
DFC2019 via `depthwizard/datasets/` adapters. The primary source is the
raw HDF5 release with lazy per-file `hf_hub_download` (float32 heights
preserved; only needed tiles downloaded); a local-directory mode serves
offline/pre-downloaded subsets; the EarthNets/Dataset4EO streaming
re-packaging is an optional backend knob that deliberately raises
`NotImplementedError` with the float16 + license rationale. Semantic
labels are normalized to a ONE-HOT over the 6 project classes (building,
vegetation, road, water, ground, other) with an explicit ignore mask —
raw ids are preserved and every meaning is traceable to a verified
official legend (pubgeo/dfc2019 and EarthNets/RSI-MMSegmentation). Depth
caches are namespaced per dataset
(`<model_tag>/{dfc2019,gamus}/{sample_id}.npy`, legacy flat DFC layouts
still resolve). Mixing GAMUS with DFC2019 in training is REFUSED until a
human-verified per-dataset statistics artifact exists
(`model.py stats` → `dataset_stats.json` → `verified: true`).

### Experiment ladder & gating (binding)

| Exp | What | Status | Command |
|---|---|---|---|
| 1 | Dn → global affine | preserved (frozen) | `fit-baseline` / `eval-baseline` |
| 2 | Dn → U-Net → height | preserved (frozen) | `train` (no `dataset:` section) |
| 3 | RGB+Dn → U-Net | preserved (frozen) | `train --use-rgb` |
| 4 | +semantic one-hot (in_ch 10) | **NEW** | `train --use-rgb --use-sem` (configs/exp4_sem.yaml) |
| 5 | +semantic+DEM (in_ch 11) | **NEW** | `+ --use-dem --synth-dem` (configs/exp5_sem_dem.yaml) |
| 6 | LoRA on DAv2 ViT-B + calibration | **GATED — NOT implemented** | — |

Exp 6 (LoRA / "Architecture C") is deliberately deferred per the binding
user constraint: it may be implemented only AFTER Exp 1–5 are reproducibly
evaluated (citable `evaluate` runs on both datasets). The same gate applies
to mixed-dataset training (stats artifact must be human-verified first).
Phase-4 loss extras (`--w-grad/--w-smooth/--w-sem`) default to 0, which is
bit-identical to the pre-GAMUS training loss.

## 2. Backend quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The full pipeline, in order (paths in configs/phase1.yaml):
python model.py inspect      --rgb-dir rgb_data/Train-Track1-RGB/Track1-RGB \
                            --truth-dir rgb_data_truth/Train-Track1-Truth/Track1-Truth
python model.py splits       --rgb-dir ... --truth-dir ... --out outputs/splits/splits.json
python model.py depth        --rgb-dir ... --device auto
python model.py fit-baseline --config configs/phase1.yaml
python model.py eval-baseline --config configs/phase1.yaml
python model.py dummies      --config configs/phase1.yaml
python model.py reference    --config configs/phase1.yaml
python model.py train        --config configs/phase2.yaml --use-rgb --cosine --out-tag rgb_cos
python model.py evaluate     --config configs/phase2.yaml \
                            --checkpoint outputs/calib_net/rgb_cos/best.pt   # CITABLE

# Demo inference (arbitrary image; falls back to LIVE DAv2 when no cache):
python model.py infer --config configs/infer.yaml --input some_scene.tif

# Track-2 absolute DSM (arithmetic anchoring, labeled ANCHORED):
python model.py infer --input scene.tif --anchor-dem dem.tif
python model.py infer --input scene.tif --ground-elev 12.5

# Diagnostics:
python model.py eval-scene --pred outputs/infer/X/dsm.tif --truth <X>_AGL.tif
python model.py gt-check   --pred outputs/infer/X/dsm.npy --truth <X>_AGL.tif
python model.py diag       --config configs/phase2.yaml
```

`python model.py --help` lists everything; `python model.py <cmd> --help` for
details. Command modules load lazily — help works instantly, even before
`torch` is installed.

### Smoke test the whole thing (no dataset needed)

```bash
python tools/make_fake_dataset.py --root fake_data --n 12 --size 256 --rows 4 --cols 3
python model.py inspect --rgb-dir fake_data/rgb_data/Train-Track1-RGB/Track1-RGB \
                       --truth-dir fake_data/rgb_data_truth/Train-Track1-Truth/Track1-Truth \
                       --report-dir outputs/reports
# GAMUS side (network-free):
python tools/make_fake_gamus.py --root fake_gamus --per-split 3 --size 128
python model.py splits --dataset gamus --gamus-source local \
    --gamus-local-root fake_gamus --out outputs/splits/gamus_splits.json
python model.py inspect --dataset gamus --gamus-source local \
    --gamus-local-root fake_gamus --report-dir outputs/reports_gamus
pytest model_tests/ -q     # full depthwizard suite (frozen-path pins + GAMUS)
pytest tests/ -q           # src GeoTIFF pipeline tests
python tools/e2e_bridge_check.py   # CLI + service contract, all-green
```

### 2b. GAMUS multi-dataset quickstart

```bash
# 1) freeze the OFFICIAL GAMUS splits (never re-split them) — needs network
#    once; the manifest makes later steps offline-capable
python model.py splits --dataset gamus --gamus-source hf \
    --out outputs/splits/gamus_splits.json

# 2) inspect a bounded sample (hf mode downloads each inspected triple)
python model.py inspect --dataset gamus --gamus-source hf --limit 16 \
    --report-dir outputs/reports_gamus

# 3) precompute the ViT-B depth cache — namespaced under <tag>/gamus/
python model.py depth --dataset gamus --gamus-source hf \
    --out-dir outputs/depth_cache --limit 512 --device auto

# 4) THE GATE: per-dataset statistics (binding constraint — no mixing
#    until verified). Review the artifact, then --mark-verified.
python model.py stats --config configs/gamus.yaml \
    --datasets dfc2019 gamus --out outputs/stats/dataset_stats.json

# 5) GAMUS experiment ladder (configs/gamus_experiments.yaml has the full
#    runbook): Exp 1 fit-baseline --dataset gamus; Exp 2/3/4 train with
#    --out-tag gamus_*; CITABLE eval via evaluate --dataset gamus.
python model.py fit-baseline --config configs/gamus.yaml --dataset gamus
python model.py train --config configs/gamus.yaml --out-tag gamus_dn
python model.py evaluate --config configs/gamus.yaml --dataset gamus \
    --checkpoint outputs/calib_net/gamus_dn/best.pt

# 6) cross-dataset generalization probes (DFC flagship on GAMUS):
python model.py evaluate --config configs/gamus.yaml --dataset gamus \
    --checkpoint outputs/calib_net/rgb_cos/best.pt --splits val
```

Mixed GAMUS+DFC training additionally requires a `dataset: name: mixed`
config with `verified_stats:` pointing at the reviewed stats artifact —
see `depthwizard/datasets/mixed.py` (it refuses without one).

## 3. Frontend quickstart

```bash
cd frontend
npm install
npm run dev               # http://localhost:5173
```

Point it at the backend with `frontend/.env`:
`VITE_API_BASE_URL=http://localhost:8000/api/v1` (see `frontend/.env.example`).
Supabase supplies auth when `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`
are set; without them the backend runs with auth disabled locally.

The viewer (`frontend/src/components/TerrainViewer/`): GPU-shader
heightmap with RGB texture projection and progressive LOD, orbit mode,
layer switching (RGB / Depth / DSM / Slope), a pointer-lock first-person
walkthrough mode, hover elevation/slope readout, minimap, and an
honesty-first stats panel (`CRS UNKNOWN`, `ANCHORED (not learned)`).
Rendered evidence lives in `docs/screenshots/`.

## 4. Docker (all services, one command)

```bash
# expects the serving checkpoint at
# outputs/calib_net/gamus_rgb_grad/best.pt (committed; see scripts/fetch_checkpoint.sh)
docker compose up --build
#   backend  → http://localhost:8000  (health: /api/v1/health, docs: /docs)
#   frontend → http://localhost:5173
#   rustfs   → http://localhost:9000 (console :9001)
```

## 5. The connection (how "everything is wired")

```
browser ── /api/v1/scenes (upload GeoTIFF/PNG/JPG)
   │       POST /api/v1/scenes/{id}/process → job
   │       GET  /api/v1/jobs/{job_id}       → status/progress
   │       GET  /api/v1/scenes/{id}/results → payload for the 3D viewer
   │       GET  /api/v1/scenes/{id}/terrain, /validation, /reference
   ▼
backend/app/api/router.py  (FastAPI routers — the ONLY backend doorway)
   │
   ▼
backend/app/services/processing_service.py :: process_scene
    → depthwizard/inference.py :: run_inference
      (checkpoint via tifops, Dn via cache/live DAv2,
       flagship CalibrationNet, optional anchoring)
    → scene payload + terrain tiles + validation artifacts
         │
         ▼
    TerrainViewer (Three.js heightmap shader + RGB drape, orbit /
    walkthrough) ── stats panel ── validation panel
```

## 6. Migration notes (old numbered scripts → commands)

| Old | New |
|---|---|
| `scripts/01_inspect_dataset.py` | `python model.py inspect` |
| `scripts/02_make_splits.py` | `python model.py splits` |
| `scripts/03_precompute_depth.py` | `python model.py depth` |
| `scripts/04_fit_baseline.py` | `python model.py fit-baseline` |
| `scripts/05_eval_baseline.py` | `python model.py eval-baseline` |
| `scripts/06_dummy_baselines.py` | `python model.py dummies` |
| `scripts/07_reference_table.py` | `python model.py reference` |
| `scripts/08_train_calibration.py` | `python model.py train` |
| `scripts/09_eval_calibration.py` | `python model.py evaluate` (still the ONLY citable source) |
| `scripts/10_infer_single.py` | `python model.py infer` (+ live DAv2, anchoring, JSON payload) |
| `scripts/gt_check.py` | `python model.py gt-check` |
| `scripts/diag_rgb_arm.py` | `python model.py diag` |
| — | `python model.py eval-scene`, `python model.py serve` (new) |

Frozen library modules were moved **verbatim** — `normalize.py`,
`metrics.py`, `splits.py`, `geo.py`, `dataset.py`, `calibration_net.py` are
byte-identical to the certified versions.

## 7. Verification checklist

```bash
pytest tests/ -q                                   # src pipeline tests
pytest backend_tests/ -q                           # backend API suite
python model.py --help                              # instant, no torch needed
python model.py infer --help                        # lazy per-command import
cd frontend && npm run lint && npm run build        # frontend checks
```
