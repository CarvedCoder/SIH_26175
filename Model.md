# DepthWizard (SIH26175) — Modular Monorepo

**Single-view aerial RGB → LiDAR-derived height (AGL) estimation + 3D flythrough.**

One repository, two connected halves, **one entry point each**:

| Half | Entry point | What it does |
|---|---|---|
| Backend (Python) | `python main.py <command>` | dataset audit → splits → depth cache → baselines → training → **citable evaluation** → inference → FastAPI service |
| Frontend (Next.js) | `webapp/` → `npm run dev` | upload → heights → interactive 3D flythrough (Three.js) |

They meet at **one contract**: `depthwizard/inference.py :: build_scene_payload` —
the CLI (`main.py infer`), the service (`main.py serve`), and the webapp all
funnel through the same code path, so the viewer can never drift from the
certified inference pipeline.

> **Governance (frozen):** FINAL / citable numbers come ONLY from
> `python main.py evaluate`. Everything else is exploration, demonstration,
> or bookkeeping. Unknowns print `UNKNOWN` — never guessed.

---

## 1. Layout

```
main.py                    THE backend entry point (lazy subcommand dispatch)
depthwizard/               core library — frozen contracts, import never rewrite
  normalize.py               Dn min-max + AGL cleaning   [single source of truth]
  metrics.py                 masked MAE/RMSE/MedAE/bias/r, pooled + per-tile
  splits.py                  scene-level tile/block splits + overlap guards
  geo.py                     rasterio I/O, tile pairing, quicklooks
  dataset.py                 DFC2019Dataset (joint crop/aug, ImageNet RGB)
  calibration_net.py         CalibrationNet  H = clamp(a(x,y)·Dn + b(x,y), 0)
  tifops.py                  checkpoint loading + predict-fn factory
  backbone.py                LIVE Depth-Anything-V2 (cache-miss fallback)
  inference.py               image → AGL/DSM + scene payload (CLI & service)
  anchoring.py               AGL + ground/DTM → absolute DSM  [ANCHORED, not learned]
  streaming.py               memory-light pooled metric accumulators
  cli/                      one module per pipeline command (no numbered scripts)
service/api.py             FastAPI bridge: /health, /predict (webapp backend)
configs/                   phase1.yaml · phase2.yaml · infer.yaml
tests/                     68 tests (contracts, metrics, splits, inference,
                           anchoring, CLI registry, streaming equivalence)
tools/make_fake_dataset.py synthetic mini-dataset for end-to-end smoke tests
webapp/                    Next.js frontend (its own README has details)
docker-compose.yml         backend + webapp, pre-wired
worklog.md                 campaign ledger (append-only)
```

## 2. Backend quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The full pipeline, in order (paths in configs/phase1.yaml):
python main.py inspect      --rgb-dir rgb_data/Train-Track1-RGB/Track1-RGB \
                            --truth-dir rgb_data_truth/Train-Track1-Truth/Track1-Truth
python main.py splits       --rgb-dir ... --truth-dir ... --out outputs/splits/splits.json
python main.py depth        --rgb-dir ... --device auto
python main.py fit-baseline --config configs/phase1.yaml
python main.py eval-baseline --config configs/phase1.yaml
python main.py dummies      --config configs/phase1.yaml
python main.py reference    --config configs/phase1.yaml
python main.py train        --config configs/phase2.yaml --use-rgb --cosine --out-tag rgb_cos
python main.py evaluate     --config configs/phase2.yaml \
                            --checkpoint outputs/calib_net/rgb_cos/best.pt   # CITABLE

# Demo inference (arbitrary image; falls back to LIVE DAv2 when no cache):
python main.py infer --config configs/infer.yaml --input some_scene.tif

# Track-2 absolute DSM (arithmetic anchoring, labeled ANCHORED):
python main.py infer --input scene.tif --anchor-dem dem.tif
python main.py infer --input scene.tif --ground-elev 12.5

# Diagnostics:
python main.py eval-scene --pred outputs/infer/X/dsm.tif --truth <X>_AGL.tif
python main.py gt-check   --pred outputs/infer/X/dsm.npy --truth <X>_AGL.tif
python main.py diag       --config configs/phase2.yaml
```

`python main.py --help` lists everything; `python main.py <cmd> --help` for
details. Command modules load lazily — help works instantly, even before
`torch` is installed.

### Smoke test the whole thing (no dataset needed)

```bash
python tools/make_fake_dataset.py --root fake_data --n 12 --size 256 --rows 4 --cols 3
python main.py inspect --rgb-dir fake_data/rgb_data/Train-Track1-RGB/Track1-RGB \
                       --truth-dir fake_data/rgb_data_truth/Train-Track1-Truth/Track1-Truth \
                       --report-dir outputs/reports
pytest tests/ -q          # 68 tests
```

## 3. Frontend quickstart

```bash
cd webapp
npm install
npm run dev               # http://localhost:3000
```

With no extra env, the `/api/predict` route **spawns the CLI directly**
(`python main.py infer --json-out …` from the repo root). For the service
transport:

```bash
# terminal 1 (repo root)
python main.py serve --port 8000
# terminal 2
cd webapp && DW_API_URL=http://localhost:8000 npm run dev
```

The viewer: RGB-draped DSM mesh, orbit + auto-fly camera, hover
elevation/slope readout, vertical exaggeration, wireframe overlay,
screenshot, session gallery, and an honesty-first stats panel
(`CRS UNKNOWN`, `ANCHORED (not learned)`).

## 4. Docker (both halves, one command)

```bash
# expects your trained flagship at outputs/calib_net/rgb_cos/best.pt
docker compose up --build
#   webapp  → http://localhost:3000
#   backend → http://localhost:8000/health
```

## 5. The connection (how "everything is wired")

```
browser ── POST /api/predict (multipart: image, anchor_dem?, ground_elev?, mode)
   │
   ▼
webapp/src/app/api/predict/route.ts        (the ONLY backend doorway)
   │  lib/bridge.ts picks the transport:
   │
   ├── DW_API_URL set?  ──► FastAPI  service/api.py  ─┐
   │                                                   │  both call
   └── otherwise      ──► spawn `python main.py infer` ─┤
                                                         ▼
                                    depthwizard/inference.py :: run_inference
                                    (checkpoint via tifops, Dn via cache/live DAv2,
                                     flagship CalibrationNet, optional anchoring)
                                                         │
                                    build_scene_payload (grid + RGB + stats)
                                                         ▼
                                    Viewer3D mesh ── gallery ── stats panel
```

## 6. Migration notes (old numbered scripts → commands)

| Old | New |
|---|---|
| `scripts/01_inspect_dataset.py` | `python main.py inspect` |
| `scripts/02_make_splits.py` | `python main.py splits` |
| `scripts/03_precompute_depth.py` | `python main.py depth` |
| `scripts/04_fit_baseline.py` | `python main.py fit-baseline` |
| `scripts/05_eval_baseline.py` | `python main.py eval-baseline` |
| `scripts/06_dummy_baselines.py` | `python main.py dummies` |
| `scripts/07_reference_table.py` | `python main.py reference` |
| `scripts/08_train_calibration.py` | `python main.py train` |
| `scripts/09_eval_calibration.py` | `python main.py evaluate` (still the ONLY citable source) |
| `scripts/10_infer_single.py` | `python main.py infer` (+ live DAv2, anchoring, JSON payload) |
| `scripts/gt_check.py` | `python main.py gt-check` |
| `scripts/diag_rgb_arm.py` | `python main.py diag` |
| — | `python main.py eval-scene`, `python main.py serve` (new) |

Frozen library modules were moved **verbatim** — `normalize.py`,
`metrics.py`, `splits.py`, `geo.py`, `dataset.py`, `calibration_net.py` are
byte-identical to the certified versions (the worklog ledger stays valid).

## 7. Verification checklist

```bash
pytest tests/ -q                                   # 68 pass
python main.py --help                              # instant, no torch needed
python main.py infer --help                        # lazy per-command import
cd webapp && npm run typecheck && npm run build    # frontend types + build
```
