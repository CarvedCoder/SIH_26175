# DepthWizard — SIH 26175

Monocular height estimation from aerial/satellite imagery: a frozen
**Depth Anything V2** backbone feeds a lightweight **CalibrationNet** that
predicts above-ground-level height (metres) per pixel, with optional
ground anchoring to an absolute DSM.

## Repository layout

| Path                                       | What it is                                                                                                                                          |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `depthwizard/`                             | The ML library: backbone, CalibrationNet, dataset, training/eval CLIs, and the ONE certified inference path (`depthwizard.inference.run_inference`) |
| `backend/app/`                             | **The canonical FastAPI backend** (`backend.app.main:app`) — the `/api/v1` scene/job API the frontend uses                                          |
| `service/`                                 | LEGACY stateless `/predict` service (manual smoke-test door only; not deployed)                                                                     |
| `frontend/`                                | React/Vite frontend (out of backend scope)                                                                                                          |
| `model.py`                                 | CLI shim (`python model.py train/evaluate/infer/depth/...`)                                                                                         |
| `model_tests/`, `tests/`, `backend_tests/` | ML core, DEM pipeline, and backend API test suites                                                                                                  |

## Quick start (backend)

```bash
# 1) install (uv manages the lockfile; Python >= 3.12)
uv sync --extra dev

# 2) provide a trained checkpoint (default resolution order):
#    $DW_CKPT  else  outputs/calib_net/gamus_rgb_grad/best.pt
#    Train one: python model.py train --use-rgb --out-tag gamus_rgb_grad
#    model_loaded in /api/v1/health reports honestly whether it exists.

# 3) run the canonical backend
uv run uvicorn backend.app.main:app --reload --reload-dir backend

# 4) run the frontend (separate terminal)
cd frontend && npm install && npm run dev
# frontend/.env: VITE_API_BASE_URL=http://localhost:8000/api/v1
```

The API is then at `http://localhost:8000/api/v1` with docs at `/docs`.

## Configuration (environment)

| Variable                 | Default                                       | Meaning                                                            |
| ------------------------ | --------------------------------------------- | ------------------------------------------------------------------ |
| `DW_CKPT`                | `outputs/calib_net/gamus_rgb_grad/best.pt`    | CalibrationNet checkpoint                                          |
| `DW_CKPT_SHA256`         | unset                                         | Optional checkpoint integrity check                                |
| `DW_DEVICE`              | `auto`                                        | `cuda` / `cpu` / `auto`                                            |
| `DW_BACKBONE`            | `depth-anything/Depth-Anything-V2-Base-hf`    | Live DAv2 fallback model                                           |
| `DW_CORS_ORIGINS`        | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated explicit allowlist (never `*` by default)          |
| `DW_API_KEY`             | unset                                         | When set, all `/api/v1` routes (except health) require `X-API-Key` |
| `DW_MAX_UPLOAD_BYTES`    | `524288000` (500 MB)                          | Upload cap, enforced during streaming                              |
| `DW_MAX_CONCURRENT_JOBS` | `1`                                           | Simultaneous inference runs                                        |
| `DW_JOB_TTL_SECONDS`     | `86400`                                       | Job record retention                                               |

## Docker (the canonical backend)

You need to get a minio free lisence to start the server as of now (get the lisence through https://www.min.io/pricing)
Store it under .secrets/minio.license/minio.license.txt
then run

```bash

mkdir -p ~/aistor-binaries
curl --progress-bar -L \
  https://dl.min.io/aistor/mc/release/darwin-arm64/mc \
  -o ~/aistor-binaries/mc
chmod +x ~/aistor-binaries/mc
export PATH="$HOME/aistor-binaries:$PATH" # works on linux and mac systems only

mc alias set depthwizard \
  http://localhost:9000 \
  "$MINIO_ACCESS_KEY" \
  "$MINIO_SECRET_KEY" # put actual data from .env in here

mc license update depthwizard .secrets/minio.license/minio.license.txt

mc license info depthwizard # verification

```

```bash
docker compose up --build
# backend -> http://localhost:8000  (health: /api/v1/health, docs: /docs)
```

The image runs `uvicorn backend.app.main:app` as a non-root user, installs
dependencies from `pyproject.toml` + `uv.lock` (`--frozen`), and mounts
`./outputs` (checkpoints) and `./data` (scene storage).

## Tests

```bash
uv pytest            # full suite: ML core + DEM pipeline + backend API
# or:
uv run pytest model_tests tests backend_tests
```

## Honesty contracts (do not break them)

- FINAL/citable numbers come ONLY from `model.py evaluate` — the serving
  path is a demonstration, never a metric source.
- Dn is min-max normalized **per 1024 tile** at training AND inference
  (`depthwizard/inference.py:normalize_dn_per_tile`).
- Slope is computed only when a real GSD exists; otherwise the API reports
  `gsd_available: false` with null degrees — never a guess.
- Anchoring is arithmetic and always labelled `ANCHORED (not learned)`.
- The API never fabricates success: unknown scenes 404 before jobs are
  created, result `job_id` is null when unknown, missing artifacts 404.
