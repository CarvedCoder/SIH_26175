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

## Quick start (the master CLI)

```bash
git clone <repository>
cd SIH_26175

./dw setup      # detects prerequisites, asks before installing/downloading,
                # creates .env (local SQLite), syncs deps, starts RustFS,
                # verifies the depthwizard bucket + checkpoints
./dw start      # preflight, then launches backend + frontend (Ctrl+C stops)
```

Then open:

```text
Frontend        http://localhost:5173
Backend         http://localhost:8010
API docs        http://localhost:8010/docs
RustFS console  http://localhost:9001   (S3 API: http://localhost:9000)
```

`dw setup` works from a fresh clone: it only asks for confirmation before
installing anything or downloading model weights, never uses sudo/admin
rights, and configures the zero-config local default
(`DATABASE_URL=sqlite:///data/depthwizard.db` — no Supabase/external
PostgreSQL required to run the demo). Checkpoints download automatically
(RDAH ~65 MB, MD5-verified; DAv2 weights ~0.3-1.3 GB on first use, only
after you confirm).

### All CLI commands

| Command          | What it does                                                              |
| ---------------- | ------------------------------------------------------------------------- |
| `./dw doctor`    | read-only diagnostics (add `--verbose`)                                   |
| `./dw setup`     | install/configure/verify everything needed to run (idempotent, resumable) |
| `./dw start`     | preflight, then hand over to `scripts/start.py` (Ctrl+C stops all)        |
| `./dw test`      | config + RustFS roundtrip + backend tests (`--full` adds ML tests + a real inference on the demo tile) |
| `./dw status`    | component status table + URLs                                             |
| `./dw logs`      | `rustfs` \| `backend` \| `frontend` \| all                                |
| `./dw stop`      | stop dev processes (deletes nothing)                                      |
| `./dw restart`   | stop + start                                                              |
| `./dw clean`     | `--cache` \| `--deps` \| `--data` \| `--all` (destructive: confirmed)     |

Windows: use `dw.cmd` with the same commands.

### What `dw setup` automates — and what it needs from you

**Automatically installed/configured (with a confirmation prompt first):**
uv (user-level, no admin rights), Python dependencies via `uv sync`
(python deps only; torch wheels ~2–4 GB on a truly fresh machine, cached
afterwards), frontend deps via `npm ci`, `.env` bootstrap with the
zero-config SQLite default, RustFS container + `depthwizard` bucket
creation, RDAH-Net checkpoint (~65 MB, figshare, MD5-verified on arrival).

**Confirmed separately (large downloads — never silent):**

| Asset | Size | Destination |
| ----- | ---- | ----------- |
| Depth Anything V2 weights | 0.3–1.3 GB | `~/.cache/huggingface/hub/` |
| RDAH-Net released checkpoint | ~65 MB | `checkpoints/rdah/` |
| RustFS Docker image | ~60 MB | Docker |
| torch/CUDA wheels (first sync only) | ~2–4 GB | `.venv` (via uv cache) |

**Requires manual installation (admin rights — this script never uses
sudo/apt/brew/choco):** Python ≥ 3.12, Node.js + npm, Docker Desktop /
Docker Engine. If any are missing, `dw setup` prints exact per-platform
instructions and stops; after installing them, re-run `./dw setup` — it is
idempotent and resumes where it left off.

**Run without confirmation:** add `--yes` (e.g. `./dw setup --yes`). It
accepts normal install/download confirmations but never bypasses
destructive confirmations (typing `delete`).

### `dw` local defaults

| Thing | Default |
| ----- | ------- |
| Database | `DATABASE_URL=sqlite:///data/depthwizard.db` (created on first start; no Supabase/PostgreSQL needed for the demo) |
| Object storage | RustFS (S3 API `http://localhost:9000`, console `http://localhost:9001`), bucket `depthwizard` auto-created |
| Presigned URLs | signed against `S3_PUBLIC_ENDPOINT` (browser-facing) while data ops use `S3_ENDPOINT` (internal) |
| Backend | `http://localhost:8010` (host dev, `DW_PORT`), port 8000 in docker-compose |
| Frontend | `http://localhost:5173` |

### `dw` exit codes

`0` success · `1` general failure · `2` invalid configuration ·
`3` missing prerequisite · `4` test failure · `5` cancelled

### `dw test`

```bash
./dw test        # config + RustFS roundtrip (write/read/delete/presign)
                 # + pytest backend_tests (+ live backend/frontend checks
                 #   when the stack is running)
./dw test --full # + pytest model_tests + a real inference pass on the
                 #   demo tile (needs the checkpoints; slow on CPU)
```

### Troubleshooting

- `dw doctor` is always safe to run — it changes nothing and tells you
  exactly which prerequisite is missing.
- A `STOPPED` RustFS after a reboot: `docker compose start rustfs`, or just
  `./dw start` (it brings storage up before launching the app).
- Port already in use: `dw start` detects a running backend and refuses to
  launch duplicates — use `./dw status`, `./dw stop` first.
- `dw stop` / `dw clean --data` never touch RustFS objects, model
  checkpoints, or model weights.

## Advanced / manual setup

Prefer to drive each piece yourself? Everything the CLI automates can be run
by hand.

Backend (uv manages the lockfile; Python >= 3.12):

```bash
# 1) install
uv sync --extra dev

# 2) provide a trained checkpoint (default resolution order):
#    $DW_CKPT  else  outputs/calib_net/postproc_flagship/best.pt  else  ./best.pt
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

Object storage is **RustFS** — a self-hosted, S3-compatible server with **no
license mechanism** and no per-device activation. Credentials come from `.env`
(`S3_ACCESS_KEY` / `S3_SECRET_KEY`); the `depthwizard` bucket is created
automatically by the backend on startup.

```bash
docker compose up --build
# backend  -> http://localhost:8000  (health: /api/v1/health, docs: /docs)
# RustFS S3 API  -> http://localhost:9000
# RustFS console -> http://localhost:9001  (sign in with S3_ACCESS_KEY/S3_SECRET_KEY)
```

Objects persist in the named Docker volume `rustfs_data` and survive
container restarts. Inside the compose network the backend talks to
`http://rustfs:9000`; presigned URLs handed to the browser use
`S3_PUBLIC_ENDPOINT` (default `http://localhost:9000`).

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
