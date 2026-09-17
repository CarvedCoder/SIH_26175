# Backend Architecture Audit

Date: 2026-09-17. Scope: `backend/app/**`, `backend_tests/**`,
`depthwizard/backbone.py`, `depthwizard/inference.py` (orchestration
surface), `depthwizard/tifops.py` (checkpoint loading). Every claim below
was verified by reading the code; file:line references are given where
they matter.

---

## 1. Executive summary

The backend is in **better architectural shape than a "modular,
stateless" rewrite brief typically assumes** — a previous refactor
(the `c4fae0d` "stateless code refine" commit) already made job state
disk-backed and removed several globals. The remaining gaps are real but
bounded, and they fall into four groups:

1. **Job interruption detection uses PID liveness as truth**
   (`backend/app/jobs/manager.py:158` `_revive_or_fail`). On a
   multi-host deployment a PID is meaningless: PID reuse produces false
   "alive", and a worker on another host is always falsely dead or falsely
   alive. §5 of the refactor brief forbids this. Fix: durable
   lease/heartbeat; PID demoted to diagnostics.
2. **The in-process job cache can serve stale state across processes**
   (`manager.py:102` `_jobs` + `manager.py:239` `get_job` returns the
   cached copy without consulting disk). Two API instances polling the
   same job can permanently disagree: whichever process first cached the
   job never re-reads the disk record. This breaks Test B (any instance
   answers any job) in a subtle way: it works on *first* read, then
   freezes.
3. **Storage paths are constructed by business logic.** `paths.py` is a
   clean, security-reviewed layout module, but `result_service`,
   `terrain_service`, `export_service`, and `processing_service` all
   build raw `Path`s from scene ids and open files directly
   (`result_service.py:9`, `terrain_service.py:9,26,49,309,388`). There
   is no `ArtifactStore` seam, so pointing the system at S3/MinIO
   requires touching every consumer. (Local-disk works; horizontal scale
   with shared volumes only half-works — `paths.py` hard-codes
   `PROJECT_ROOT`.)
4. **Job execution is in-process** (`jobs.py:84`
   `background_tasks.add_task`). The API process owns GPU inference. Job
   *records* survive a restart, but in-flight work does not (honestly
   recorded as `JOB_INTERRUPTED`). There is no `TaskQueue`/worker seam,
   so "any worker can process queued work" is not yet reachable.

Plus the documented-but-not-yet-load-bearing items: model singletons
(`backbone._DEFAULT`, `job_manager`, `processing_service`,
`result_service`), env-var reads outside `core/config.py`
(`processing_service.py:48-53`, `:37-39`), and a monolithic 981-line
`depthwizard/inference.py`.

## 2. What is already right (preserve these)

* **Job records are disk-first, write-through** — every mutation writes a
  JSON document under `data/process/scenes/<id>/jobs/<job_id>.json` with
  atomic replace (`manager.py:130-139`). Restart loses nothing; the
  state machine (queued → processing → completed/failed/cancelled) is
  explicit and terminal states are locked.
* **Typed error contract** (`core/errors.py`): stable
  `{error:{code,message,details,recoverable}}` shape, registered
  handlers; no raw tracebacks reach the client.
* **Upload path is stateless-safe**: staging → rasterio content
  validation → atomic move → `scene.json` metadata with the designated
  `input.<ext>` determinism contract (`routes/scenes.py:247-279`).
* **Path-traversal defenses** in `core/paths.py` (id format validation,
  symlink-escape guard on deletion).
* **Request-ID middleware + access logs** (`core/middleware.py`).
* **Health endpoint honestly reports checkpoint resolvability** without
  loading models (`routes/health.py:18-25`).
* **backbone.py is a disposable cache done right**: lazy, thread-safe,
  re-derivable from config; nothing depends on it being warm.

## 3. Detailed findings

### 3.1 State inventory (who owns what today)

| State | Owner | Durable? | Correctness-critical? |
|---|---|---|---|
| Job records | JSON files under scene process dir | yes | yes — but see §1.1/§1.2 |
| Scene input + metadata | `data/raw/scenes/<id>/` | yes | yes |
| Processing outputs | `data/output/scenes/<id>/` | yes | yes |
| Upload staging | `data/staging/` | yes (transient) | no |
| In-memory job cache | `JobManager._jobs` | **no** | **yes — bug §1.2** |
| `_INFERENCE_SEMAPHORE` | process-local | no | concurrency policy only |
| DAv2 weights | `backbone._DEFAULT` (process-local) | no | no — reloadable |
| Checkpoint | file path from `DW_CKPT` | yes | yes |
| Settings | `core/config.py` env read, frozen | n/a | no |

### 3.2 Dependency direction

`routes → services → jobs/manager → core/paths` and
`services → depthwizard.inference`. The ML layer does **not** import
`backend.*` (verified: `depthwizard/inference.py` has no backend
imports) — the domain boundary the brief demands already exists in that
direction. The violations are all *within* the backend: routes contain
business logic (scenes.py upload/mosaic ≈ 430 lines of use-case code),
and services construct storage paths directly.

### 3.3 Specific defects

* **PID as truth** — `manager.py:76-86, 158`: `_pid_alive()` decides
  whether a processing job is marked `JOB_INTERRUPTED`. Wrong on any
  multi-host topology; PID reuse on a single host can keep a dead job
  "running" forever.
* **Stale cross-process reads** — `manager.py:191-192` prefers the
  in-memory cache over disk; `get_job` at `:235-243` returns cached
  without disk validation. Multi-instance polling disagrees.
* **Background-task execution** — `routes/jobs.py:39-66`: the worker is
  the API process. A crash loses in-flight work (recorded honestly, but
  the queue/worker seam of brief §6 does not exist).
* **Env reads outside config** — `processing_service.py:37-39, 48-53`
  reads `DW_MAX_CONCURRENT_JOBS`, `DW_DEVICE`, `DW_BACKBONE`,
  `DW_NO_LIVE`, `DW_CKPT`, `DW_CKPT_SHA256` directly; `core/config.py`
  claims to be the single source of truth.
* **Module-level singletons as service locators** —
  `jobs/manager.py:412`, `processing_service.py:376`,
  `result_service.py:112`. All import-time constructed; tests monkeypatch
  around them. Injectable construction is missing.
* **No artifact abstraction** — binary outputs (dsm.npy/tif/preview) are
  addressed by path convention across four services; artifact identity
  (id, checksum, size, content type) is nowhere recorded.
* **`depthwizard/inference.py`** (981 lines) mixes: input reading,
  tiling, backbone dispatch, normalization, calibration, post-processing
  dispatch, anchoring, georeferencing, artifact writing, and payload
  assembly. It is the numerical heart of the product and is covered by
  `model_tests` — a decomposition must be golden-regression-gated
  (refactor brief §52-54).

### 3.4 Concurrency

`threading.Semaphore` serializes GPU work per process
(`processing_service.py:37`). Correct for single-process; a no-op across
instances. Concurrency policy belongs to the worker/queue layer once one
exists.

### 3.5 Test coverage today

`backend_tests/` uses a mocked `run_inference` (no GPU) and redirects
storage into tmp dirs (`conftest.py` isolated_storage). Tests exist for
jobs, scenes, results, errors, frontend contract, terrain, health. They
do **not** test cross-instance reads, lease expiry, or cache-clear
invariance — the statelessness properties the refactor targets.

## 4. Migration plan (incremental; brief §51 order adapted)

Tranche 1 — **delivered** (backend seams, zero behavior change
except the two defect fixes):
1. ✅ `backend/app/domain/` — Job entity + `JobRepository` / `ArtifactStore`
   protocols (no FastAPI imports).
2. ✅ `FileJobRepository` — disk-first reads (fixes §1.2), durable
   lease/heartbeat replaces PID-as-truth (fixes §1.1); `owner_pid` kept
   as a diagnostic; legacy job files (PID-era) migrate on read.
3. ✅ `JobService` application layer; old `job_manager` import becomes a
   facade over the service (no caller broken). Split health endpoints
   `/health/live` + `/health/ready` added.
4. ✅ `ArtifactStore` protocol + `LocalArtifactStore` + contract tests
   (S3-compatible implementation is the next tranche; services rewiring
   follows).
5. ✅ Statelessness test suite (restart / two instances / lease expiry /
   cache-clear / legacy migration).

Tranche 2 — **delivered** (execution split + config):
6. ✅ `TaskQueue` protocol + request-scoped `BackgroundTaskQueue` (inline
   mode) + external worker `python -m backend.app.worker` claiming from
   the durable store (`claim_queued` / `list_queued`); `Job.request`
   persisted so any worker can execute any job.
7. ✅ Env reads centralized into `core/config.py` (`DW_CKPT`,
   `DW_CKPT_SHA256`, `DW_DEVICE`, `DW_BACKBONE`, `DW_NO_LIVE`,
   `DW_WORKER_MODE`); `ProcessingService` takes Settings only.
8. ✅ Golden regression fixture generated + pinned
   (`model_tests/test_golden_regression.py`) — tranche 3's gate.

Tranche 3a — **delivered** (use-case layer):
9. ✅ `FileSceneRepository` (`infrastructure/persistence/`) — owns the
   scene record (scene.json), designated-input determinism, scene-id
   listing; layout/security invariants stay in core/paths.py.
10. ✅ `SceneService` (`application/scenes/`) — inspection, staging→
    validate→commit, mosaic orchestration, validation, deletion. Routes
    now only stream uploads and shape Pydantic responses; the six
    route-to-route `_require_scene` imports replaced by
    `api/routes/_deps.py`. Typed `RasterInvalid`/`SceneInputAmbiguous`
    errors replace HTTPException-in-business-logic.
11. ✅ `processing_service` input resolution delegates to the scene
    service; `record_failure` maps `AppError` codes first so typed
    errors reach the job record verbatim.

Tranche 3b — **delivered** (storage addressing + durable claims):
12. ✅ All backend services (result/terrain/export/validation/processing)
    address artifacts via `scene_artifacts.py` key builders +
    `ArtifactStore.path_for` (keys mirror the on-disk layout; an
    S3/MinIO store now only changes that module). Local Paths still flow
    to rasterio/FileResponse until the object-store implementation lands.
13. ✅ `SqliteJobRepository` — DB-backed JobRepository (stdlib SQLite,
    WAL): `claim_queued` runs inside BEGIN IMMEDIATE, so claims are
    exactly-once ACROSS worker processes. Selected by `DW_JOB_STORE`
    ("file" default | "sqlite"), `DW_JOB_DB`.
14. ✅ State-machine policy extracted to `job_ops.py` (shared by both
    repositories — implementations cannot drift); the statelessness
    contract suite now runs against BOTH implementations (§46).
15. ✅ `ProcessingService` job-store access is constructor-injected
    (`job_service=`) — no service-locator globals in the worker path.

Tranche 3c — **started** (inference.py decomposition, golden-gated):
16. ✅ Scene outputs + payload assembly moved to
    `depthwizard/pipeline/scene_outputs.py` (pure moves; names
    re-exported from inference.py so every import path is unchanged;
    golden regression pins the behavior).
17. Remaining: split the DepthWizardPredictor into DepthProvider /
    CalibrationModel / Tiling stages behind protocols; artifact writing
    behind an explicit ArtifactWriter that targets the ArtifactStore.

Tranche 3 — ML pipeline decomposition (golden-regression-gated):
9. Golden fixture from a fixed input + checkpoint; byte-compare raw Dn /
   Dn / calibrated AGL / post-processed / DSM before & after.
10. Split `inference.py` into pipeline stages behind protocols
    (DepthProvider / CalibrationModel / PostProcessor / Anchor / DSM /
    ArtifactWriter) — stages individually testable, CLI and API share
    them.
11. `ModelRuntime` owning backbone+calibration caches explicitly.

## 5. Explicitly out of scope (and why)

* **Postgres/Redis/S3 forced adoption** — the deployment today is
  single-host Docker; the brief itself says to choose the smallest
  reliable mechanism. The protocols make Postgres/S3 drop-in later.
* **API contract changes** — none. All routes keep their request/response
  shapes (frontend contract test pins this).
* **Numerical behavior** — untouched in this tranche; the ML pipeline is
  called exactly as before.
