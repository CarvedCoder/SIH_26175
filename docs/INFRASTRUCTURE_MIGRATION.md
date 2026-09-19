# DepthWizard Infrastructure Migration — Supabase Auth · PostgreSQL · MinIO

This document describes the infrastructure migration: Supabase owns
authentication, PostgreSQL (Supabase-hosted in production) owns relational
persistence, MinIO (S3 API) owns object storage, and the DepthWizard ML
inference pipeline is untouched.

## From a clean machine

```bash
cp .env.example .env        # fill in Supabase + credentials
python scripts/start.py     # venv, deps, docker infra, backend, frontend
```

The startup script is idempotent: containers are reused, dependencies are
only installed when missing/out of sync, and Ctrl+C stops the child servers
cleanly (PostgreSQL/MinIO are left running; `docker compose stop` stops them).

## What changed (source of truth)

| Concern            | Before                              | After |
|--------------------|-------------------------------------|-------|
| Authentication     | static `X-API-Key` (or disabled)    | **Supabase JWT** (`Authorization: Bearer`), verified in `backend/app/core/auth.py`; legacy API-key mode and explicit local-dev disabled mode retained |
| Scene/job metadata | `scene.json` files + JSON job files | **SQL tables** (`scenes`, `jobs`) via SQLAlchemy 2 (`backend/app/db/`) |
| Object storage     | local filesystem only               | **MinIO/S3** durable store (`backend/app/storage/`); local disk remains the processing workspace + read-through cache |
| Result delivery    | `FileResponse` streams              | **short-lived presigned URLs** (307 redirect on file routes; absolute presigned URLs inside JSON payloads) or direct streams with the local backend |

## Authentication flow

1. Frontend signs in/up through `@supabase/supabase-js` (`frontend/src/lib/supabase.js`,
   real Supabase calls in `src/store/authContext.jsx` + `AuthPage`).
2. Every API call (`frontend/src/api/client.js apiFetch`) attaches
   `Authorization: Bearer <access_token>` and retries once after a session
   refresh on 401.
3. Backend `get_current_user` verifies the JWT (HS256 via `SUPABASE_JWT_SECRET`,
   or ES256/RS256 via the project JWKS) and exposes the verified identity.
4. The scene guard (`_require_scene`) checks the scene's `owner_id` against
   the verified `sub`: foreign scenes get **403**, unknown ones 404, missing/
   invalid tokens **401**. User ids are never taken from request payloads.

## Object storage flow

* Upload: validated raster is saved locally (deterministic `input.<ext>`),
  uploaded to `scenes/{owner}/{scene}/input.<ext>`, and its key is stored in
  the `scenes` row.
* Processing: inference keeps writing to the local output dir (unchanged
  pipeline). On job completion every artifact is uploaded under
  `results/{owner}/{scene}/` and registered in `scenes.artifacts` (JSON map
  of name → {key, size}).
* Reads: missing local artifacts are re-downloaded from the object store
  (read-through cache in `result_service`), so a restart on a fresh machine
  still serves results. File routes redirect (307) to a presigned GET URL
  generated **only after** the ownership check; JSON payload URLs are
  absolute presigned URLs so `<img>`/three.js loaders work without headers.
* TTL: `STORAGE_SIGNED_URL_TTL` (default 300 s). MinIO credentials never
  reach the browser; the bucket is never public; keys are built only from
  validated scene ids + the verified owner id.

## Database

* Engine from `DATABASE_URL` (single source of truth, `backend/app/db/database.py`).
  Default is the compose PostgreSQL; there is **no silent SQLite fallback** —
  an unreachable database fails the process at startup.
* Schema is created with `Base.metadata.create_all` at startup. No SQLite
  data migration is needed: the previous implementation had **no database**
  (scene metadata lived in per-scene `scene.json` files). Legacy scene
  directories without a row still resolve in `_require_scene` for local-dev
  continuity; re-uploading a scene registers it properly.
* Jobs survive backend restarts; a job orphaned by a crash is marked
  `JOB_INTERRUPTED` on the next read (PID-liveness check preserved).

## Environment variables

See `.env.example` for the full annotated list (Supabase, DATABASE_URL,
MINIO_*, STORAGE_SIGNED_URL_TTL, DW_* inference knobs, VITE_* frontend).

## Supabase setup

1. Create a project; note the **project URL**, **anon key**, and
   **JWT secret** (or use asymmetric JWKS keys — the backend supports both).
2. Backend: set `SUPABASE_URL` + `SUPABASE_JWT_SECRET` (or just `SUPABASE_URL`
   for JWKS) and `SUPABASE_JWT_AUDIENCE=authenticated`.
3. Frontend (`frontend/.env`): `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`.
4. Database: use Supabase's PostgreSQL connection string as `DATABASE_URL`
   (pooler URI, `+psycopg` driver). Schema is created automatically.

## Tests

`pytest backend_tests` covers API contract, jobs, scenes, results, terrain,
errors, the frontend contract, plus the new `test_auth.py` (401/invalid/
expired JWT, valid user, cross-user 403 on scenes/jobs/exports) and
`test_storage.py` (key-safety, upload/exists, presign TTL, sync + cache
re-materialization, deletion) with a fake S3 client. All pre-existing ML
tests are untouched; `model_tests/test_checkpoint_security.py` already
enforces `weights_only=True` checkpoint loading and still passes.

## Known limitations

* `tests/test_scenes.py::test_png_scene_processes_end_to_end` fails on the
  pre-migration baseline too (DSM-availability assertion vs. the `dsm.npy`
  fallback in `result_service`) — pre-existing, unrelated to this migration.
* `uv.lock` predates the four new dependencies; the Dockerfile now resolves
  at build time (`uv sync --no-dev`). Run `uv lock` to restore frozen installs.
* Social login buttons remain non-functional (Supabase provider config is a
  dashboard task).
* Alembic migrations are not introduced yet; `create_all` is idempotent and
  additive. Add Alembic once the schema starts evolving.
