"""Central environment-driven configuration — the single source of truth.

Precedence: environment variable > default here. No other backend module
may read os.environ for serving behavior directly (checkpoint resolution
stays in the processing service because it is inference-domain config).

All knobs are namespaced ``DW_*``. Wildcard CORS is never a default.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()


class Settings:
    """Frozen-at-first-use serving configuration."""

    def __init__(self) -> None:
        # --- CORS -------------------------------------------------------
        # Comma-separated explicit origins. Development defaults cover the
        # Vite dev server; production MUST set DW_CORS_ORIGINS explicitly.
        raw_origins = os.environ.get(
            "DW_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        if raw_origins.strip() == "*":
            # Explicit opt-in wildcard (never the default); kept for local
            # demos only.
            self.cors_origins: list[str] = ["*"]
        else:
            self.cors_origins = [
                origin.strip()
                for origin in raw_origins.split(",")
                if origin.strip()
            ]

        # --- Auth --------------------------------------------------------
        # AUTHORITY: Supabase owns authentication. When Supabase is
        # configured (SUPABASE_JWT_SECRET for HS256 projects, or
        # SUPABASE_URL for JWKS/asymmetric keys) every /api/v1 route
        # requires a valid Bearer access token issued by Supabase.
        # DW_API_KEY remains an explicit LEGACY fallback (CI/tooling);
        # when neither is configured auth is DISABLED for local
        # development — an explicit, logged state, never a silent bypass.
        self.supabase_url: str | None = os.environ.get("SUPABASE_URL") or None
        self.supabase_jwt_secret: str | None = (
            os.environ.get("SUPABASE_JWT_SECRET") or None
        )
        self.supabase_jwt_audience: str = (
            os.environ.get("SUPABASE_JWT_AUDIENCE") or "authenticated"
        )
        self.api_key: str | None = os.environ.get("DW_API_KEY") or None

        # --- Database ------------------------------------------------------
        # Single source of truth for SQL persistence (scene/job/result
        # metadata). Defaults to the docker-compose PostgreSQL; there is
        # no silent SQLite fallback — an unreachable database fails fast
        # at startup. Tests override DATABASE_URL with a per-test SQLite
        # file.
        self.database_url: str = (
            os.environ.get("DATABASE_URL")
            or "postgresql+psycopg://depthwizard:depthwizard@localhost:5432/depthwizard"
        )

        # --- Object storage ---------------------------------------------------
        # MinIO (S3 API) is the durable object store; the local filesystem
        # remains the processing workspace + read-through cache. "local"
        # keeps the pre-migration behavior (no object store).
        self.storage_backend: str = os.environ.get("STORAGE_BACKEND", "minio")
        self.minio_endpoint: str = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        self.minio_access_key: str = os.environ.get("MINIO_ACCESS_KEY", "")
        self.minio_secret_key: str = os.environ.get("MINIO_SECRET_KEY", "")
        self.minio_bucket: str = os.environ.get("MINIO_BUCKET", "depthwizard")
        self.minio_region: str | None = os.environ.get("MINIO_REGION") or None
        self.minio_secure: bool = os.environ.get("MINIO_SECURE", "false") == "true"
        self.storage_signed_url_ttl: int = _int_env("STORAGE_SIGNED_URL_TTL", 300)

        # --- Upload limits ------------------------------------------------
        self.max_upload_bytes: int = _int_env("DW_MAX_UPLOAD_BYTES", 500 * 1024 * 1024)

        # --- Jobs ----------------------------------------------------------
        self.max_concurrent_jobs: int = _int_env("DW_MAX_CONCURRENT_JOBS", 1)
        self.job_retention_limit: int = _int_env("DW_JOB_RETENTION_LIMIT", 500)
        self.job_ttl_seconds: int = _int_env("DW_JOB_TTL_SECONDS", 24 * 3600)
        # Durable lease for claimed jobs (worker heartbeats renew it at
        # lease/3). Must comfortably exceed the longest single inference
        # run even WITHOUT heartbeats, as a belt-and-braces margin.
        self.job_lease_seconds: int = _int_env("DW_JOB_LEASE_SECONDS", 1800)
        # Durable job store: "file" (JSON per scene dir) or "sqlite"
        # (transactional claims; exactly-once across worker processes).
        self.job_store: str = os.environ.get("DW_JOB_STORE", "file")
        self.job_db_path: str = os.environ.get("DW_JOB_DB", "") or ""

        # --- Serving identity ------------------------------------------------
        self.version: str = "1.0.0"

        # --- Inference / worker domain (tranche 2: centralized here so no
        # other module reads os.environ for serving behavior) ----------------
        self.device: str = os.environ.get("DW_DEVICE", "auto")
        self.backbone_id: str = os.environ.get(
            "DW_BACKBONE",
            "depth-anything/Depth-Anything-V2-Base-hf",
        )
        # DW_NO_LIVE=1 disables the live DAv2 fallback (offline honesty).
        self.live_backbone: bool = os.environ.get("DW_NO_LIVE") != "1"
        self.checkpoint: str | None = os.environ.get("DW_CKPT") or None
        self.checkpoint_sha256: str | None = (
            os.environ.get("DW_CKPT_SHA256") or None
        )
        # Job execution: "inline" (default; API threadpool — dev mode) or
        # "external" (API only records jobs; backend.app.worker claims them).
        self.worker_mode: str = os.environ.get("DW_WORKER_MODE", "inline")
        self.worker_poll_seconds: float = float(
            os.environ.get("DW_WORKER_POLL_SECONDS", "1.0")
        )


settings = get_settings()
