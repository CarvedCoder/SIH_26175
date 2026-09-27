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
        # metadata). SQLite IS the default main database (zero-config local
        # development); Supabase/PostgreSQL is the opt-in upgrade — set
        # DATABASE_URL to the Supabase connection string to use it (same
        # schema, same code). Tests override DATABASE_URL with a per-test
        # SQLite file.
        self.database_url: str = (
            os.environ.get("DATABASE_URL")
            or "sqlite:///data/depthwizard.db"
        )

        # --- Object storage ---------------------------------------------------
        # RustFS (S3 API) is the durable object store; the local filesystem
        # remains the processing workspace + read-through cache. "local"
        # keeps the pre-migration behavior (no object store). The legacy
        # value "minio" is accepted as an alias of "s3" (the S3 client code
        # is server-agnostic — any S3-compatible endpoint works).
        self.storage_backend: str = os.environ.get("STORAGE_BACKEND", "s3").lower()
        # S3_ENDPOINT is what the BACKEND talks to (inside Docker:
        # rustfs:9000; host-side dev: localhost:9000). Accepts either
        # "host:port" (scheme chosen via S3_SECURE) or a full URL.
        self.s3_endpoint: str = os.environ.get("S3_ENDPOINT", "localhost:9000")
        # S3_PUBLIC_ENDPOINT is what PRESIGNED URLs embed — the host the
        # BROWSER must reach. Unset => presign against S3_ENDPOINT (correct
        # for host-side dev where they are the same; wrong inside Docker,
        # where the browser cannot resolve the internal service name).
        self.s3_public_endpoint: str | None = (
            os.environ.get("S3_PUBLIC_ENDPOINT") or None
        )
        self.s3_access_key: str = os.environ.get("S3_ACCESS_KEY", "")
        self.s3_secret_key: str = os.environ.get("S3_SECRET_KEY", "")
        self.s3_bucket: str = os.environ.get("S3_BUCKET", "depthwizard")
        self.s3_region: str | None = os.environ.get("S3_REGION") or None
        self.s3_secure: bool = os.environ.get("S3_SECURE", "false") == "true"
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
        # Per-backend checkpoint overrides (RDAH integration): these WIN over
        # the generic DW_CKPT for their backend, so the webapp backend switch
        # can serve a fine-tuned RDAH net AND a CalibrationNet checkpoint at
        # the same time (a generic DW_CKPT only fits one architecture).
        self.ckpt_rdah: str | None = os.environ.get("DW_CKPT_RDAH") or None
        self.ckpt_calib: str | None = os.environ.get("DW_CKPT_CALIB") or None
        self.checkpoint_sha256: str | None = (
            os.environ.get("DW_CKPT_SHA256") or None
        )
        # Optional inference inputs (unset => auto/None, as before):
        # DW_DN_PATH forces a specific Dn normalization raster; DW_ANCHOR_DEM
        # anchors the DSM to a reference DEM; DW_CACHE_DIR is the backbone
        # weight cache.
        self.dn_path: str | None = os.environ.get("DW_DN_PATH") or None
        self.anchor_dem: str | None = os.environ.get("DW_ANCHOR_DEM") or None
        self.cache_dir: str | None = os.environ.get("DW_CACHE_DIR") or None

        # Post-processing (run_inference postprocess= preset + params).
        # Presets: none|median|guided|bilateral|wls|conf|semantic|full
        self.postprocess: str = os.environ.get("DW_POSTPROCESS", "wls")
        self.wls_lambda: float = float(os.environ.get("DW_WLS_LAMBDA", "2.0"))
        self.wls_sigma_rgb: float = float(
            os.environ.get("DW_WLS_SIGMA_RGB", "0.08")
        )
        self.wls_max_iter: int = _int_env("DW_WLS_MAX_ITER", 300)
        # Flip/rotate ensemble inside the refinement (runtime ~3-4x).
        self.tta: bool = os.environ.get("DW_TTA", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # Job execution: "inline" (default; API threadpool — dev mode) or
        # "external" (API only records jobs; backend.app.worker claims them).
        self.worker_mode: str = os.environ.get("DW_WORKER_MODE", "inline")
        self.worker_poll_seconds: float = float(
            os.environ.get("DW_WORKER_POLL_SECONDS", "1.0")
        )

        # --- Semantic segmentation -----------------------------------------
        # DW_SEMANTIC=false disables semantic prediction entirely (no
        # sem artifacts generated, Route Assist uses geometry only).
        self.semantic_enabled: bool = (
            os.environ.get("DW_SEMANTIC", "true").strip().lower()
            in {"1", "true", "yes"}
        )
        self.semantic_confidence_threshold: float = float(
            os.environ.get("DW_SEM_CONFIDENCE", "0.70")
        )
        self.semantic_hard_block_threshold: float = float(
            os.environ.get("DW_SEM_HARD_BLOCK", "0.90")
        )

        # --- Route semantic integration ------------------------------------
        # DW_ROUTE_SEMANTIC=false makes Route Assist ignore semantics even
        # when the artifacts exist (geometry-only mode, like pre-upgrade).
        self.route_semantic_enabled: bool = (
            os.environ.get("DW_ROUTE_SEMANTIC", "true").strip().lower()
            in {"1", "true", "yes"}
        )


settings = get_settings()
