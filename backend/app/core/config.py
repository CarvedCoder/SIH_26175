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
        # When DW_API_KEY is set, every /api/v1 route except /health
        # requires the X-API-Key header. When unset, auth is DISABLED —
        # an explicit, logged local-development configuration (never a
        # silent bypass).
        self.api_key: str | None = os.environ.get("DW_API_KEY") or None

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
