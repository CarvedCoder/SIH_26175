"""Health endpoints — canonical contract + platform compatibility.

    GET /api/v1/health  -> frontend contract {status, version, model_loaded}
    GET /health         -> identical payload for Docker/platform healthchecks

    GET /api/v1/health/live  -> liveness: the process is up (no deps touched)
    GET /api/v1/health/ready -> readiness: able to accept work (cheap check)

``model_loaded`` honestly reports whether the serving stack can serve
inference: a height-model checkpoint is resolvable (DW_CKPT or the repo
default for the requested backend — RDAH by default, else CalibrationNet).
It is NOT a claim that the DAv2 backbone is resident (it
lazy-loads on first use). Readiness performs the same cheap filesystem
check — no GPU/model loading happens on any health request.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse
from backend.app.services.processing_service import processing_service

router = APIRouter(
    prefix="/api/v1",
    tags=["Health"],
)


def _health_payload() -> HealthResponse:
    try:
        checkpoint, _architecture = processing_service._resolve_checkpoint()
        model_loaded = checkpoint.is_file()
    except (FileNotFoundError, ValueError):
        model_loaded = False

    return HealthResponse(
        status="ok",
        version=settings.version,
        model_loaded=model_loaded,
    )


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return _health_payload()


@router.get("/health/live", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Liveness: process is serving. Deliberately touches NOTHING — no
    storage, no config file IO, no model state — so it can never fail
    because a dependency is down (that is readiness's job)."""
    return HealthResponse(status="ok", version=settings.version,
                          model_loaded=False)


@router.get("/health/ready", response_model=HealthResponse)
def readiness() -> HealthResponse:
    """Readiness: this instance can accept work (checkpoint resolvable,
    storage root writable). Cheap by design — never loads models."""
    model_loaded = False
    try:
        checkpoint, _architecture = processing_service._resolve_checkpoint()
        model_loaded = checkpoint.is_file()
    except (FileNotFoundError, ValueError):
        pass
    # a readiness probe also proves the durable roots are reachable
    from backend.app.core.paths import ensure_directories

    try:
        ensure_directories()
        ok = True
    except OSError:
        ok = False
    return HealthResponse(
        status="ok" if ok else "degraded",
        version=settings.version,
        model_loaded=model_loaded,
    )


@router.get("/health/models")
def models_health(load_and_probe: bool = Query(default=False)) -> dict:
    """Per-backend model health report (TerraHeight integration).

    Cheap by default: each backend resolves its checkpoint and verifies it
    EXISTS on the filesystem. ``?load_and_probe=true`` additionally loads
    the TerraHeight-S weights and runs one small (154x154) forward with a
    NaN/Inf check — deliberately NOT part of the plain health/readiness
    contracts (no GPU work on ordinary health requests) and never a full
    scene run."""
    backends: dict = {}
    for arch in ("rdah", "calibration_net", "terraheight_s"):
        entry: dict = {"checkpoint": None, "checkpoint_exists": False}
        try:
            checkpoint, _ = processing_service._resolve_checkpoint(arch)
            entry["checkpoint"] = str(checkpoint)
            entry["checkpoint_exists"] = checkpoint.is_file()
        except (FileNotFoundError, ValueError) as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        backends[arch] = entry
    report = {
        "status": "ok",
        "version": settings.version,
        "backends": backends,
        "terraheight_probe": None,
    }
    if load_and_probe:
        from depthwizard.terraheight import terraheight_health

        try:
            checkpoint, _ = processing_service._resolve_checkpoint("terraheight_s")
        except (FileNotFoundError, ValueError):
            checkpoint = None
        report["terraheight_probe"] = terraheight_health(checkpoint)
    return report
