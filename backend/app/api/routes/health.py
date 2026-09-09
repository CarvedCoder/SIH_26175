"""Health endpoints — canonical contract + platform compatibility.

    GET /api/v1/health  -> frontend contract {status, version, model_loaded}
    GET /health         -> identical payload for Docker/platform healthchecks

``model_loaded`` honestly reports whether the serving stack can serve
inference: the calibration checkpoint is resolvable (DW_CKPT or the repo
default). It is NOT a claim that the DAv2 backbone is resident (it
lazy-loads on first use).
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse
from backend.app.services.processing_service import processing_service

router = APIRouter(
    prefix="/api/v1",
    tags=["Health"],
)


def _health_payload() -> HealthResponse:
    try:
        checkpoint = processing_service._resolve_checkpoint()
        model_loaded = checkpoint.is_file()
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok",
        version=settings.version,
        model_loaded=model_loaded,
    )


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return _health_payload()
