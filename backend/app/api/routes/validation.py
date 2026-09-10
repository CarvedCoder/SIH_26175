"""Validation routes — comparison against reference elevation data.

GET  /{scene_id}/validation           -> metrics + artifacts (honest: only
                                         what actually exists in the output
                                         directory)
GET  /{scene_id}/validation/error-map -> JSON metadata {url, format}; the
                                         image file itself is served from
                                         the canonical allowlisted route
                                         /results/error-map (the frontend
                                         reads ``url`` from the JSON)
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes.scenes import _require_scene
from backend.app.core.errors import AppError
from backend.app.schemas.result import ArtifactUrlResponse
from backend.app.services.validation_service import get_validation

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Validation"],
)


@router.get("/{scene_id}/validation")
async def get_scene_validation(scene_id: str):
    """Return validation results for a processed scene."""
    _require_scene(scene_id)
    return get_validation(scene_id)


@router.get("/{scene_id}/validation/error-map")
async def get_validation_error_map(scene_id: str):
    """Return error-map metadata (the frontend follows the ``url``)."""
    _require_scene(scene_id)

    validation = get_validation(scene_id)
    if validation.error_map is None:
        raise AppError(
            status_code=404,
            code="ERROR_MAP_NOT_FOUND",
            message="No validation error map exists for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )

    return ArtifactUrlResponse(
        scene_id=scene_id,
        url=validation.error_map.url,
        format=validation.error_map.format or "png",
    )
