from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.core.paths import get_scene_output_dir
from backend.app.schemas.validation import ValidationResponse
from backend.app.services.validation_service import get_validation


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Validation"],
)


@router.get(
    "/{scene_id}/validation",
    response_model=ValidationResponse,
)
async def get_scene_validation(scene_id: str) -> ValidationResponse:
    """Return validation results for a processed scene."""
    return get_validation(scene_id)


@router.get("/{scene_id}/validation/error-map")
async def get_validation_error_map(scene_id: str):
    """Return the generated validation error map."""
    output_dir = get_scene_output_dir(scene_id)

    candidates = [
        output_dir / "error_map.png",
        output_dir / "validation_error_map.png",
        output_dir / "error_map.tif",
    ]

    for path in candidates:
        if path.is_file():
            return FileResponse(path=Path(path))

    raise HTTPException(
        status_code=404,
        detail=f"No validation error map found for scene '{scene_id}'.",
    )