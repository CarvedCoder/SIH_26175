"""Export routes — downloadable scene artifacts via export_service.

Export types: depth | dsm | terrain | validation. Unknown types get a
typed 400; missing artifacts an honest 404 (never a fabricated file).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.api.routes.scenes import _require_scene
from backend.app.core.errors import AppError
from backend.app.services.export_service import (
    ExportFileNotFound,
    get_export,
)

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Export"],
)


@router.get("/{scene_id}/export/{export_type}")
async def export_scene(scene_id: str, export_type: str):
    """Download an exported scene artifact (file stream)."""
    _require_scene(scene_id)

    try:
        path = get_export(scene_id, export_type)
    except ExportFileNotFound as exc:
        raise AppError(
            status_code=404,
            code="EXPORT_NOT_FOUND",
            message="No export exists for this scene and type.",
            details={"scene_id": scene_id, "export_type": export_type},
            recoverable=True,
        ) from exc
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="INVALID_EXPORT_TYPE",
            message=str(exc),
            recoverable=False,
        ) from exc

    return FileResponse(
        path=path, filename=f"{scene_id}_{export_type}{path.suffix}"
    )
