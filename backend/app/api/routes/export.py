from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.export_service import (
    ExportFileNotFound,
    get_export,
)


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Export"],
)


@router.get("/{scene_id}/export/{export_type}")
async def export_scene(
    scene_id: str,
    export_type: str,
):
    """Download an exported scene artifact."""
    try:
        path = get_export(scene_id, export_type)
    except ExportFileNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return FileResponse(path=path)