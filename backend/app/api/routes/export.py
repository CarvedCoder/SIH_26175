"""Export routes — downloadable scene artifacts via export_service.

Export types: depth | dsm | terrain | validation. Unknown types get a
typed 400; missing artifacts an honest 404 (never a fabricated file).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

from backend.app.api.routes._deps import require_scene as _require_scene
from backend.app.core.auth import current_user
from backend.app.core.errors import AppError
from backend.app.db.database import session_scope
from backend.app.db.models import SceneRow
from backend.app.services.export_service import (
    ExportFileNotFound,
    get_export,
)
from backend.app.storage.service import storage_service

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

    # S3/RustFS backend: deliver via a short-lived presigned URL (the
    # ownership guard above has already passed); local backend streams.
    if storage_service.is_object_store:
        with session_scope() as session:
            row = session.get(SceneRow, scene_id)
            owner_id = row.owner_id if row is not None else current_user().user_id
        url = storage_service.publish_and_presign(owner_id, scene_id, path)
        if url is not None:
            return RedirectResponse(url=url, status_code=307)

    return FileResponse(
        path=path, filename=f"{scene_id}_{export_type}{path.suffix}"
    )
