from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.result_service import result_service


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Result Files"],
)


def _get_result_file(scene_id: str, result_name: str) -> FileResponse:
    """Return a generated result file for a scene."""

    files = result_service.get_result_files(scene_id)
    path = files.get(result_name)

    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Result '{result_name}' not found "
                f"for scene '{scene_id}'."
            ),
        )

    return FileResponse(path=path)


@router.get("/{scene_id}/results/preview")
async def get_preview(scene_id: str):
    """Return the generated DSM preview image."""

    return _get_result_file(scene_id, "preview")


@router.get("/{scene_id}/results/dsm")
async def get_dsm(scene_id: str):
    """Return the generated georeferenced DSM GeoTIFF."""

    return _get_result_file(scene_id, "dsm")


@router.get("/{scene_id}/results/depth")
async def get_depth(scene_id: str):
    """Return the generated numerical depth/DSM array."""

    return _get_result_file(scene_id, "depth")