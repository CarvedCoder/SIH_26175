"""Result file serving — fixed allowlist, no arbitrary path requests.

Every servable artifact name maps to an exact filename inside the scene's
output directory; the allowlist is the ONLY way a request resolves to a
file (audit: arbitrary filesystem paths must never be requestable).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.api.routes.scenes import _require_scene
from backend.app.core.errors import AppError
from backend.app.services.result_service import result_service

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Result Files"],
)

# Extended allowlist beyond the core pipeline outputs: artifacts generated
# by the validation/reference/minimap features.
_EXTRA_ALLOWED_FILES = {
    "reference": ("reference.tif", "reference_dem.tif", "ref_dem.tif"),
    "minimap": ("minimap.png",),
    "heightmap": ("heightmap.png",),
    "rgb": ("rgb_preview.png",),
    "error-map": (
        "error_map.png",
        "validation_error_map.png",
        "error_map.tif",
    ),
}


def _get_result_file(scene_id: str, result_name: str) -> FileResponse:
    """Return a generated result file for a scene."""

    files = result_service.get_result_files(scene_id)
    path = files.get(result_name)

    if path is None and result_name in _EXTRA_ALLOWED_FILES:
        output_dir = result_service.get_output_dir(scene_id)
        path = next(
            (
                output_dir / name
                for name in _EXTRA_ALLOWED_FILES[result_name]
                if (output_dir / name).is_file()
            ),
            None,
        )

    if path is None:
        raise AppError(
            status_code=404,
            code="RESULT_NOT_FOUND",
            message="The requested result artifact does not exist for this scene.",
            details={"scene_id": scene_id, "result": result_name},
            recoverable=True,
        )

    return FileResponse(path=path, filename=path.name)


@router.get("/{scene_id}/results/preview")
async def get_preview(scene_id: str):
    """Return the generated DSM preview image."""
    _require_scene(scene_id)
    return _get_result_file(scene_id, "preview")


@router.get("/{scene_id}/results/dsm")
async def get_dsm(scene_id: str):
    """Return the generated georeferenced DSM GeoTIFF."""
    _require_scene(scene_id)
    return _get_result_file(scene_id, "dsm")


@router.get("/{scene_id}/results/depth")
async def get_depth(scene_id: str):
    """Return the generated numerical depth/DSM array."""
    _require_scene(scene_id)
    return _get_result_file(scene_id, "depth")


@router.get("/{scene_id}/results/reference")
async def get_reference_file(scene_id: str):
    """Return the reference DEM raster, when one exists."""
    _require_scene(scene_id)
    return _get_result_file(scene_id, "reference")


@router.get("/{scene_id}/results/minimap")
async def get_minimap_file(scene_id: str):
    """Return the terrain minimap image."""
    _require_scene(scene_id)
    return _get_result_file(scene_id, "minimap")


@router.get("/{scene_id}/results/error-map")
async def get_error_map_file(scene_id: str):
    """Return the validation error-map image."""
    _require_scene(scene_id)
    return _get_result_file(scene_id, "error-map")


@router.get("/{scene_id}/results/heightmap")
async def get_heightmap_file(scene_id: str):
    """Return the browser-friendly heightmap PNG (R channel = elevation)."""
    _require_scene(scene_id)
    from backend.app.services.terrain_service import terrain_service

    # generate on demand (idempotent) so the URL is always real
    try:
        terrain_service.get_heightmap_path(scene_id)
    except (FileNotFoundError, ValueError):
        pass
    return _get_result_file(scene_id, "heightmap")


@router.get("/{scene_id}/results/rgb")
async def get_rgb_file(scene_id: str):
    """Return the RGB preview of the source imagery (texture layer)."""
    _require_scene(scene_id)
    from backend.app.services.terrain_service import terrain_service

    try:
        terrain_service.get_rgb_preview_path(scene_id)
    except (FileNotFoundError, ValueError):
        pass
    return _get_result_file(scene_id, "rgb")
