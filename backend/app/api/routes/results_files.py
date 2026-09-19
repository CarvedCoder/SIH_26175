"""Result file serving — fixed allowlist, no arbitrary path requests.

Every servable artifact name maps to an exact filename inside the scene's
output directory; the allowlist is the ONLY way a request resolves to a
file (audit: arbitrary filesystem paths must never be requestable).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

from backend.app.api.routes._deps import require_scene as _require_scene
from backend.app.core.errors import AppError
from backend.app.db.database import session_scope
from backend.app.db.models import SceneRow
from backend.app.services.result_service import result_service
from backend.app.storage.service import storage_service

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


def _get_result_file(scene_id: str, result_name: str):
    """Return a generated result file for a scene.

    Delivery: with the S3/MinIO backend the client receives a 307 redirect
    to a SHORT-LIVED presigned URL (generated only after the ownership
    guard passed; no credentials are exposed). With the local development
    backend the file is streamed directly."""
    from backend.app.core.auth import current_user

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

    if storage_service.is_object_store:
        with session_scope() as session:
            row = session.get(SceneRow, scene_id)
            owner_id = row.owner_id if row is not None else current_user().user_id
        url = storage_service.publish_and_presign(owner_id, scene_id, path)
        if url is not None:
            return RedirectResponse(url=url, status_code=307)

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


# ── Viewer texture layers (greyscale data PNGs, shader-colormapped) ──


def _layer_texture_response(scene_id: str, generator):
    """Serve a generated viewer layer texture, or a clean 404."""
    from backend.app.core.auth import current_user
    from backend.app.db.database import session_scope as _scope
    from backend.app.db.models import SceneRow as _SceneRow
    from backend.app.services.terrain_service import terrain_service

    try:
        path = generator(terrain_service, scene_id)
    except (FileNotFoundError, ValueError) as exc:
        raise AppError(
            status_code=404,
            code="RESULT_NOT_FOUND",
            message="That texture layer is not available for this scene yet.",
            details={"scene_id": scene_id, "reason": str(exc)},
            recoverable=True,
        )
    if storage_service.is_object_store:
        with _scope() as session:
            row = session.get(_SceneRow, scene_id)
            owner_id = row.owner_id if row is not None else current_user().user_id
        url = storage_service.publish_and_presign(owner_id, scene_id, path)
        if url is not None:
            return RedirectResponse(url=url, status_code=307)
    return FileResponse(path=path, filename=path.name)


@router.get("/{scene_id}/results/dsm-texture")
async def get_dsm_texture_file(scene_id: str):
    """Return the greyscale DSM texture for the viewer's DSM layer."""
    _require_scene(scene_id)
    return _layer_texture_response(
        scene_id, lambda svc, sid: svc.get_dsm_layer_path(sid)
    )


@router.get("/{scene_id}/results/slope")
async def get_slope_texture_file(scene_id: str):
    """Return the greyscale slope texture for the viewer's Slope layer."""
    _require_scene(scene_id)
    return _layer_texture_response(
        scene_id, lambda svc, sid: svc.get_slope_layer_path(sid)
    )


@router.get("/{scene_id}/results/reference-preview")
async def get_reference_texture_file(scene_id: str):
    """Return the greyscale reference-DEM texture for the Compare menu."""
    _require_scene(scene_id)
    return _layer_texture_response(
        scene_id, lambda svc, sid: svc.get_reference_layer_path(sid)
    )
