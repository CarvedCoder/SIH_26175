"""Geometry-aware 3D building reconstruction API routes.

GET /api/v1/scenes/{scene_id}/buildings3d           metadata + availability
GET /api/v1/scenes/{scene_id}/results/buildings3d   footprints + DSM heights JSON
GET /api/v1/scenes/{scene_id}/results/buildings3d-preview     preview PNG
"""

from __future__ import annotations

import json

from fastapi import APIRouter

from backend.app.api.routes._deps import require_scene
from backend.app.api.routes.disaster import _artifact_url, _serve_artifact

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Buildings3D"],
)


@router.get("/{scene_id}/buildings3d")
async def get_buildings3d_meta(scene_id: str):
    """3D building reconstruction metadata and availability.

    availability follows the artifact on disk — a scene processed before
    this feature (or with no building candidates) reports available=false
    instead of an invented empty payload.
    """
    require_scene(scene_id)

    from backend.app.services.result_service import result_service

    output_dir = result_service.get_output_dir(scene_id)
    meta_path = output_dir / "buildings3d.json"
    if not meta_path.is_file():
        return {
            "available": False,
            "count": 0,
            "mask_source": None,
            "height_source": None,
            "georeferenced": False,
            "url": None,
        }

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            reconstruction = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {
            "available": False,
            "count": 0,
            "mask_source": None,
            "height_source": None,
            "georeferenced": False,
            "url": None,
        }

    base = f"/api/v1/scenes/{scene_id}/results"
    return {
        "available": bool(reconstruction.get("available")),
        "count": int(reconstruction.get("count", 0)),
        "mask_source": reconstruction.get("mask_source"),
        "height_source": reconstruction.get("height_source"),
        "georeferenced": bool(reconstruction.get("georeferenced")),
        "crs": reconstruction.get("crs"),
        "units": reconstruction.get("units", "m"),
        "damage_classified": int(reconstruction.get("damage_classified", 0)),
        "damage_classes": reconstruction.get("damage_classes", []),
        "tree_count": int(reconstruction.get("tree_count", 0)),
        "has_ground": bool(reconstruction.get("has_ground")),
        "ground_heightmap_url": _artifact_url(
            scene_id, "ground_heightmap.png", f"{base}/ground-heightmap"
        ) if reconstruction.get("has_ground") else None,
        "url": _artifact_url(scene_id, "buildings3d.json", f"{base}/buildings3d"),
        "preview_url": _artifact_url(
            scene_id, "buildings3d_preview.png", f"{base}/buildings3d-preview"
        ),
    }


@router.get("/{scene_id}/results/buildings3d")
async def get_buildings3d_json(scene_id: str):
    """Serve the reconstruction JSON (footprints + heights + confidence)."""
    require_scene(scene_id)

    from backend.app.services.result_service import result_service

    path = result_service.get_output_dir(scene_id) / "buildings3d.json"
    if not path.is_file():
        from backend.app.core.errors import AppError

        raise AppError(
            status_code=404,
            code="RESULT_NOT_FOUND",
            message="No 3D building reconstruction exists for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/results/buildings3d-preview")
async def get_buildings3d_preview(scene_id: str):
    """Serve the reconstruction preview PNG (footprint overlay)."""
    require_scene(scene_id)

    from backend.app.core.errors import AppError
    from backend.app.services.result_service import result_service

    path = result_service.get_output_dir(scene_id) / "buildings3d_preview.png"
    if not path.is_file():
        raise AppError(
            status_code=404,
            code="RESULT_NOT_FOUND",
            message="No 3D building reconstruction preview exists for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)
