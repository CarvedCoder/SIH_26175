"""Building detection and damage assessment API routes.

GET /api/v1/scenes/{scene_id}/buildings           building metadata + availability
GET /api/v1/scenes/{scene_id}/damage              damage metadata + availability
GET /api/v1/scenes/{scene_id}/results/buildings-geojson     buildings GeoJSON
GET /api/v1/scenes/{scene_id}/results/buildings-preview     buildings preview PNG
GET /api/v1/scenes/{scene_id}/results/damage-geojson        damage GeoJSON
GET /api/v1/scenes/{scene_id}/results/damage-preview        damage preview PNG
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.api.routes._deps import require_scene
from backend.app.core.errors import AppError

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Disaster"],
)


def _disaster_files(scene_id: str) -> dict[str, Path]:
    """Return existing disaster artifact paths for a scene."""
    from backend.app.services.result_service import result_service

    files = result_service.get_result_files(scene_id)
    return {
        k: v for k, v in files.items()
        if k.startswith("building") or k.startswith("damage")
    }


def _serve_artifact(scene_id: str, path: Path):
    """Serve an artifact via presigned URL or direct file response."""
    from backend.app.storage.service import storage_service

    if storage_service.is_object_store:
        url = storage_service.presign_artifact(scene_id, path)
        if url is not None:
            from fastapi.responses import RedirectResponse
            return RedirectResponse(
                url=url,
                status_code=307,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Cache-Control": "no-cache, must-revalidate",
                },
            )
    return FileResponse(
        path=path,
        filename=path.name,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache, must-revalidate",
        },
    )


def _artifact_url(scene_id: str, filename: str, fallback: str) -> str:
    """Build a browser-accessible URL for a disaster artifact."""
    from backend.app.services.result_service import result_service

    output_dir = result_service.get_output_dir(scene_id)
    artifact_path = output_dir / filename
    if not artifact_path.exists():
        return fallback

    from backend.app.storage.service import storage_service
    url = storage_service.presign_artifact(scene_id, artifact_path)
    return url or fallback


@router.get("/{scene_id}/buildings")
async def get_buildings_meta(scene_id: str):
    """Building detection metadata and availability."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)

    meta_path = files.get("buildings_meta")
    if not meta_path or not meta_path.exists():
        return {
            "available": False,
            "count": 0,
            "mode": None,
            "georeferenced": False,
            "url": None,
            "geojson_url": None,
        }

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base = f"/api/v1/scenes/{scene_id}/results"
    return {
        "available": True,
        "count": meta.get("count", 0),
        "mode": meta.get("mode", "post_only"),
        "georeferenced": meta.get("georeferenced", False),
        "coordinate_space": meta.get("coordinate_space", "pixel_space"),
        "url": _artifact_url(
            scene_id, "buildings_preview.png", f"{base}/buildings-preview"
        ),
        "geojson_url": _artifact_url(
            scene_id, "buildings.geojson", f"{base}/buildings-geojson"
        ),
    }


@router.get("/{scene_id}/damage")
async def get_damage_meta(scene_id: str):
    """Damage assessment metadata and availability."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)

    meta_path = files.get("damage_meta")
    if not meta_path or not meta_path.exists():
        return {
            "available": False,
            "mode": None,
            "building_count": 0,
            "damage_counts": {},
            "review_count": 0,
            "geojson_url": None,
            "preview_url": None,
        }

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base = f"/api/v1/scenes/{scene_id}/results"
    return {
        "available": True,
        "mode": meta.get("mode", "post_only"),
        "building_count": meta.get("building_count", 0),
        "damage_counts": meta.get("damage_counts", {}),
        "review_count": meta.get("review_count", 0),
        "mean_confidence": meta.get("mean_confidence"),
        "recovered_destroyed_areas": meta.get("recovered_destroyed_areas", 0),
        "georeferenced": meta.get("georeferenced", False),
        "colors": meta.get("colors", {}),
        "geojson_url": _artifact_url(
            scene_id, "damage_buildings.geojson", f"{base}/damage-geojson"
        ),
        "preview_url": _artifact_url(
            scene_id, "damage_preview.png", f"{base}/damage-preview"
        ),
        "url": _artifact_url(
            scene_id, "damage_preview.png", f"{base}/damage-preview"
        ),
    }


@router.get("/{scene_id}/results/buildings-geojson")
@router.get("/{scene_id}/buildings/geojson")
async def buildings_geojson(scene_id: str):
    """Detected buildings as GeoJSON FeatureCollection."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)
    path = files.get("buildings_geojson")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="BUILDINGS_NOT_AVAILABLE",
            message="Building detection results are not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/results/buildings-preview")
@router.get("/{scene_id}/buildings/preview")
async def buildings_preview(scene_id: str):
    """Building detection preview PNG."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)
    path = files.get("buildings_preview")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="BUILDINGS_NOT_AVAILABLE",
            message="Building detection preview is not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/buildings/mask")
async def buildings_mask(scene_id: str):
    """Building detection binary mask (.npy)."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)
    path = files.get("building_mask")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="BUILDINGS_NOT_AVAILABLE",
            message="Building mask array is not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/results/damage-geojson")
@router.get("/{scene_id}/damage/geojson")
async def damage_geojson(scene_id: str):
    """Damage assessment as GeoJSON with per-building damage classes."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)
    path = files.get("damage_geojson")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="DAMAGE_NOT_AVAILABLE",
            message="Damage assessment results are not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/results/damage-preview")
@router.get("/{scene_id}/damage/preview")
async def damage_preview(scene_id: str):
    """Damage assessment color-coded preview PNG."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)
    path = files.get("damage_preview")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="DAMAGE_NOT_AVAILABLE",
            message="Damage assessment preview is not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/damage/labels")
async def damage_labels(scene_id: str):
    """Damage assessment per-pixel labels array (.npy)."""
    require_scene(scene_id)
    files = _disaster_files(scene_id)
    path = files.get("damage_labels")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="DAMAGE_NOT_AVAILABLE",
            message="Damage labels array is not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)
