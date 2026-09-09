from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.terrain import TerrainResponse
from backend.app.services.terrain_service import terrain_service


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Terrain"],
)


@router.get(
    "/{scene_id}/terrain",
    response_model=TerrainResponse,
)
async def get_scene_terrain(scene_id: str):
    """Return the normalized terrain representation for a scene."""

    if not terrain_service.terrain_available(scene_id):
        raise HTTPException(
            status_code=404,
            detail=f"Terrain not available for scene '{scene_id}'.",
        )

    try:
        terrain = terrain_service.get_terrain(scene_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return TerrainResponse(
        scene_id=scene_id,
        available=True,
        terrain=terrain,
    )