"""Terrain routes: normalized terrain, tiles, minimap, elevation probes,
and measurement tools.

Honesty contract: measurements are computed from the PREDICTED DSM. Slope
requires a real ground sample distance — non-georeferenced scenes get the
typed refusal shape (``gsd_available: false``, null degrees) instead of a
fabricated metric slope, mirroring depthwizard's GSD-honesty rules.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.api.routes.scenes import _require_scene
from backend.app.core.errors import AppError
from backend.app.schemas.measurement import (
    ElevationResponse,
    HeightMeasureRequest,
    HeightMeasureResponse,
    SlopeMeasureRequest,
    SlopeMeasureResponse,
    TerrainTilesResponse,
)
from backend.app.schemas.result import ArtifactUrlResponse
from backend.app.schemas.terrain import TerrainResponse
from backend.app.services.terrain_service import terrain_service

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Terrain"],
)


def _require_results(scene_id: str) -> None:
    _require_scene(scene_id)
    if not terrain_service.terrain_available(scene_id):
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No terrain results exist for this scene yet — process it first.",
            details={"scene_id": scene_id},
            recoverable=True,
        )


def _require_depth(scene_id: str) -> None:
    """Point products work off the numerical DSM array (which exists for
    non-georeferenced scenes too)."""
    _require_scene(scene_id)
    if not terrain_service.depth_available(scene_id):
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No depth results exist for this scene yet — process it first.",
            details={"scene_id": scene_id},
            recoverable=True,
        )


@router.get("/{scene_id}/terrain", response_model=TerrainResponse)
async def get_scene_terrain(scene_id: str):
    """Return the normalized terrain representation for a scene."""
    _require_results(scene_id)

    terrain = terrain_service.get_terrain(scene_id)
    return TerrainResponse(
        scene_id=scene_id,
        available=True,
        terrain=terrain,
    )


@router.get("/{scene_id}/terrain/tiles", response_model=TerrainTilesResponse)
async def get_scene_terrain_tiles(
    scene_id: str,
    tile_size: int = Query(default=256, ge=32, le=1024),
):
    """Return the tile grid over the scene's terrain."""
    _require_depth(scene_id)
    tiles = terrain_service.terrain_tiles(scene_id, tile_size=tile_size)
    return TerrainTilesResponse(
        scene_id=scene_id,
        tile_size=tiles["tile_size"],
        grid_x=tiles["grid_x"],
        grid_y=tiles["grid_y"],
        tiles=tiles["tiles"],
    )


@router.get("/{scene_id}/minimap", response_model=ArtifactUrlResponse)
async def get_scene_minimap(scene_id: str):
    """Return a small overview image of the predicted terrain."""
    _require_depth(scene_id)
    try:
        terrain_service.get_minimap_path(scene_id)
    except FileNotFoundError:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No depth results exist for this scene yet.",
            recoverable=True,
        )
    return ArtifactUrlResponse(
        scene_id=scene_id,
        url=f"/api/v1/scenes/{scene_id}/results/minimap",
        format="png",
    )


@router.get("/{scene_id}/elevation", response_model=ElevationResponse)
async def get_scene_elevation(
    scene_id: str,
    x: int = Query(..., ge=0),
    y: int = Query(..., ge=0),
):
    """Point elevation probe on the predicted DSM."""
    _require_depth(scene_id)
    try:
        elevation = terrain_service.sample_elevation(scene_id, x, y)
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="POINT_OUT_OF_BOUNDS",
            message=str(exc),
            recoverable=True,
        )
    return ElevationResponse(
        scene_id=scene_id, x=x, y=y, elevation=elevation, units="meters"
    )


@router.post(
    "/{scene_id}/measure/height", response_model=HeightMeasureResponse
)
async def measure_height(scene_id: str, request: HeightMeasureRequest):
    """Vertical delta between two points sampled from the predicted DSM."""
    _require_depth(scene_id)
    try:
        result = terrain_service.measure_height(
            scene_id,
            (request.ground.x, request.ground.y),
            (request.top.x, request.top.y),
        )
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="POINT_OUT_OF_BOUNDS",
            message=str(exc),
            recoverable=True,
        )
    return HeightMeasureResponse(scene_id=scene_id, **result)


@router.post(
    "/{scene_id}/measure/slope", response_model=SlopeMeasureResponse
)
async def measure_slope(scene_id: str, request: SlopeMeasureRequest):
    """Slope between two points. Non-georeferenced scenes get an honest
    refusal (null slope, gsd_available=false) — pixel-space slopes are not
    metric and are never guessed."""
    _require_depth(scene_id)
    try:
        result = terrain_service.measure_slope(
            scene_id,
            (request.point_a.x, request.point_a.y),
            (request.point_b.x, request.point_b.y),
        )
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="POINT_OUT_OF_BOUNDS",
            message=str(exc),
            recoverable=True,
        )
    return SlopeMeasureResponse(scene_id=scene_id, **result)
