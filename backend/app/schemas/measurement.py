from __future__ import annotations

from pydantic import BaseModel, Field


class Point(BaseModel):
    """Pixel-space point on the scene grid."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class ElevationResponse(BaseModel):
    """Point elevation probe (GET /scenes/{id}/elevation)."""

    scene_id: str
    x: int
    y: int
    elevation: float | None
    units: str = "meters"


class HeightMeasureRequest(BaseModel):
    """Body of POST /scenes/{id}/measure/height."""

    ground: Point
    top: Point


class HeightMeasureResponse(BaseModel):
    """Vertical delta between two points, sampled from the DSM."""

    scene_id: str
    ground_elevation: float
    top_elevation: float
    height: float
    units: str = "meters"


class SlopeMeasureRequest(BaseModel):
    """Body of POST /scenes/{id}/measure/slope."""

    point_a: Point
    point_b: Point


class SlopeMeasureResponse(BaseModel):
    """Slope between two points. Computed ONLY when the scene carries a
    real ground sample distance (GSD) — the repo's honesty contract refuses
    to fabricate metric slopes from pixel-space guesses."""

    scene_id: str
    slope_degrees: float | None = None
    slope_percent: float | None = None
    horizontal_distance_m: float | None = None
    elevation_change_m: float
    units: str = "meters"
    gsd_available: bool = False


class TerrainTile(BaseModel):
    """One terrain tile descriptor for the frontend's tile loader."""

    x: int
    y: int
    width: int
    height: int
    url: str


class TerrainTilesResponse(BaseModel):
    scene_id: str
    tile_size: int
    grid_x: int
    grid_y: int
    tiles: list[TerrainTile]


class MinimapResponse(BaseModel):
    scene_id: str
    url: str
    format: str = "png"
