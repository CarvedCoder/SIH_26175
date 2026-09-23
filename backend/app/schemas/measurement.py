from __future__ import annotations

from pydantic import BaseModel, Field


class Point(BaseModel):
    """Pixel-space point on the scene grid."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class ElevationConfidence(BaseModel):
    """Honest accuracy statement for a metered elevation sample."""

    level: str  # 'high' | 'medium' | 'low' | 'none'
    percent: int | None = None  # only when validation RMSE exists
    basis: str  # what the level is grounded in


class ElevationResponse(BaseModel):
    """Point elevation probe (GET /scenes/{id}/elevation)."""

    scene_id: str
    x: int
    y: int
    elevation: float | None
    units: str = "meters"
    metered: bool = False
    precision_m: float | None = None
    confidence: ElevationConfidence | None = None
    # Local ground level (low percentile of the DSM around the point) and
    # the sampled pixel's height above it — a geometric structure-height
    # estimate from the predicted surface, present when the sample is valid.
    ground_elevation: float | None = None
    height_above_ground_m: float | None = None
    is_structure: bool = False
    height_confidence: ElevationConfidence | None = None


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
