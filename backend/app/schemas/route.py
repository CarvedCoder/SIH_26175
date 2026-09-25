"""Route-assist schemas (jury round 1: vehicle passability + routing)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PointPixel(BaseModel):
    x: int = Field(..., ge=0, description="Column (x) in source-raster pixels")
    y: int = Field(..., ge=0, description="Row (y) in source-raster pixels")


class RouteAssessRequest(BaseModel):
    start: PointPixel
    end: PointPixel
    vehicles: list[str] = Field(
        default_factory=lambda: ["fire_truck"],
        min_length=1,
        max_length=5,
        description=(
            "Vehicle profile keys: fire_truck | ambulance | rescue_atv | "
            "suv_4x4 | rescue_chopper (aerial: landing-zone assessment)"
        ),
    )


class VehicleAssessment(BaseModel):
    vehicle: str
    vehicle_label: str
    verdict: str  # CAN_GO | CAUTION | CANNOT_GO
    reasons: list[str]
    limits: dict[str, float]
    path: list[PointPixel] | None = None
    path_length_m: float | None = None
    path_length_px: float | None = None
    max_slope_on_path_deg: float | None = None
    max_step_on_path_m: float | None = None
    caution_fraction: float | None = None
    estimated_travel_seconds: float | None = None
    detour_pixel: PointPixel | None = None
    # Which picks were auto-moved to standable/reachable ground ('start'
    # and/or 'end') — the reasons list carries the human explanation.
    snapped: dict[str, bool] | None = None
    # Aerial vehicles only: the selected touchdown point near the
    # destination (flat, open patch) instead of a ground path.
    landing_zone: dict[str, Any] | None = None
    path_geojson: dict[str, Any] | None = None

    # Semantic attribution on the path
    road_fraction: float | None = None
    building_fraction: float | None = None
    water_fraction: float | None = None
    vegetation_fraction: float | None = None
    ground_fraction: float | None = None
    other_fraction: float | None = None
    semantic_risk_fraction: float | None = None
    semantic_aware: bool = False


class RouteAssessResponse(BaseModel):
    scene_id: str
    start_pixel: PointPixel
    end_pixel: PointPixel
    units: str
    georeferenced: bool
    disclaimer: str
    semantic_available: bool = False
    path_crs: str | None = None
    vehicles: list[VehicleAssessment]


class HeatmapResponse(BaseModel):
    scene_id: str
    vehicle: str
    url: str
    format: str = "png"
    blocked_pct: float
    caution_pct: float
    free_pct: float
    georeferenced: bool
