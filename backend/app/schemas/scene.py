from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SceneStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingPath(str, Enum):
    RELATIVE = "relative"
    CALIBRATED = "calibrated"
    ABSOLUTE = "absolute"


class SceneCapabilities(BaseModel):
    """Capabilities available for a scene based on its input/reference data."""

    absolute_elevation: bool = False
    relative_elevation: bool = True
    reference_comparison: bool = False
    slope: bool = False
    height_measurement: bool = False
    error_map: bool = False
    local_refinement: bool = False


class SceneDimensions(BaseModel):
    """Raster/image dimensions."""

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    channels: int | None = Field(default=None, ge=1)


class SceneGeoReference(BaseModel):
    """Geospatial information extracted from the input."""

    available: bool = False
    crs: str | None = None

    min_x: float | None = None
    min_y: float | None = None
    max_x: float | None = None
    max_y: float | None = None

    pixel_width: float | None = None
    pixel_height: float | None = None


class SceneCreateResponse(BaseModel):
    """Response returned after a scene is uploaded."""

    scene_id: str
    filename: str
    status: SceneStatus

    format: str | None = None

    dimensions: SceneDimensions | None = None
    georeference: SceneGeoReference = Field(
        default_factory=SceneGeoReference
    )

    processing_path: ProcessingPath = ProcessingPath.RELATIVE

    capabilities: SceneCapabilities = Field(
        default_factory=SceneCapabilities
    )


class SceneResponse(SceneCreateResponse):
    """Complete scene metadata returned by the API."""

    created_at: str | None = None
    updated_at: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

class SceneSummary(BaseModel):
    """Entry of the scene list (GET /scenes)."""

    scene_id: str
    filename: str
    status: SceneStatus
    has_results: bool
    created_at: str | None = None
