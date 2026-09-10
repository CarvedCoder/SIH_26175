from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ValidationMetrics(BaseModel):
    """Numerical comparison metrics between generated DSM and reference data."""

    rmse: float | None = Field(default=None, ge=0.0)
    mae: float | None = Field(default=None, ge=0.0)
    correlation: float | None = None

    sample_count: int | None = Field(
        default=None,
        ge=0,
    )

    units: str = "meters"


class ValidationAsset(BaseModel):
    """A validation-related file exposed by the backend."""

    name: str
    url: str
    format: str | None = None


class ReferenceDEM(BaseModel):
    """Metadata describing the reference elevation dataset."""

    available: bool = False

    name: str | None = None
    crs: str | None = None
    units: str | None = None

    width: int | None = Field(
        default=None,
        ge=1,
    )

    height: int | None = Field(
        default=None,
        ge=1,
    )


class ValidationResponse(BaseModel):
    """Validation result for a processed scene."""

    scene_id: str

    available: bool = False

    reference: ReferenceDEM = Field(
        default_factory=ReferenceDEM
    )

    metrics: ValidationMetrics = Field(
        default_factory=ValidationMetrics
    )

    error_map: ValidationAsset | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

class ValidationDimensions(BaseModel):
    width: int
    height: int
    channels: int | None = None


class ValidationCheckResponse(BaseModel):
    """Result of POST /scenes/{id}/validate — an input-raster health check
    performed BEFORE processing. This is distinct from the processed-scene
    validation comparison (GET /scenes/{id}/validation)."""

    scene_id: str
    valid: bool
    issues: list[str] = Field(default_factory=list)
    georeferenced: bool | None = None
    dimensions: ValidationDimensions | None = None
