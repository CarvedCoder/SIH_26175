from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResultAsset(BaseModel):
    """A browser-accessible or downloadable result asset."""

    name: str
    url: str
    format: str | None = None
    size_bytes: int | None = None


class ElevationStats(BaseModel):
    """Elevation/depth statistics for a generated terrain product."""

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    relief: float | None = None
    units: str = "meters"


class DepthResult(BaseModel):
    """Depth estimation result."""

    available: bool = False

    preview: ResultAsset | None = None
    raw: ResultAsset | None = None

    width: int | None = None
    height: int | None = None

    statistics: ElevationStats | None = None


class DSMResult(BaseModel):
    """Digital Surface Model result."""

    available: bool = False

    preview: ResultAsset | None = None
    raster: ResultAsset | None = None

    width: int | None = None
    height: int | None = None

    statistics: ElevationStats | None = None

    crs: str | None = None

    bounds: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )


class ResultCapabilities(BaseModel):
    """Capabilities supported by the generated results."""

    depth: bool = False
    dsm: bool = False
    terrain: bool = False
    validation: bool = False
    reference_comparison: bool = False
    export: bool = False


class SceneResultsResponse(BaseModel):
    """Complete result response for a processed scene.

    ``job_id`` is null when result artifacts exist but no job record does
    (evicted or post-restart) — the backend never invents ids."""
    scene_id: str
    job_id: str | None = None

    status: str

    depth: DepthResult = Field(
        default_factory=DepthResult
    )

    dsm: DSMResult = Field(
        default_factory=DSMResult
    )

    capabilities: ResultCapabilities = Field(
        default_factory=ResultCapabilities
    )

    assets: list[ResultAsset] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

class DepthMetaResponse(BaseModel):
    """Depth-layer metadata for the frontend viewer (GET /scenes/{id}/depth).

    ``url`` is the visual (preview PNG); ``download_url`` is the raw
    product (float32 .npy). Both are real, servable endpoints."""

    scene_id: str
    available: bool
    url: str | None = None
    download_url: str | None = None
    format: str | None = None
    width: int | None = None
    height: int | None = None
    statistics: ElevationStats | None = None


class DsmMetaResponse(BaseModel):
    """DSM-layer metadata for the frontend viewer (GET /scenes/{id}/dsm)."""

    scene_id: str
    available: bool
    url: str | None = None
    download_url: str | None = None
    format: str | None = None
    width: int | None = None
    height: int | None = None
    crs: str | None = None


class ArtifactUrlResponse(BaseModel):
    """A small JSON descriptor pointing at a servable artifact."""

    scene_id: str
    url: str
    format: str | None = None
