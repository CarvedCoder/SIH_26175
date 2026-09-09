from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TerrainDimensions(BaseModel):
    """Dimensions of the terrain heightmap."""

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class TerrainBounds(BaseModel):
    """Spatial bounds of the terrain."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float


class CoordinateSystem(BaseModel):
    """Coordinate reference system used by the terrain."""

    crs: str | None = None
    units: str = "meters"


class TerrainAsset(BaseModel):
    """A terrain-related asset exposed to the frontend."""

    name: str
    url: str
    format: str | None = None
    width: int | None = None
    height: int | None = None


class TerrainLayer(BaseModel):
    """A visual layer that can be enabled or disabled."""

    name: str
    type: str
    visible: bool = True
    url: str | None = None


class TerrainCapabilities(BaseModel):
    """Operations supported by the generated terrain."""

    absolute_elevation: bool = False
    relative_elevation: bool = True
    reference_comparison: bool = False
    slope: bool = False
    height_measurement: bool = False
    error_map: bool = False
    local_refinement: bool = False


class TerrainReference(BaseModel):
    """Reference DEM information, when available."""

    available: bool = False

    name: str | None = None
    url: str | None = None

    crs: str | None = None
    units: str | None = None

    width: int | None = None
    height: int | None = None


class TerrainScene(BaseModel):
    """
    Normalized terrain representation consumed by the frontend.

    The frontend renderer should not need to know how DepthWizard
    generated the terrain.
    """

    scene_id: str

    dimensions: TerrainDimensions

    bounds: TerrainBounds

    coordinate_system: CoordinateSystem = Field(
        default_factory=CoordinateSystem
    )

    elevation_mode: str = "relative"

    units: str = "meters"

    heightmap: TerrainAsset | None = None

    texture: TerrainAsset | None = None

    minimap: TerrainAsset | None = None

    reference: TerrainReference = Field(
        default_factory=TerrainReference
    )

    layers: list[TerrainLayer] = Field(
        default_factory=list
    )

    capabilities: TerrainCapabilities = Field(
        default_factory=TerrainCapabilities
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class TerrainResponse(BaseModel):
    """API response returned for a scene's terrain."""

    scene_id: str

    available: bool = False

    terrain: TerrainScene | None = None