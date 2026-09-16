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
    """API response returned for a scene's terrain.

    Top-level convenience fields are the 3D renderer's contract
    (TerrainCanvas destructures heightmap_url / texture_url /
    height_scale / min_elevation / max_elevation); the nested ``terrain``
    object keeps the full typed detail."""

    scene_id: str

    available: bool = False

    terrain: TerrainScene | None = None

    # -- renderer contract (top level) -----------------------------------
    heightmap_url: str | None = None
    texture_url: str | None = None

    height_scale: float | None = Field(
        default=None,
        description="Metres of vertical displacement for a fully-saturated "
        "heightmap pixel (the max AGL of the scene).",
    )
    min_elevation: float | None = None
    max_elevation: float | None = None

    # -- physical world footprint (real-world terrain scale) ----------------
    world_width_m: float | None = Field(
        default=None,
        description="Real-world width (east-west) of the terrain footprint "
        "in metres = raster width × GSD. For non-georeferenced scenes an "
        "assumed 1 m/pixel fallback is used and "
        "is_georeferenced_scale is False.",
    )
    world_depth_m: float | None = Field(
        default=None,
        description="Real-world depth (north-south) of the terrain footprint "
        "in metres = raster height × GSD (same fallback rule as "
        "world_width_m).",
    )
    is_georeferenced_scale: bool | None = Field(
        default=None,
        description="True when world_width_m/world_depth_m come from a real "
        "CRS+transform GSD; False means the 1 m/pixel fallback was applied "
        "(never invented GPS coordinates).",
    )