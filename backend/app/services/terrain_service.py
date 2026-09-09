from __future__ import annotations

from pathlib import Path

import rasterio

from backend.app.core.paths import get_scene_output_dir
from backend.app.schemas.terrain import (
    CoordinateSystem,
    TerrainAsset,
    TerrainBounds,
    TerrainCapabilities,
    TerrainDimensions,
    TerrainLayer,
    TerrainReference,
    TerrainScene,
)

class TerrainService:
    """Builds a normalized terrain representation from DepthWizard outputs."""

    def get_output_dir(self, scene_id: str) -> Path:
        """Return the output directory for a scene."""
        return get_scene_output_dir(scene_id)

    def get_dsm_path(self, scene_id: str) -> Path:
        """Return the generated DSM GeoTIFF path."""
        return self.get_output_dir(scene_id) / "dsm.tif"

    def terrain_available(self, scene_id: str) -> bool:
        """Return whether a generated DSM is available."""
        return self.get_dsm_path(scene_id).is_file()

    def _read_dsm_metadata(self, path: Path) -> dict:
        """Read the spatial metadata required by the terrain API."""

        with rasterio.open(path) as dataset:
            return {
                "width": dataset.width,
                "height": dataset.height,
                "crs": (
                    dataset.crs.to_string()
                    if dataset.crs is not None
                    else None
                ),
                "bounds": {
                    "min_x": float(dataset.bounds.left),
                    "min_y": float(dataset.bounds.bottom),
                    "max_x": float(dataset.bounds.right),
                    "max_y": float(dataset.bounds.top),
                },
                "pixel_width": abs(float(dataset.transform.a)),
                "pixel_height": abs(float(dataset.transform.e)),
                "dtype": dataset.dtypes[0],
                "count": dataset.count,
            }

    def get_terrain(self, scene_id: str) -> TerrainScene:
        """Build the normalized terrain representation for a scene."""

        dsm_path = self.get_dsm_path(scene_id)

        if not dsm_path.is_file():
            raise FileNotFoundError(
                f"DSM result not found for scene '{scene_id}'."
            )

        metadata = self._read_dsm_metadata(dsm_path)

        heightmap = TerrainAsset(
            name="dsm",
            url=f"/api/v1/scenes/{scene_id}/results/dsm",
            format="tif",
            width=metadata["width"],
            height=metadata["height"],
        )

        texture = TerrainAsset(
            name="preview",
            url=f"/api/v1/scenes/{scene_id}/results/preview",
            format="png",
            width=metadata["width"],
            height=metadata["height"],
        )

        layers = [
            TerrainLayer(
                name="Elevation",
                type="elevation",
                visible=True,
                url=heightmap.url,
            ),
            TerrainLayer(
                name="Texture",
                type="texture",
                visible=True,
                url=texture.url,
            ),
        ]

        georeferenced = metadata["crs"] is not None

        capabilities = TerrainCapabilities(
            absolute_elevation=georeferenced,
            relative_elevation=True,
            reference_comparison=False,
            slope=False,
            height_measurement=False,
            error_map=False,
            local_refinement=False,
        )

        return TerrainScene(
            scene_id=scene_id,
            dimensions=TerrainDimensions(
                width=metadata["width"],
                height=metadata["height"],
            ),
            bounds=TerrainBounds(
                **metadata["bounds"],
            ),
            coordinate_system=CoordinateSystem(
                crs=metadata["crs"],
                units="meters",
            ),
            elevation_mode=(
                "absolute"
                if georeferenced
                else "relative"
            ),
            units="meters",
            heightmap=heightmap,
            texture=texture,
            minimap=None,
            reference=TerrainReference(
                available=False,
            ),
            layers=layers,
            capabilities=capabilities,
            metadata={
                "dtype": metadata["dtype"],
                "bands": metadata["count"],
                "pixel_width": metadata["pixel_width"],
                "pixel_height": metadata["pixel_height"],
            },
        )


terrain_service = TerrainService()
