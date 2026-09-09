from __future__ import annotations

from pathlib import Path

import numpy as np
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



    # ------------------------------------------------------------------
    # Point products: elevation probes, measurements, tiles, minimap
    # ------------------------------------------------------------------

    def _load_dsm_array(self, scene_id: str) -> np.ndarray:
        """Load the predicted DSM array (float32 [H,W], metres)."""
        from backend.app.services.result_service import result_service

        files = result_service.get_result_files(scene_id)
        depth_path = files.get("depth")
        if depth_path is None:
            raise FileNotFoundError(
                f"No depth results available for scene '{scene_id}'."
            )
        return np.load(depth_path).astype(np.float32, copy=False)

    def _pixel_size(self, scene_id: str) -> tuple[float, float] | None:
        """(gsd_x, gsd_y) metres per pixel — None when not georeferenced.

        Honesty contract (mirrors depthwizard.geo.pixel_size_metres): a
        scene without a real CRS has NO metric scale and this service
        refuses to invent one.
        """
        dsm_path = self.get_dsm_path(scene_id)
        if not dsm_path.is_file():
            return None
        with rasterio.open(dsm_path) as ds:
            if ds.crs is None:
                return None
            return abs(float(ds.transform.a)), abs(float(ds.transform.e))

    def sample_elevation(self, scene_id: str, x: int, y: int) -> float:
        dsm = self._load_dsm_array(scene_id)
        height, width = dsm.shape
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"point ({x}, {y}) is outside the {width}x{height} scene grid."
            )
        return float(dsm[y, x])

    def measure_height(
        self, scene_id: str, ground: tuple[int, int], top: tuple[int, int]
    ) -> dict:
        dsm = self._load_dsm_array(scene_id)
        height, width = dsm.shape
        for x, y in (ground, top):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"point ({x}, {y}) is outside the {width}x{height} scene grid."
                )
        ground_elev = float(dsm[ground[1], ground[0]])
        top_elev = float(dsm[top[1], top[0]])
        return {
            "ground_elevation": ground_elev,
            "top_elevation": top_elev,
            "height": top_elev - ground_elev,
            "units": "meters",
        }

    def measure_slope(
        self, scene_id: str, point_a: tuple[int, int], point_b: tuple[int, int]
    ) -> dict:
        dsm = self._load_dsm_array(scene_id)
        height, width = dsm.shape
        for x, y in (point_a, point_b):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"point ({x}, {y}) is outside the {width}x{height} scene grid."
                )

        a_elev = float(dsm[point_a[1], point_a[0]])
        b_elev = float(dsm[point_b[1], point_b[0]])
        elevation_change = b_elev - a_elev

        gsd = self._pixel_size(scene_id)
        if gsd is None:
            # No CRS -> no metric horizontal scale; a slope in degrees would
            # be fabricated. Return the honest refusal shape.
            return {
                "slope_degrees": None,
                "slope_percent": None,
                "horizontal_distance_m": None,
                "elevation_change_m": elevation_change,
                "units": "meters",
                "gsd_available": False,
            }

        dx_px = point_b[0] - point_a[0]
        dy_px = point_b[1] - point_a[1]
        horizontal_m = float(np.hypot(dx_px * gsd[0], dy_px * gsd[1]))
        if horizontal_m <= 0:
            raise ValueError("points must differ horizontally to measure slope.")

        slope = float(np.degrees(np.arctan2(abs(elevation_change), horizontal_m)))
        return {
            "slope_degrees": slope,
            "slope_percent": float(np.tan(np.radians(slope)) * 100.0),
            "horizontal_distance_m": horizontal_m,
            "elevation_change_m": elevation_change,
            "units": "meters",
            "gsd_available": True,
        }

    def terrain_tiles(
        self, scene_id: str, tile_size: int = 256
    ) -> dict:
        """Tile grid over the DSM for the frontend's progressive loader."""
        dsm = self._load_dsm_array(scene_id)
        height, width = dsm.shape
        grid_x = -(-width // tile_size)
        grid_y = -(-height // tile_size)
        tiles = []
        for ty in range(grid_y):
            for tx in range(grid_x):
                t_width = min(tile_size, width - tx * tile_size)
                t_height = min(tile_size, height - ty * tile_size)
                tiles.append(
                    {
                        "x": tx,
                        "y": ty,
                        "width": t_width,
                        "height": t_height,
                        "url": f"/api/v1/scenes/{scene_id}/results/preview",
                    }
                )
        return {
            "tile_size": tile_size,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "tiles": tiles,
        }

    def get_minimap_path(self, scene_id: str) -> Path:
        """Return a small PNG minimap, generating it from the DSM if absent.

        Idempotent: an existing up-to-date minimap.png is reused. The
        minimap is a rendering of the predicted DSM (grayscale, height =
        brightness), NOT new model output.
        """
        from PIL import Image

        dsm = self._load_dsm_array(scene_id)
        out_path = self.get_output_dir(scene_id) / "minimap.png"

        if out_path.is_file():
            return out_path

        valid = dsm[np.isfinite(dsm)]
        if valid.size == 0:
            raise ValueError(f"DSM for scene '{scene_id}' has no finite values.")
        lo, hi = float(valid.min()), float(valid.max())
        if hi - lo < 1e-6:
            normalized = np.zeros_like(dsm, dtype=np.uint8) + 128
        else:
            normalized = (
                np.clip((dsm - lo) / (hi - lo), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            normalized[np.isnan(dsm)] = 0

        # Cap the minimap size for cheap transport.
        height, width = normalized.shape
        stride = max(1, int(np.ceil(max(height, width) / 512)))
        if stride > 1:
            normalized = normalized[::stride, ::stride]

        from PIL import Image as PILImage

        out_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(normalized, mode="L").save(out_path, format="PNG")
        return out_path


terrain_service = TerrainService()
