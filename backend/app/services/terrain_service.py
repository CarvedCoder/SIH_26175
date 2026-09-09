from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from backend.app.core.logging import logger
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
        """Return whether the georeferenced DSM raster is available."""
        return self.get_dsm_path(scene_id).is_file()

    def depth_available(self, scene_id: str) -> bool:
        """Return whether the numerical DSM array is available.

        Non-georeferenced scenes produce dsm.npy WITHOUT a dsm.tif, so
        point products (elevation/measurements) must gate on this, not on
        the raster."""
        from backend.app.services.result_service import result_service

        return result_service.get_result_files(scene_id).get("depth") is not None

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
        """Build the normalized terrain representation for a scene.

        Works for BOTH georeferenced and non-georeferenced scenes: the
        geometry comes from the predicted depth array (dsm.npy); the CRS/
        bounds come from the DSM GeoTIFF when one exists, and fall back to
        honest pixel-space bounds otherwise (JPG/PNG uploads have no CRS).
        """
        from backend.app.services.result_service import result_service

        files = result_service.get_result_files(scene_id)
        depth_path = files.get("depth")
        if depth_path is None:
            raise FileNotFoundError(
                f"DSM result not found for scene '{scene_id}'."
            )

        dsm = np.load(depth_path).astype(np.float32, copy=False)
        height, width = dsm.shape

        dsm_raster = files.get("dsm")
        if dsm_raster is not None:
            metadata = self._read_dsm_metadata(dsm_raster)
        else:
            metadata = {
                "width": width,
                "height": height,
                "crs": None,
                "bounds": {
                    "min_x": 0.0,
                    "min_y": 0.0,
                    "max_x": float(width),
                    "max_y": float(height),
                },
                "pixel_width": None,
                "pixel_height": None,
                "dtype": "float32",
                "count": 1,
            }

        georeferenced = metadata["crs"] is not None

        # Renderer artifacts (idempotent, browser-friendly):
        heightmap_asset = self._heightmap_asset(scene_id, dsm)
        texture_asset = self._texture_asset(scene_id)

        layers = [
            TerrainLayer(
                name="Elevation",
                type="elevation",
                visible=True,
                url=heightmap_asset.url,
            ),
            TerrainLayer(
                name="Texture",
                type="texture",
                visible=True,
                url=texture_asset.url,
            ),
        ]

        gsd = self._pixel_size(scene_id)

        capabilities = TerrainCapabilities(
            absolute_elevation=georeferenced,
            relative_elevation=True,
            reference_comparison=False,
            slope=gsd is not None,
            height_measurement=True,
            error_map=False,
            local_refinement=True,
        )

        return TerrainScene(
            scene_id=scene_id,
            dimensions=TerrainDimensions(
                width=width,
                height=height,
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
            heightmap=heightmap_asset,
            texture=texture_asset,
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
    # Browser artifacts: heightmap + RGB texture (idempotent generation)
    # ------------------------------------------------------------------

    def get_heightmap_path(self, scene_id: str) -> Path:
        """8-bit grayscale PNG of the predicted heights, normalized to
        [0, 255] (the 3D renderer reads the RED channel as elevation).

        Idempotent: regenerated only when missing. Capped at 1024 px on
        the long side (the mesh geometry never needs more).
        """
        from PIL import Image

        from backend.app.services.result_service import result_service

        files = result_service.get_result_files(scene_id)
        depth_path = files.get("depth")
        if depth_path is None:
            raise FileNotFoundError(
                f"No depth results available for scene '{scene_id}'."
            )

        out_path = self.get_output_dir(scene_id) / "heightmap.png"
        if out_path.is_file():
            return out_path

        dsm = np.load(depth_path).astype(np.float32, copy=False)
        valid = dsm[np.isfinite(dsm)]
        if valid.size == 0:
            raise ValueError(f"DSM for scene '{scene_id}' has no finite values.")
        lo, hi = float(valid.min()), float(valid.max())
        if hi - lo < 1e-6:
            normalized = np.full_like(dsm, 128, dtype=np.uint8)
        else:
            normalized = (
                np.clip((dsm - lo) / (hi - lo), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            normalized[np.isnan(dsm)] = 0

        stride = max(1, int(np.ceil(max(normalized.shape) / 1024)))
        if stride > 1:
            normalized = normalized[::stride, ::stride]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normalized, mode="L").save(out_path, format="PNG")
        logger.info(
            "terrain artifact generated: %s (lo=%.3f hi=%.3f)",
            out_path.name, lo, hi,
        )
        return out_path

    def get_rgb_preview_path(self, scene_id: str) -> Path:
        """PNG preview of the SOURCE imagery (the 'rgb' texture layer).

        Generated from the scene's stored input raster — works for JPG,
        PNG and GeoTIFF inputs alike. Idempotent.
        """
        from PIL import Image

        import rasterio

        from backend.app.services.processing_service import processing_service

        out_path = self.get_output_dir(scene_id) / "rgb_preview.png"
        if out_path.is_file():
            return out_path

        input_path = processing_service.resolve_scene_input(scene_id)
        with rasterio.open(input_path) as ds:
            bands = [1, 2, 3] if ds.count >= 3 else [1]
            raw = ds.read(bands)
            if ds.count < 3:
                raw = np.stack([raw[0]] * 3)
            rgb = np.ascontiguousarray(
                raw.transpose(1, 2, 0), dtype=np.uint8
            )[:, :, :3]

        height, width = rgb.shape[:2]
        stride = max(1, int(np.ceil(max(height, width) / 2048)))
        if stride > 1:
            rgb = rgb[::stride, ::stride]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")
        logger.info(
            "terrain artifact generated: %s (%dx%d)", out_path.name, width, height
        )
        return out_path

    def _heightmap_asset(self, scene_id: str, dsm: np.ndarray) -> TerrainAsset:
        try:
            path = self.get_heightmap_path(scene_id)
        except (FileNotFoundError, ValueError):
            return None
        return TerrainAsset(
            name="heightmap",
            url=f"/api/v1/scenes/{scene_id}/results/heightmap",
            format="png",
            width=path and dsm.shape[1],
            height=dsm.shape[0],
        )

    def _texture_asset(self, scene_id: str) -> TerrainAsset:
        try:
            self.get_rgb_preview_path(scene_id)
        except (FileNotFoundError, ValueError, RuntimeError):
            return TerrainAsset(
                name="preview",
                url=f"/api/v1/scenes/{scene_id}/results/preview",
                format="png",
            )
        return TerrainAsset(
            name="rgb",
            url=f"/api/v1/scenes/{scene_id}/results/rgb",
            format="png",
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

    def renderer_fields(self, scene_id: str) -> dict:
        """Top-level fields of the 3D renderer contract (TerrainCanvas):
        heightmap_url, texture_url, height_scale, min/max_elevation.

        ``height_scale`` is the max AGL in metres so vertical displacement
        is at true metric proportion; the frontend's exaggeration control
        scales from there.
        """
        from backend.app.services.result_service import result_service

        scene = self.get_terrain(scene_id)
        files = result_service.get_result_files(scene_id)
        depth_path = files["depth"]
        stats = result_service._load_array_stats(depth_path)

        return {
            "heightmap_url": (
                scene.heightmap.url if scene.heightmap is not None else None
            ),
            "texture_url": scene.texture.url if scene.texture is not None else None,
            "height_scale": stats["maximum"],
            "min_elevation": stats["minimum"],
            "max_elevation": stats["maximum"],
        }



terrain_service = TerrainService()
