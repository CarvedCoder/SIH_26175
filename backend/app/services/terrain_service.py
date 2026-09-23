from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from backend.app.core.logging import logger
from backend.app.infrastructure.storage.scene_artifacts import (
    scene_artifact_store,
    scene_output_dir_key,
)
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

def get_validation_rmse(scene_id: str):
    """Validation RMSE for the scene, or None when unavailable."""
    from backend.app.services.validation_service import get_validation
    try:
        return get_validation(scene_id).metrics.rmse
    except Exception:
        return None


class TerrainService:
    """Builds a normalized terrain representation from DepthWizard outputs."""

    def get_output_dir(self, scene_id: str) -> Path:
        """Output directory for a scene, addressed via the artifact store."""
        return scene_artifact_store().path_for(scene_output_dir_key(scene_id))

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
        if dsm_raster is not None and dsm_raster.suffix == ".tif":
            metadata = self._read_dsm_metadata(dsm_raster)
        else:
            # dsm.npy fallback (or no raster): honest pixel-space metadata.
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

    @staticmethod
    def _fill_invalid(arr: np.ndarray) -> np.ndarray:
        """Replace NaN/inf samples with the mean of their finite 8-neighbours,
        iterating until filled (crater-free hole filling without scipy).

        Depth-model DSMs occasionally contain small invalid patches; writing
        them to the scene minimum punched visible pits into the terrain.
        """
        mask = ~np.isfinite(arr)
        if not mask.any():
            return arr
        arr = arr.astype(np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        remaining = mask
        # Each pass fills every hole adjacent to a finite pixel; hole depth
        # shrinks from both sides, so the cap is never hit in practice.
        for _ in range(64):
            if not remaining.any():
                break
            filled_sum = np.zeros_like(arr)
            filled_count = np.zeros_like(arr, dtype=np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    shifted = np.full_like(arr, np.nan)
                    ys = slice(max(0, -dy), arr.shape[0] - max(0, dy))
                    xs = slice(max(0, -dx), arr.shape[1] - max(0, dx))
                    ys_src = slice(max(0, dy), arr.shape[0] - max(0, -dy))
                    xs_src = slice(max(0, dx), arr.shape[1] - max(0, -dx))
                    shifted[ys, xs] = arr[ys_src, xs_src]
                    finite = np.isfinite(shifted)
                    filled_sum[finite] += np.nan_to_num(shifted[finite])
                    filled_count += finite.astype(np.int32)
            fillable = remaining & (filled_count > 0)
            arr[fillable] = filled_sum[fillable] / filled_count[fillable]
            remaining = ~np.isfinite(arr)
        # Any survivor (an all-invalid raster edge region) gets the global mean
        if remaining.any():
            valid = arr[np.isfinite(arr)]
            arr[remaining] = valid.mean() if valid.size else 0.0
        return arr

    def get_heightmap_path(self, scene_id: str) -> Path:
        """16-bit grayscale PNG of the predicted heights, normalised to
        [0, 65535]. The renderer decodes both bytes — the previous 8-bit
        encoding left only 256 elevation levels, which terraced visibly
        once vertical exaggeration was applied.

        Quality pipeline (each step matters for sharp building edges):
          1. NaN/inf holes are neighbour-filled BEFORE normalisation.
          2. Downsample to the 1024-px cap with a LANCZOS filter — a real
             low-pass before decimation — instead of dropping every
             stride-th pixel, which aliased single-pixel noise into
             spikes. LANCZOS also keeps step edges (building walls) far
             sharper than a box average.

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
        dsm = self._fill_invalid(dsm)
        if not np.isfinite(dsm).all() and dsm.size:
            valid = dsm[np.isfinite(dsm)]
            if valid.size == 0:
                raise ValueError(
                    f"DSM for scene '{scene_id}' has no finite values."
                )
        lo, hi = float(dsm.min()), float(dsm.max())
        if hi - lo < 1e-6:
            normalized = np.full_like(dsm, 0.5)
        else:
            normalized = np.clip((dsm - lo) / (hi - lo), 0.0, 1.0)

        # Filter-then-decimate via LANCZOS (float mode keeps precision).
        max_side = max(normalized.shape)
        if max_side > 1024:
            scale = 1024.0 / max_side
            new_size = (
                max(1, round(normalized.shape[1] * scale)),
                max(1, round(normalized.shape[0] * scale)),
            )
            img = Image.fromarray(normalized.astype(np.float32), mode="F")
            img = img.resize(new_size, Image.LANCZOS)
            normalized = np.asarray(img, dtype=np.float32)

        encoded = (np.clip(normalized, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encoded, mode="I;16").save(out_path, format="PNG")
        logger.info(
            "terrain artifact generated: %s (16-bit, lo=%.3f hi=%.3f)",
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

    # ------------------------------------------------------------------
    # Browser texture layers for the terrain viewer's Layers/Compare menus.
    # Each writes a single-channel greyscale PNG whose R channel encodes the
    # normalised value — the viewer's fragment shader applies the viridis /
    # diverging colormap itself, so the PNG must be colour-free data, not a
    # matplotlib preview.
    # ------------------------------------------------------------------

    _LAYER_PNG_CAP = 2048

    @staticmethod
    def _normalise_grey(arr: np.ndarray) -> np.ndarray:
        """Fill invalid cells and squeeze the value range into [0, 1]."""
        arr = TerrainService._fill_invalid(arr.astype(np.float32, copy=False))
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.full_like(arr, 0.5)
        lo, hi = np.percentile(valid, 2.0), np.percentile(valid, 98.0)
        if hi - lo < 1e-9:
            return np.full_like(arr, 0.5)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    @classmethod
    def _save_grey_png(cls, normalized: np.ndarray, out_path: Path) -> Path:
        from PIL import Image

        height, width = normalized.shape
        stride = max(1, int(np.ceil(max(height, width) / cls._LAYER_PNG_CAP)))
        if stride > 1:
            normalized = normalized[::stride, ::stride]
        encoded = (np.clip(normalized, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encoded, mode="L").save(out_path, format="PNG")
        logger.info(
            "terrain artifact generated: %s (%dx%d)",
            out_path.name, normalized.shape[1], normalized.shape[0],
        )
        return out_path

    def get_dsm_layer_path(self, scene_id: str) -> Path:
        """Greyscale DSM texture for the viewer's 'Estimated DSM' layer."""
        from backend.app.services.result_service import result_service

        dsm_path = result_service.get_result_files(scene_id).get("dsm")
        if dsm_path is None:
            raise FileNotFoundError(
                f"No DSM results available for scene '{scene_id}'."
            )
        out_path = self.get_output_dir(scene_id) / "dsm_layer.png"
        if out_path.is_file():
            return out_path
        if dsm_path.suffix == ".npy":
            dsm = np.load(dsm_path, mmap_mode="r")
        else:
            with rasterio.open(dsm_path) as ds:
                dsm = ds.read(1)
        return self._save_grey_png(self._normalise_grey(np.asarray(dsm)), out_path)

    def get_slope_layer_path(self, scene_id: str) -> Path:
        """Greyscale slope texture (per-pixel gradient, in degrees)."""
        from backend.app.services.result_service import result_service

        dsm_path = result_service.get_result_files(scene_id).get("depth")
        if dsm_path is None:
            raise FileNotFoundError(
                f"No depth results available for scene '{scene_id}'."
            )
        out_path = self.get_output_dir(scene_id) / "slope_layer.png"
        if out_path.is_file():
            return out_path
        dsm = self._fill_invalid(
            np.load(dsm_path, mmap_mode="r").astype(np.float32, copy=False)
        )
        # Unit pixel spacing — the layer is a relative-gradient visual, and
        # the shader renders it through the same viridis ramp as the DSM.
        gy, gx = np.gradient(dsm)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        return self._save_grey_png(self._normalise_grey(slope_deg), out_path)

    def get_reference_layer_path(self, scene_id: str) -> Path:
        """Greyscale reference-DEM texture for the 'Reference DEM' layer."""
        output_dir = self.get_output_dir(scene_id)
        out_path = output_dir / "reference_layer.png"
        if out_path.is_file():
            return out_path

        raster_ref = next(
            (
                output_dir / name
                for name in ("reference.tif", "reference_dem.tif", "ref_dem.tif")
                if (output_dir / name).is_file()
            ),
            None,
        )
        array_ref = next(
            (
                output_dir / name
                for name in ("reference.npy", "reference_dem.npy", "ref_dem.npy")
                if (output_dir / name).is_file()
            ),
            None,
        )

        if raster_ref is not None:
            with rasterio.open(raster_ref) as ds:
                ref = ds.read(1).astype(np.float32, copy=False)
        elif array_ref is not None:
            ref = np.load(array_ref, mmap_mode="r").astype(np.float32, copy=False)
        else:
            raise FileNotFoundError(
                f"No reference elevation available for scene '{scene_id}'."
            )
        return self._save_grey_png(self._normalise_grey(np.asarray(ref)), out_path)

    def _artifact_url(self, scene_id: str, filename: str, legacy_path: str) -> str:
        """Presigned URL when the object store is active, else the legacy
        API-relative path."""
        from backend.app.storage.service import storage_service

        url = storage_service.presign_artifact(
            scene_id, self.get_output_dir(scene_id) / filename
        )
        return url or legacy_path

    def _heightmap_asset(self, scene_id: str, dsm: np.ndarray) -> TerrainAsset:
        try:
            path = self.get_heightmap_path(scene_id)
        except (FileNotFoundError, ValueError):
            return None
        return TerrainAsset(
            name="heightmap",
            url=self._artifact_url(
                scene_id, "heightmap.png",
                f"/api/v1/scenes/{scene_id}/results/heightmap",
            ),
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
                url=self._artifact_url(
                    scene_id, "dsm_preview.png",
                    f"/api/v1/scenes/{scene_id}/results/preview",
                ),
                format="png",
            )
        return TerrainAsset(
            name="rgb",
            url=self._artifact_url(
                scene_id, "rgb_preview.png",
                f"/api/v1/scenes/{scene_id}/results/rgb",
            ),
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

        Routes through depthwizard.geo.pixel_size_metres (the single source
        of truth for metric pixel size) instead of reading raw transform
        coefficients: a geographic-CRS scene would otherwise report degrees
        mislabeled as metres. A scene without a real CRS has NO metric
        scale and this service refuses to invent one.
        """
        from depthwizard.geo import pixel_size_metres

        dsm_path = self.get_dsm_path(scene_id)
        if not dsm_path.is_file():
            return None
        with rasterio.open(dsm_path) as ds:
            return pixel_size_metres(ds.crs, ds.transform)

    def sample_elevation(self, scene_id: str, x: int, y: int) -> float:
        dsm = self._load_dsm_array(scene_id)
        height, width = dsm.shape
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"point ({x}, {y}) is outside the {width}x{height} scene grid."
            )
        return float(dsm[y, x])

    def sample_elevation_with_confidence(
        self, scene_id: str, x: int, y: int
    ) -> dict:
        """Exact metered DSM sample plus an honest accuracy assessment.

        The viewer's on-mesh readouts derive elevation from the 16-bit
        heightmap texture (LANCZOS-resampled, quantised to 65536 levels);
        this samples the float32 DSM array directly — the metered value —
        and reports:
          * precision_m — the quantisation bound of the visual readout
            (half a heightmap level), i.e. how far the on-screen estimate
            can sit from the metered value;
          * confidence — grounded in validation RMSE when a reference DEM
            exists; otherwise a qualitative level with the reason stated.
            A percentage is NEVER fabricated without validation evidence.
        """
        dsm = self._load_dsm_array(scene_id)
        height, width = dsm.shape
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"point ({x}, {y}) is outside the {width}x{height} scene grid."
            )

        raw = float(dsm[y, x])
        span = float(np.nanmax(dsm) - np.nanmin(dsm)) if dsm.size else 0.0
        precision_m = max(span / 65535.0 / 2.0, 1e-4)

        if not np.isfinite(raw):
            return {
                "elevation": None,
                "metered": False,
                "precision_m": precision_m,
                "confidence": {
                    "level": "none",
                    "percent": None,
                    "basis": "No valid DSM sample at this pixel",
                },
                "units": "meters",
            }

        rmse = get_validation_rmse(scene_id)

        if rmse is not None and span > 1e-6:
            percent = int(round(max(5.0, min(99.0, 100.0 * (1.0 - rmse / span)))))
            level = "high" if percent >= 85 else "medium" if percent >= 60 else "low"
            confidence = {
                "level": level,
                "percent": percent,
                "basis": f"Validated against reference DEM (RMSE {rmse:.2f} m)",
            }
        elif self._pixel_size(scene_id) is not None:
            confidence = {
                "level": "medium",
                "percent": None,
                "basis": "GCP-calibrated metric DSM — not validated against ground truth",
            }
        else:
            confidence = {
                "level": "low",
                "percent": None,
                "basis": "Relative DSM — scene-scale depths, not metric-calibrated",
            }

        # Local ground level + structure height: the ground around the
        # sampled pixel is the low percentile of the DSM in a window
        # (~GROUND_WINDOW px) — roofs and other elevated structures stand
        # out against it. This is a GEOMETRIC estimate from the predicted
        # surface, stated as such (no fabricated accuracy).
        gw = 64
        hy, hx = dsm.shape
        y0, y1 = max(0, y - gw // 2), min(hy, y + gw // 2)
        x0, x1 = max(0, x - gw // 2), min(hx, x + gw // 2)
        patch = dsm[y0:y1, x0:x1]
        ground = float(np.nanpercentile(patch, 10.0))
        height_ag = raw - ground
        is_structure = bool(np.isfinite(height_ag) and height_ag >= 2.0)

        return {
            "elevation": raw,
            "metered": True,
            "precision_m": precision_m,
            "confidence": confidence,
            "ground_elevation": ground if np.isfinite(ground) else None,
            "height_above_ground_m": float(height_ag) if np.isfinite(height_ag) else None,
            "is_structure": is_structure,
            "height_confidence": self._height_confidence(scene_id),
            "units": "meters",
        }

    def _height_confidence(self, scene_id: str) -> dict:
        """Honest accuracy statement for the geometric structure-height
        estimate. Without a validated reference the scale itself is the
        limiting factor, so the basis says exactly what is trusted."""
        try:
            rmse = get_validation_rmse(scene_id)
            if rmse is not None:
                return {
                    "level": "high",
                    "percent": None,
                    "basis": f"Geometric height on a DSM validated against a reference DEM (RMSE {rmse:.2f} m)",
                }
        except Exception:
            pass
        if self._pixel_size(scene_id) is not None:
            return {
                "level": "medium",
                "percent": None,
                "basis": "Geometric height on a CRS-calibrated metric DSM",
            }
        return {
            "level": "medium",
            "percent": None,
            "basis": "Geometric height at the documented 1 m/pixel fallback scale — no ground truth available",
        }

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
                        "url": self._artifact_url(
                            scene_id, "dsm_preview.png",
                            f"/api/v1/scenes/{scene_id}/results/preview",
                        ),
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
        heightmap_url, texture_url, height_scale, min/max_elevation, and
        the physical world footprint (world_width_m / world_depth_m) the
        mesh is sized against.

        ``height_scale`` is the max AGL in metres so vertical displacement
        is at true metric proportion; the frontend's exaggeration control
        scales from there. The world footprint is raster dims × GSD via
        pixel_size_metres — for non-georeferenced scenes (JPG/PNG, no CRS)
        there is no honest metric scale, so a DOCUMENTED fallback of one
        metre per pixel is used and ``is_georeferenced_scale`` reports
        False so the frontend/UI can label it as an assumed scale. No GPS
        coordinates are ever invented.
        """
        from backend.app.services.result_service import result_service

        scene = self.get_terrain(scene_id)
        files = result_service.get_result_files(scene_id)
        depth_path = files["depth"]
        stats = result_service._load_array_stats(depth_path)

        gsd = self._pixel_size(scene_id)
        if gsd is not None:
            world_width_m = scene.dimensions.width * gsd[0]
            world_depth_m = scene.dimensions.height * gsd[1]
            is_georeferenced_scale = True
        else:
            world_width_m = float(scene.dimensions.width)
            world_depth_m = float(scene.dimensions.height)
            is_georeferenced_scale = False

        return {
            "heightmap_url": (
                scene.heightmap.url if scene.heightmap is not None else None
            ),
            "texture_url": scene.texture.url if scene.texture is not None else None,
            "height_scale": stats["maximum"],
            "min_elevation": stats["minimum"],
            "max_elevation": stats["maximum"],
            "world_width_m": float(world_width_m),
            "world_depth_m": float(world_depth_m),
            "is_georeferenced_scale": is_georeferenced_scale,
        }



terrain_service = TerrainService()
