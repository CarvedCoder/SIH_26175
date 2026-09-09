from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from backend.app.core.paths import get_scene_output_dir


class ResultService:
    """Reads and describes outputs produced by the DepthWizard pipeline."""

    def __init__(self) -> None:
        # Stats cache keyed by (path, mtime): polling the results endpoint
        # must not rescan multi-hundred-MB arrays on every GET.
        self._stats_cache: dict[Path, tuple[float, dict[str, Any]]] = {}

    def get_output_dir(self, scene_id: str) -> Path:
        """Return the output directory for a scene."""
        return get_scene_output_dir(scene_id)

    def scene_has_results(self, scene_id: str) -> bool:
        """Return True when the scene has generated results."""
        return bool(self.get_result_files(scene_id))

    def get_result_files(self, scene_id: str) -> dict[str, Path]:
        """Return known DepthWizard result files that exist."""
        output_dir = self.get_output_dir(scene_id)

        if not output_dir.exists():
            return {}

        known_files = {
            "depth": output_dir / "dsm.npy",
            "dsm": output_dir / "dsm.tif",
            "dsm_anchored": output_dir / "dsm_anchored.tif",
            "preview": output_dir / "dsm_preview.png",
        }

        return {
            name: path
            for name, path in known_files.items()
            if path.exists() and path.is_file()
        }

    def _load_array_stats(self, path: Path) -> dict[str, Any]:
        """Calculate statistics from a generated NumPy raster (mtime-cached)."""
        stat = path.stat()
        cached = self._stats_cache.get(path)
        if cached is not None and cached[0] == stat.st_mtime:
            return cached[1]

        array = np.load(path)

        stats = {
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "relief": float(np.max(array) - np.min(array)),
            "units": "meters",
        }
        self._stats_cache[path] = (stat.st_mtime, stats)
        return stats

    def _read_dsm_metadata(self, path: Path) -> dict[str, Any]:
        """Read spatial metadata from the generated DSM GeoTIFF."""
        with rasterio.open(path) as dataset:
            return {
                "width": dataset.width,
                "height": dataset.height,
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "bounds": [
                    float(dataset.bounds.left),
                    float(dataset.bounds.bottom),
                    float(dataset.bounds.right),
                    float(dataset.bounds.top),
                ],
            }

    def get_result_summary(self, scene_id: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of generated scene results."""
        output_dir = self.get_output_dir(scene_id)
        files = self.get_result_files(scene_id)

        summary: dict[str, Any] = {
            "scene_id": scene_id,
            "available": bool(files),
            "output_dir": str(output_dir),
            "files": {
                name: str(path)
                for name, path in files.items()
            },
        }

        depth_path = files.get("depth")
        if depth_path is not None:
            summary["depth"] = self._load_array_stats(depth_path)

        dsm_path = files.get("dsm")
        if dsm_path is not None:
            summary["dsm"] = self._read_dsm_metadata(dsm_path)

        return summary


result_service = ResultService()