from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from backend.app.infrastructure.storage.scene_artifacts import (
    RESULT_ARTIFACT_KEYS,
    scene_artifact_store,
    scene_output_dir_key,
)


class ResultService:
    """Reads and describes outputs produced by the DepthWizard pipeline.

    STATELESS: every method is a pure function of the files on disk — no
    in-memory caches, no cross-request state. Array statistics are computed
    over a memory-mapped view so repeated polling never materialises the
    full raster, without needing a memo dict whose staleness would have to
    be reasoned about (and which would be per-process anyway).
    """

    def get_output_dir(self, scene_id: str) -> Path:
        """Output directory for a scene, addressed via the artifact store."""
        return scene_artifact_store().path_for(scene_output_dir_key(scene_id))

    def scene_has_results(self, scene_id: str) -> bool:
        """Return True when the scene has generated results."""
        return bool(self.get_result_files(scene_id))

    def get_result_files(self, scene_id: str) -> dict[str, Path]:
        """Known DepthWizard result artifacts that exist, addressed via
        the artifact store (keys in scene_artifacts.RESULT_ARTIFACT_KEYS)."""
        store = scene_artifact_store()
        files: dict[str, Path] = {}
        for name, suffix in RESULT_ARTIFACT_KEYS.items():
            try:
                path = store.path_for(scene_output_dir_key(scene_id) + "/" + suffix)
            except Exception:  # noqa: BLE001 — an invalid scene id has no artifacts
                return {}
            if path.exists() and path.is_file():
                files[name] = path
        return files

    def _load_array_stats(self, path: Path) -> dict[str, Any]:
        """Calculate statistics from a generated NumPy raster.

        Memory-mapped read: min/max/median walk the array on disk without
        loading it whole, so the call is cheap enough to stay cache-free
        and therefore free of any staleness semantics.
        """
        array = np.load(path, mmap_mode="r")

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