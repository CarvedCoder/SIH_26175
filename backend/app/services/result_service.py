from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from backend.app.core.paths import get_scene_output_dir


class ResultService:
    """Reads and describes outputs produced by the DepthWizard pipeline.

    STATELESS: every method is a pure function of the files on disk — no
    in-memory caches, no cross-request state. Array statistics are computed
    over a memory-mapped view so repeated polling never materialises the
    full raster, without needing a memo dict whose staleness would have to
    be reasoned about (and which would be per-process anyway).
    """

    def get_output_dir(self, scene_id: str) -> Path:
        """Return the output directory for a scene."""
        return get_scene_output_dir(scene_id)

    def scene_has_results(self, scene_id: str) -> bool:
        """Return True when the scene has generated results."""
        return bool(self.get_result_files(scene_id))

    def get_result_files(self, scene_id: str) -> dict[str, Path]:
        """Return known DepthWizard result files that exist.

        Read-through cache: artifacts registered in the SQL scene row are
        re-downloaded from the object store when missing locally (e.g.
        after a backend restart on a fresh machine)."""
        output_dir = self.get_output_dir(scene_id)

        if not output_dir.exists():
            self._materialize_from_object_store(scene_id)
            if not output_dir.exists():
                return {}

        known_files = {
            "depth": output_dir / "dsm.npy",
            "dsm": output_dir / "dsm.tif",
            "dsm_anchored": output_dir / "dsm_anchored.tif",
            "preview": output_dir / "dsm_preview.png",
        }

        # The pipeline writes the DSM surface as dsm.npy; the GeoTIFF twin
        # exists only for georeferenced exports. Accept either so DSM
        # availability matches what the pipeline actually produces — a
        # non-georeferenced scene (dsm.npy only) still has a usable DSM.
        if not known_files["dsm"].exists() and known_files["depth"].exists():
            known_files["dsm"] = known_files["depth"]

        return {
            name: path
            for name, path in known_files.items()
            if path.exists() and path.is_file()
        }

    def _materialize_from_object_store(self, scene_id: str) -> None:
        """Read-through cache miss: pull DB-registered artifacts from the
        object store back into the local output dir (no-op with the local
        backend or when the scene has no registered artifacts)."""
        try:
            from backend.app.db.database import session_scope
            from backend.app.db.models import SceneRow
            from backend.app.storage.service import storage_service

            with session_scope() as session:
                row = session.get(SceneRow, scene_id)
                if row is None or not row.artifacts:
                    return
                owner_id, artifacts = row.owner_id, dict(row.artifacts)
            storage_service.ensure_scene_outputs_local(
                owner_id, scene_id, artifacts
            )
        except Exception:
            # A cache-miss download failure must not break result reads —
            # the local files (if any) remain the answer.
            from backend.app.core.logging import logger

            logger.exception("artifact materialization failed for %s", scene_id)

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
        """Read spatial metadata from the generated DSM (GeoTIFF or .npy)."""
        if path.suffix == ".npy":
            # Non-georeferenced DSM surface: no CRS/bounds, shape only.
            array = np.load(path, mmap_mode="r")
            return {
                "width": int(array.shape[1]),
                "height": int(array.shape[0]),
                "crs": None,
                "bounds": None,
            }
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
            # Never expose server filesystem paths in API payloads.
            "files": {name: path.name for name, path in files.items()},
        }

        depth_path = files.get("depth")
        if depth_path is not None:
            summary["depth"] = self._load_array_stats(depth_path)

        dsm_path = files.get("dsm")
        if dsm_path is not None:
            summary["dsm"] = self._read_dsm_metadata(dsm_path)

        return summary


result_service = ResultService()