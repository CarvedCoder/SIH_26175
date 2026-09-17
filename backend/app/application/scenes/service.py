"""SceneService — the scene lifecycle use cases (brief §10/§11).

Owns the business rules the old routes/scenes.py mixed with HTTP:
    * content validation (rasterio parse = the real file-type gate);
    * staging -> validate -> atomic commit (no permanent state on failure);
    * the designated input.<ext> determinism contract;
    * mosaic merge orchestration (depthwizard.mosaic);
    * scene inspection metadata (persisted once at upload, never
      re-inspected per request).

Routes keep ONLY HTTP concerns: streaming an UploadFile to staging and
shaping Pydantic responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import rasterio

from backend.app.core.errors import (
    AppError,
    InvalidSceneId,
    RasterInvalid,
    SceneInputAmbiguous,
    SceneNotFound,
)
from backend.app.core.logging import logger
from backend.app.core.paths import (
    UPLOAD_STAGING_DIR,
    delete_scene_directories,
    ensure_directories,
    get_scene_raw_dir,
)
from backend.app.infrastructure.persistence.scene_repository import (
    FileSceneRepository,
    designated_input_name,
)

# Raster size floor: below this the scene cannot carry usable signal.
MIN_DIMENSION = 64


class SceneService:
    def __init__(self, repository: FileSceneRepository | None = None) -> None:
        self._repo = repository or FileSceneRepository()

    # -- existence / lookup -------------------------------------------------

    def require_scene(self, scene_id: str) -> None:
        """Raise the typed 404/400 errors unless the scene exists."""
        if not scene_id or not scene_id.startswith("scene_"):
            raise InvalidSceneId(scene_id)
        if not self._repo.exists(scene_id):
            raise SceneNotFound(scene_id)

    def get_record(self, scene_id: str) -> dict[str, Any]:
        self.require_scene(scene_id)
        stored = self._repo.load_record(scene_id)
        if stored is None:
            raise SceneNotFound(scene_id)
        return stored

    def find_scene_input(self, scene_id: str) -> Path | None:
        """The designated input raster, or None (deterministic order)."""
        return self._repo.find_designated_input(scene_id)

    def resolve_scene_input(self, scene_id: str) -> Path:
        """The ONE input raster of a scene — ambiguous scenes are
        REJECTED, never resolved by iteration order."""
        designated = self.find_scene_input(scene_id)
        if designated is not None:
            return designated
        input_dir = get_scene_raw_dir(scene_id)
        if not input_dir.exists():
            raise FileNotFoundError(f"Scene '{scene_id}' has no stored input.")
        rasters = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {"." + e.split(".")[-1] for e in
                                                     ("tif", "tiff", "png", "jpg", "jpeg")}
        )
        if not rasters:
            raise FileNotFoundError(f"No image input found for scene '{scene_id}'.")
        if len(rasters) > 1:
            raise SceneInputAmbiguous(scene_id)
        return rasters[0]

    # -- raster inspection ------------------------------------------------------

    def inspect_raster(self, path: Path) -> dict[str, Any]:
        """Inspect an uploaded raster WITHOUT running model inference.

        rasterio.open() is the actual content validation: a file that
        cannot be parsed as a raster is rejected before it becomes scene
        state. Raises the typed 400 (never leaks filesystem paths)."""
        try:
            with rasterio.open(path) as dataset:
                crs = dataset.crs
                bounds = dataset.bounds
                transform = dataset.transform
                georeferenced = crs is not None
                return {
                    "width": dataset.width,
                    "height": dataset.height,
                    "channels": dataset.count,
                    "format": dataset.driver,
                    "georeferenced": georeferenced,
                    "crs": crs.to_string() if crs else None,
                    "min_x": bounds.left,
                    "min_y": bounds.bottom,
                    "max_x": bounds.right,
                    "max_y": bounds.top,
                    "pixel_width": abs(transform.a),
                    "pixel_height": abs(transform.e),
                    "processing_path": "absolute" if georeferenced else "relative",
                }
        except AppError:
            raise
        except Exception:
            logger.exception("raster inspection failed for scene upload")
            raise RasterInvalid() from None

    # -- upload commit ------------------------------------------------------------

    def commit_upload(
        self,
        scene_id: str,
        staging_path: Path,
        extension: str,
        original_filename: str,
        *,
        on_failure: Callable[[], None],
    ) -> dict[str, Any]:
        """Validate the staged file, then atomically commit it as the
        scene's designated input + record. ``on_failure`` is the HTTP
        layer's cleanup hook (remove staging + partial scene dir)."""
        try:
            metadata = self.inspect_raster(staging_path)
            scene_dir = get_scene_raw_dir(scene_id)
            scene_dir.mkdir(parents=True, exist_ok=True)
            destination = scene_dir / designated_input_name(extension)
            staging_path.replace(destination)
            self._repo.write_record(scene_id, original_filename, metadata)
            return metadata
        except AppError:
            on_failure()
            raise
        except OSError:
            logger.exception("failed to persist upload for scene %s", scene_id)
            on_failure()
            raise AppError(
                status_code=500,
                code="STORAGE_FAILURE",
                message="Failed to store the uploaded file.",
            ) from None

    def commit_mosaic(
        self,
        scene_id: str,
        staged_paths: list[Path],
        staged_names: list[str],
    ) -> dict[str, Any]:
        """Merge validated staged GeoTIFFs into one contiguous scene input
        (depthwizard.mosaic) and commit. Inputs are never silently
        resized or reprojected — a mismatch is a typed 400."""
        from depthwizard.mosaic import MosaicError, mosaic_rasters, write_mosaic

        mosaic_staging = UPLOAD_STAGING_DIR / f"{scene_id}_mosaic_merged.tif"
        try:
            try:
                result = mosaic_rasters(staged_paths)
            except MosaicError as exc:
                raise AppError(
                    status_code=400,
                    code="MOSAIC_INVALID",
                    message=str(exc),
                ) from exc
            write_mosaic(result, mosaic_staging)
            metadata = self.inspect_raster(mosaic_staging)
            self._repo.replace_input(scene_id, mosaic_staging)
            self._repo.write_record(
                scene_id,
                f"mosaic({'+'.join(staged_names)})",
                metadata,
            )
            logger.info(
                "scene mosaic: %s inputs=%d merged=%dx%d",
                scene_id, len(staged_paths),
                metadata["width"], metadata["height"],
            )
            return metadata
        finally:
            mosaic_staging.unlink(missing_ok=True)

    def cleanup_failed_upload(self, staging_path: Path, scene_dir: Path | None) -> None:
        """Remove staging + any partially created scene dir (idempotent)."""
        staging_path.unlink(missing_ok=True)
        if scene_dir is not None and scene_dir.is_dir():
            for child in scene_dir.iterdir():
                child.unlink(missing_ok=True)
            scene_dir.rmdir()

    # -- lifecycle ---------------------------------------------------------------

    def validate(self, scene_id: str) -> dict[str, Any]:
        """Re-check a scene's stored input raster without processing it."""
        self.require_scene(scene_id)
        input_path = self.find_scene_input(scene_id)
        if input_path is None:
            return {"valid": False, "issues": ["Scene input raster is missing."]}
        issues: list[str] = []
        try:
            metadata = self.inspect_raster(input_path)
            if metadata["width"] < MIN_DIMENSION or metadata["height"] < MIN_DIMENSION:
                issues.append(
                    f"Raster is smaller than {MIN_DIMENSION}x{MIN_DIMENSION} pixels."
                )
        except AppError:
            return {
                "valid": False,
                "issues": ["Input raster is not a readable GeoTIFF."],
            }
        return {"valid": not issues, "issues": issues, "metadata": metadata}

    def delete(self, scene_id: str, *, delete_jobs: Callable[[str], int]) -> dict:
        """Delete a scene: jobs first (via the injected callback), then all
        stored artifacts (symlink-safe per core/paths.py)."""
        self.require_scene(scene_id)
        removed_jobs = delete_jobs(scene_id)
        delete_scene_directories(scene_id)
        logger.info("scene deleted: %s (jobs removed: %d)",
                    scene_id, removed_jobs)
        return {"scene_id": scene_id, "deleted": True, "jobs_removed": removed_jobs}

    def ensure_storage(self) -> None:
        ensure_directories()


scene_service = SceneService()
