"""Scene lifecycle routes: upload, inspect, list, get, delete, validate.

Upload safety (audit C4/M3/S7):
    * extension whitelist + declared max size (DW_MAX_UPLOAD_BYTES, default
      500 MB) enforced DURING the streaming write — oversized bodies are
      rejected without buffering them in memory;
    * the upload lands in a staging directory and is validated with
      rasterio BEFORE anything permanent is created; only then is it
      atomically moved into the scene's raw directory;
    * every failure path cleans up its temp files.

Determinism (audit M9): the validated raster is stored under the exact
name ``input.<ext>`` (its original extension) and its inspected metadata
is persisted in ``scene.json``, so every later reader sees the same file
and metadata — no unsorted iterdir, no per-request re-inspection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.core.config import get_settings
from backend.app.core.errors import AppError, InvalidSceneId, SceneNotFound
from backend.app.core.logging import logger
from backend.app.core.paths import (
    UPLOAD_STAGING_DIR,
    delete_scene_directories,
    ensure_directories,
    get_scene_output_dir,
    get_scene_raw_dir,
    get_scene_process_dir,
    valid_scene_id,
)
from backend.app.jobs.manager import job_manager
from backend.app.schemas.scene import (
    ProcessingPath,
    SceneCapabilities,
    SceneCreateResponse,
    SceneDimensions,
    SceneGeoReference,
    SceneResponse,
    SceneStatus,
    SceneSummary,
)
from backend.app.schemas.validation import (
    ValidationCheckResponse,
    ValidationDimensions,
)
from backend.app.services.result_service import result_service

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Scenes"],
)

# PNG/JPG are first-class inputs (the frontend advertises them): rasterio
# reads them via the GDAL PNG/JPEG drivers, and the certified inference
# path (read_image) consumes any rasterio-readable RGB raster.
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
SCENE_METADATA_NAME = "scene.json"
_UPLOAD_CHUNK = 1024 * 1024


def designated_input_name(extension: str) -> str:
    """Exact filename the validated upload is stored under — determinism
    contract of the processing service (input.<original extension>)."""
    return f"input{extension}"


# ---------------------------------------------------------------------------
# Inspection (rasterio parse = the real file-type gate)
# ---------------------------------------------------------------------------

def _inspect_raster(path: Path) -> dict:
    """Inspect an uploaded raster WITHOUT running model inference.

    rasterio.open() is the actual content validation: a file that cannot
    be parsed as a raster is rejected here, before it becomes scene state.
    """
    try:
        with rasterio.open(path) as dataset:
            crs = dataset.crs
            bounds = dataset.bounds
            transform = dataset.transform

            georeferenced = crs is not None
            processing_path = (
                ProcessingPath.ABSOLUTE if georeferenced else ProcessingPath.RELATIVE
            )

            return {
                "width": dataset.width,
                "height": dataset.height,
                "channels": dataset.count,
                "format": dataset.driver,  # GTiff | PNG | JPEG — honest
                "georeferenced": georeferenced,
                "crs": crs.to_string() if crs else None,
                "min_x": bounds.left,
                "min_y": bounds.bottom,
                "max_x": bounds.right,
                "max_y": bounds.top,
                "pixel_width": abs(transform.a),
                "pixel_height": abs(transform.e),
                "processing_path": processing_path,
            }
    except HTTPException:
        raise
    except Exception:
        # Raw exception text may embed filesystem paths; the client gets a
        # stable message, the detail is logged server-side.
        logger.exception("raster inspection failed for scene upload")
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be parsed as a supported image raster (PNG, JPEG, or GeoTIFF).",
        )


def _load_scene_metadata(scene_id: str) -> dict | None:
    path = get_scene_raw_dir(scene_id) / SCENE_METADATA_NAME
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _store_scene_metadata(
    scene_id: str, filename: str, metadata: dict
) -> None:
    payload = {
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            key: (
                value.value if hasattr(value, "value") else value
            )
            for key, value in metadata.items()
        },
    }
    path = get_scene_raw_dir(scene_id) / SCENE_METADATA_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_capabilities(georeferenced: bool) -> SceneCapabilities:
    """Capabilities available from the scene input (honest: what the input
    can support, not what has been processed)."""
    return SceneCapabilities(
        absolute_elevation=georeferenced,
        relative_elevation=True,
        reference_comparison=False,
        slope=georeferenced,
        height_measurement=True,
        error_map=False,
        local_refinement=True,
    )


def _build_scene_response(
    scene_id: str,
    filename: str,
    metadata: dict,
) -> SceneResponse:
    return SceneResponse(
        scene_id=scene_id,
        filename=filename,
        status=SceneStatus.READY,
        format=metadata["format"],
        dimensions=SceneDimensions(
            width=metadata["width"],
            height=metadata["height"],
            channels=metadata["channels"],
        ),
        georeference=SceneGeoReference(
            available=metadata["georeferenced"],
            crs=metadata["crs"],
            min_x=metadata["min_x"],
            min_y=metadata["min_y"],
            max_x=metadata["max_x"],
            max_y=metadata["max_y"],
            pixel_width=metadata["pixel_width"],
            pixel_height=metadata["pixel_height"],
        ),
        processing_path=metadata["processing_path"],
        capabilities=_build_capabilities(metadata["georeferenced"]),
    )


def _require_scene(scene_id: str) -> None:
    if not valid_scene_id(scene_id):
        raise InvalidSceneId(scene_id)
    if not get_scene_raw_dir(scene_id).is_dir():
        raise SceneNotFound(scene_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=SceneCreateResponse)
async def create_scene(
    file: UploadFile = File(...),
    reference_type: str | None = Form(None),
):
    """Upload, validate, and register a new scene."""
    if reference_type is not None:
        raise HTTPException(
            status_code=422,
            detail="Reference data upload is not supported yet; upload a "
            "plain GeoTIFF scene.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported formats: "
            ".png, .jpg, .jpeg, .tif, .tiff",
        )

    declared_length = file.size
    if declared_length is not None and declared_length > get_settings().max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the maximum allowed size.",
        )

    ensure_directories()
    from uuid import uuid4

    scene_id = f"scene_{uuid4().hex[:12]}"
    scene_dir = get_scene_raw_dir(scene_id)
    staging_path = UPLOAD_STAGING_DIR / f"{scene_id}{extension}"

    try:
        # 1) stream to staging with a hard byte cap (never memory-buffered)
        received = 0
        try:
            with staging_path.open("wb") as destination:
                while chunk := await file.read(_UPLOAD_CHUNK):
                    received += len(chunk)
                    if received > get_settings().max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Uploaded file exceeds the maximum allowed size.",
                        )
                    destination.write(chunk)
        finally:
            await file.close()

        # 2) validate content BEFORE committing any permanent state
        metadata = _inspect_raster(staging_path)

        # 3) commit: atomically move the validated raster into place,
        #    under the designated input.<ext> name (determinism contract)
        scene_dir.mkdir(parents=True, exist_ok=True)
        atomic_destination = scene_dir / designated_input_name(extension)
        staging_path.replace(atomic_destination)
        _store_scene_metadata(scene_id, original_filename, metadata)

    except HTTPException:
        _cleanup_failed_upload(staging_path, scene_dir)
        raise
    except OSError:
        logger.exception("failed to persist upload for scene %s", scene_id)
        _cleanup_failed_upload(staging_path, scene_dir)
        raise HTTPException(status_code=500, detail="Failed to store the uploaded file.")

    logger.info(
        "scene created: %s file=%s bytes=%d", scene_id, original_filename, received
    )

    scene = _build_scene_response(scene_id, original_filename, metadata)
    return SceneCreateResponse(
        scene_id=scene.scene_id,
        filename=scene.filename,
        status=scene.status,
        format=scene.format,
        dimensions=scene.dimensions,
        georeference=scene.georeference,
        processing_path=scene.processing_path,
        capabilities=scene.capabilities,
    )


def _cleanup_failed_upload(staging_path: Path, scene_dir: Path) -> None:
    """Remove staging + any partially created scene dir (idempotent)."""
    staging_path.unlink(missing_ok=True)
    if scene_dir is not None and scene_dir.is_dir():
        for child in scene_dir.iterdir():
            child.unlink(missing_ok=True)
        scene_dir.rmdir()


@router.post("/{scene_id}/mosaic", response_model=SceneCreateResponse)
async def mosaic_scene_inputs(
    scene_id: str,
    files: list[UploadFile] = File(...),
):
    """Opt-in multi-file upload: merge spatially adjacent GeoTIFFs into one
    contiguous scene input BEFORE processing (depthwizard.mosaic).

    The single-file upload contract is untouched — this endpoint exists for
    survey datasets that arrive as several adjacent GeoTIFFs. All inputs
    must be genuinely georeferenced, share one CRS and one ground
    resolution, and touch/overlap each other; otherwise the mosaic is
    rejected with 400 (inputs are never silently resized or reprojected).
    Mosaicking a processed scene is refused with 409 so results can never
    go stale against a replaced input.
    """
    from depthwizard.mosaic import MosaicError, mosaic_rasters, write_mosaic

    _require_scene(scene_id)
    if result_service.get_result_files(scene_id).get("depth") is not None:
        raise HTTPException(
            status_code=409,
            detail="Scene already has processing results — mosaicking would "
            "replace its input and leave them stale. Delete the scene and "
            "upload the mosaic instead.",
        )
    if len(files) < 2:
        raise HTTPException(
            status_code=422,
            detail="Mosaic upload needs at least 2 GeoTIFF files "
            "(a single file needs no mosaic).",
        )

    ensure_directories()
    settings = get_settings()
    staged: list[tuple[str, Path]] = []
    received_total = 0
    try:
        for index, file in enumerate(files):
            original_filename = Path(file.filename or "").name
            extension = Path(original_filename).suffix.lower()
            if extension not in GEOTIFF_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{original_filename or 'file ' + str(index + 1)}' is not "
                    "a GeoTIFF (.tif/.tiff) — only georeferenced rasters can "
                    "be mosaicked.",
                )
            staging_path = UPLOAD_STAGING_DIR / (
                f"{scene_id}_mosaic_{index}{extension}"
            )
            received = 0
            try:
                with staging_path.open("wb") as destination:
                    while chunk := await file.read(_UPLOAD_CHUNK):
                        received += len(chunk)
                        received_total += len(chunk)
                        if received > settings.max_upload_bytes or (
                            received_total > settings.max_upload_bytes
                        ):
                            raise HTTPException(
                                status_code=413,
                                detail="Uploaded files exceed the maximum "
                                "allowed size.",
                            )
                        destination.write(chunk)
            finally:
                await file.close()
            staged.append((original_filename, staging_path))

        # Validate + merge BEFORE touching any permanent scene state.
        try:
            result = mosaic_rasters([p for _, p in staged])
        except MosaicError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        mosaic_staging = UPLOAD_STAGING_DIR / f"{scene_id}_mosaic_merged.tif"
        write_mosaic(result, mosaic_staging)
        metadata = _inspect_raster(mosaic_staging)

        scene_dir = get_scene_raw_dir(scene_id)
        for old in scene_dir.glob("input.*"):
            old.unlink()
        mosaic_staging.replace(scene_dir / designated_input_name(".tif"))
        _store_scene_metadata(
            scene_id, f"mosaic({'+'.join(name for name, _ in staged)})", metadata
        )
        logger.info(
            "scene mosaic: %s inputs=%d merged=%dx%d",
            scene_id, len(staged), metadata["width"], metadata["height"],
        )
    except HTTPException:
        for _, path in staged:
            path.unlink(missing_ok=True)
        (UPLOAD_STAGING_DIR / f"{scene_id}_mosaic_merged.tif").unlink(missing_ok=True)
        raise
    except OSError:
        logger.exception("failed to persist mosaic for scene %s", scene_id)
        raise HTTPException(status_code=500, detail="Failed to store the mosaic.")

    return SceneCreateResponse(
        scene_id=scene_id,
        filename=f"mosaic({'+'.join(name for name, _ in staged)})",
        status=SceneStatus.READY,
        format=metadata["format"],
        dimensions=SceneDimensions(
            width=metadata["width"],
            height=metadata["height"],
            channels=metadata["channels"],
        ),
        georeference=SceneGeoReference(
            available=metadata["georeferenced"],
            crs=metadata["crs"],
            min_x=metadata["min_x"],
            min_y=metadata["min_y"],
            max_x=metadata["max_x"],
            max_y=metadata["max_y"],
            pixel_width=metadata["pixel_width"],
            pixel_height=metadata["pixel_height"],
        ),
        processing_path=metadata["processing_path"],
        capabilities=_build_capabilities(metadata["georeferenced"]),
    )


@router.get("", response_model=list[SceneSummary])
async def list_scenes():
    """List registered scenes (most recent first)."""
    from backend.app.core.paths import SCENES_RAW_DIR

    summaries: list[SceneSummary] = []
    if not SCENES_RAW_DIR.is_dir():
        return summaries

    for scene_dir in SCENES_RAW_DIR.iterdir():
        if not scene_dir.is_dir() or not valid_scene_id(scene_dir.name):
            continue
        stored = _load_scene_metadata(scene_dir.name)
        if stored is None:
            continue
        summaries.append(
            SceneSummary(
                scene_id=scene_dir.name,
                filename=stored.get("filename", "unknown"),
                status=SceneStatus.READY,
                has_results=bool(
                    result_service.get_result_files(scene_dir.name)
                ),
                created_at=stored.get("created_at"),
            )
        )

    summaries.sort(key=lambda s: s.created_at or s.scene_id, reverse=True)
    return summaries


@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str):
    """Return metadata for an existing scene (from stored metadata — no
    re-inspection of the raster per request)."""
    _require_scene(scene_id)

    stored = _load_scene_metadata(scene_id)
    if stored is None:
        raise SceneNotFound(scene_id)

    metadata = stored["metadata"]
    metadata["processing_path"] = ProcessingPath(metadata["processing_path"])
    return _build_scene_response(
        scene_id=scene_id,
        filename=stored.get("filename", "unknown"),
        metadata=metadata,
    )


@router.delete("/{scene_id}", status_code=200)
async def delete_scene(scene_id: str):
    """Delete a scene and ALL of its stored artifacts (inputs,
    intermediates, results) plus its job records."""
    _require_scene(scene_id)

    deleted_jobs = job_manager.delete_jobs_for_scene(scene_id)
    delete_scene_directories(scene_id)
    logger.info(
        "scene deleted: %s (jobs removed: %d)", scene_id, deleted_jobs
    )
    return {"scene_id": scene_id, "deleted": True, "jobs_removed": deleted_jobs}


@router.post("/{scene_id}/validate", response_model=ValidationCheckResponse)
async def validate_scene(scene_id: str):
    """Re-check a scene's stored input raster without processing it."""
    _require_scene(scene_id)

    from backend.app.services.processing_service import processing_service

    input_path = processing_service.find_scene_input(scene_id)
    if input_path is None:
        return ValidationCheckResponse(
            scene_id=scene_id, valid=False, issues=["Scene input raster is missing."]
        )

    issues: list[str] = []
    try:
        metadata = _inspect_raster(input_path)
        if metadata["width"] < 64 or metadata["height"] < 64:
            issues.append("Raster is smaller than 64x64 pixels.")
    except HTTPException:
        return ValidationCheckResponse(
            scene_id=scene_id, valid=False, issues=["Input raster is not a readable GeoTIFF."]
        )

    return ValidationCheckResponse(
        scene_id=scene_id,
        valid=not issues,
        issues=issues,
        georeferenced=metadata["georeferenced"],
        dimensions=ValidationDimensions(
            width=metadata["width"],
            height=metadata["height"],
            channels=metadata["channels"],
        ),
    )
