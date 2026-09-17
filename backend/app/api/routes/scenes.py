"""Scene lifecycle routes — THIN HTTP adapters (refactor brief §10).

The business rules live in the application layer
(``backend.app.application.scenes.service.SceneService``) and the scene
record in ``infrastructure.persistence.scene_repository``. Routes keep
only HTTP concerns: streaming an UploadFile to staging (with the hard
byte cap enforced DURING the write) and shaping Pydantic responses.

Upload safety (audit C4/M3/S7, preserved):
    * extension whitelist + declared max size (DW_MAX_UPLOAD_BYTES);
    * the upload lands in staging and is validated with rasterio BEFORE
      anything permanent is created; only then is it atomically moved
      into the scene's raw directory;
    * every failure path cleans up its temp files.

Determinism (audit M9, preserved): the validated raster is stored under
the exact name ``input.<ext>`` and its inspected metadata is persisted in
``scene.json`` — no unsorted iterdir, no per-request re-inspection.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.api.routes._deps import require_scene
from backend.app.appstate import settings_for
from backend.app.core.logging import logger
from backend.app.application.scenes.service import scene_service
from backend.app.core.errors import AppError
from backend.app.core.paths import UPLOAD_STAGING_DIR
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
_UPLOAD_CHUNK = 1024 * 1024


# ---------------------------------------------------------------------------
# Response assembly (pure schema shaping — HTTP layer's job)
# ---------------------------------------------------------------------------

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
    if isinstance(metadata.get("processing_path"), str):
        metadata = dict(metadata)
        metadata["processing_path"] = ProcessingPath(metadata["processing_path"])
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


# ---------------------------------------------------------------------------
# Upload streaming (the only HTTP-bound part of scene creation)
# ---------------------------------------------------------------------------

async def _stream_to_staging(
    file: UploadFile, staging_path: Path, max_bytes: int
) -> int:
    """Stream an upload to staging with a hard byte cap — oversized bodies
    are rejected without being buffered in memory."""
    received = 0
    try:
        with staging_path.open("wb") as destination:
            while chunk := await file.read(_UPLOAD_CHUNK):
                received += len(chunk)
                if received > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the maximum allowed size.",
                    )
                destination.write(chunk)
    finally:
        await file.close()
    return received


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

    settings = settings_for()
    declared_length = file.size
    if declared_length is not None and declared_length > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the maximum allowed size.",
        )

    scene_service.ensure_storage()
    from uuid import uuid4

    scene_id = f"scene_{uuid4().hex[:12]}"
    scene_dir = scene_service._repo.raw_dir(scene_id)
    staging_path = UPLOAD_STAGING_DIR / f"{scene_id}{extension}"

    try:
        received = await _stream_to_staging(
            file, staging_path, settings.max_upload_bytes
        )
        metadata = scene_service.commit_upload(
            scene_id,
            staging_path,
            extension,
            original_filename,
            on_failure=lambda: scene_service.cleanup_failed_upload(
                staging_path, scene_dir
            ),
        )
    except OSError:
        scene_service.cleanup_failed_upload(staging_path, scene_dir)
        logger.exception("failed to persist upload for scene %s", scene_id)
        raise HTTPException(status_code=500, detail="Failed to store the uploaded file.")

    logger.info(
        "scene created: %s file=%s bytes=%d", scene_id, original_filename, received
    )
    scene = _build_scene_response(scene_id, original_filename, metadata)
    return SceneCreateResponse(
        scene_id=scene_id,
        **_response_fields(scene),
    )


def _response_fields(response: SceneResponse) -> dict:
    return {
        "filename": response.filename,
        "status": response.status,
        "format": response.format,
        "dimensions": response.dimensions,
        "georeference": response.georeference,
        "processing_path": response.processing_path,
        "capabilities": response.capabilities,
    }


@router.post("/{scene_id}/mosaic", response_model=SceneCreateResponse)
async def mosaic_scene_inputs(
    scene_id: str,
    files: list[UploadFile] = File(...),
):
    """Opt-in multi-file upload: merge spatially adjacent GeoTIFFs into one
    contiguous scene input BEFORE processing (depthwizard.mosaic).

    All inputs must be genuinely georeferenced, share one CRS and one
    ground resolution, and touch/overlap each other; otherwise the mosaic
    is rejected with 400 (inputs are never silently resized or
    reprojected). Mosaicking a processed scene is refused with 409 so
    results can never go stale against a replaced input."""
    require_scene(scene_id)
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

    scene_service.ensure_storage()
    settings = settings_for()
    staged: list[tuple[str, Path]] = []
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
            await _stream_to_staging(file, staging_path, settings.max_upload_bytes)
            staged.append((original_filename, staging_path))

        metadata = scene_service.commit_mosaic(
            scene_id, [p for _, p in staged], [n for n, _ in staged]
        )
    except AppError as exc:
        for _, path in staged:
            path.unlink(missing_ok=True)
        _to_http_error(exc)
    except HTTPException:
        for _, path in staged:
            path.unlink(missing_ok=True)
        raise
    except OSError:
        for _, path in staged:
            path.unlink(missing_ok=True)
        logger.exception("failed to persist mosaic for scene %s", scene_id)
        raise HTTPException(status_code=500, detail="Failed to store the mosaic.")

    merged_name = f"mosaic({'+'.join(name for name, _ in staged)})"
    scene = _build_scene_response(scene_id, merged_name, metadata)
    return SceneCreateResponse(
        scene_id=scene_id,
        **_response_fields(scene),
    )


def _to_http_error(exc: AppError) -> None:
    """Convert a typed domain error into the HTTP envelope shape."""
    raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("", response_model=list[SceneSummary])
async def list_scenes():
    """List registered scenes (most recent first)."""
    summaries: list[SceneSummary] = []
    for scene_id in scene_service._repo.list_scene_ids():
        stored = scene_service._repo.load_record(scene_id)
        if stored is None:
            continue
        summaries.append(
            SceneSummary(
                scene_id=scene_id,
                filename=stored.get("filename", "unknown"),
                status=SceneStatus.READY,
                has_results=bool(
                    result_service.get_result_files(scene_id)
                ),
                created_at=stored.get("created_at"),
            )
        )
    return summaries


@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str):
    """Return metadata for an existing scene (from stored metadata — no
    re-inspection of the raster per request)."""
    stored = scene_service.get_record(scene_id)
    metadata = dict(stored["metadata"])
    return _build_scene_response(
        scene_id=scene_id,
        filename=stored.get("filename", "unknown"),
        metadata=metadata,
    )


@router.delete("/{scene_id}", status_code=200)
async def delete_scene(scene_id: str):
    """Delete a scene and ALL of its stored artifacts (inputs,
    intermediates, results) plus its job records."""
    return scene_service.delete(scene_id, delete_jobs=job_manager.delete_jobs_for_scene)


@router.post("/{scene_id}/validate", response_model=ValidationCheckResponse)
async def validate_scene(scene_id: str):
    """Re-check a scene's stored input raster without processing it."""
    result = scene_service.validate(scene_id)
    metadata = result.get("metadata")
    return ValidationCheckResponse(
        scene_id=scene_id,
        valid=result["valid"],
        issues=result["issues"],
        georeferenced=metadata["georeferenced"] if metadata else None,
        dimensions=ValidationDimensions(
            width=metadata["width"],
            height=metadata["height"],
            channels=metadata["channels"],
        )
        if metadata
        else None,
    )
