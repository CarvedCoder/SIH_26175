from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import rasterio
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.paths import (
    ensure_directories,
    get_scene_raw_dir,
)
from backend.app.schemas.scene import (
    ProcessingPath,
    SceneCapabilities,
    SceneCreateResponse,
    SceneDimensions,
    SceneGeoReference,
    SceneResponse,
    SceneStatus,
)


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Scenes"],
)


ALLOWED_EXTENSIONS = {".tif", ".tiff"}


def _inspect_raster(path: Path) -> dict:
    """Inspect an uploaded raster without running model inference."""

    try:
        with rasterio.open(path) as dataset:
            crs = dataset.crs
            bounds = dataset.bounds
            transform = dataset.transform

            georeferenced = crs is not None

            if georeferenced:
                processing_path = ProcessingPath.ABSOLUTE
            else:
                processing_path = ProcessingPath.RELATIVE

            return {
                "width": dataset.width,
                "height": dataset.height,
                "channels": dataset.count,
                "format": "GeoTIFF",
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

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to inspect raster: {exc}",
        ) from exc


def _build_capabilities(georeferenced: bool) -> SceneCapabilities:
    """Build capabilities available from the scene input."""

    return SceneCapabilities(
        absolute_elevation=georeferenced,
        relative_elevation=True,
        reference_comparison=False,
        slope=False,
        height_measurement=False,
        error_map=False,
        local_refinement=False,
    )


def _build_scene_response(
    scene_id: str,
    filename: str,
    metadata: dict,
) -> SceneResponse:
    """Build the API scene response from inspected metadata."""

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
        capabilities=_build_capabilities(
            metadata["georeferenced"]
        ),
    )


@router.post("", response_model=SceneCreateResponse)
async def create_scene(
    file: UploadFile = File(...),
):
    """Upload and inspect a new scene."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )

    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Currently supported formats: "
                ".tif and .tiff"
            ),
        )

    scene_id = f"scene_{uuid4().hex[:12]}"

    ensure_directories()

    scene_dir = get_scene_raw_dir(scene_id)
    scene_dir.mkdir(parents=True, exist_ok=True)

    input_path = scene_dir / original_filename

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc

    finally:
        await file.close()

    metadata = _inspect_raster(input_path)

    scene = _build_scene_response(
        scene_id=scene_id,
        filename=original_filename,
        metadata=metadata,
    )

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


@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str):
    """Return metadata for an existing scene."""

    scene_dir = get_scene_raw_dir(scene_id)

    if not scene_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{scene_id}' not found.",
        )

    files = [
        path
        for path in scene_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"No raster input found for scene '{scene_id}'.",
        )

    input_path = files[0]

    metadata = _inspect_raster(input_path)

    return _build_scene_response(
        scene_id=scene_id,
        filename=input_path.name,
        metadata=metadata,
    )