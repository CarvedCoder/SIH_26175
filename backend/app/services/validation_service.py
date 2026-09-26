from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.infrastructure.storage.scene_artifacts import (
    scene_artifact_store,
    scene_output_dir_key,
)
from backend.app.schemas.validation import (
    ReferenceDEM,
    ValidationAsset,
    ValidationMetrics,
    ValidationResponse,
)


def _find_first_existing(
    directory: Path,
    filenames: list[str],
) -> Path | None:
    """Return the first matching file that exists."""
    for filename in filenames:
        path = directory / filename
        if path.exists():
            return path

    return None


def _calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> ValidationMetrics:
    """Calculate basic DSM/reference comparison metrics."""

    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if prediction.shape != reference.shape:
        raise ValueError(
            "Prediction and reference arrays must have the same shape."
        )

    valid = np.isfinite(prediction) & np.isfinite(reference)

    if not np.any(valid):
        return ValidationMetrics(
            sample_count=0,
        )

    prediction = prediction[valid]
    reference = reference[valid]

    error = prediction - reference

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))

    correlation: float | None = None

    if prediction.size > 1:
        prediction_std = float(np.std(prediction))
        reference_std = float(np.std(reference))

        if prediction_std > 0 and reference_std > 0:
            correlation = float(
                np.corrcoef(prediction, reference)[0, 1]
            )

    return ValidationMetrics(
        rmse=rmse,
        mae=mae,
        correlation=correlation,
        sample_count=int(prediction.size),
        units="meters",
    )


def _load_array(path: Path) -> np.ndarray:
    """Load a NumPy- or GeoTIFF-based raster result."""

    if path.suffix.lower() == ".npy":
        return np.load(path)

    if path.suffix.lower() == ".npz":
        data = np.load(path)

        if not data.files:
            raise ValueError(f"No arrays found in {path}")

        return data[data.files[0]]

    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as dataset:
            return dataset.read(1)

    raise ValueError(
        f"Unsupported validation array format: {path.suffix}"
    )


def _find_prediction_profile(
    output_dir: Path,
    prediction_path: Path,
) -> dict[str, Any] | None:
    """The prediction's rasterio grid (crs/transform/width/height), or
    None when the prediction carries no georeferencing on disk."""

    import rasterio

    candidates = (
        [prediction_path]
        if prediction_path.suffix.lower() in {".tif", ".tiff"}
        else []
    ) + [output_dir / "dsm.tif", output_dir / "dsm_anchored.tif"]

    for path in candidates:
        if not path.exists():
            continue
        with rasterio.open(path) as dataset:
            return {
                "crs": dataset.crs,
                "transform": dataset.transform,
                "width": dataset.width,
                "height": dataset.height,
            }

    return None


def _reproject_reference_if_needed(
    output_dir: Path,
    prediction_path: Path,
    reference_path: Path,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Bring the reference onto the prediction's EXACT grid before any
    pixel comparison — mirroring
    depthwizard.anchoring.resample_dem_to_tile: CRS required on both
    sides, bilinear resampling, partial coverage REFUSED.

    Returns the reference unchanged when no georeferencing can drive a
    reprojection (pixel-space artifacts such as reference.npy written
    on-grid by the processing pipeline) — the caller's shape check in
    _calculate_metrics then guards the plain array case.
    """

    import rasterio
    from rasterio.warp import Resampling, reproject

    del prediction  # documented via the return contract above

    if reference_path.suffix.lower() not in {".tif", ".tiff"}:
        return reference

    with rasterio.open(reference_path) as ref:
        if ref.crs is None:
            # Pixel-registered reference: no CRS to reproject with — the
            # caller's shape check governs the comparison.
            return reference

        profile = _find_prediction_profile(output_dir, prediction_path)
        if profile is None or profile["crs"] is None:
            raise ValueError(
                "reference is georeferenced but the prediction grid is "
                "not — refusing a pixel-space comparison"
            )

        dst = np.full(
            (profile["height"], profile["width"]), np.nan, dtype=np.float32
        )
        reproject(
            source=rasterio.band(ref, 1),
            destination=dst,
            src_transform=ref.transform,
            src_crs=ref.crs,
            src_nodata=ref.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    if np.isnan(dst).any():
        raise ValueError(
            f"reference '{reference_path.name}' does not fully cover the "
            f"prediction footprint ({np.isnan(dst).mean():.1%} missing "
            "pixels) — partial-coverage validation is refused."
        )

    return dst


def get_validation(
    scene_id: str,
) -> ValidationResponse:
    """
    Build the validation response for a processed scene.

    This service first looks for existing validation artifacts generated
    by the processing pipeline. If numerical DSM/reference arrays are
    available, it calculates the comparison metrics.
    """

    output_dir = scene_artifact_store().path_for(scene_output_dir_key(scene_id))

    if not output_dir.exists():
        return ValidationResponse(
            scene_id=scene_id,
            available=False,
        )

    # ---------------------------------------------------------------
    # Look for generated validation metadata
    # ---------------------------------------------------------------

    metadata_path = _find_first_existing(
        output_dir,
        [
            "validation.json",
            "validation_metrics.json",
        ],
    )

    metadata: dict[str, Any] = {}

    if metadata_path is not None:
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, dict):
                metadata = loaded

        except (OSError, json.JSONDecodeError):
            metadata = {}

    # ---------------------------------------------------------------
    # Locate prediction/reference arrays
    # ---------------------------------------------------------------

    prediction_path = _find_first_existing(
        output_dir,
        [
            "dsm.npy",
            "dsm_array.npy",
            "anchored_dsm.npy",
        ],
    )

    reference_path = _find_first_existing(
        output_dir,
        [
            "reference.npy",
            "reference_dem.npy",
            "ref_dem.npy",
        ],
    )

    metrics = ValidationMetrics()

    if prediction_path is not None and reference_path is not None:
        try:
            prediction = _load_array(prediction_path)
            reference = _load_array(reference_path)

            reference = _reproject_reference_if_needed(
                output_dir,
                prediction_path,
                reference_path,
                prediction,
                reference,
            )

            metrics = _calculate_metrics(
                prediction,
                reference,
            )

        except (OSError, ValueError, TypeError):
            # Preserve the API contract even when numerical validation
            # cannot be performed.
            metrics = ValidationMetrics()

    # ---------------------------------------------------------------
    # Reference metadata
    # ---------------------------------------------------------------

    reference_available = reference_path is not None

    reference = ReferenceDEM(
        available=reference_available,
        name=reference_path.name if reference_path else None,
        crs=metadata.get("reference_crs"),
        units=metadata.get("reference_units", "meters")
        if reference_available
        else None,
    )

    # ---------------------------------------------------------------
    # Error map
    # ---------------------------------------------------------------

    error_map_path = _find_first_existing(
        output_dir,
        [
            "error_map.png",
            "validation_error_map.png",
            "error_map.tif",
        ],
    )

    error_map = None

    if error_map_path is not None:
        from backend.app.storage.service import storage_service

        error_map_url = (
            storage_service.presign_artifact(scene_id, error_map_path)
            or f"/api/v1/scenes/{scene_id}/results/error-map"
        )
        error_map = ValidationAsset(
            name=error_map_path.name,
            url=error_map_url,
            format=error_map_path.suffix.lstrip("."),
        )

    # ---------------------------------------------------------------
    # Determine whether validation is actually available
    # ---------------------------------------------------------------

    available = (
        reference_available
        or error_map is not None
        or metrics.sample_count is not None
        or bool(metadata)
    )

    return ValidationResponse(
        scene_id=scene_id,
        available=available,
        reference=reference,
        metrics=metrics,
        error_map=error_map,
        metadata=metadata,
    )