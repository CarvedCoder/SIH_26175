"""Stage 3: DEM preprocessing / normalization.

Runs on the whole cleaned raster, BEFORE tiling. This is a deliberate
change from the prototype (which normalized each 256x256 tile
independently using that tile's own local min/max): per-tile normalization
gives every tile a different, inconsistent scale, which is not a
meaningful "model-friendly representation" of the full scene and cannot be
reversed without storing per-tile parameters. Normalizing once, globally,
before tiling gives every tile from the same source a consistent,
reproducible scale, and only one set of parameters needs to be recorded.

Design (per task requirements):
- NaN/inf/nodata pixels are excluded from the min/max computation and are
  never assigned a fabricated normalized elevation; they are re-flagged as
  nodata in the output (using a sentinel outside the valid normalized
  range so it can never be confused with real data - see NORMALIZED_NODATA).
- Output dtype is float32 (config.NORMALIZED_DTYPE), matching the task's
  "model-friendly" requirement and Blueprint Section 9's float32/mixed
  precision training assumption.
- The exact method and parameters used (source min/max, or mean/std for
  standardization) are recorded in NormalizationParams and returned to the
  caller, so the transform can be reproduced or inverted later, and so a
  different model-required strategy can be swapped in via
  config.NORMALIZATION_METHOD without changing this module's structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

import config
from src.exceptions import NoValidDataError, NormalizationError, OutputDirectoryError
from src.raster_io import build_invalid_mask, open_geotiff
from src.schemas import NormalizationParams, PreprocessResult

#: Sentinel for the *normalized* output raster's nodata value. Chosen
#: outside [0, 1] (the minmax target range) so it can never collide with a
#: legitimate normalized elevation value.
NORMALIZED_NODATA = -1.0


def preprocess_dem(
    input_path: Path,
    output_dir: Path,
    method: str = config.NORMALIZATION_METHOD,
) -> PreprocessResult:
    """Normalize a cleaned elevation GeoTIFF into a model-friendly raster.

    Parameters
    ----------
    input_path:
        Path to the cleaned GeoTIFF (output of clean_dem()).
    output_dir:
        Directory the normalized GeoTIFF is written into.
    method:
        "minmax" (default) rescales valid pixels to [0, 1].
        "standardize" applies z-score normalization ((x - mean) / std).

    Raises
    ------
    NoValidDataError
        If there are no valid pixels to compute statistics from.
    NormalizationError
        If the valid-data range (or std, for standardize) is degenerate
        (e.g. a perfectly flat raster with max == min).
    OutputDirectoryError
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot create output directory {output_dir}: {exc}") from exc

    with open_geotiff(input_path) as dataset:
        elevation = dataset.read(1).astype(np.float64)
        profile = dataset.profile.copy()
        nodata = dataset.nodata

    _, _, _, invalid_mask = build_invalid_mask(elevation, nodata)
    valid_mask = ~invalid_mask
    valid_pixel_count = int(np.sum(valid_mask))

    if valid_pixel_count == 0:
        raise NoValidDataError(f"No valid elevation pixels to normalize in {input_path}")

    valid_values = elevation[valid_mask]
    normalized = np.full(elevation.shape, NORMALIZED_NODATA, dtype=np.float64)

    if method == "minmax":
        source_min = float(np.min(valid_values))
        source_max = float(np.max(valid_values))
        value_range = source_max - source_min
        if value_range == 0:
            raise NormalizationError(
                f"Cannot min-max normalize {input_path}: all valid pixels "
                f"share the same elevation value ({source_min}); the range is zero."
            )
        normalized[valid_mask] = (valid_values - source_min) / value_range
        params = NormalizationParams(
            method="minmax",
            source_min=source_min,
            source_max=source_max,
            nodata_value=NORMALIZED_NODATA,
        )

    elif method == "standardize":
        mean = float(np.mean(valid_values))
        std = float(np.std(valid_values))
        if std == 0:
            raise NormalizationError(
                f"Cannot standardize {input_path}: valid pixels have zero "
                f"standard deviation (mean={mean})."
            )
        normalized[valid_mask] = (valid_values - mean) / std
        params = NormalizationParams(
            method="standardize",
            source_min=float(np.min(valid_values)),
            source_max=float(np.max(valid_values)),
            nodata_value=NORMALIZED_NODATA,
            mean=mean,
            std=std,
        )
    else:
        raise NormalizationError(f"Unknown normalization method: {method!r}")

    normalized_dtype = np.dtype(config.NORMALIZED_DTYPE)
    normalized = normalized.astype(normalized_dtype)

    profile.update(dtype=str(normalized_dtype), nodata=NORMALIZED_NODATA)

    output_path = output_dir / f"{input_path.stem}_normalized.tif"
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(normalized, 1)

    valid_normalized = normalized[valid_mask]
    return PreprocessResult(
        output_path=output_path,
        params=params,
        normalized_min=float(np.min(valid_normalized)),
        normalized_max=float(np.max(valid_normalized)),
        valid_pixel_count=valid_pixel_count,
    )
