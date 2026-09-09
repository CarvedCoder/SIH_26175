"""Low-level, format-agnostic raster I/O helpers.

This module is the seam where future non-GeoTIFF input formats (JPG/JPEG/
PNG, per the project's future scope) get plugged in. Today it only
implements the GeoTIFF path via rasterio, but every function that is
GeoTIFF-specific is named accordingly (`open_geotiff`, `read_geotiff_band`)
so a future `open_png`/`open_jpeg` can sit next to it and be dispatched
from `detect_format` without touching any downstream pipeline stage -
those stages only ever see the common internal representation
(a numpy array + a profile dict), never a format-specific object.

    Input
      v
    detect_format()          <- implemented, GeoTIFF only for now
      v
    format-specific loader   <- open_geotiff() implemented;
      v                         open_png()/open_jpeg() are future work
    common internal representation (ndarray + profile dict)
      v
    common preprocessing pipeline (clean_dem, preprocess_dem, ...)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import rasterio

from config import FUTURE_EXTENSIONS, SUPPORTED_EXTENSIONS
from src.exceptions import InputPathError, InvalidGeoTiffError, UnsupportedFormatError


def detect_format(path: Path) -> str:
    """Identify the input format from its extension and validate support.

    Raises
    ------
    InputPathError
        If the path does not exist or is not a file.
    UnsupportedFormatError
        If the extension is recognized as a *future* format
        (JPG/JPEG/PNG) or is not recognized at all.
    """
    if not path.exists():
        raise InputPathError(f"Input path does not exist: {path}")
    if not path.is_file():
        raise InputPathError(f"Input path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS:
        return "geotiff"

    if suffix in FUTURE_EXTENSIONS:
        raise UnsupportedFormatError(
            f"'{suffix}' input is planned but not implemented yet in this "
            f"version of the pipeline (GeoTIFF only). File: {path}"
        )

    raise UnsupportedFormatError(
        f"Unsupported file extension '{suffix}'. Supported now: "
        f"{SUPPORTED_EXTENSIONS}. Planned for later: {FUTURE_EXTENSIONS}. "
        f"File: {path}"
    )


def open_geotiff(path: Path):
    """Open a GeoTIFF and return the rasterio dataset handle.

    Raises
    ------
    InvalidGeoTiffError
        If rasterio cannot open the file, or it has no bands.
    """
    try:
        dataset = rasterio.open(path)
    except rasterio.RasterioIOError as exc:
        raise InvalidGeoTiffError(
            f"Could not open '{path}' as a GeoTIFF: {exc}"
        ) from exc

    if dataset.count < 1:
        dataset.close()
        raise InvalidGeoTiffError(f"GeoTIFF has no bands: {path}")

    if dataset.width <= 0 or dataset.height <= 0:
        dataset.close()
        raise InvalidGeoTiffError(
            f"GeoTIFF has invalid dimensions ({dataset.width}x{dataset.height}): {path}"
        )

    return dataset


def read_geotiff_band(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """Read a single band plus its rasterio profile.

    Returns
    -------
    (array, profile) where array is a 2D numpy array and profile is the
    rasterio profile dict (contains crs, transform, nodata, dtype, etc.).
    """
    with open_geotiff(path) as dataset:
        array = dataset.read(band)
        profile = dataset.profile.copy()
    return array, profile


def build_invalid_mask(
    array: np.ndarray, nodata: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build boolean masks for NaN, inf, nodata, and their union.

    Returns (nan_mask, inf_mask, nodata_mask, combined_invalid_mask).
    Never raises; nodata=None simply yields an all-False nodata_mask.
    """
    nan_mask = np.isnan(array)
    inf_mask = np.isinf(array)
    if nodata is not None:
        nodata_mask = array == nodata
    else:
        nodata_mask = np.zeros_like(array, dtype=bool)
    combined = nan_mask | inf_mask | nodata_mask
    return nan_mask, inf_mask, nodata_mask, combined
