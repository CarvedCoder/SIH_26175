"""Stage 6 (and internal checks): verification.

Consolidates cleaned-raster, preprocessed-raster, and tile verification
into one module (rather than three near-identical tiny files) since they
share the same "open it, check it, report structured results" shape.
Each check is read-only and never raises for an individual bad tile/file -
failures are collected into a VerificationResult so main.py can report a
complete picture rather than stopping at the first problem. main.py
decides whether any failures should halt the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

import config
from src.raster_io import build_invalid_mask
from src.schemas import TileInfo, TileVerificationEntry, VerificationResult


def _check_single_raster(
    path: Path, expected_shape: tuple[int, int] | None = None
) -> tuple[bool, str | None]:
    """Open a raster and sanity-check it. Returns (ok, reason_if_not_ok)."""
    try:
        with rasterio.open(path) as dataset:
            if dataset.count < 1:
                return False, "no bands"
            if expected_shape is not None:
                actual = (dataset.height, dataset.width)
                if actual != expected_shape:
                    return False, f"shape {actual} != expected {expected_shape}"
            data = dataset.read(1)
            _, _, _, invalid_mask = build_invalid_mask(data, dataset.nodata)
            valid_count = int(np.sum(~invalid_mask))
            if valid_count == 0:
                return False, "all pixels invalid (nodata/NaN/inf)"
    except rasterio.errors.RasterioIOError as exc:
        return False, f"could not open: {exc}"
    return True, None


def verify_cleaned(path: Path) -> VerificationResult:
    """Verify the cleaned-DEM output of clean_dem()."""
    ok, reason = _check_single_raster(path)
    entry = TileVerificationEntry(tile_path=path, ok=ok, reason=reason)
    return VerificationResult(
        all_ok=ok, checked_count=1, failed_count=0 if ok else 1, entries=[entry]
    )


def  verify_preprocessed(
     path: Path, method: str = config.NORMALIZATION_METHOD
 ) -> VerificationResult:
    """Verify the normalized-DEM output of preprocess_dem().

   For the "minmax" method, additionally confirms the valid data range
     falls within [0, 1] (allowing a small numerical tolerance)
    """
    ok, reason = _check_single_raster(path)
    if ok and method == "minmax":
        with rasterio.open(path) as dataset:
            data = dataset.read(1)
            _, _, _, invalid_mask = build_invalid_mask(data, dataset.nodata)
            valid_values = data[~invalid_mask]
            if valid_values.size > 0:
                vmin, vmax = float(np.min(valid_values)), float(np.max(valid_values))
                if vmin < -1e-6 or vmax > 1 + 1e-6:
                    ok = False
                    reason = f"normalized values out of [0, 1] range: min={vmin}, max={vmax}"
    entry = TileVerificationEntry(tile_path=path, ok=ok, reason=reason)
    return VerificationResult(
        all_ok=ok, checked_count=1, failed_count=0 if ok else 1, entries=[entry]
    )


def verify_tiles(tiles: list[TileInfo], tile_size: int = config.TILE_SIZE) -> VerificationResult:
    """Verify every generated tile: openable, correct shape, has valid data."""
    entries: list[TileVerificationEntry] = []
    for tile in tiles:
        ok, reason = _check_single_raster(tile.tile_path, expected_shape=(tile_size, tile_size))
        entries.append(TileVerificationEntry(tile_path=tile.tile_path, ok=ok, reason=reason))

    failed_count = sum(1 for e in entries if not e.ok)
    return VerificationResult(
        all_ok=failed_count == 0,
        checked_count=len(entries),
        failed_count=failed_count,
        entries=entries,
    )
