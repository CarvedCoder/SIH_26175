"""Stage 2: DEM cleaning.

Design decisions (explicit, per task requirements - "cleaning decisions
must be explicit and documented", "do not silently replace meaningful
elevation values"):

1. NaN, +/-inf, and the source's declared `nodata` value are all treated
   as *invalid* and are the ONLY pixels this stage ever touches. Every
   other pixel value is passed through completely unchanged - the
   prototype already read the array and wrote it back untouched, which
   this preserves for valid data, but the prototype never actually did
   anything with the invalid pixels it detected (it computed
   `invalid_count` and then wrote the raw array back verbatim). That is
   the primary bug this stage fixes.

2. All invalid pixels are written out as a single, explicit nodata
   sentinel value in the output raster (and recorded in the output
   GeoTIFF's `nodata` field), rather than being replaced with any
   numeric elevation guess (0, mean, interpolation, ...). Fabricating an
   elevation value for a pixel with no data would violate "do not
   silently replace meaningful elevation values" - marking it as nodata
   is the only transformation that doesn't invent information.

3. If the source already declares a nodata value, that same value is
   reused as the sentinel (so pixels already correctly marked nodata are
   left bit-for-bit identical). If the source has no declared nodata
   value but does contain NaN/inf pixels, a new sentinel
   (CLEAN_NODATA_SENTINEL) is chosen and every NaN/inf pixel is
   set to it; this is the one case where pixel *values* change, and it
   is logged in CleaningResult.notes.

4. A raster that is entirely invalid, or has zero valid pixels, is a
   pipeline-stopping error (NoValidDataError) - there is nothing for any
   later stage to work with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from src.exceptions import NoValidDataError, OutputDirectoryError
from src.raster_io import build_invalid_mask, open_geotiff
from src.schemas import CleaningResult

#: Sentinel used only when the source GeoTIFF has no declared nodata value
#: but still contains NaN/inf pixels that must be marked invalid.
CLEAN_NODATA_SENTINEL = -9999.0


def clean_dem(input_path: Path, output_dir: Path) -> CleaningResult:
    """Clean a single-band elevation GeoTIFF.

    Parameters
    ----------
    input_path:
        Source GeoTIFF (typically the raw input, or a prior stage's output).
    output_dir:
        Directory the cleaned GeoTIFF is written into. Created if missing.

    Returns
    -------
    CleaningResult with the output path and the exact counts/decisions made.

    Raises
    ------
    NoValidDataError
        If every pixel is NaN, inf, or nodata.
    OutputDirectoryError
        If output_dir cannot be created.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot create output directory {output_dir}: {exc}") from exc

    notes: list[str] = []

    with open_geotiff(input_path) as dataset:
        elevation = dataset.read(1)
        profile = dataset.profile.copy()
        source_nodata = dataset.nodata

    nan_mask, inf_mask, nodata_mask, invalid_mask = build_invalid_mask(elevation, source_nodata)
    total_pixels = elevation.size
    valid_pixel_count = int(total_pixels - np.sum(invalid_mask))

    if valid_pixel_count == 0:
        raise NoValidDataError(
            f"No valid elevation pixels found in {input_path} "
            f"(all {total_pixels} pixels are NaN, inf, or nodata)."
        )

    if source_nodata is not None:
        nodata_value_used = float(source_nodata)
    else:
        nodata_value_used = CLEAN_NODATA_SENTINEL
        notes.append(
            f"Source had no declared nodata value; NaN/inf pixels were "
            f"assigned sentinel {CLEAN_NODATA_SENTINEL} and this value was "
            f"set as the output GeoTIFF's nodata."
        )

    cleaned = elevation.astype(np.float64, copy=True)
    # Only invalid pixels are touched; every valid elevation value is
    # passed through bit-for-bit.
    cleaned[invalid_mask] = nodata_value_used
    cleaned = cleaned.astype(elevation.dtype, copy=False)

    profile.update(nodata=nodata_value_used)

    output_path = output_dir / f"{input_path.stem}_cleaned.tif"
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cleaned, 1)

    return CleaningResult(
        output_path=output_path,
        invalid_pixel_count=int(np.sum(invalid_mask)),
        nan_pixel_count=int(np.sum(nan_mask)),
        inf_pixel_count=int(np.sum(inf_mask)),
        nodata_pixel_count=int(np.sum(nodata_mask)),
        total_pixels=int(total_pixels),
        valid_pixel_count=valid_pixel_count,
        nodata_value_used=nodata_value_used,
        notes=tuple(notes),
    )
