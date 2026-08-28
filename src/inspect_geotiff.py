"""Stage 1: GeoTIFF inspection.

Reads and validates a GeoTIFF's structural and geospatial metadata plus
basic data-quality statistics, without mutating anything on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.exceptions import InvalidGeoTiffError, MissingCRSError
from src.raster_io import build_invalid_mask, open_geotiff
from src.schemas import GeoTiffMetadata


def inspect_geotiff(path: Path, require_crs: bool = True) -> GeoTiffMetadata:
    """Inspect a GeoTIFF and return its metadata and data-quality stats.

    Parameters
    ----------
    path:
        Path to the GeoTIFF.
    require_crs:
        If True (default), a missing CRS raises MissingCRSError. The
        Blueprint's geospatial pipeline (Section 12) requires the CRS to be
        read from the source and preserved throughout, so a georeferenced
        pipeline run should fail fast rather than silently proceeding
        without one.

    Raises
    ------
    InvalidGeoTiffError, MissingCRSError
    """
    with open_geotiff(path) as dataset:
        crs = dataset.crs
        if require_crs and crs is None:
            raise MissingCRSError(
                f"GeoTIFF has no CRS defined, but the pipeline requires "
                f"georeferenced input: {path}"
            )

        band = dataset.read(1)
        nodata = dataset.nodata

        nan_mask, inf_mask, nodata_mask, invalid_mask = build_invalid_mask(band, nodata)
        total_pixels = band.size
        valid_mask = ~invalid_mask
        valid_pixel_count = int(np.sum(valid_mask))

        if valid_pixel_count == 0:
            elevation_min = elevation_max = elevation_mean = None
        else:
            valid_values = band[valid_mask]
            elevation_min = float(np.min(valid_values))
            elevation_max = float(np.max(valid_values))
            elevation_mean = float(np.mean(valid_values))

        return GeoTiffMetadata(
            path=path,
            width=dataset.width,
            height=dataset.height,
            band_count=dataset.count,
            dtypes=tuple(dataset.dtypes),
            crs=crs.to_string() if crs is not None else None,
            transform=tuple(dataset.transform)[:6],
            bounds=tuple(dataset.bounds),
            resolution=tuple(dataset.res),
            nodata=nodata,
            total_pixels=total_pixels,
            nodata_pixel_count=int(np.sum(nodata_mask)),
            nan_pixel_count=int(np.sum(nan_mask)),
            inf_pixel_count=int(np.sum(inf_mask)),
            valid_pixel_count=valid_pixel_count,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            elevation_mean=elevation_mean,
        )
