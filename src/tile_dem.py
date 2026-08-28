"""Stage 5: Tiling.

Implements Blueprint Section 11 ("Large Image / Tiling Pipeline"):
  - windowed reads via rasterio.windows (never load the full scene twice),
  - fixed TILE_SIZE (default 512, config.py),
  - configurable OVERLAP (default 64, config.py),
  - reflect-padding at scene borders so every tile is exactly
    tile_size x tile_size even at the image edge or for a source smaller
    than tile_size,
  - CRS/transform preserved and correctly recomputed per tile (a tile's
    transform is NOT the source transform - it must account for the
    tile's pixel offset within the source grid).

Tiling algorithm
-----------------
Tile top-left corners advance with stride = tile_size - overlap, starting
at (0, 0), for as long as the current corner is still inside the raster.
Because stride <= tile_size whenever overlap >= 0, consecutive tiles
always touch or overlap - there is no coverage gap regardless of whether
the source dimensions are smaller than, exactly divisible by, or not
evenly divisible by tile_size. Any tile that would read past the raster
edge (including the degenerate case where the whole raster is smaller
than tile_size) is padded up to tile_size x tile_size using reflect
padding (config.PAD_MODE), falling back to edge-replication only if
reflect padding is mathematically impossible (pad width >= data extent -
only relevant for rasters a few pixels wide/tall).

Gaussian-weighted blending for reconstructing a seamless mosaic from
overlapping tiles (Blueprint Section 11 point 4) is a stitching-time
concern for a later stage (model inference is out of scope for this
task); it is not implemented here. Every TileInfo does, however, record
its grid position, pixel offsets, and real-vs-padded extents, which is
exactly what a future stitching module needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import config
from src.exceptions import TilingError
from src.raster_io import open_geotiff
from src.schemas import TileInfo, TilingResult


def _tile_starts(dimension: int, tile_size: int, overlap: int) -> list[int]:
    """Compute tile start offsets along one axis with full coverage."""
    if dimension <= tile_size:
         return [0]

    
    stride = max(tile_size - overlap, 1)
    max_start = dimension - tile_size
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
         starts.append(max_start)
    return starts


def _pad_to_tile_size(
    array: np.ndarray, tile_size: int
) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    """Pad a (possibly undersized) 2D array up to tile_size x tile_size.

    Padding is only ever added at the bottom/right, since tile windows
    always start within the raster (row_off, col_off >= 0) - the deficit
    can only occur at the far edge of the read window.

    Returns (padded_array, (top, bottom, left, right), pad_mode_used).
    """
    height, width = array.shape
    pad_bottom = tile_size - height
    pad_right = tile_size - width

    if pad_bottom == 0 and pad_right == 0:
        return array, (0, 0, 0, 0), "none"

    mode = config.PAD_MODE
    # numpy's reflect mode requires pad width <= (extent - 1) along that
    # axis; fall back to edge-replication for pathologically small inputs.
    if mode == "reflect" and (
        (pad_bottom > 0 and pad_bottom >= height)
        or (pad_right > 0 and pad_right >= width)
    ):
        mode = config.PAD_MODE_FALLBACK

    padded = np.pad(array, ((0, pad_bottom), (0, pad_right)), mode=mode)
    return padded, (0, pad_bottom, 0, pad_right), mode


def tile_dem(
    input_path: Path,
    output_dir: Path,
    tile_size: int = config.TILE_SIZE,
    overlap: int = config.OVERLAP,
) -> TilingResult:
    """Split a (cleaned + normalized) GeoTIFF into overlapping, padded tiles.

    Parameters
    ----------
    input_path:
        Source GeoTIFF to tile.
    output_dir:
        Directory tiles are written into (created if missing).
    tile_size, overlap:
        See config.py for defaults and their Blueprint provenance.

    Raises
    ------
    TilingError
        If tile_size <= overlap (would never advance) or no tiles could
        be produced.
    """
    if overlap >= tile_size:
        raise TilingError(
            f"overlap ({overlap}) must be smaller than tile_size ({tile_size})."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[TileInfo] = []

    with open_geotiff(input_path) as dataset:
        row_starts = _tile_starts(dataset.height, tile_size, overlap)
        col_starts = _tile_starts(dataset.width, tile_size, overlap)

        for row_index, row_off in enumerate(row_starts):
            for col_index, col_off in enumerate(col_starts):
                valid_height = min(tile_size, dataset.height - row_off)
                valid_width = min(tile_size, dataset.width - col_off)

                window = Window(col_off, row_off, valid_width, valid_height)
                data = dataset.read(1, window=window)
                window_transform = dataset.window_transform(window)
                source_bounds = rasterio.windows.bounds(window, dataset.transform)

                padded_data, padding, pad_mode_used = _pad_to_tile_size(data, tile_size)

                profile = dataset.profile.copy()
                profile.update(
                    height=tile_size,
                    width=tile_size,
                    transform=window_transform,
                )

                tile_name = f"tile_r{row_index:03d}_c{col_index:03d}.tif"
                tile_path = output_dir / tile_name

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(padded_data, 1)

                tiles.append(
                    TileInfo(
                        tile_path=tile_path,
                        row_index=row_index,
                        col_index=col_index,
                        row_offset=row_off,
                        col_offset=col_off,
                        width=tile_size,
                        height=tile_size,
                        valid_width=valid_width,
                        valid_height=valid_height,
                        padding=padding,
                        pad_mode_used=pad_mode_used,
                        transform=tuple(window_transform)[:6],
                        source_bounds=tuple(source_bounds),
                        overlap=overlap,
                    )
                )

    if not tiles:
        raise TilingError(f"No tiles were generated for {input_path}.")

    return TilingResult(
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        tiles=tiles,
        grid_rows=len(row_starts),
        grid_cols=len(col_starts),
    )
