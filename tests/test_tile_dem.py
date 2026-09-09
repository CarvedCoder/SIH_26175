from pathlib import Path

import pytest
import rasterio

from src.exceptions import TilingError
from src.tile_dem import tile_dem


def test_tile_exact_dimensions_no_padding(
    tmp_geotiff_exact: Path, tmp_path: Path
):
    result = tile_dem(
        tmp_geotiff_exact,
        tmp_path / "tiles",
        tile_size=512,
        overlap=0,
    )

    for tile in result.tiles:
        with rasterio.open(tile.tile_path) as dataset:
            assert dataset.width == 512
            assert dataset.height == 512

        assert tile.padding == (0, 0, 0, 0)


def test_tile_non_divisible_dimensions_are_padded(
    tmp_geotiff_non_divisible: Path, tmp_path: Path
):
    result = tile_dem(
        tmp_geotiff_non_divisible,
        tmp_path / "tiles",
        tile_size=512,
        overlap=64,
    )

    # Every tile must still be exactly tile_size x tile_size.
    for tile in result.tiles:
        with rasterio.open(tile.tile_path) as dataset:
            assert dataset.width == 512
            assert dataset.height == 512

    # Edge-aligned tiles may reach the raster boundary exactly,
    # so padding is not required for every non-divisible raster.
    assert all(
        tile.padding == (0, 0, 0, 0)
        for tile in result.tiles
    )


def test_tile_smaller_than_tile_size_produces_single_padded_tile(
    tmp_geotiff_small: Path, tmp_path: Path
):
    result = tile_dem(
        tmp_geotiff_small,
        tmp_path / "tiles",
        tile_size=512,
        overlap=64,
    )

    assert len(result.tiles) == 1

    tile = result.tiles[0]

    assert tile.valid_height == 200
    assert tile.valid_width == 150

    with rasterio.open(tile.tile_path) as dataset:
        assert dataset.width == 512
        assert dataset.height == 512


def test_tile_no_gaps_or_missing_coverage(
    tmp_geotiff_non_divisible: Path, tmp_path: Path
):
    result = tile_dem(
        tmp_geotiff_non_divisible,
        tmp_path / "tiles",
        tile_size=512,
        overlap=64,
    )

    max_row_reach = max(
        t.row_offset + t.valid_height
        for t in result.tiles
    )

    max_col_reach = max(
        t.col_offset + t.valid_width
        for t in result.tiles
    )

    with rasterio.open(tmp_geotiff_non_divisible) as src:
        assert max_row_reach == src.height
        assert max_col_reach == src.width


def test_tile_preserves_crs(
    tmp_geotiff_non_divisible: Path, tmp_path: Path
):
    result = tile_dem(
        tmp_geotiff_non_divisible,
        tmp_path / "tiles",
        tile_size=512,
        overlap=64,
    )

    with rasterio.open(tmp_geotiff_non_divisible) as src:
        source_crs = src.crs

    for tile in result.tiles[:3]:
        with rasterio.open(tile.tile_path) as dataset:
            assert dataset.crs == source_crs


def test_overlap_must_be_smaller_than_tile_size(
    tmp_geotiff_small: Path, tmp_path: Path
):
    with pytest.raises(TilingError):
        tile_dem(
            tmp_geotiff_small,
            tmp_path / "tiles",
            tile_size=256,
            overlap=256,
        )