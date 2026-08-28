from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.clean_dem import clean_dem
from src.exceptions import NoValidDataError


def test_clean_dem_marks_invalid_pixels_as_nodata(tmp_geotiff_with_invalids: Path, tmp_path: Path):
    output_dir = tmp_path / "process"
    result = clean_dem(tmp_geotiff_with_invalids, output_dir)

    assert result.invalid_pixel_count == 75
    assert result.output_path.exists()

    with rasterio.open(result.output_path) as dataset:
        cleaned = dataset.read(1)
        assert dataset.nodata == result.nodata_value_used
        # Previously-NaN region is now nodata.
        assert np.all(cleaned[0:5, 0:5] == result.nodata_value_used)
        # Previously-inf region is now nodata.
        assert np.all(cleaned[10:15, 10:15] == result.nodata_value_used)


def test_clean_dem_preserves_valid_values_unchanged(tmp_geotiff_with_invalids: Path, tmp_path: Path):
    with rasterio.open(tmp_geotiff_with_invalids) as src:
        original = src.read(1)

    result = clean_dem(tmp_geotiff_with_invalids, tmp_path / "process")

    with rasterio.open(result.output_path) as dataset:
        cleaned = dataset.read(1)

    valid_region = original[100:150, 100:150]
    cleaned_region = cleaned[100:150, 100:150]
    np.testing.assert_array_equal(valid_region, cleaned_region)


def test_clean_dem_preserves_crs_and_transform(tmp_geotiff: Path, tmp_path: Path):
    result = clean_dem(tmp_geotiff, tmp_path / "process")
    with rasterio.open(tmp_geotiff) as src, rasterio.open(result.output_path) as dst:
        assert src.crs == dst.crs
        assert src.transform == dst.transform


def test_clean_dem_all_invalid_raises(tmp_geotiff_all_nodata: Path, tmp_path: Path):
    with pytest.raises(NoValidDataError):
        clean_dem(tmp_geotiff_all_nodata, tmp_path / "process")
