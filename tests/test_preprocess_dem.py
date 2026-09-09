from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.clean_dem import clean_dem
from src.exceptions import NoValidDataError, NormalizationError
from src.preprocess_dem import NORMALIZED_NODATA, preprocess_dem


def test_preprocess_minmax_normalizes_to_unit_range(tmp_geotiff: Path, tmp_path: Path):
    cleaning_result = clean_dem(tmp_geotiff, tmp_path / "process")
    result = preprocess_dem(cleaning_result.output_path, tmp_path / "process", method="minmax")

    assert 0.0 <= result.normalized_min <= result.normalized_max <= 1.0

    with rasterio.open(result.output_path) as dataset:
        assert dataset.dtypes[0] == "float32"
        assert dataset.nodata == NORMALIZED_NODATA


def test_preprocess_excludes_nodata_from_normalization(
    tmp_geotiff_with_invalids: Path, tmp_path: Path
):
    cleaning_result = clean_dem(tmp_geotiff_with_invalids, tmp_path / "process")
    result = preprocess_dem(cleaning_result.output_path, tmp_path / "process")

    with rasterio.open(result.output_path) as dataset:
        normalized = dataset.read(1)
        # Cleaned nodata pixels must map to the normalized nodata sentinel,
        # not to a value inside [0, 1].
        assert np.all(normalized[0:5, 0:5] == NORMALIZED_NODATA)


def test_preprocess_records_reversible_params(tmp_geotiff: Path, tmp_path: Path):
    cleaning_result = clean_dem(tmp_geotiff, tmp_path / "process")
    result = preprocess_dem(cleaning_result.output_path, tmp_path / "process")

    assert result.params.method == "minmax"
    assert result.params.source_max > result.params.source_min


def test_preprocess_flat_raster_raises(tmp_path: Path):
    flat = np.full((50, 50), 5.0, dtype=np.float32)
    path = tmp_path / "flat.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        count=1,
        height=50,
        width=50,
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(flat, 1)

    with pytest.raises(NormalizationError):
        preprocess_dem(path, tmp_path / "process")
