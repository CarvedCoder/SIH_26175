from pathlib import Path

import pytest

from src.exceptions import (
    InputPathError,
    MissingCRSError,
    UnsupportedFormatError,
)
from src.inspect_geotiff import inspect_geotiff
from src.raster_io import detect_format


def test_inspect_valid_geotiff(tmp_geotiff: Path):
    meta = inspect_geotiff(tmp_geotiff)
    assert meta.width == 300
    assert meta.height == 300
    assert meta.band_count == 1
    assert meta.crs is not None
    assert meta.valid_pixel_count == meta.total_pixels


def test_inspect_reports_invalid_pixel_breakdown(tmp_geotiff_with_invalids: Path):
    meta = inspect_geotiff(tmp_geotiff_with_invalids)
    assert meta.nan_pixel_count == 25
    assert meta.inf_pixel_count == 25
    assert meta.nodata_pixel_count == 25
    assert meta.valid_pixel_count == meta.total_pixels - 75


def test_missing_crs_raises(tmp_geotiff_no_crs: Path):
    with pytest.raises(MissingCRSError):
        inspect_geotiff(tmp_geotiff_no_crs, require_crs=True)


def test_missing_crs_allowed_when_not_required(tmp_geotiff_no_crs: Path):
    meta = inspect_geotiff(tmp_geotiff_no_crs, require_crs=False)
    assert meta.crs is None


def test_invalid_input_path_raises():
    with pytest.raises(InputPathError):
        detect_format(Path("/nonexistent/path/does_not_exist.tif"))


def test_unsupported_extension_raises(tmp_path: Path):
    bogus = tmp_path / "not_a_tif.txt"
    bogus.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        detect_format(bogus)


def test_future_extension_raises_clear_message(tmp_path: Path):
    bogus_png = tmp_path / "future.png"
    bogus_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(UnsupportedFormatError, match="not implemented yet"):
        detect_format(bogus_png)
