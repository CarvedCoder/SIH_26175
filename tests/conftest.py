"""Shared pytest fixtures: small synthetic rasters, not the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_raster(
    path: Path,
    array: np.ndarray,
    nodata: float | None,
    crs: str = "EPSG:32632",
) -> Path:
    transform = from_origin(500000, 5600000, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "dtype": str(array.dtype),
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture
def tmp_geotiff(tmp_path: Path) -> Path:
    """A clean 300x300 float32 GeoTIFF with a valid CRS and no invalid pixels."""
    rng = np.random.default_rng(42)
    array = (rng.random((300, 300)) * 100).astype(np.float32)
    return _write_raster(tmp_path / "clean_300.tif", array, nodata=-9999.0)


@pytest.fixture
def tmp_geotiff_with_invalids(tmp_path: Path) -> Path:
    """A 300x300 float32 GeoTIFF with NaN, inf, and nodata pixels mixed in."""
    rng = np.random.default_rng(7)
    array = (rng.random((300, 300)) * 100).astype(np.float32)
    array[0:5, 0:5] = np.nan
    array[10:15, 10:15] = np.inf
    array[20:25, 20:25] = -9999.0
    return _write_raster(tmp_path / "invalid_300.tif", array, nodata=-9999.0)


@pytest.fixture
def tmp_geotiff_no_crs(tmp_path: Path) -> Path:
    """A GeoTIFF with no CRS defined."""
    array = np.ones((50, 50), dtype=np.float32) * 10.0
    path = tmp_path / "no_crs.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": 50,
        "width": 50,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture
def tmp_geotiff_all_nodata(tmp_path: Path) -> Path:
    """A GeoTIFF where every pixel is nodata."""
    array = np.full((50, 50), -9999.0, dtype=np.float32)
    return _write_raster(tmp_path / "all_nodata.tif", array, nodata=-9999.0)


@pytest.fixture
def tmp_geotiff_small(tmp_path: Path) -> Path:
    """A raster smaller than the default tile size (512) in both dimensions."""
    rng = np.random.default_rng(1)
    array = (rng.random((200, 150)) * 50).astype(np.float32)
    return _write_raster(tmp_path / "small_200x150.tif", array, nodata=-9999.0)


@pytest.fixture
def tmp_geotiff_non_divisible(tmp_path: Path) -> Path:
    """A raster whose dimensions do not divide evenly by the tile size."""
    rng = np.random.default_rng(2)
    array = (rng.random((900, 700)) * 50).astype(np.float32)
    return _write_raster(tmp_path / "non_divisible_900x700.tif", array, nodata=-9999.0)


@pytest.fixture
def tmp_geotiff_exact(tmp_path: Path) -> Path:
    """A raster whose dimensions are exactly divisible by the tile size (512)."""
    rng = np.random.default_rng(3)
    array = (rng.random((1024, 512)) * 50).astype(np.float32)
    return _write_raster(tmp_path / "exact_1024x512.tif", array, nodata=-9999.0)
