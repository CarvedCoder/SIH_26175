"""Tests for depthwizard.anchoring — Track-2 absolute DSM (arithmetic, honest).

Pins:
  * DSM = AGL + ground, exactly (constant datum)
  * DEM resampling onto the image grid (rasterio warp, bilinear)
  * REFUSAL semantics: no CRS -> raise; partial coverage -> raise
    (anchoring must fail loudly, never silently produce garbage)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.anchoring import (ANCHORED_LABEL, anchor, anchor_with_constant,
                                   resample_dem_to_tile)


def _write_dem(path: Path, arr: np.ndarray, crs="EPSG:32617") -> None:
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs=crs,
                       transform=from_origin(500000, 4000000, 30.0, 30.0)) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# Constant datum
# ---------------------------------------------------------------------------

def test_anchor_with_constant_is_exact_addition():
    agl = np.array([[0.0, 3.5], [12.0, 7.25]], dtype=np.float32)
    res = anchor_with_constant(agl, 101.5)
    np.testing.assert_allclose(res.dsm, agl + 101.5, atol=1e-6)
    assert res.label == ANCHORED_LABEL == "ANCHORED (not learned)"
    assert res.source.startswith("constant:")


def test_anchor_with_constant_rejects_nan():
    with pytest.raises(ValueError):
        anchor_with_constant(np.zeros((2, 2)), float("nan"))


# ---------------------------------------------------------------------------
# DEM resampling
# ---------------------------------------------------------------------------

def test_resample_dem_same_grid_is_identity(tmp_path):
    dem = np.tile(np.linspace(10, 20, 16, dtype=np.float32), (16, 1))
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 30.0, 30.0),
               "height": 16, "width": 16}
    out = resample_dem_to_tile(tmp_path / "dem.tif", profile)
    np.testing.assert_allclose(out, dem, atol=1e-4)


def test_resample_dem_upsample_shape_matches_tile(tmp_path):
    dem = np.linspace(10, 20, 8, dtype=np.float32).reshape(8, 1)
    dem = np.tile(dem, (1, 8))                     # 8x8 source
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 15.0, 15.0),
               "height": 16, "width": 16}          # 16x16 target grid
    out = resample_dem_to_tile(tmp_path / "dem.tif", profile)
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()


def test_resample_dem_requires_crs_on_tile(tmp_path):
    dem = np.zeros((8, 8), dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": None, "transform": None, "height": 8, "width": 8}
    with pytest.raises(ValueError, match="NO CRS"):
        resample_dem_to_tile(tmp_path / "dem.tif", profile)


def test_resample_dem_requires_crs_on_dem(tmp_path):
    dem = np.zeros((8, 8), dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem, crs=None)
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 30.0, 30.0),
               "height": 8, "width": 8}
    with pytest.raises(ValueError, match="NO CRS"):
        resample_dem_to_tile(tmp_path / "dem.tif", profile)


def test_resample_dem_partial_coverage_refuses(tmp_path):
    """DEM covering only the top-left quarter of the tile -> ValueError."""
    dem = np.zeros((8, 8), dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem)
    # tile extends 4x the DEM footprint at the same resolution
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 30.0, 30.0),
               "height": 16, "width": 16}
    with pytest.raises(ValueError, match="does not fully cover"):
        resample_dem_to_tile(tmp_path / "dem.tif", profile)


# ---------------------------------------------------------------------------
# Full anchor dispatch
# ---------------------------------------------------------------------------

def test_anchor_dem_addition_on_aligned_grids(tmp_path):
    dem = np.full((8, 8), 42.0, dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 30.0, 30.0),
               "height": 8, "width": 8}
    agl = np.full((8, 8), 5.5, dtype=np.float32)
    res = anchor(agl, tmp_path / "dem.tif", None, profile)
    np.testing.assert_allclose(res.dsm, 47.5, atol=1e-4)
    assert res.source.startswith("dem:")
    assert res.dem_stats["min"] == pytest.approx(42.0)


def test_anchor_constant_beats_none(tmp_path):
    agl = np.ones((4, 4), dtype=np.float32)
    assert anchor(agl, None, None) is None
    res = anchor(agl, None, 10.0)
    np.testing.assert_allclose(res.dsm, 11.0)


def test_anchor_dem_without_profile_raises(tmp_path):
    dem = np.zeros((4, 4), dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem)
    with pytest.raises(ValueError, match="profile"):
        anchor(np.ones((4, 4)), tmp_path / "dem.tif", None, None)
