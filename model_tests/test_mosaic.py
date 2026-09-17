"""Tests for depthwizard.mosaic — adjacent GeoTIFFs -> one contiguous raster.

Covers: correct merged bounds/dimensions/transform, CRS preservation,
nodata handling, and every rejection path (CRS mismatch, resolution
mismatch, non-georeferenced input, scattered tiles, single file).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.mosaic import MosaicError, mosaic_rasters, write_mosaic


def _write_tif(path, *, left, top, width=64, height=64, res=0.5, value=0.0,
               crs="EPSG:32617", dtype="float32", nodata=None):
    """Tiny GeoTIFF placed at (left, top) with the given resolution."""
    data = np.full((1, height, width), value, dtype=dtype)
    profile = dict(
        driver="GTiff", height=height, width=width, count=1, dtype=dtype,
        crs=crs, transform=from_origin(left, top, res, res), nodata=nodata,
        compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


@pytest.fixture()
def adjacent_pair(tmp_path):
    """Two 32m x 32m tiles side by side: A [500000, 4000000], B to its east."""
    a = _write_tif(tmp_path / "a.tif", left=500000.0, top=4000000.0, value=1.0)
    b = _write_tif(tmp_path / "b.tif", left=500032.0, top=4000000.0, value=2.0)
    return a, b


def test_adjacent_pair_merges_into_one_raster(adjacent_pair):
    a, b = adjacent_pair
    result = mosaic_rasters(list(adjacent_pair))
    assert result.data.shape == (1, 64, 128)  # two 32m @ 0.5 m/px tiles side by side
    assert result.profile["width"] == 128 and result.profile["height"] == 64
    # CRS + resolution preserved from the inputs
    assert result.profile["crs"].to_string() == "EPSG:32617"
    assert abs(result.profile["transform"].a) == pytest.approx(0.5)
    # union bounds: A's left .. B's right
    with rasterio.open(a) as ds:
        left_a = ds.bounds.left
    with rasterio.open(b) as ds:
        right_b = ds.bounds.right
    assert result.profile["transform"].c == pytest.approx(left_a)
    # written output round-trips with the same geometry
    out = write_mosaic(result, Path(str(a) + ".mosaic.tif"))
    with rasterio.open(out) as ds:
        assert ds.width == 128 and ds.height == 64
        assert ds.crs.to_string() == "EPSG:32617"
        assert ds.bounds.right == pytest.approx(right_b)


def test_overlapping_inputs_feather_smoothly_in_overlap(tmp_path):
    """Overlapping (not just adjacent) tiles merge with a FEATHERED blend:
    the overlap band ramps smoothly from A's value to B's value instead of
    the old hard last-wins cut, which showed as a seam in the projected
    RGB texture. Single-source regions stay pixel-exact (normalisation)."""
    a = _write_tif(tmp_path / "a.tif", left=0.0, top=100.0, value=1.0)
    b = _write_tif(tmp_path / "b.tif", left=16.0, top=100.0, value=2.0)
    result = mosaic_rasters([a, b])
    assert result.profile["width"] == 96  # 64 + 64 - 32 overlap

    row = result.data[0, 32, :].astype(np.float64)  # mid-height transect
    # A-only region: exact source value (weight normalises back to 1)
    assert row[4] == pytest.approx(1.0, abs=1e-6)
    # B-only region: exact source value
    assert row[-5] == pytest.approx(2.0, abs=1e-6)
    # The overlap band (cols 32..64) is a smooth ramp: its interior is
    # strictly between the two sources, and no adjacent-column jump is
    # larger than a fraction of the total 1.0 step (the old hard cut
    # jumped 1.0 at a single pixel boundary)
    assert np.all(row[36:60] > 1.0) and np.all(row[36:60] < 2.0)
    assert np.abs(np.diff(row[30:66])).max() < 0.2


def test_nodata_gaps_are_explicit_not_interpolated(tmp_path):
    """A float mosaic with an uncovered corner reports -9999 nodata in the
    profile — coverage gaps stay detectable, never interpolated away."""
    a = _write_tif(tmp_path / "a.tif", left=0.0, top=100.0, width=64, height=32, value=1.0)
    b = _write_tif(tmp_path / "b.tif", left=32.0, top=100.0, width=64, height=32, value=2.0)
    result = mosaic_rasters([a, b])
    assert result.profile["nodata"] == -9999.0
    # rows 32..64 are covered by neither tile
    assert (result.data[0, 32:, :] == -9999.0).all()
    # and an explicit input nodata is carried through unchanged
    c = _write_tif(tmp_path / "c.tif", left=0.0, top=100.0, nodata=0.0, value=0.0)
    d = _write_tif(tmp_path / "d.tif", left=32.0, top=100.0, nodata=0.0, value=3.0)
    result2 = mosaic_rasters([c, d])
    assert result2.profile["nodata"] == 0.0


def test_rejects_crs_mismatch(tmp_path):
    a = _write_tif(tmp_path / "a.tif", left=0.0, top=100.0, crs="EPSG:32617")
    b = _write_tif(tmp_path / "b.tif", left=32.0, top=100.0, crs="EPSG:4326")
    with pytest.raises(MosaicError, match="CRS mismatch"):
        mosaic_rasters([a, b])


def test_rejects_resolution_mismatch_instead_of_resizing(tmp_path):
    a = _write_tif(tmp_path / "a.tif", left=0.0, top=100.0, res=0.5)
    b = _write_tif(tmp_path / "b.tif", left=32.0, top=100.0, res=0.25)
    with pytest.raises(MosaicError, match="resolution mismatch"):
        mosaic_rasters([a, b])


def test_rejects_non_georeferenced_input(tmp_path):
    pass

    from PIL import Image

    png = tmp_path / "plane.png"
    Image.new("RGB", (64, 64)).save(png, format="PNG")
    geo = _write_tif(tmp_path / "geo.tif", left=0.0, top=100.0)
    with pytest.raises(MosaicError, match="no CRS"):
        mosaic_rasters([png, geo])


def test_rejects_scattered_tiles_with_gaps(tmp_path):
    a = _write_tif(tmp_path / "a.tif", left=0.0, top=100.0)
    b = _write_tif(tmp_path / "b.tif", left=1000.0, top=100.0)  # 468 m gap
    with pytest.raises(MosaicError, match="does not touch or overlap"):
        mosaic_rasters([a, b])


def test_rejects_single_file_and_missing_file(tmp_path):
    only = _write_tif(tmp_path / "only.tif", left=0.0, top=100.0)
    with pytest.raises(MosaicError, match="at least 2"):
        mosaic_rasters([only])
    with pytest.raises(MosaicError, match="not found"):
        mosaic_rasters([only, tmp_path / "ghost.tif"])


def test_rejects_band_count_mismatch(tmp_path):
    a = _write_tif(tmp_path / "a.tif", left=0.0, top=100.0)
    profile = dict(
        driver="GTiff", height=64, width=64, count=3, dtype="uint8",
        crs="EPSG:32617", transform=from_origin(32.0, 100.0, 0.5, 0.5),
    )
    b = tmp_path / "rgb.tif"
    with rasterio.open(b, "w", **profile) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    with pytest.raises(MosaicError, match="band count mismatch"):
        mosaic_rasters([a, b])
