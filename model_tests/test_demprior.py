"""Tests for depthwizard.demprior — the Method-D DEM conditioning utilities.

Mirrors the test style of ``tests/test_anchoring.py``: toy arrays, exact
assertions, extent-mismatch raises, every output tagged with the literal
SYNTHETIC-DEM-PROXY string when the DEM was synthesised from AGL.

Honesty contract pinned here:

  * ``resample_dem_to_tile`` is the SAME function ``anchoring.py`` uses —
    we re-import it through demprior so consumers depend on one
    implementation, not two. The CRS-gate and full-coverage-or-raise
    behavior tested in ``tests/test_anchoring.py`` applies verbatim here.
  * ``synth_dem_from_agl`` MUST tag its output via the
    ``SYNTHETIC_DEM_TAG`` constant. Every consumer of a synthetic DEM
    MUST surface that tag in its outputs / worklog lines (the contract
    documented in demprior.py). The test suite asserts the tag exists as
    a module constant so callers can grep for it.
  * A constant-AGL input MUST produce a constant DEM (the Gaussian of a
    constant is the constant — the trivial case).
  * A spike AGL (one cell with a tall building; rest zero) MUST be
    smoothed away — the DEM at the spike location MUST be strictly less
    than the AGL at that location (low-pass characteristic).
  * GSD conversion: passing a real ``gsd_m`` MUST shrink the effective
    pixel-space kernel (sigma_px = sigma_m / gsd_m); passing ``None``
    MUST assume 1 m / px and the caller MUST carry the SYNTHETIC-DEM-PROXY
    tag in their output (which the dataset does automatically).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard import demprior
from depthwizard.anchoring import resample_dem_to_tile as anchoring_resample
from depthwizard.demprior import (SYNTHETIC_DEM_TAG, synth_dem_from_agl,
                                  resample_dem_to_tile as demprior_resample)


# ---------------------------------------------------------------------------
# SYNTHETIC_DEM_TAG — the contract every output / worklog line MUST carry
# when the DEM was synthesised rather than read from a real DEM file.
# ---------------------------------------------------------------------------

def test_synthetic_dem_tag_is_literal_string():
    """The tag must be exactly the literal string 'SYNTHETIC-DEM-PROXY' so
    downstream consumers (worklog, train log, checkpoint metadata) can grep
    for it."""
    assert SYNTHETIC_DEM_TAG == "SYNTHETIC-DEM-PROXY"


def test_demprior_resample_dem_is_same_function_as_anchoring():
    """demprior.resample_dem_to_tile MUST be the same callable as
    anchoring.resample_dem_to_tile — single source of truth. If they ever
    diverge, this test catches the silent-misalignment bug class."""
    assert demprior_resample is anchoring_resample


# ---------------------------------------------------------------------------
# synth_dem_from_agl — pure-numpy ablation
# ---------------------------------------------------------------------------

def test_synth_dem_constant_agl_is_constant():
    """Gaussian low-pass of a constant field is the constant itself."""
    agl = np.full((32, 32), 5.0, dtype=np.float32)
    out = synth_dem_from_agl(agl, sigma_m=2.0)
    assert out.shape == agl.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 5.0, atol=1e-5)


def test_synth_dem_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    agl = rng.random((64, 48)).astype(np.float32) * 10
    out = synth_dem_from_agl(agl, sigma_m=4.0)
    assert out.shape == (64, 48)
    assert out.dtype == np.float32


def test_synth_dem_smooths_away_spike():
    """A one-cell spike (tall building) MUST be suppressed — the DEM at
    that location MUST be strictly less than the AGL there.

    Hand reasoning: with sigma_m=4 m and gsd_m=1 m/px, sigma_px=4 px,
    radius=12 px, kernel size 25x25. The spike's energy is spread across
    ~625 cells -> the spike value at the spike location drops by at
    least 99% (the Gaussian weight at distance 0 is 1; the sum over the
    kernel is ~ 2*pi*sigma^2 ~ 100, so each cell holds ~1% of the spike).
    """
    agl = np.zeros((33, 33), dtype=np.float32)        # 33 so the spike sits
                                                       # exactly at the center
                                                       # of a 25x25 kernel
    agl[16, 16] = 30.0                                 # tall building
    out = synth_dem_from_agl(agl, sigma_m=4.0, gsd_m=1.0)
    # the smoothed value at the spike MUST be less than the spike
    assert out[16, 16] < 5.0, (
        f"DEM at spike location {out[16, 16]:.3f} should be < 5.0 — "
        "the spike was NOT smoothed away; the Gaussian kernel is broken")
    # the smoothed value must be strictly positive (the spike energy is
    # spread to the rest of the array)
    assert out[16, 16] > 0.1


def test_synth_dem_gsd_m_shrinks_kernel_proportionally():
    """With gsd_m=2 m/px, sigma_m=4 m -> sigma_px=2 px (half the kernel
    of sigma_m=4 m at gsd_m=1 m/px). The spike should be LESS smoothed
    (less spread) -> HIGHER peak value at the spike location.

    This pins the metres->pixels conversion rule (worklog Section 4:
    silent-wrong-unit bugs are forbidden here).
    """
    agl = np.zeros((33, 33), dtype=np.float32)
    agl[16, 16] = 30.0
    out_gsd1 = synth_dem_from_agl(agl, sigma_m=4.0, gsd_m=1.0)
    out_gsd2 = synth_dem_from_agl(agl, sigma_m=4.0, gsd_m=2.0)
    # Smaller pixel-space sigma -> less spread -> more energy retained
    # at the spike location.
    assert out_gsd2[16, 16] > out_gsd1[16, 16], (
        f"DEM at spike with gsd=2 ({out_gsd2[16, 16]:.3f}) should be > "
        f"DEM at spike with gsd=1 ({out_gsd1[16, 16]:.3f}) — the metres-"
        "to-pixels sigma conversion is reversed")


def test_synth_dem_rejects_2d_only_inputs():
    """The function MUST refuse 3-D inputs (a common bug: feeding a [1,H,W]
    array from a torch-tensor-to-numpy round trip without squeezing)."""
    with pytest.raises(ValueError, match="2-D"):
        synth_dem_from_agl(np.zeros((1, 16, 16), dtype=np.float32), sigma_m=2.0)


def test_synth_dem_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma"):
        synth_dem_from_agl(np.zeros((16, 16), dtype=np.float32), sigma_m=0)
    with pytest.raises(ValueError, match="sigma"):
        synth_dem_from_agl(np.zeros((16, 16), dtype=np.float32), sigma_m=-1.0)


def test_synth_dem_large_sigma_returns_constant_mean():
    """A sigma larger than the whole tile (pathological case) MUST return
    the AGL mean — the mathematical limit of the Gaussian as sigma -> inf.
    No silent zero-padding."""
    agl = np.full((4, 4), 7.0, dtype=np.float32)
    agl[0, 0] = 100.0
    # sigma_m = 1e6 m, gsd_m = None -> sigma_px = 1e6 px >> 4 px tile ->
    # falls into the "constant mean" path.
    out = synth_dem_from_agl(agl, sigma_m=1e6)
    expected_mean = float(agl.mean())   # (7*15 + 100) / 16 = 12.8125
    np.testing.assert_allclose(out, expected_mean, atol=1e-5)


# ---------------------------------------------------------------------------
# Real-DEM resampling — re-exercised through demprior's re-export
# (the full CRS-gate / coverage suite lives in test_anchoring.py; here we
# only confirm the re-exported function STILL works end-to-end)
# ---------------------------------------------------------------------------

def _write_dem(path: Path, arr: np.ndarray, crs="EPSG:32617") -> None:
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs=crs,
                       transform=from_origin(500000, 4000000, 30.0, 30.0)) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_demprior_resample_dem_same_grid_is_identity(tmp_path):
    """End-to-end smoke: the re-exported resample_dem_to_tile still works
    identically to anchoring's version (CRS-gate, bilinear, full-coverage)."""
    dem = np.tile(np.linspace(10, 20, 16, dtype=np.float32), (16, 1))
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 30.0, 30.0),
               "height": 16, "width": 16}
    out = demprior_resample(tmp_path / "dem.tif", profile)
    np.testing.assert_allclose(out, dem, atol=1e-4)


def test_demprior_resample_dem_partial_coverage_raises(tmp_path):
    """Extent-mismatch raises — the DEM covering only the top-left quarter
    of the tile MUST refuse (never silently zero-pad the missing pixels)."""
    dem = np.zeros((8, 8), dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": "EPSG:32617",
               "transform": from_origin(500000, 4000000, 30.0, 30.0),
               "height": 16, "width": 16}
    with pytest.raises(ValueError, match="does not fully cover"):
        demprior_resample(tmp_path / "dem.tif", profile)


def test_demprior_resample_dem_no_crs_raises(tmp_path):
    """CRS-gate: anchoring-style refusal on a non-georeferenced tile."""
    dem = np.zeros((8, 8), dtype=np.float32)
    _write_dem(tmp_path / "dem.tif", dem)
    profile = {"crs": None, "transform": None, "height": 8, "width": 8}
    with pytest.raises(ValueError, match="NO CRS"):
        demprior_resample(tmp_path / "dem.tif", profile)


# ---------------------------------------------------------------------------
# Module-level surface pin — if these names move, downstream consumers break
# ---------------------------------------------------------------------------

def test_demprior_public_surface_is_stable():
    """Pin the public API names so a refactor that silently renames
    ``synth_dem_from_agl`` or ``SYNTHETIC_DEM_TAG`` breaks the build."""
    assert hasattr(demprior, "synth_dem_from_agl")
    assert callable(demprior.synth_dem_from_agl)
    assert hasattr(demprior, "SYNTHETIC_DEM_TAG")
    assert hasattr(demprior, "resample_dem_to_tile")
    assert callable(demprior.resample_dem_to_tile)
