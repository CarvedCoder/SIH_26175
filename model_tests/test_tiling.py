"""Tests for depthwizard.tiling — overlapping windows + weighted stitching.

Covers exactly the cases the architecture report's Section 12 promised:
exact-tile-size, smaller-than-tile, larger, non-divisible dimensions,
overlapping windows, edge windows, weighted stitching, NaN handling,
nodata/mask handling, and full coverage (no gaps) at every size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.tiling import (
    OverlapStitcher,
    TilingConfig,
    blend_weight_window,
    iter_tile_windows,
)


# ---------------------------------------------------------------------------
# TilingConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults_match_blueprint_example():
    cfg = TilingConfig()
    assert cfg.tile_size == 1024
    assert cfg.overlap == 256
    assert cfg.stride == 768


def test_config_rejects_overlap_ge_tile_size():
    with pytest.raises(ValueError):
        TilingConfig(tile_size=512, overlap=512)
    with pytest.raises(ValueError):
        TilingConfig(tile_size=512, overlap=600)


def test_config_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError):
        TilingConfig(tile_size=0)
    with pytest.raises(ValueError):
        TilingConfig(tile_size=-10)


def test_config_rejects_negative_overlap():
    with pytest.raises(ValueError):
        TilingConfig(overlap=-1)


def test_config_rejects_unknown_blend():
    with pytest.raises(ValueError):
        TilingConfig(blend="gaussian_but_misspelled")


def test_config_zero_overlap_is_valid_stride_equals_tile_size():
    cfg = TilingConfig(tile_size=256, overlap=0)
    assert cfg.stride == 256


# ---------------------------------------------------------------------------
# iter_tile_windows — coverage across all the required cases
# ---------------------------------------------------------------------------


def _assert_full_coverage(windows, height, width, cfg):
    """Every pixel of (height, width) is covered by at least one window,
    and no window reads past the source when the source >= tile_size."""
    covered = np.zeros((height, width), dtype=bool)
    for w in windows:
        vh, vw = w.valid_height, w.valid_width
        covered[w.row_off : w.row_off + vh, w.col_off : w.col_off + vw] = True
        if height >= cfg.tile_size:
            assert w.row_off + cfg.tile_size <= height
        if width >= cfg.tile_size:
            assert w.col_off + cfg.tile_size <= width
    assert covered.all(), "iter_tile_windows left a gap in coverage"


def test_exact_tile_sized_image_single_window():
    cfg = TilingConfig(tile_size=512, overlap=64)
    windows = iter_tile_windows(512, 512, cfg)
    assert len(windows) == 1
    assert windows[0].row_off == 0 and windows[0].col_off == 0
    assert not windows[0].needs_padding
    _assert_full_coverage(windows, 512, 512, cfg)


def test_image_smaller_than_tile_size_needs_padding():
    cfg = TilingConfig(tile_size=512, overlap=64)
    windows = iter_tile_windows(200, 150, cfg)
    assert len(windows) == 1
    w = windows[0]
    assert w.valid_height == 200 and w.valid_width == 150
    assert w.needs_padding


def test_image_larger_than_tile_produces_overlapping_windows():
    cfg = TilingConfig(tile_size=512, overlap=64)
    windows = iter_tile_windows(1200, 1200, cfg)
    assert len(windows) > 1
    _assert_full_coverage(windows, 1200, 1200, cfg)
    # Consecutive row starts must actually overlap (stride < tile_size).
    row_starts = sorted({w.row_off for w in windows})
    for a, b in zip(row_starts, row_starts[1:]):
        assert b - a < cfg.tile_size


@pytest.mark.parametrize(
    "height,width",
    [
        (1025, 1025),  # one pixel past two tiles
        (1500, 2049),  # asymmetric, non-divisible on both axes
        (2048, 2048),  # exactly divisible
        (300, 5000),  # extreme aspect ratio
    ],
)
def test_non_divisible_and_asymmetric_dimensions(height, width):
    cfg = TilingConfig(tile_size=1024, overlap=256)
    windows = iter_tile_windows(height, width, cfg)
    _assert_full_coverage(windows, height, width, cfg)


def test_edge_windows_snap_to_exact_boundary_no_overshoot():
    cfg = TilingConfig(tile_size=512, overlap=64)
    windows = iter_tile_windows(1000, 1000, cfg)
    max_row_off = max(w.row_off for w in windows)
    max_col_off = max(w.col_off for w in windows)
    assert max_row_off + cfg.tile_size == 1000
    assert max_col_off + cfg.tile_size == 1000


def test_rejects_nonpositive_dimensions():
    cfg = TilingConfig()
    with pytest.raises(ValueError):
        iter_tile_windows(0, 100, cfg)
    with pytest.raises(ValueError):
        iter_tile_windows(100, -5, cfg)


# ---------------------------------------------------------------------------
# blend_weight_window
# ---------------------------------------------------------------------------


def test_blend_weight_window_shape_and_range():
    w = blend_weight_window(512, 64, blend="cosine")
    assert w.shape == (512, 512)
    assert w.min() > 0.0  # never a hard zero (see _MIN_WEIGHT)
    assert np.isclose(w.max(), 1.0)


def test_blend_weight_window_peaks_at_center_tapers_at_border():
    w = blend_weight_window(512, 128, blend="cosine")
    center = w[256, 256]
    corner = w[0, 0]
    edge_mid = w[0, 256]
    assert center > edge_mid > corner


def test_blend_weight_window_none_is_uniform():
    w = blend_weight_window(256, 64, blend="none")
    assert np.all(w == 1.0)


def test_blend_weight_window_zero_overlap_is_uniform():
    w = blend_weight_window(256, 0, blend="cosine")
    assert np.all(w == 1.0)


def test_blend_weight_window_rejects_unknown_blend():
    with pytest.raises(ValueError):
        blend_weight_window(256, 32, blend="nope")


# ---------------------------------------------------------------------------
# OverlapStitcher — weighted blending, NaN handling, nodata/mask handling
# ---------------------------------------------------------------------------


def test_stitcher_single_tile_passthrough():
    cfg = TilingConfig(tile_size=64, overlap=16)
    windows = iter_tile_windows(64, 64, cfg)
    stitcher = OverlapStitcher(64, 64, cfg)
    pred = np.full((64, 64), 7.0, dtype=np.float32)
    stitcher.add_tile(windows[0], pred)
    out = stitcher.finalize()
    assert np.allclose(out, 7.0)


def test_stitcher_no_seam_at_tile_boundary_for_smooth_signal():
    """A smooth ground-truth ramp, tiled + independently 'predicted' with
    small per-tile noise, must stitch back with no discontinuity at the
    old tile boundary — the exact defect the task describes."""
    cfg = TilingConfig(tile_size=64, overlap=16)
    h = w = 100
    yy, xx = np.mgrid[0:h, 0:w]
    truth = (xx + yy).astype(np.float32)  # smooth linear ramp

    rng = np.random.default_rng(0)
    stitcher = OverlapStitcher(h, w, cfg)
    for window in iter_tile_windows(h, w, cfg):
        y0, x0 = window.row_off, window.col_off
        tile_truth = truth[y0 : y0 + cfg.tile_size, x0 : x0 + cfg.tile_size]
        # Simulate independent per-tile inference noise (this is exactly
        # what makes non-overlapping stitching seam: two tiles disagree
        # at their shared border because each was predicted alone).
        noisy = tile_truth + rng.normal(0, 0.05, size=tile_truth.shape).astype(np.float32)
        padded = np.zeros((cfg.tile_size, cfg.tile_size), dtype=np.float32)
        padded[: noisy.shape[0], : noisy.shape[1]] = noisy
        stitcher.add_tile(window, padded)

    out = stitcher.finalize()
    assert not np.isnan(out).any()
    # Error should track the injected per-pixel noise scale, not spike at
    # the boundary that used to sit at x=48,64,... (non-overlapping grid
    # for tile_size=64 would have placed hard seams every 64px).
    err = np.abs(out - truth)
    assert err.max() < 0.5  # generous bound; a hard seam would be O(1)+


def test_stitcher_nan_prediction_excluded_from_weighting():
    cfg = TilingConfig(tile_size=32, overlap=8, blend="none")
    windows = iter_tile_windows(32, 32, cfg)
    stitcher = OverlapStitcher(32, 32, cfg)
    pred = np.full((32, 32), 3.0, dtype=np.float32)
    pred[5, 5] = np.nan
    stitcher.add_tile(windows[0], pred)
    out = stitcher.finalize()
    assert np.isnan(out[5, 5])  # only tile covering it was NaN there -> honest NaN
    assert out[0, 0] == 3.0


def test_stitcher_nan_tile_does_not_poison_overlapping_valid_tile():
    """Two overlapping tiles cover the same pixel; one is NaN there, the
    other is valid — the valid one must win outright, not average with NaN."""
    cfg = TilingConfig(tile_size=32, overlap=16, blend="cosine")
    stitcher = OverlapStitcher(48, 32, cfg)
    windows = iter_tile_windows(48, 32, cfg)
    assert len(windows) == 2  # rows 0 and 16, both cols 0

    tile_a = np.full((32, 32), 10.0, dtype=np.float32)
    tile_b = np.full((32, 32), 20.0, dtype=np.float32)
    # Overlap region is rows [16, 32) of the source. Make tile_a NaN
    # there so only tile_b contributes at those rows.
    tile_a[16:32, :] = np.nan

    stitcher.add_tile(windows[0], tile_a)
    stitcher.add_tile(windows[1], tile_b)
    out = stitcher.finalize()
    assert np.allclose(out[16:32, :], 20.0)
    assert np.allclose(out[:16, :], 10.0)


def test_stitcher_valid_mask_excludes_nodata_pixels():
    cfg = TilingConfig(tile_size=16, overlap=0, blend="none")
    windows = iter_tile_windows(16, 16, cfg)
    stitcher = OverlapStitcher(16, 16, cfg)
    pred = np.full((16, 16), 5.0, dtype=np.float32)
    mask = np.ones((16, 16), dtype=bool)
    mask[0, 0] = False  # simulate a GeoTIFF nodata pixel
    stitcher.add_tile(windows[0], pred, valid_mask=mask)
    out = stitcher.finalize()
    assert np.isnan(out[0, 0])
    assert out[1, 1] == 5.0


def test_stitcher_uncovered_pixel_is_nan_never_fabricated():
    cfg = TilingConfig(tile_size=16, overlap=0)
    stitcher = OverlapStitcher(20, 20, cfg)  # never add any tile
    out = stitcher.finalize()
    assert np.isnan(out).all()


def test_stitcher_rejects_wrong_prediction_shape():
    cfg = TilingConfig(tile_size=32, overlap=8)
    windows = iter_tile_windows(32, 32, cfg)
    stitcher = OverlapStitcher(32, 32, cfg)
    with pytest.raises(ValueError):
        stitcher.add_tile(windows[0], np.zeros((16, 16), dtype=np.float32))


def test_stitcher_matches_non_overlapping_result_when_blend_none_and_overlap_zero():
    """blend='none' + overlap=0 must reproduce a plain non-overlapping
    passthrough (the pre-fix numeric contract for the interior of tiles),
    which the report flags as a needed regression check."""
    cfg = TilingConfig(tile_size=32, overlap=0, blend="none")
    h = w = 64
    stitcher = OverlapStitcher(h, w, cfg)
    windows = iter_tile_windows(h, w, cfg)
    assert len(windows) == 4  # 2x2 grid, no overlap
    expected = np.zeros((h, w), dtype=np.float32)
    for i, window in enumerate(windows):
        val = float(i + 1)
        pred = np.full((cfg.tile_size, cfg.tile_size), val, dtype=np.float32)
        stitcher.add_tile(window, pred)
        y0, x0 = window.row_off, window.col_off
        expected[y0 : y0 + cfg.tile_size, x0 : x0 + cfg.tile_size] = val
    out = stitcher.finalize()
    assert np.array_equal(out, expected)