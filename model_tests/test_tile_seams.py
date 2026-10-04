"""Regression: overlap-aware tiling on a KNOWN surface (Part G).

Constructs a synthetic gradient, pushes it through the SAME tiled path
inference uses (iter_tile_windows + OverlapStitcher with cosine blending),
and verifies NUMERICALLY (not visually):
  * full spatial coverage — no NaN holes
  * weighted normalization is exact for a linear surface
  * seam discontinuity at interior tile boundaries is bounded

The max seam step is REPORTED by the test output so the number exists,
independently of the pass/fail assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.tiling import (
    TilingConfig,
    OverlapStitcher,
    iter_tile_windows,
)


def _tile_through(cfg: TilingConfig, surface: np.ndarray) -> np.ndarray:
    """Push `surface` through the tiled path (identity 'model')."""
    h, w = surface.shape
    stitcher = OverlapStitcher(h, w, cfg)
    for win in iter_tile_windows(h, w, cfg):
        tile = np.full((cfg.tile_size, cfg.tile_size), np.nan, np.float32)
        vh, vw = win.valid_height, win.valid_width
        # 'inference': edge-replicate pad when the window overruns the source
        patch = np.pad(
            surface[win.row_off : win.row_off + vh, win.col_off : win.col_off + vw],
            ((0, cfg.tile_size - vh), (0, cfg.tile_size - vw)),
            mode="edge",
        )
        tile[: vh + (cfg.tile_size - vh), : vw + (cfg.tile_size - vw)] = patch
        stitcher.add_tile(win, tile)
    return stitcher.finalize()


def _max_interior_seam_step(out: np.ndarray, cfg: TilingConfig, axis: int):
    """Largest |diff| across tile-boundary lines inside the stitched array."""
    stride = cfg.stride
    limit_h = out.shape[0] if axis == 0 else out.shape[0]
    limit_w = out.shape[1]
    steps = []
    if axis == 0:
        for r in range(stride, out.shape[0] - 1, stride):
            if r + 1 >= out.shape[0]:
                continue
            steps.append(np.abs(out[r + 1] - out[r]).max())
    else:
        for c in range(stride, out.shape[1] - 1, stride):
            if c + 1 >= out.shape[1]:
                continue
            steps.append(np.abs(out[:, c + 1] - out[:, c]).max())
    return max(steps) if steps else 0.0


@pytest.mark.parametrize("cfg", [
    TilingConfig(tile_size=64, overlap=16),
    TilingConfig(tile_size=64, overlap=32),
    TilingConfig(tile_size=128, overlap=64),
])
def test_gradient_surface_stitched_without_seams(cfg):
    h, w = 300, 220
    yy, xx = np.mgrid[0:h, 0:w]
    surface = (0.37 * yy + 1.13 * xx + 5.0).astype(np.float32)

    out = _tile_through(cfg, surface)

    # 1. full coverage: no holes, no fabricated NaNs for a fully valid input
    assert np.isfinite(out).all()
    # 2. the identity 'model' must be reproduced almost exactly
    err = np.abs(out - surface)
    assert err.max() < 1e-3, f"stitched surface deviates: max err {err.max():.4g}"
    # 3. seam discontinuities across tile-boundary lines stay bounded by the
    #    SMOOTH surface's own natural gradient (no stitching spikes)
    natural_step_axis0 = np.abs(np.diff(surface, axis=0)).max()
    natural_step_axis1 = np.abs(np.diff(surface, axis=1)).max()
    seam0 = _max_interior_seam_step(out, cfg, axis=0)
    seam1 = _max_interior_seam_step(out, cfg, axis=1)
    print(f"[seam] cfg={cfg} max seam step axis0={seam0:.4g} "
          f"axis1={seam1:.4g} (natural {natural_step_axis0:.4g}/"
          f"{natural_step_axis1:.4g})")
    assert seam0 <= natural_step_axis0 * 1.01 + 1e-6
    assert seam1 <= natural_step_axis1 * 1.01 + 1e-6


def test_seam_error_reported_numerically_for_noisy_surface():
    """A realistic noisy surface through overlapping tiles: report the seam
    error (max deviation vs the untiled reference) as a NUMBER."""
    cfg = TilingConfig(tile_size=64, overlap=16)
    rng = np.random.default_rng(42)
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w]
    surface = (0.5 * yy + 0.25 * xx).astype(np.float32)
    noisy = surface + rng.normal(0, 0.05, surface.shape).astype(np.float32)

    out = _tile_through(cfg, noisy)
    assert np.isfinite(out).all()
    seam_err = float(np.abs(out - noisy).max())
    print(f"[seam] noisy surface max deviation vs untiled input: "
          f"{seam_err:.4g} m")
    assert seam_err < 0.5  # blending must not amplify pixel noise into seams


def test_overlapping_tiles_match_non_overlapping_global_pass():
    """Stitched overlap path == direct evaluation on the full grid, for a
    NONLINEAR (quadratic) surface — blending must not bias shape."""
    cfg = TilingConfig(tile_size=96, overlap=24)
    h, w = 250, 180
    yy, xx = np.mgrid[0:h, 0:w]
    surface = (0.001 * xx**2 + 0.002 * yy**2 + 0.5).astype(np.float32)

    out = _tile_through(cfg, surface)
    err = np.abs(out - surface)
    assert np.isfinite(out).all()
    print(f"[seam] quadratic surface max stitched error: {err.max():.4g} m")
    assert err.max() < 5e-2
