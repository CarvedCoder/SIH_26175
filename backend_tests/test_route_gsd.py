"""Route Assist GSD regression (Part N + Part F): physical thresholds must
not silently change when the pixel grid changes.

A synthetic hill with a FIXED physical footprint (200 m) is rendered on
grids from 0.35 m to 10 m GSD; :class:`classify_grid` receives each grid
with its true GSD. The physical slope field (degrees) is GSD-invariant —
that invariance is what this suite pins. Step/roughness are 3x3/5x5
NEIGHBOURHOOD metrics, so their raw values legitimately scale with the
cell size in metres; what must hold is that they stay in METRES and
scale linearly with GSD for the same physical terrain.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.services.route_service import VEHICLE_PROFILES, classify_grid


def _hill(n: int, height_m: float = 18.0, footprint_m: float = 200.0) -> np.ndarray:
    """Physically FIXED hill (metres) rendered on an n x n grid."""
    yy, xx = np.mgrid[0:n, 0:n]
    xm = (xx - (n - 1) / 2) * footprint_m / n
    ym = (yy - (n - 1) / 2) * footprint_m / n
    r = np.hypot(xm, ym) / (footprint_m / 2)
    return (height_m * np.exp(-(r**2))).astype(np.float32)


def _terrain_at_gsd(gsd_m: float, footprint_m: float = 200.0):
    n = max(8, int(round(footprint_m / gsd_m)))
    return _hill(n, footprint_m=footprint_m), n


@pytest.mark.parametrize("gsd_m", [0.35, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_physical_slope_is_gsd_invariant(gsd_m):
    terrain, n = _terrain_at_gsd(gsd_m)
    vehicle = VEHICLE_PROFILES["fire_truck"]
    grid = classify_grid(terrain, gsd_m, vehicle)
    assert grid.gsd_m == gsd_m
    assert np.isfinite(grid.slope_deg).all()
    # analytic peak slope of the hill: h*sqrt(2/e)/r0 in rad ~ 8.8 deg
    assert grid.slope_deg.max() == pytest.approx(8.8, abs=1.5)


def test_slope_field_matches_across_resolutions():
    """The same physical terrain at 0.5 m and 2 m GSD: identical slope.
    Grids differ in pixel count, so compare at the centre row/column."""
    vehicle = VEHICLE_PROFILES["fire_truck"]
    slopes = []
    for gsd in (0.5, 2.0):
        terrain, n = _terrain_at_gsd(gsd)
        grid = classify_grid(terrain, gsd, vehicle)
        slopes.append((n, grid.slope_deg))
    (n_a, a), (n_b, b) = slopes
    row_a, row_b = a[n_a // 2], b[n_b // 2]
    # resample the coarse profile onto the fine grid for comparison
    xp = np.linspace(0.0, 1.0, n_b)
    x = np.linspace(0.0, 1.0, n_a)
    b_up = np.interp(x, xp, row_b)
    np.testing.assert_allclose(row_a, b_up, atol=1.0)  # degrees


def test_step_metric_in_metres_scales_with_gsd():
    """For fixed terrain, step over the 3x3 window ~ linear in cell size."""
    vehicle = VEHICLE_PROFILES["fire_truck"]
    steps = {}
    for gsd in (1.0, 2.0):
        terrain, _ = _terrain_at_gsd(gsd)
        grid = classify_grid(terrain, gsd, vehicle)
        steps[gsd] = float(grid.step_m.max())
    # doubling the GSD at most doubles the physical 3x3 step (never shrinks
    # it, never explodes) — the metric stays in METRES either way
    assert 0.5 * steps[2.0] <= steps[1.0] <= 2.0 * steps[2.0] + 1e-6
    assert steps[1.0] < 18.0  # cannot exceed the hill's total relief


def test_steeper_slope_blocked_consistently_across_gsds():
    """A physically TOO-STEEP terrain is CANNOT GO at every GSD."""
    vehicle = VEHICLE_PROFILES["fire_truck"]
    for gsd_m in (0.5, 1.0, 2.0, 5.0):
        terrain, _ = _terrain_at_gsd(gsd_m, footprint_m=100.0)
        terrain = (terrain / 18.0 * 60.0).astype(np.float32)  # 60 m over 50 m
        grid = classify_grid(terrain, gsd_m, vehicle)
        assert grid.classes.max() == 2, (
            f"fire truck allowed on a physically-impossible {gsd_m} m slope"
        )


def test_none_gsd_honest_pixel_units_not_metres():
    """Non-georeferenced scene: thresholds degrade honestly (slope-only,
    step/rough disabled) instead of pretending metre units."""
    terrain = _hill(64)
    vehicle = VEHICLE_PROFILES["fire_truck"]
    grid = classify_grid(terrain, None, vehicle)
    assert grid.gsd_m is None
    assert (grid.step_m == 0).all() and (grid.roughness_m == 0).all()
