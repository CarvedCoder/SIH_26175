"""Regression tests for the native acceleration layer + Python fallback.

The contract: backend.app.services.native_accel's functions are drop-in
identical to the pure-Python implementations — same numerics whether or
not the C++ extension (native/build/dw_native) is loadable.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.services import native_accel


def reference_fill_invalid(arr: np.ndarray) -> np.ndarray:
    """The ORIGINAL NumPy implementation (kept verbatim as the oracle)."""
    mask = ~np.isfinite(arr)
    if not mask.any():
        return np.asarray(arr, dtype=np.float32).copy()
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    remaining = mask
    for _ in range(64):
        if not remaining.any():
            break
        filled_sum = np.zeros_like(out)
        filled_count = np.zeros_like(out, dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.full_like(out, np.nan)
                ys = slice(max(0, -dy), out.shape[0] - max(0, dy))
                xs = slice(max(0, -dx), out.shape[1] - max(0, dx))
                ys_src = slice(max(0, dy), out.shape[0] - max(0, -dy))
                xs_src = slice(max(0, dx), out.shape[1] - max(0, -dx))
                shifted[ys, xs] = out[ys_src, xs_src]
                finite = np.isfinite(shifted)
                filled_sum[finite] += np.nan_to_num(shifted[finite])
                filled_count += finite.astype(np.int32)
        fillable = remaining & (filled_count > 0)
        out[fillable] = filled_sum[fillable] / filled_count[fillable]
        remaining = ~np.isfinite(out)
    if remaining.any():
        valid = out[np.isfinite(out)]
        out[remaining] = valid.mean() if valid.size else 0.0
    return out


@pytest.fixture()
def force_python_fallback(monkeypatch):
    """Force the pure-Python path regardless of native availability."""
    monkeypatch.setattr(native_accel, "_native", None)
    monkeypatch.setattr(native_accel, "_native_state", "unavailable")
    yield


@pytest.mark.parametrize(
    "holes",
    [
        "single",
        "block",
        "random",
        "edges",
        "none",
    ],
)
def test_fill_invalid_matches_reference_numpy(holes):
    rng = np.random.default_rng(42)
    dsm = rng.uniform(0, 18, (97, 121)).astype(np.float32)
    if holes == "single":
        dsm[50, 60] = np.nan
    elif holes == "block":
        dsm[20:30, 40:60] = np.nan
    elif holes == "random":
        dsm[rng.random(dsm.shape) < 0.01] = np.nan
    elif holes == "edges":
        dsm[0, :] = np.nan
        dsm[-1, :] = np.nan
        dsm[:, 0] = np.inf

    expected = reference_fill_invalid(dsm.copy())
    actual = native_accel.fill_invalid(dsm)
    assert np.array_equal(expected, actual)
    # The input must never be mutated.
    if holes == "none":
        assert np.isfinite(dsm).all()


def test_fill_invalid_all_invalid_grid():
    out = native_accel.fill_invalid(np.full((4, 5), np.nan, dtype=np.float32))
    assert not np.isnan(out).any()
    assert np.all(out == 0.0)


def test_fill_invalid_fallback_when_native_unavailable(force_python_fallback):
    rng = np.random.default_rng(7)
    dsm = rng.uniform(0, 10, (64, 64)).astype(np.float32)
    dsm[10:14, 20:24] = np.nan
    expected = reference_fill_invalid(dsm.copy())
    actual = native_accel.fill_invalid(dsm)
    assert np.array_equal(expected, actual)
    assert native_accel.native_available() is False


def test_stretch_to_01_matches_numpy_and_handles_nan():
    rng = np.random.default_rng(3)
    arr = rng.uniform(-5, 25, (33, 44)).astype(np.float32)
    arr[5, 5] = np.nan
    arr[6, 6] = np.inf

    expected = np.clip((arr - 2.0) / (16.0 - 2.0), 0, 1)
    expected[~np.isfinite(arr)] = 0.0  # NaN AND Inf map to 0 (fallback semantics)
    actual = native_accel.stretch_to_01(arr, 2.0, 16.0)
    assert np.allclose(expected, actual, atol=1e-6)


def test_quantize_u8_matches_numpy():
    rng = np.random.default_rng(5)
    arr = rng.uniform(-0.2, 1.2, (17, 23)).astype(np.float32)
    expected = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    actual = native_accel.quantize_u8(arr)
    assert np.array_equal(expected, actual)


def test_terrain_service_uses_accelerated_fill_with_fallback(force_python_fallback):
    from backend.app.services.terrain_service import TerrainService

    rng = np.random.default_rng(9)
    dsm = rng.uniform(0, 18, (32, 32)).astype(np.float32)
    dsm[3:6, 3:6] = np.nan
    filled = TerrainService._fill_invalid(dsm)
    assert np.array_equal(filled, reference_fill_invalid(dsm.copy()))


@pytest.mark.skipif(
    not native_accel.native_available(),
    reason="native dw_native extension not built (python native/build.py)",
)
def test_native_module_loads_and_agrees():
    """When built, the C++ kernels must agree exactly with the reference."""
    assert native_accel.native_available()
    rng = np.random.default_rng(11)
    dsm = rng.uniform(0, 18, (64, 80)).astype(np.float32)
    dsm[10:20, 30:40] = np.nan
    expected = reference_fill_invalid(dsm.copy())
    actual = native_accel.fill_invalid(dsm)
    assert np.array_equal(expected, actual)
