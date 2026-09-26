"""Native raster acceleration with a pure-Python fallback.

The C++ SIMD kernels (native/, built via `python native/build.py` or
cmake) implement the measured CPU hotspots of the serving stack:

    * fill_invalid   — iterative neighbour-mean NaN/Inf fill (the #1
      hotspot: runs on every height-tile request and every viewer-layer
      PNG generation; 0.6 s on a 2540x2180 DSM in NumPy)
    * stretch_to_01  — affine normalization to [0, 1] (every layer PNG)
    * quantize_u8    — [0, 1] float -> uint8 PNG samples

CONTRACT: the functions here are drop-in identical to the pure-Python
implementations below — same inputs, same outputs (bit-for-bit for
fill_invalid on any input that converges before the 64-iteration cap;
the pathological cap case can differ in <float32-epsilon on leftover
cells because numpy's np.mean uses pairwise summation). The application
MUST work when the native extension is unavailable: if the module is
missing or fails to load, the pure-Python path below is used verbatim.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_native = None
_native_state = "unprobed"  # "ok" | "unavailable" | "unprobed"


def _try_import_native():
    """Locate the built dw_native extension (native/build/ in the repo,
    or anywhere on sys.path). Never raises."""
    global _native, _native_state
    if _native_state == "ok":
        return _native
    if _native_state == "unavailable":
        return None
    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[3]
    native_build = repo_root / "native" / "build"
    if native_build.is_dir():
        candidates.extend(sorted(native_build.glob("dw_native*.pyd")))
        candidates.extend(sorted(native_build.glob("dw_native*.so")))
    import sys

    for cand in candidates:
        if str(cand.parent) not in sys.path:
            sys.path.insert(0, str(cand.parent))
    try:
        import dw_native as mod  # type: ignore

        _native = mod
        _native_state = "ok"
        logger.info("native raster kernels loaded: %s", mod.__file__)
    except Exception as exc:  # pragma: no cover — depends on build presence
        _native = None
        _native_state = "unavailable"
        logger.info(
            "native raster kernels unavailable (%s) — using the pure-Python "
            "fallback (identical numerics, slower)",
            exc,
        )
    return _native


def native_available() -> bool:
    """True when the C++ kernels loaded. Diagnostics only — callers must
    not branch numerics on this (both paths are equivalent)."""
    return _try_import_native() is not None


def fill_invalid(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf samples with the mean of their finite 8-neighbours,
    iterating until filled (crater-free hole filling without scipy).

    Returns a NEW float32 array; the input is never modified.
    """
    mod = _try_import_native()
    if mod is not None:
        filled, _iterations = mod.fill_invalid(
            np.ascontiguousarray(arr, dtype=np.float32)
        )
        return filled

    # ── Pure-Python fallback (the original NumPy implementation) ──
    mask = ~np.isfinite(arr)
    if not mask.any():
        return np.asarray(arr, dtype=np.float32).copy()
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    remaining = mask
    # Each pass fills every hole adjacent to a finite pixel; hole depth
    # shrinks from both sides, so the cap is never hit in practice.
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
    # Any survivor (an all-invalid raster edge region) gets the global mean
    if remaining.any():
        valid = out[np.isfinite(out)]
        out[remaining] = valid.mean() if valid.size else 0.0
    return out


def stretch_to_01(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Affine stretch (arr - lo) / (hi - lo), clamped to [0, 1];
    non-finite inputs map to 0."""
    mod = _try_import_native()
    if mod is not None:
        return mod.stretch_to_01(
            np.ascontiguousarray(arr, dtype=np.float32),
            float(lo),
            float(hi),
        )
    out = np.zeros(arr.shape, dtype=np.float32)
    valid = np.isfinite(arr)
    span = hi - lo
    if abs(span) > 1e-9:
        out[valid] = np.clip(
            (arr[valid].astype(np.float32) - lo) / span, 0.0, 1.0
        )
    return out


def quantize_u8(arr: np.ndarray) -> np.ndarray:
    """[0, 1] floats -> uint8 display samples (x255, rounded)."""
    mod = _try_import_native()
    if mod is not None:
        return mod.quantize_u8(np.ascontiguousarray(arr, dtype=np.float32))
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
