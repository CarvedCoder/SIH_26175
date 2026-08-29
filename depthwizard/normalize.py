"""Target cleaning + depth normalization — SINGLE SOURCE OF TRUTH.

Consistency rule (from the problem spec, sec. 4):
  * The RAW Depth Anything V2 tensor is preserved on disk (float32 .npy).
  * Dn = min-max normalized raw depth, computed per tile with THESE functions,
    identically at fit time, eval time, and inference time. No other file may
    define its own normalization.
  * AGL is used raw for evaluation bookkeeping and clamped at 0 m for
    training/eval targets (AGL is "above ground level"; the observed
    min ~ -0.28 m is LiDAR/DTM residual noise, not real negative height).
    clamping happens ONLY through clean_agl().
"""

from __future__ import annotations

import numpy as np


def minmax_normalize(raw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-tile min-max normalization of a relative depth map to [0, 1].

    Notes:
      * Depth Anything V2 emits a relative (inverse-depth-like) signal where
        LARGER values = CLOSER to the sensor. For a nadir aerial/satellite
        view that usually correlates POSITIVELY with elevation, but the
        relation is not metric and not guaranteed linear — which is exactly
        why the calibration stage exists.
      * Per-tile normalization discards any cross-tile magnitude information
        the backbone might carry. That is acceptable (and matches your
        JAX_004_006 baseline), but it is a documented decision: revisit only
        if Phase 2 shows the calibration head starved of signal.
    """
    raw = raw.astype(np.float32, copy=False)
    lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
    if hi - lo < eps:
        # Flat depth (e.g., uniform texture-less tile). Return mid-gray.
        return np.full_like(raw, 0.5, dtype=np.float32)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def clean_agl(agl: np.ndarray, clamp_min: float = 0.0) -> np.ndarray:
    """Return the supervised/eval target: finite AGL clamped at ground level."""
    a = np.where(np.isfinite(agl), agl, clamp_min).astype(np.float32, copy=False)
    return np.maximum(a, clamp_min)


def valid_target_mask(agl: np.ndarray) -> np.ndarray:
    """Pixels eligible for metric evaluation: finite AGL values."""
    return np.isfinite(agl)
