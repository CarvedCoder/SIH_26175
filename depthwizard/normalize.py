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


# Log-stats epsilon: raw DAv2 relative depth is strictly positive (observed
# range ~1-6 on GAMUS/DFC tiles), but eps keeps the log finite for any
# degenerate window (all-zero crops, edge-padded strips).
_DN_STATS_EPS = 1e-3


def dn_tile_stats(raw: np.ndarray, eps: float = _DN_STATS_EPS) -> np.ndarray:
    """The 4-scalar RAW-tile conditioning vector for FiLM statistics (Exp 1):

        [ log(raw_min + eps), log(raw_max + eps),
          log(raw_max - raw_min + eps), log(raw_mean + eps) ]

    Computed from the SAME raw array minmax_normalize normalizes (the exact
    statistics that per-tile normalization DISCARDS). Dihedral- and
    crop-invariant by construction at fixed granularity; recompute per
    normalization unit (per training tile / per inference window), never
    from an already-normalized array (log of [0,1] values is a different,
    wrong signal). Returns float32 [4].
    """
    raw = raw.astype(np.float32, copy=False)
    lo = float(np.nanmin(raw))
    hi = float(np.nanmax(raw))
    mean = float(np.nanmean(raw))
    return np.array(
        [
            np.log(lo + eps),
            np.log(hi + eps),
            np.log(hi - lo + eps),
            np.log(mean + eps),
        ],
        dtype=np.float32,
    )


def minmax_normalize_with_stats(
    raw: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """(minmax_normalize(raw), dn_tile_stats(raw)) — one call for callers
    that need BOTH the normalized tile and the discarded raw statistics
    (FiLM conditioning, Exp 1). The stats eps is independent of the
    normalization eps on purpose (see dn_tile_stats)."""
    return minmax_normalize(raw, eps=eps), dn_tile_stats(raw)


def valid_target_mask(agl: np.ndarray) -> np.ndarray:
    """Pixels eligible for metric evaluation: finite AGL values."""
    return np.isfinite(agl)
