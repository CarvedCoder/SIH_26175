"""Conservative confidence estimation — documented, NOT calibrated.

The repository's model produces no calibrated uncertainty (the ``w_conf``
loss knob is explicitly reserved/unimplemented in depthwizard.losses).
This module therefore constructs a RELATIVE confidence map in [0,1] from
observable deployment-time signals only, and it is used ONLY to modulate
the WLS data term (high confidence -> stay at AGL_raw; low confidence ->
allow stronger spatial refinement).

Signals combined (all multiplicative, each documented):

1. local_stability   1 - normalized |x - local median| — how much a pixel
                     deviates from its robust neighbourhood consensus.
                     Spikes/hot pixels get low confidence.
2. edge_consistency  penalizes height edges WITHOUT image support (the
                     classic monocular hallucination: a depth discontinuity
                     at a smooth RGB location) AND image edges without
                     height response are left alone (texture is common).
3. tta_agreement     (optional) 1 - normalized per-pixel spread across
                     test-time-augmented predictions — model instability
                     evidence, the strongest available signal when TTA runs.
4. validity          non-finite AGL or RGB -> confidence 0.

These values are RELATIVE WEIGHTS, not probabilities: no calibration
curve, coverage guarantee, or NLL claim is made or implied (anti-
fabrication contract). The floor used downstream (confidence_floor)
guarantees every pixel keeps a minimal data anchor regardless.
"""

from __future__ import annotations

import numpy as np


def _local_median(a: np.ndarray, radius: int, valid: np.ndarray) -> np.ndarray:
    """Median over a (2r+1)^2 window, NaN-aware, vectorized via shifts."""
    r = radius
    h, w = a.shape
    stack = []
    a0 = np.where(valid, a, np.nan)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ys = slice(max(0, -dy), min(h, h - dy))
            yd = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, -dx), min(w, w - dx))
            xd = slice(max(0, dx), min(w, w + dx))
            v = np.full_like(a0, np.nan)
            v[ys, xs] = a0[yd, xd]
            stack.append(v)
    return np.nanmedian(np.stack(stack), axis=0)


def _norm01(x: np.ndarray) -> np.ndarray:
    """Scale a non-negative map to [0,1] by its 98th percentile (robust)."""
    hi = float(np.nanpercentile(x, 98)) if np.isfinite(x).any() else 0.0
    if hi <= 1e-9:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0)


def estimate_confidence(
    agl: np.ndarray,
    rgb_u8: np.ndarray,
    tta_stack: list[np.ndarray] | None = None,
    valid: np.ndarray | None = None,
    local_radius: int = 2,
) -> np.ndarray:
    """Relative confidence in [0,1]; 0 exactly where inputs are invalid.

    ``tta_stack``: aligned AGL predictions from the TTA variants (identity
    NOT required to be included; spread is computed over the stack).
    """
    agl = np.asarray(agl, dtype=np.float32)
    rgb = np.asarray(rgb_u8)
    if valid is None:
        valid = np.isfinite(agl)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(agl)
    rgb_ok = np.isfinite(rgb[..., :3]).all(axis=2) if rgb.ndim == 3 else True
    valid = valid & rgb_ok

    conf = np.ones(agl.shape, dtype=np.float64)

    # 1) local stability ------------------------------------------------
    med = _local_median(agl.astype(np.float64), local_radius, valid)
    dev = np.abs(np.where(valid, agl, 0.0) - np.nan_to_num(med, nan=0.0))
    dev = np.where(valid, dev, 0.0)
    conf *= 1.0 - _norm01(dev)

    # 2) edge consistency: height gradient without RGB support -------------
    f = np.where(valid, agl, 0.0).astype(np.float64)
    gy_h = np.abs(np.gradient(f, axis=0))
    gx_h = np.abs(np.gradient(f, axis=1))
    g_h = np.maximum(gy_h, gx_h)
    rgbf = rgb[..., :3].astype(np.float64) / 255.0
    gy_c = np.abs(np.gradient(rgbf, axis=0)).sum(axis=2)
    gx_c = np.abs(np.gradient(rgbf, axis=1)).sum(axis=2)
    g_c = np.maximum(gy_c, gx_c)
    unsupported = np.clip(g_h * _safe_inv(_norm01(g_c) + 0.2), 0.0, None)
    conf *= 1.0 - _norm01(unsupported)

    # 3) TTA disagreement --------------------------------------------------
    if tta_stack:
        stack = np.stack(
            [np.where(valid, np.asarray(s, dtype=np.float64), np.nan)
             for s in tta_stack]
        )
        spread = np.nanstd(stack, axis=0)
        conf *= 1.0 - _norm01(np.nan_to_num(spread, nan=0.0))

    conf = np.where(valid, np.clip(conf, 0.0, 1.0), 0.0)
    return conf.astype(np.float32)


def _safe_inv(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(x, 1e-6)
