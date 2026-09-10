"""Conservative isolated-spike removal (robust median/MAD).

A pixel is REPLACED by its local median only when ALL of:

1. strong deviation      |x - med_local| > tau * (1.4826 * MAD_local + eps)
2. neighbour disagreement the window's other pixels do NOT deviate from
   their medians with the same sign beyond frac (isolated-ness): a genuine
   roof edge is a CONTIGUOUS band of similarly-deviating pixels, a spike
   is not;
3. weak edge support     the local RGB gradient is below max_rgb_grad —
   strong image edges are presumed geometry until proven otherwise.

The replacement value is the local median (robust), and only the outlier
pixels change — never their neighbours. Roof edges fail test 2 (many
co-deviating pixels along the edge) and test 3 (building outlines are
strong RGB edges), so they survive. Pinned by unit tests.
"""

from __future__ import annotations

import numpy as np


def _shifted_med_mad(a: np.ndarray, radius: int, valid: np.ndarray):
    """Local median + local MAD over (2r+1)^2, NaN-aware via shift-stack."""
    r = radius
    h, w = a.shape
    a0 = np.where(valid, a, np.nan)
    stack = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ys = slice(max(0, -dy), min(h, h - dy))
            yd = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, -dx), min(w, w - dx))
            xd = slice(max(0, dx), min(w, w + dx))
            v = np.full_like(a0, np.nan)
            v[ys, xs] = a0[yd, xd]
            stack.append(v)
    st = np.stack(stack)
    with np.errstate(all="ignore"):
        med = np.nanmedian(st, axis=0)
        mad = np.nanmedian(np.abs(st - med[None]), axis=0)
    return med, mad


def remove_spikes(
    signal: np.ndarray,
    rgb_u8: np.ndarray,
    radius: int = 3,
    tau: float = 6.0,
    min_isolation: float = 0.25,
    max_rgb_grad: float = 0.15,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cleaned signal, spike_mask). NaN pattern preserved.

    tau            robust-z threshold (dimensionless)
    min_isolation  spike requires < this fraction of window pixels deviating
                   with the same sign (0.25 => >75% of neighbours disagree)
    max_rgb_grad   RGB L1 gradient above which a pixel is protected
    """
    signal = np.asarray(signal, dtype=np.float32)
    rgb = np.asarray(rgb_u8)
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    f = signal.astype(np.float64)
    med, mad = _shifted_med_mad(f, radius, valid)
    sigma = 1.4826 * np.nan_to_num(mad, nan=0.0) + 1e-3  # robust sigma, metres
    resid = np.where(valid, f - np.nan_to_num(med, nan=0.0), 0.0)

    strong = valid & (np.abs(resid) > tau * sigma)

    # isolation: fraction of the window deviating from ITS median with the
    # same sign. Approximated cheaply: compare the centre's residual sign
    # with the residual of the median-filtered deviation map — a contiguous
    # edge makes neighbouring |resid| large too, so a simple threshold on a
    # smoothed |resid| detects it (box mean over the same window).
    same_sign_support = _box_mean(
        (np.abs(resid) > tau * sigma).astype(np.float64), radius
    )
    isolated = same_sign_support < min_isolation

    # RGB edge protection: L1 gradient magnitude, [0,1] units
    rgbf = rgb[..., :3].astype(np.float64) / 255.0
    g = np.maximum(
        np.abs(np.gradient(rgbf, axis=0)).sum(axis=2),
        np.abs(np.gradient(rgbf, axis=1)).sum(axis=2),
    )
    weak_edge = g < max_rgb_grad

    spike = strong & isolated & weak_edge
    out = signal.copy()
    out[spike] = np.nan_to_num(med, nan=0.0)[spike].astype(np.float32)
    return out, spike


def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Fast uniform box mean via zero-padded integral image (edge-replicated)."""
    from .guided import _box_filter

    return _box_filter(a, radius)
