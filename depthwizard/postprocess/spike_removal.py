"""Conservative isolated-spike removal (robust median/MAD).

A pixel is REPLACED by its local median only when ALL of:

1. strong deviation      |x - med_local| > tau * (1.4826 * MAD_local + eps)
2. neighbour disagreement
   a. the window's other pixels do NOT deviate from their medians with
      the SAME SIGN beyond frac (isolated-ness), AND
   b. NO 4-neighbour is itself a strong same-sign outlier (contiguity):
      a genuine roof edge/corner is a CONNECTED band of co-deviating
      pixels — an isolated spike touches none. This is what distinguishes
      `isolated spike` from `building roof edge` (task Sec. 9).
3. weak edge support     the local RGB gradient is below max_rgb_grad —
   strong image edges are presumed geometry until proven otherwise.

The replacement value is the local median (robust), and only the outlier
pixels change — never their neighbours. Roof edges/corners fail test 2
(connected co-deviating ring) and roof outlines fail test 3 (strong RGB
edges), so they survive. Pinned by unit tests.
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


def _shift_or(m: np.ndarray) -> np.ndarray:
    """True where ANY 4-neighbour of the pixel is True in ``m``."""
    out = np.zeros_like(m)
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


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

    strong_pos = valid & (resid > tau * sigma)   # too high vs neighbourhood
    strong_neg = valid & (resid < -tau * sigma)  # too low
    strong = strong_pos | strong_neg

    # 2a. same-sign support fraction over the window. Sign-aware (the
    # docstring says "deviating with the same sign"): at a step edge BOTH
    # sides deviate with opposite signs, and an unsigned count would
    # over-count support; here each side only counts its own sign.
    sup_pos = _box_mean(strong_pos.astype(np.float64), radius)
    sup_neg = _box_mean(strong_neg.astype(np.float64), radius)
    same_sign_support = np.where(resid >= 0, sup_pos, sup_neg)
    isolated = same_sign_support < min_isolation

    # 2b. contiguity: a strong same-sign 4-neighbour means this pixel is
    # part of a connected band (roof edge / roof corner ring) — protected.
    nb_pos = _shift_or(strong_pos)
    nb_neg = _shift_or(strong_neg)
    has_band_neighbour = np.where(resid >= 0, nb_pos, nb_neg)

    # RGB edge protection: L1 gradient magnitude, [0,1] units
    rgbf = rgb[..., :3].astype(np.float64) / 255.0
    g = np.maximum(
        np.abs(np.gradient(rgbf, axis=0)).sum(axis=2),
        np.abs(np.gradient(rgbf, axis=1)).sum(axis=2),
    )
    weak_edge = g < max_rgb_grad

    spike = strong & isolated & ~has_band_neighbour & weak_edge
    out = signal.copy()
    out[spike] = np.nan_to_num(med, nan=0.0)[spike].astype(np.float32)
    return out, spike


def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Fast uniform box mean via zero-padded integral image (edge-replicated)."""
    from .guided import _box_filter

    return _box_filter(a, radius)
