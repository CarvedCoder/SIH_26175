"""RGB-guided edge-preserving filter (He, Sun, Tang — ECCV 2010).

Principle (the task's contract):

    RGB  = guidance  I
    AGL  = signal    p            (metric metres — never normalized here)
    out  = q         edge-aware filtered signal

Local linear model in each (2r+1)^2 window:

    q_i = a_i * I_i + b_i

    a_i = cov(I, p) / (var(I) + eps)
    b_i = mean(p) - a_i * mean(I)

    q = mean(a) * I + mean(b)        (means over the same windows)

* ``eps`` is the edge threshold: guidance variance BELOW eps is treated as
  noise (smoothed away); variance ABOVE eps is structure (kept sharp).
* Boxes are computed with integral images -> O(N) per call, independent of
  radius, fully vectorized (no Python pixel loops), float64 internally for
  numerical robustness of the cumsum trick, float32 out.
* Boundaries: edge-replicate padding (never invents new extrema).
* NaN handling: NaN signal pixels are masked out of every box statistic;
  output keeps the input NaN pattern exactly.

The filter is EXACT for a constant signal (a=0, b=c -> q=c), so it cannot
shift the global height scale of a flat region — a required calibration
property (pinned by unit tests).
"""

from __future__ import annotations

import numpy as np


def _box_filter(
    arr: np.ndarray, radius: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Windowed mean via integral image, edge-replicated borders.

    ``mask`` (bool, same shape): when given, computes the MASKED mean —
    sums are accumulated only over valid pixels and normalized by the valid
    count per window (count 0 -> 0). This keeps NaN pixels from poisoning
    every window they touch.
    """
    r = radius
    n_y, n_x = arr.shape
    a = arr.astype(np.float64, copy=False)
    if mask is not None:
        a = a * mask
        m_src = mask.astype(np.float64)
    else:
        m_src = None

    # pad ring of r+1 (edge-replicate); the integral image gets a leading
    # zero row+col so every window sum is a pure index difference (no
    # boundary special cases, no off-by-one).
    pad = r + 1
    q = np.pad(a, ((pad, pad), (pad, pad)), mode="edge")
    ii = np.zeros((q.shape[0] + 1, q.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(q, axis=0), axis=1)
    if m_src is not None:
        qm = np.pad(m_src, ((pad, pad), (pad, pad)), mode="edge")
        ic = np.zeros_like(ii)
        ic[1:, 1:] = np.cumsum(np.cumsum(qm, axis=0), axis=1)

    s_hi = 2 * r + 2
    win = (
        ii[s_hi : s_hi + n_y, s_hi : s_hi + n_x]
        - ii[1 : 1 + n_y, s_hi : s_hi + n_x]
        - ii[s_hi : s_hi + n_y, 1 : 1 + n_x]
        + ii[1 : 1 + n_y, 1 : 1 + n_x]
    )
    if m_src is None:
        area = (2 * r + 1) ** 2
        return win / area
    c = (
        ic[s_hi : s_hi + n_y, s_hi : s_hi + n_x]
        - ic[1 : 1 + n_y, s_hi : s_hi + n_x]
        - ic[s_hi : s_hi + n_y, 1 : 1 + n_x]
        + ic[1 : 1 + n_y, 1 : 1 + n_x]
    )
    return np.divide(win, c, out=np.zeros_like(win), where=c > 0)


def rgb_to_luma(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 [H,W,3] -> Rec.601 luma in [0,1] (float32). The guidance image."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    return (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def guided_filter(
    signal: np.ndarray,
    guidance: np.ndarray,
    radius: int = 4,
    eps: float = 1e-3,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Edge-preserving guided filter.

    signal    [H,W] float, metric units (AGL metres)
    guidance  [H,W] float in [0,1] (RGB luma) — NOT the signal itself
    radius    window radius in px
    eps       edge threshold in guidance-variance units
    valid     [H,W] bool; None -> finite(signal)

    Returns float32 [H,W] with the input's NaN pattern preserved.
    """
    signal = np.asarray(signal, dtype=np.float32)
    guidance = np.asarray(guidance, dtype=np.float32)
    if signal.shape != guidance.shape:
        raise ValueError(
            f"signal/guidance shape mismatch {signal.shape} vs {guidance.shape}"
        )
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    p = np.where(valid, signal, 0.0).astype(np.float64)
    I = np.where(valid, guidance, 0.0).astype(np.float64)

    mean_I = _box_filter(I, radius, valid)
    mean_p = _box_filter(p, radius, valid)
    corr_Ip = _box_filter(I * p, radius, valid)
    corr_II = _box_filter(I * I, radius, valid)

    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius, valid)
    mean_b = _box_filter(b, radius, valid)

    q = (mean_a * I + mean_b).astype(np.float32)
    return np.where(valid, q, np.nan).astype(np.float32)
