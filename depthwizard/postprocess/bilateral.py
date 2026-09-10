"""Joint bilateral filter — RGB-guided edge-preserving smoothing.

Kernel between a centre pixel i and neighbour j:

    w_ij = exp(-||pos_i - pos_j||^2 / (2 s_space^2))          (spatial)
         * exp(-||rgb_i - rgb_j||_1^2 / (2 s_color^2))        (range)

The RANGE term is computed on the GUIDANCE RGB (joint/cross bilateral):
homogeneous image regions average together; across strong colour changes
(shadow lines, roof-ground colour contrast) the weight collapses and the
height discontinuity is preserved.

Implementation notes:
* fully vectorized — one shifted-array pair per window offset, no pixel
  loops; accumulators keep memory at O(N) regardless of radius;
* edge-replicate padding for shifts;
* NaN signal pixels are excluded from every neighbourhood average, and the
  output preserves the input NaN pattern;
* known weakness (documented in docs/postprocessing.md): satellite RGB has
  texture edges (road markings, tree crowns, shadows) that are NOT height
  edges — the filter over-trusts image edges. This is why the guided filter
  and WLS variants exist.
"""

from __future__ import annotations

import numpy as np


def joint_bilateral_filter(
    signal: np.ndarray,
    rgb_u8: np.ndarray,
    radius: int = 5,
    sigma_color: float = 0.1,
    sigma_space: float = 3.0,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Filter ``signal`` [H,W] (metres) guided by ``rgb_u8`` [H,W,3].

    sigma_color is in normalized [0,1] RGB units; sigma_space in pixels.
    Returns float32 [H,W]; NaN pattern of ``signal`` preserved.
    """
    signal = np.asarray(signal, dtype=np.float32)
    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb_u8 must be [H,W,3], got {rgb.shape}")
    if signal.shape != rgb.shape[:2]:
        raise ValueError(
            f"signal/rgb shape mismatch {signal.shape} vs {rgb.shape[:2]}"
        )
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    rgbf = rgb[..., :3].astype(np.float32) / 255.0
    p = np.where(valid, signal, 0.0).astype(np.float64)
    m = valid.astype(np.float64)

    inv_2sc = 1.0 / (2.0 * max(sigma_color, 1e-6) ** 2)
    inv_2ss = 1.0 / (2.0 * max(sigma_space, 1e-6) ** 2)

    acc = np.zeros_like(p)
    wsum = np.zeros_like(p)
    r = int(radius)
    h, w = p.shape

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # shifted views with edge-replicate semantics (np.roll would wrap)
            ys = slice(max(0, -dy), min(h, h - dy))
            yd = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, -dx), min(w, w - dx))
            xd = slice(max(0, dx), min(w, w + dx))

            d_rgb = np.abs(rgbf[ys, xs] - rgbf[yd, xd]).sum(axis=2)  # L1
            w_col = np.exp(-(d_rgb**2) * inv_2sc)
            w_spa = np.exp(-float(dy * dy + dx * dx) * inv_2ss)
            w_ij = w_col * (w_spa * m[yd, xd])

            acc[ys, xs] += w_ij * p[yd, xd]
            wsum[ys, xs] += w_ij

    q = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 1e-12)
    return np.where(valid, q, np.nan).astype(np.float32)
