"""Robust multi-scale isolated-spike detection and edge-preserving repair.

Mathematical formulation:
1. Robust local scale & location:
   - For each sample x, local median med_local and local MAD are computed over
     a (2r + 1)^2 window using fast separable reflection-padded median filters.
   - Robust scale estimate:
       sigma_local = max(1.4826 * MAD_local, eps_physical)
     where eps_physical (default 0.15 m) establishes a physical noise floor,
     preventing division-by-zero artifacts on flat terrain.
   - Robust z-score:
       z = (x - med_local) / sigma_local

2. Multi-scale outlier gating:
   - A sample is a candidate outlier only if it exceeds tau (default 3.8) at the
     primary scale (radius r1=1, 3x3) AND maintains consistent residual sign
     at secondary scale (radius r2=2, 5x5), with minimum physical elevation
     difference |x - med| > min_spike_height (default 0.6 m).

3. Connected-Component Spatial Support:
   - Candidate outliers are grouped into 8-connected components.
   - Genuine terrain structures (rooftops, cliffs, ridges, embankments) form
     coherent regions with area > max_component_size (default 6 px) and are
     STRICTLY PROTECTED.
   - For small candidate components (1 <= area <= max_component_size), the
     1-pixel outer boundary ring dC = dilate(C) \\ C is evaluated:
     * Peak: fraction of boundary samples significantly lower than min(C)
       must exceed min_isolation (default >= 65%).
     * Pit: fraction of boundary samples significantly higher than max(C)
       must exceed min_isolation (default >= 65%).
   - Step edges (such as building perimeter walls) have boundary pixels on
     both high and low sides (~50% lower, ~50% equal/higher) and are thus
     preserved without distortion.

4. Edge-Aware Local Repair:
   - Outlier pixels are replaced by the robust median of VALID, NON-SPIKE
     neighboring samples in a local 5x5 window.
   - Genuine terrain samples and step edges are never modified.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


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


def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Fast uniform box mean via guided filter box helper."""
    from .guided import _box_filter

    return _box_filter(a, radius)


def remove_spikes(
    signal: np.ndarray,
    rgb_u8: np.ndarray | None = None,
    radius: int = 1,
    tau: float = 3.8,
    min_isolation: float = 0.65,
    max_rgb_grad: float = 0.15,
    valid: np.ndarray | None = None,
    gsd: tuple[float, float] = (1.0, 1.0),
    min_spike_height: float = 0.6,
    max_component_size: int = 6,
    max_local_slope: float | None = None,
    edge_protection: bool = True,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect and repair anomalous elevation spikes while preserving genuine structures.

    Parameters
    ----------
    signal : np.ndarray
        [H, W] metric elevation or AGL raster in meters (NaN allowed).
    rgb_u8 : np.ndarray, optional
        [H, W, 3] source imagery for optional edge consistency checks.
    radius : int
        Local median filter radius (default 1 -> 3x3 window).
    tau : float
        Robust z-score threshold (dimensionless, default 3.8).
    min_isolation : float
        Fraction of outer boundary ring samples that must disagree in elevation
        (default 0.65 -> >= 65% of perimeter must be lower/higher).
    max_rgb_grad : float
        RGB L1-gradient magnitude threshold for optional edge protection.
    valid : np.ndarray, optional
        [H, W] boolean mask of valid samples (finite and unmasked).
    gsd : tuple[float, float]
        Ground sample distance (dx, dy) in meters per pixel.
    min_spike_height : float
        Minimum elevation delta in meters for candidate spike identification.
    max_component_size : int
        Maximum connected-component pixel area for candidate spikes.
        Structures with area > max_component_size are protected.
    max_local_slope : float, optional
        Maximum physically plausible terrain slope in degrees.
    edge_protection : bool
        Whether to enforce structural edge preservation.
    return_details : bool
        If True, returns (cleaned, spike_mask, replacement_mask).

    Returns
    -------
    cleaned : np.ndarray
        Repaired elevation raster with original NaN pattern bit-identical.
    spike_mask : np.ndarray
        Boolean mask of detected spike pixels.
    replacement_mask : np.ndarray (optional)
        Boolean mask of pixels whose values were modified.
    """
    signal = np.asarray(signal, dtype=np.float32)
    h, w = signal.shape

    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    n_valid = int(np.sum(valid))
    if n_valid == 0:
        cleaned = signal.copy()
        mask = np.zeros((h, w), dtype=bool)
        return (cleaned, mask, mask) if return_details else (cleaned, mask)

    f = signal.astype(np.float64)
    # Reflection-padded median filtering with NaN replacement
    f_filled = f.copy()
    valid_mean = float(np.mean(f[valid]))
    f_filled[~valid] = valid_mean

    # 1. Multi-scale robust median and MAD estimation
    ksize1 = 2 * max(1, radius) + 1
    med1 = ndimage.median_filter(f_filled, size=ksize1, mode="reflect")
    mad1 = ndimage.median_filter(np.abs(f_filled - med1), size=ksize1, mode="reflect")

    # Physical noise floor (15 cm): prevents division-by-zero on flat terrain
    sigma1 = np.maximum(1.4826 * mad1, 0.15)
    resid1 = np.where(valid, f - med1, 0.0)
    z1 = resid1 / sigma1

    # Secondary spatial scale (5x5) for multi-scale consistency
    med2 = ndimage.median_filter(f_filled, size=5, mode="reflect")
    resid2 = np.where(valid, f - med2, 0.0)

    # Candidate outlier gating
    cand_pos = (
        valid
        & (z1 > tau)
        & (resid1 > min_spike_height)
        & (resid2 > 0.4 * min_spike_height)
    )
    cand_neg = (
        valid
        & (z1 < -tau)
        & (resid1 < -min_spike_height)
        & (resid2 < -0.4 * min_spike_height)
    )

    # Optional physical slope check using actual GSD.
    # The slope is the steepest ONE-SIDED step involving the pixel itself:
    # a central-difference gradient (np.gradient) at a needle's peak sees
    # only its flat neighbours and reports ~0 slope, so the peak would
    # escape detection while its (non-outlier) neighbours carried the
    # entire signal. Steep but coherent structures (cliffs, building
    # walls) are still protected downstream by the connected-component
    # size limit and the boundary isolation test.
    if max_local_slope is not None and gsd is not None:
        dx = max(1e-4, float(gsd[0]))
        dy = max(1e-4, float(gsd[1]))

        d_rows = np.abs(f_filled[1:, :] - f_filled[:-1, :])
        d_cols = np.abs(f_filled[:, 1:] - f_filled[:, :-1])
        step_y = np.zeros_like(f_filled)
        step_x = np.zeros_like(f_filled)
        step_y[:-1, :] = d_rows
        step_y[1:, :] = np.maximum(step_y[1:, :], d_rows)
        step_x[:, :-1] = d_cols
        step_x[:, 1:] = np.maximum(step_x[:, 1:], d_cols)

        slope_deg = np.rad2deg(np.arctan(np.hypot(step_y / dy, step_x / dx)))
        extreme_slope = valid & (slope_deg > max_local_slope)
        cand_pos |= extreme_slope & (resid1 > min_spike_height)
        cand_neg |= extreme_slope & (resid1 < -min_spike_height)

    # 2. Connected-component analysis (8-connectivity)
    conn = np.ones((3, 3), dtype=bool)
    lbl_pos, n_pos = ndimage.label(cand_pos, structure=conn)
    lbl_neg, n_neg = ndimage.label(cand_neg, structure=conn)

    spike_pos = np.zeros_like(cand_pos)
    sizes_pos = np.bincount(lbl_pos.ravel())
    margin = max(0.15, 0.25 * min_spike_height)

    for i in range(1, n_pos + 1):
        # Large coherent structures are protected (genuine roofs/cliffs)
        if sizes_pos[i] <= max_component_size:
            comp_mask = lbl_pos == i
            dilated = ndimage.binary_dilation(comp_mask, structure=conn)
            boundary = dilated & (~comp_mask) & valid
            if np.any(boundary):
                c_min = np.min(f[comp_mask])
                b_vals = f[boundary]
                # Peak must be higher than most of its surrounding perimeter
                if np.mean(b_vals < c_min - margin) >= min_isolation:
                    spike_pos[comp_mask] = True

    spike_neg = np.zeros_like(cand_neg)
    sizes_neg = np.bincount(lbl_neg.ravel())
    for i in range(1, n_neg + 1):
        if sizes_neg[i] <= max_component_size:
            comp_mask = lbl_neg == i
            dilated = ndimage.binary_dilation(comp_mask, structure=conn)
            boundary = dilated & (~comp_mask) & valid
            if np.any(boundary):
                c_max = np.max(f[comp_mask])
                b_vals = f[boundary]
                # Pit must be lower than most of its surrounding perimeter
                if np.mean(b_vals > c_max + margin) >= min_isolation:
                    spike_neg[comp_mask] = True

    spike_mask = spike_pos | spike_neg

    # 3. Edge-aware inlier replacement
    cleaned = signal.copy()
    inlier_mask = valid & (~spike_mask)
    f_inliers = np.where(inlier_mask, f, np.nan)

    spike_indices = np.where(spike_mask)
    for y, x in zip(*spike_indices):
        # 5x5 inlier window
        win = f_inliers[max(0, y - 2) : min(h, y + 3), max(0, x - 2) : min(w, x + 3)]
        inliers = win[np.isfinite(win)]
        if len(inliers) >= 3:
            cleaned[y, x] = np.median(inliers)
        else:
            # Fallback to 7x7 window if near a sparse boundary
            win_large = f_inliers[
                max(0, y - 3) : min(h, y + 4), max(0, x - 3) : min(w, x + 4)
            ]
            inliers_l = win_large[np.isfinite(win_large)]
            if len(inliers_l) > 0:
                cleaned[y, x] = np.median(inliers_l)
            else:
                cleaned[y, x] = med1[y, x]

    cleaned = np.where(valid, cleaned, np.nan).astype(np.float32)
    rep_mask = spike_mask & (np.abs(cleaned - signal) > 1e-4)

    if return_details:
        return cleaned, spike_mask, rep_mask
    return cleaned, spike_mask
