"""EXPERIMENTAL robust local planar refinement.

Motivation: large approximately-planar structures (flat roofs, roads,
parking lots) are where monocular prediction often shows low-frequency
waviness that edge-aware filters cannot remove (they only smooth high
frequencies). For each sufficiently large connected component of the
PREDICTED building mask, fit

    z = a*x + b*y + c        (robust IRLS with Huber weights)

and accept the plane ONLY if

    * inlier fraction >= planar_min_inlier_frac, and
    * robust inlier RMSE  <= planar_max_residual (metres)

Rejected components are left untouched (a sloped/multi-facet roof must
never be flattened). Accepted components replace ONLY inlier pixels;
outliers (superstructures on the roof: HVAC, water tanks) keep the raw
prediction.

STATUS: experimental. Known failure modes (see docs/postprocessing.md):
sloped roofs with good inlier statistics get replaced by their best-fit
plane — the residual guard bounds the introduced error but does not
eliminate it; components grown from noisy predicted masks can mix two
adjacent structures. Never enable without ablation evidence.
"""

from __future__ import annotations

import numpy as np


def _robust_plane_fit(z: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                      iters: int = 4, huber_delta: float = 1.0):
    """IRLS plane fit; returns (a, b, c, inlier_mask, inlier_rmse)."""
    A = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    w = np.ones(len(z))
    coef = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
    for _ in range(iters):
        r = z - A @ coef
        s = max(1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6, 1e-3)
        ar = np.abs(r) / (1.345 * s)  # 95% efficiency point of Huber
        w = np.where(ar <= 1.0, 1.0, 1.0 / np.maximum(ar, 1e-6))
        coef = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
    r = z - A @ coef
    return coef, r


def planar_refine(
    signal: np.ndarray,
    building_mask: np.ndarray,
    min_area: int = 400,
    max_residual: float = 1.5,
    min_inlier_frac: float = 0.6,
    inlier_tol: float = 1.0,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Robust plane replacement inside predicted building components.

    ``building_mask`` MUST come from predicted semantics (or a user
    raster) — never the GT mask at deployment. Returns (refined, stats).
    """
    from scipy import ndimage

    signal = np.asarray(signal, dtype=np.float32)
    mask = np.asarray(building_mask, dtype=bool)
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    out = signal.copy()
    labels, n_comp = ndimage.label(mask & valid)
    yy, xx = np.mgrid[0 : signal.shape[0], 0 : signal.shape[1]]
    stats = {"components": int(n_comp), "accepted": 0, "rejected_residual": 0,
             "rejected_inliers": 0, "too_small": 0, "pixels_replaced": 0}

    for lab in range(1, n_comp + 1):
        sel = labels == lab
        npx = int(sel.sum())
        if npx < min_area:
            stats["too_small"] += 1
            continue
        zs = signal[sel].astype(np.float64)
        xs = xx[sel].astype(np.float64)
        ys = yy[sel].astype(np.float64)
        coef, resid = _robust_plane_fit(zs, xs, ys)
        inlier = np.abs(resid) <= inlier_tol
        rmse = float(np.sqrt(np.mean(resid[inlier] ** 2))) if inlier.any() else np.inf
        if rmse > max_residual:
            stats["rejected_residual"] += 1
            continue
        if inlier.mean() < min_inlier_frac:
            stats["rejected_inliers"] += 1
            continue
        # replace ONLY inlier pixels; outliers keep raw (superstructures)
        plane = coef[0] * xx.astype(np.float64) + coef[1] * yy.astype(np.float64) + coef[2]
        sel_in = sel & np.zeros_like(sel)
        sel_in[sel] = inlier
        out[sel_in] = plane[sel_in].astype(np.float32)
        stats["accepted"] += 1
        stats["pixels_replaced"] += int(inlier.sum())

    return out, stats
