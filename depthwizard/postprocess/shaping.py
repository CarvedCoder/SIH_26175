"""Semantic structural shaping — building plateaus + tree canopy domes.

A post-refinement stage for RENDERING quality: after core refinement, the
predicted DSM still shows urban surfaces as noisy blobby lumps — rooftops
wobble, vegetation reads as a field of spikes. This module reshapes ONLY
pixels inside PREDICTED semantic regions (never GT, never guessed):

1. Building plateau flattening (class 0)
   The predicted building mask is morphologically cleaned (close -> open ->
   fill holes -> drop specks) into crisp, edge-detected footprints, then
   each footprint's interior is replaced by its robust best-fit plane
   (IRLS/Huber, reused from planar.py). Guards: components whose heights
   are not plane-like (low inlier fraction / huge residual) are left
   untouched — a sloped or multi-facet roof must never be force-flattened.

2. Tree canopy domes (class 1)
   Vegetation components above a minimum pixel area and a minimum
   crown-ground prominence are replaced by a smooth hemispherical dome
   per crown. Crowns are split at local height maxima (nearest-peak
   Voronoi inside the vegetation mask), so a woodland block renders as a
   cluster of rounded canopies instead of one plateau or a spike field.
   Crown peak = robust percentile of the REFINED heights (real tree height
   is preserved); crown base = median ground ring just outside the mask.
   Low-prominence vegetation (grass, shrubs below the height gate) is
   left exactly as refined.

Without ``sem_probs`` the caller must skip this stage — the functions
here never invent semantic regions. All operations keep the input NaN
pattern bit-identical and report per-stage pixel counts for the
postprocess report.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

VEGETATION_CLASS = 1  # project class index (datasets.semantics.PROJECT_CLASSES)

_CONN8 = np.ones((3, 3), dtype=bool)


def vegetation_mask_from_probs(
    sem_probs: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Soft vegetation mask from predicted probabilities (prob >= threshold)."""
    p = np.asarray(sem_probs, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"semantic_probs must be [K,H,W], got {p.shape}")
    return p[VEGETATION_CLASS] >= threshold


def clean_region_mask(
    mask: np.ndarray,
    valid: np.ndarray,
    min_area: int = 25,
    open_boundary: bool = False,
) -> np.ndarray:
    """Morphological cleanup -> crisp edge-detected footprint mask.

    Binary closing (seals 1-px pits in the footprint outline), optional
    opening (removes 1-px spurs and speckle — used for the STRUCTURE CORE
    that statistics are fitted on), hole filling, then connected
    components smaller than ``min_area`` are dropped.

    ``open_boundary=False`` keeps the full predicted extent: replacement
    geometries must cover every predicted pixel, otherwise a 1-px ring of
    unshaped original heights survives around the shaped region.
    """
    m = np.asarray(mask, dtype=bool) & np.asarray(valid, dtype=bool)
    if not m.any():
        return m
    m = ndimage.binary_closing(m, structure=_CONN8)
    if open_boundary:
        m = ndimage.binary_opening(m, structure=_CONN8)
    m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m, structure=_CONN8)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= max(1, int(min_area))
        keep[0] = False
        m = keep[lbl]
    return m


def flatten_buildings(
    signal: np.ndarray,
    building_mask: np.ndarray,
    valid: np.ndarray | None = None,
    min_area: int = 25,
    max_residual: float = 2.0,
    min_inlier_frac: float = 0.5,
    inlier_tol: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Replace cleaned building footprints with their robust best-fit plane.

    Returns (shaped, stats). Components that fail the plane-likeness
    guards are left untouched. NaN pattern is preserved bit-identically.
    """
    from .planar import _robust_plane_fit

    signal = np.asarray(signal, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    out = signal.copy()
    full = clean_region_mask(building_mask, valid, min_area=min_area)
    core = clean_region_mask(building_mask, valid, min_area=min_area, open_boundary=True)
    labels, n_comp = ndimage.label(full, structure=_CONN8)
    stats = {
        "building_components": int(n_comp),
        "building_flattened": 0,
        "building_rejected": 0,
        "building_pixels_replaced": 0,
    }
    if n_comp == 0:
        return out, stats

    yy, xx = np.mgrid[0 : signal.shape[0], 0 : signal.shape[1]]
    for lab in range(1, n_comp + 1):
        sel = labels == lab
        fit_px = sel & core  # roof interior only: the eroded, speckle-free core
        if fit_px.sum() < 8:
            stats["building_rejected"] += 1
            continue
        zs = signal[fit_px].astype(np.float64)
        coef, resid = _robust_plane_fit(
            zs, xx[fit_px].astype(np.float64), yy[fit_px].astype(np.float64)
        )
        inlier = np.abs(resid) <= inlier_tol
        rmse = float(np.sqrt(np.mean(resid[inlier] ** 2))) if inlier.any() else np.inf
        if rmse > max_residual or inlier.mean() < min_inlier_frac:
            stats["building_rejected"] += 1
            continue
        plane = coef[0] * xx + coef[1] * yy + coef[2]
        out[sel] = plane[sel].astype(np.float32)
        stats["building_flattened"] += 1
        stats["building_pixels_replaced"] += int(sel.sum())

    out = np.where(valid, out, np.nan).astype(np.float32)
    return out, stats


def shape_tree_canopies(
    signal: np.ndarray,
    vegetation_mask: np.ndarray,
    valid: np.ndarray | None = None,
    min_area: int = 12,
    min_tree_height: float = 1.0,
    peak_window: int = 9,
    peak_percentile: float = 90.0,
    max_crowns: int = 4000,
    smoothing_fallback_sigma: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Replace vegetation regions with smooth per-crown canopy domes.

    Crowns are the nearest-peak Voronoi cells inside the cleaned
    vegetation mask; each crown becomes a hemispherical dome rising from
    the surrounding ground ring to a robust peak percentile of the
    refined heights. Crowns whose prominence (peak - base) is below
    ``min_tree_height`` are left as refined (grass/shrubs are not domed).
    When the crown count exceeds ``max_crowns`` the per-crown geometry is
    skipped and vegetation is only gently smoothed (degradation reported).
    """
    signal = np.asarray(signal, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(signal)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(signal)

    out = signal.copy()
    mask = clean_region_mask(vegetation_mask, valid, min_area=min_area)
    stats = {
        "veg_components": 0,
        "veg_crowns": 0,
        "veg_crowns_domed": 0,
        "veg_crowns_low_prominence": 0,
        "veg_pixels_replaced": 0,
        "veg_fallback": None,
    }
    if not mask.any():
        return out, stats

    h, w = signal.shape
    finite = np.where(mask, np.nan_to_num(signal, nan=0.0), 0.0).astype(np.float64)

    # Crown centers: local maxima of moderately smoothed heights inside the
    # mask. The sigma matters for crown segmentation: too little smoothing
    # fragments one noisy canopy into many spurious crowns.
    smoothed = ndimage.gaussian_filter(finite, sigma=1.5, mode="nearest")
    locmax = smoothed == ndimage.maximum_filter(
        smoothed, size=max(3, int(peak_window) | 1), mode="nearest"
    )
    peaks = locmax & mask

    # Every vegetation component must own at least one crown seed: give a
    # component without a local maximum its highest pixel as the seed.
    comp_lbl, n_comp = ndimage.label(mask, structure=_CONN8)
    stats["veg_components"] = int(n_comp)
    for lab in range(1, n_comp + 1):
        comp = comp_lbl == lab
        if not (peaks & comp).any():
            idx = np.argmax(np.where(comp, finite, -np.inf))
            peaks.flat[idx] = True

    peak_lbl, n_peaks = ndimage.label(peaks, structure=_CONN8)
    stats["veg_crowns"] = int(n_peaks)
    if n_peaks > max_crowns:
        # Degradation path: dense forest scenes — dome geometry per crown is
        # not worth the cost, gently smooth canopies instead (reported).
        filled = np.where(mask, np.nan_to_num(signal, nan=0.0), 0.0)
        soft = ndimage.gaussian_filter(filled, sigma=smoothing_fallback_sigma, mode="nearest")
        out[mask] = soft[mask].astype(np.float32)
        out = np.where(valid, out, np.nan).astype(np.float32)
        stats["veg_fallback"] = f"gaussian sigma={smoothing_fallback_sigma}"
        return out, stats

    # Nearest-peak Voronoi inside the mask: EDT from non-peak pixels, with
    # return_indices giving each pixel's nearest peak seed.
    _, inds = ndimage.distance_transform_edt(~peaks, return_indices=True)
    crown = peak_lbl[inds[0], inds[1]]
    crown[~mask] = 0

    # Ground base per crown: median of the valid ring just OUTSIDE the veg
    # mask around the crown's own component (the terrain the canopy sits on).
    ring = ndimage.binary_dilation(mask, structure=_CONN8) & ~mask & valid
    base_map = np.zeros(n_peaks + 1, dtype=np.float64)
    peak_map = np.zeros(n_peaks + 1, dtype=np.float64)
    domed = np.zeros(n_peaks + 1, dtype=bool)

    for lab in range(1, n_peaks + 1):
        sel = crown == lab
        vals = signal[sel & valid].astype(np.float64)
        if vals.size == 0:
            continue
        # Ground estimate: ring pixels belonging to the crown's component
        # neighbourhood, falling back to the crown's own minimum.
        ring_global = ndimage.binary_dilation(sel, structure=_CONN8) & ring
        ground_vals = signal[ring_global]
        if ground_vals.size:
            base = float(np.median(ground_vals))
        else:
            base = float(vals.min())
        peak_h = float(np.percentile(vals, peak_percentile))
        base_map[lab] = base
        peak_map[lab] = peak_h
        domed[lab] = (peak_h - base) >= min_tree_height and peak_h > base

    if not domed[1:].any():
        return out, stats

    # Per-crown hemispherical dome: d = distance to the crown CELL's own
    # boundary (the crown's Voronoi cell inside the vegetation mask),
    # R = the cell's maximum depth. shape = sqrt(1-(1-d/R)^2) rises from
    # 0 at the cell edge (ground/base) to 1 at the crown center (peak).
    replaced = 0
    for lab in range(1, n_peaks + 1):
        if not domed[lab]:
            continue
        sel = crown == lab
        d = ndimage.distance_transform_edt(sel)
        r_max = float(d.max())
        if r_max <= 0:
            continue
        t = np.clip(d / r_max, 0.0, 1.0)
        shape = np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))
        dome = base_map[lab] + (peak_map[lab] - base_map[lab]) * shape
        take = sel & valid
        out[take] = dome[take].astype(np.float32)
        replaced += int(take.sum())
    stats["veg_crowns_domed"] = int(domed[1:].sum())
    stats["veg_crowns_low_prominence"] = int(n_peaks - stats["veg_crowns_domed"])
    stats["veg_pixels_replaced"] = replaced

    out = np.where(valid, out, np.nan).astype(np.float32)
    return out, stats
