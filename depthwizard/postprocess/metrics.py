"""Boundary-aware + geometry evaluation metrics (post-processing suite).

EXTENDS (never replaces) the frozen ``depthwizard.metrics``: MAE/RMSE/r/
bias and the building metrics stay byte-identical where they are; this
module ADDS the geometric dimensions the height metrics cannot see:

    gradient_error      mean |grad(pred) - grad(gt)|  (both axes, metres)
    boundary_band_*     MAE/RMSE in a k-px band around GT class boundaries
                        (building-ground, building-other, any-change)
    building_edge_*     the boundary band restricted to building outlines
    region_*            height metrics per project class region
                        (building interior / ground / vegetation / other)
    negative_fraction   fraction of predicted AGL < 0 (physics violation)
    seam_error          mean |pred_left_edge - pred_right_edge| across
                        vertical 1024-tile seams (tile-consistency check)

All functions are pure NumPy, NaN-safe, and share the valid-mask convention
of depthwizard.metrics (finite target by default). Boundary DEFINITIONS
always come from the GT semantic layers at EVALUATION time only — deployed
refinement never sees them.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..metrics import height_metrics


# ---------------------------------------------------------------------------
# Boundary band construction (GT side, evaluation only)
# ---------------------------------------------------------------------------


def _binary_dilate(m: np.ndarray, iters: int) -> np.ndarray:
    """4-connectivity dilation without scipy (cheap shift-OR)."""
    out = m.copy()
    for _ in range(iters):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def boundary_bands(
    onehot: np.ndarray, band_px: int = 2
) -> Dict[str, np.ndarray]:
    """GT boundary masks from the project one-hot [K,H,W].

    Returns {'any', 'building', 'building_ground', 'building_veg', ...}
    -> bool [H,W] each: True within ``band_px`` of the respective boundary.
    """
    cls = onehot.argmax(axis=0)
    has_label = onehot.sum(axis=0) > 0.5  # ignore pixels excluded
    edge = np.zeros(cls.shape, dtype=bool)
    edge[1:, :] |= (cls[1:, :] != cls[:-1, :]) & has_label[1:] & has_label[:-1]
    edge[:-1, :] |= (cls[1:, :] != cls[:-1, :]) & has_label[1:] & has_label[:-1]
    edge[:, 1:] |= (cls[:, 1:] != cls[:, :-1]) & has_label[:, 1:] & has_label[:, :-1]
    edge[:, :-1] |= (cls[:, 1:] != cls[:, :-1]) & has_label[:, 1:] & has_label[:, :-1]
    edge &= has_label

    out = {"any": _binary_dilate(edge, band_px)}
    building = (cls == 0) & has_label
    bnd_b = _binary_dilate(edge & _touches(building), band_px)
    out["building"] = bnd_b
    for name, cid in (("ground", 4), ("vegetation", 1), ("road", 2), ("water", 3)):
        other = (cls == cid) & has_label
        out[f"building_{name}"] = bnd_b & _binary_dilate(
            _touches(other), band_px
        )
    return out


def _touches(region: np.ndarray) -> np.ndarray:
    """Edge pixels that are adjacent (4-neigh) to the given region."""
    t = np.zeros_like(region)
    t[1:, :] |= region[:-1, :]
    t[:-1, :] |= region[1:, :]
    t[:, 1:] |= region[:, :-1]
    t[:, :-1] |= region[:, 1:]
    return t & region | t


# ---------------------------------------------------------------------------
# Gradient + discontinuity metrics
# ---------------------------------------------------------------------------


def gradient_error(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Mean absolute gradient difference (metres/px) over valid pixels.

    Central np.gradient on both axes; the 1-px border is excluded
    (undefined central difference). Captures 'sharpness fidelity': a
    predictor that blurs building edges has large gradient error at
    boundaries even when its MAE is fine.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    mask = valid_mask if valid_mask is not None else np.isfinite(t)
    mask = np.asarray(mask, dtype=bool).copy()
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    if mask.sum() == 0:
        return {"n": 0, "grad_mae": float("nan")}
    gp_y, gp_x = np.gradient(p)
    gt_y, gt_x = np.gradient(t)
    d = (
        np.abs(gp_y - gt_y) + np.abs(gp_x - gt_x)
    ) * 0.5  # mean over the two axes
    return {
        "n": int(mask.sum()),
        "grad_mae": float(d[mask].mean()),
        "grad_rmse": float(np.sqrt((d[mask] ** 2).mean())),
    }


def discontinuity_preservation(
    pred: np.ndarray,
    target: np.ndarray,
    onehot: np.ndarray,
    band_px: int = 2,
    jump_threshold: float = 2.0,
) -> Dict[str, float]:
    """Surface-discontinuity preservation across GT building boundaries.

    For GT boundary pixels where the GT height jumps more than
    ``jump_threshold`` metres across the boundary, measure the predicted
    jump across the same pair. Reports the ratio mean(|pred jump|) /
    mean(|gt jump|): ~1 preserves discontinuities, <1 = blurred edges
    (roof height bleeding into ground), >1 = newly invented jumps.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    bands = boundary_bands(onehot, band_px)
    bmask = bands["building"] & np.isfinite(t) & np.isfinite(p)
    # vertical + horizontal neighbour pairs straddling the band
    num, den, n = 0.0, 0.0, 0
    for sl_a, sl_b in (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ):
        pair_mask = bmask[sl_a] & bmask[sl_b]
        gt_jump = np.abs(t[sl_a] - t[sl_b])
        sel = pair_mask & np.isfinite(gt_jump) & (gt_jump > jump_threshold)
        pr_jump = np.abs(p[sl_a] - p[sl_b])
        num += float(pr_jump[sel].sum())
        den += float(gt_jump[sel].sum())
        n += int(sel.sum())
    if n == 0:
        return {"n_pairs": 0, "jump_ratio": float("nan")}
    return {"n_pairs": n, "jump_ratio": float(num / max(den, 1e-9))}


# ---------------------------------------------------------------------------
# Seam consistency (tile artifacts)
# ---------------------------------------------------------------------------


def seam_error(
    pred: np.ndarray, tile: int = 1024, valid_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Mean |jump| across internal vertical/horizontal tile seams.

    Per-tile min-max Dn normalization can create step artefacts at 1024-px
    tile boundaries; this metric measures the mean absolute height
    difference across seam-adjacent pixel columns/rows (a smooth surface
    has small values). Only seams strictly inside the array count.
    """
    p = np.asarray(pred, dtype=np.float64)
    h, w = p.shape
    mask = valid_mask if valid_mask is not None else np.ones((h, w), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    vals: List[float] = []
    for x in range(tile, w, tile):
        col = np.abs(p[:, x] - p[:, x - 1])[mask[:, x] & mask[:, x - 1]]
        if col.size:
            vals.append(float(col.mean()))
    for y in range(tile, h, tile):
        row = np.abs(p[y] - p[y - 1])[mask[y] & mask[y - 1]]
        if row.size:
            vals.append(float(row.mean()))
    if not vals:
        return {"n_seams": 0, "seam_mae": float("nan")}
    return {"n_seams": len(vals), "seam_mae": float(np.mean(vals))}


# ---------------------------------------------------------------------------
# Full post-processing evaluation for one tile
# ---------------------------------------------------------------------------


def postprocess_tile_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    onehot: Optional[np.ndarray] = None,
    band_px: int = 2,
    tile: int = 1024,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Complete metric dict for one tile (frozen metrics + geometric suite).

    pred/target [H,W] metres; onehot [K,H,W] GT project one-hot (optional —
    band/region metrics are skipped when absent). The returned dict has
    groups: 'global', 'building', 'boundary_*', 'gradient', 'seam',
    'region_<name>'.
    """
    mask = valid_mask if valid_mask is not None else np.isfinite(target)
    out: Dict[str, Dict[str, float]] = {
        "global": height_metrics(pred, target, mask),
        "gradient": gradient_error(pred, target, mask),
        "seam": seam_error(pred, tile=tile, valid_mask=mask),
    }
    neg = np.asarray(pred, dtype=np.float64)[mask]
    out["global"]["negative_fraction"] = (
        float((neg < 0).mean()) if neg.size else float("nan")
    )
    if onehot is not None:
        onehot = np.asarray(onehot, dtype=np.float32)
        bands = boundary_bands(onehot, band_px)
        for name, bm in bands.items():
            m = bm & mask
            if m.sum() > 0:
                out[f"boundary_{name}"] = height_metrics(pred, target, m)
        from ..metrics import stratified_by_project_class

        strat = stratified_by_project_class(pred, target, onehot)
        for name, m in strat.items():
            out[f"region_{name}"] = m
        out["discontinuity"] = discontinuity_preservation(
            pred, target, onehot, band_px
        )
    return out


def summarize_tile_metrics(tiles: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Mean over per-tile metric dicts (per group, per key; NaN-skipping)."""
    out: Dict[str, Dict[str, float]] = {}
    groups = sorted({g for t in tiles for g in t})
    for g in groups:
        rows = [t[g] for t in tiles if g in t]
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        agg: Dict[str, float] = {}
        for k in keys:
            vals = np.array(
                [r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=np.float64
            )
            agg[k] = float(vals.mean()) if vals.size else float("nan")
        out[g] = agg
    return out
