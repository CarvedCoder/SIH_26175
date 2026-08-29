"""Height-error metrics with explicit valid-pixel masking.

Definition of "correct evaluation" for this project (blueprint Sec. 14):
    MAE  = (1/N) sum |H_pred - H_gt|          over VALID pixels only
    RMSE = sqrt((1/N) sum (H_pred - H_gt)^2)
    plus Pearson r, median abs error, mean bias.

Two aggregation modes, BOTH reported:
    * pooled   : concatenate all valid pixels of all tiles, compute once.
                 This is the headline number (weights big tiles fairly).
    * per-tile : mean +- std of tile-level metrics (shows variance).

Everything is plain NumPy so it works identically on a 512 crop or a
stitched full-scene DSM.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


def height_metrics(pred: np.ndarray,
                   target: np.ndarray,
                   valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Metric dict for one tile or one stitched scene. Scalars NaN-safe.

    pred / target : [H,W] float arrays (metres)
    valid_mask    : [H,W] bool; None -> use finite(target)
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"pred/target shape mismatch {p.shape} vs {t.shape}")

    mask = valid_mask if valid_mask is not None else np.isfinite(t)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != t.shape:
        raise ValueError(f"mask shape mismatch {mask.shape} vs {t.shape}")

    if mask.sum() == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "medae": float("nan"), "bias": float("nan"), "pearson_r": float("nan"),
                "neg_frac_pred": float("nan")}

    d = p[mask] - t[mask]
    ad = np.abs(d)
    n = int(mask.sum())

    tp, tt = p[mask], t[mask]
    if tp.std() > 0 and tt.std() > 0:
        r = float(np.corrcoef(tp, tt)[0, 1])
    else:
        r = float("nan")

    return {
        "n": n,
        "mae": float(ad.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "medae": float(np.median(ad)),
        "bias": float(d.mean()),                      # >0 = model overestimates
        "pearson_r": r,
        "neg_frac_pred": float((tp < 0).mean()),      # fraction of predicted heights < 0
    }


def pooled_metrics(pixels_pred: Iterable[np.ndarray],
                   pixels_target: Iterable[np.ndarray],
                   pixels_mask: Optional[Iterable[np.ndarray]] = None) -> Dict[str, float]:
    """Pooled metrics over an iterable of per-tile arrays (memory-light:
    accumulates sums, never stores everything)."""
    s_abs = s_sq = s_d = 0.0
    n = 0
    s_x = s_y = s_xx = s_yy = s_xy = 0.0
    neg = 0
    for i, (p, t) in enumerate(zip(pixels_pred, pixels_target)):
        m = pixels_mask[i] if pixels_mask is not None else np.isfinite(t)
        m = np.asarray(m, dtype=bool)
        pf = np.asarray(p, dtype=np.float64)[m]
        tf = np.asarray(t, dtype=np.float64)[m]
        d = pf - tf
        s_abs += np.abs(d).sum(); s_sq += (d ** 2).sum(); s_d += d.sum()
        s_x += pf.sum(); s_y += tf.sum()
        s_xx += (pf ** 2).sum(); s_yy += (tf ** 2).sum(); s_xy += (pf * tf).sum()
        neg += int((pf < 0).sum())
        n += pf.size
    if n == 0:
        return height_metrics(np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool))
    mean_x, mean_y = s_x / n, s_y / n
    cov = s_xy / n - mean_x * mean_y
    var_x = s_xx / n - mean_x ** 2
    var_y = s_yy / n - mean_y ** 2
    r = float(cov / np.sqrt(var_x * var_y)) if var_x > 0 and var_y > 0 else float("nan")
    return {
        "n": n,
        "mae": float(s_abs / n),
        "rmse": float(np.sqrt(s_sq / n)),
        "bias": float(s_d / n),
        "pearson_r": r,
        "neg_frac_pred": float(neg / n),
    }


def mean_std_over_tiles(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Summarize per-tile metric dicts as mean +- std (NaNs ignored)."""
    out: Dict[str, float] = {}
    if not metric_dicts:
        return out
    for key in metric_dicts[0]:
        vals = np.array([d[key] for d in metric_dicts], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{key}_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
    return out


def stratified_by_class(pred: np.ndarray, target: np.ndarray,
                        cls: np.ndarray, class_ids: List[int]) -> Dict[str, Dict[str, float]]:
    """Metrics per raw CLS id. NOTE: keys are raw ids (e.g. 'cls_65') on
    purpose — do NOT rename to 'building'/'water' until the DFC2019 class
    table has been verified from the dataset documentation."""
    out = {}
    for cid in class_ids:
        m = (cls == cid) & np.isfinite(target)
        if m.sum() > 0:
            out[f"cls_{cid}"] = height_metrics(pred, target, m)
    return out


def format_metric_row(name: str, m: Dict[str, float]) -> str:
    return (f"| {name} | {m.get('mae', float('nan')):.3f} | {m.get('rmse', float('nan')):.3f} "
            f"| {m.get('medae', float('nan')):.3f} | {m.get('bias', float('nan')):+.3f} "
            f"| {m.get('pearson_r', float('nan')):.3f} | {m.get('neg_frac_pred', float('nan')):.3f} "
            f"| {m.get('n', 0):,} |")
