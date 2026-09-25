"""Comprehensive terrain diagnostics and statistical evaluation for DSM pipeline.

Calculates:
- Distribution statistics: min, max, mean, median, std, P1, P5, P25, P50, P75, P95, P99, P99.9
- Error metrics: MAE, RMSE, MedAE, MaxAE, P95AE, P99AE
- Gradient and slope distributions (using physical GSD)
- Spatial anomaly & outlier counts (pixels changed / total pixels)
- Structural edge preservation metrics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class DistributionStats:
    min: float
    max: float
    mean: float
    median: float
    std: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    p99_9: float

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def compute_distribution_stats(arr: np.ndarray, valid: Optional[np.ndarray] = None) -> DistributionStats:
    """Compute complete percentile and moment profile for valid samples."""
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(arr)

    vals = arr[valid].astype(np.float64)
    if vals.size == 0:
        return DistributionStats(*(0.0,) * 13)

    percentiles = np.percentile(vals, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9])
    return DistributionStats(
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        mean=float(np.mean(vals)),
        median=float(percentiles[3]),
        std=float(np.std(vals)),
        p1=float(percentiles[0]),
        p5=float(percentiles[1]),
        p25=float(percentiles[2]),
        p50=float(percentiles[3]),
        p75=float(percentiles[4]),
        p95=float(percentiles[5]),
        p99=float(percentiles[6]),
        p99_9=float(percentiles[7]),
    )


def compute_terrain_gradients(
    dsm: np.ndarray, gsd: Tuple[float, float] = (1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute physical dz/dx, dz/dy and slope in degrees."""
    dx = max(1e-4, float(gsd[0]))
    dy = max(1e-4, float(gsd[1]))
    f = np.where(np.isfinite(dsm), dsm, np.nanmean(dsm))
    gy, gx = np.gradient(f, dy, dx)
    grad_mag = np.sqrt(gx**2 + gy**2)
    slope_deg = np.rad2deg(np.arctan(grad_mag))
    return gx, gy, slope_deg


def evaluate_postprocess_results(
    raw: np.ndarray,
    refined: np.ndarray,
    gsd: Tuple[float, float] = (1.0, 1.0),
    spike_mask: Optional[np.ndarray] = None,
    replacement_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate full before/after comparative evaluation metrics."""
    valid = np.isfinite(raw) & np.isfinite(refined)
    diff = np.abs(refined[valid].astype(np.float64) - raw[valid].astype(np.float64))

    raw_stats = compute_distribution_stats(raw, valid)
    refined_stats = compute_distribution_stats(refined, valid)

    gx_r, gy_r, slope_raw = compute_terrain_gradients(raw, gsd)
    gx_f, gy_f, slope_ref = compute_terrain_gradients(refined, gsd)

    grad_raw_stats = compute_distribution_stats(np.sqrt(gx_r**2 + gy_r**2), valid)
    grad_ref_stats = compute_distribution_stats(np.sqrt(gx_f**2 + gy_f**2), valid)

    n_valid = int(np.sum(valid))
    n_changed = int(np.sum(diff > 1e-4))
    n_spikes = int(np.sum(spike_mask)) if spike_mask is not None else 0
    n_replaced = int(np.sum(replacement_mask)) if replacement_mask is not None else n_changed

    # Error metrics between raw and refined
    mae = float(np.mean(diff)) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0
    med_ae = float(np.median(diff)) if diff.size else 0.0
    max_ae = float(np.max(diff)) if diff.size else 0.0
    p95_ae = float(np.percentile(diff, 95.0)) if diff.size else 0.0
    p99_ae = float(np.percentile(diff, 99.0)) if diff.size else 0.0

    # Structural edge preservation
    # Identify high-confidence edges in raw (gradient > P90 of raw gradients)
    edge_thresh = grad_raw_stats.p90 if hasattr(grad_raw_stats, "p90") else grad_raw_stats.p95
    structural_edges = valid & (np.sqrt(gx_r**2 + gy_r**2) > edge_thresh)
    if np.any(structural_edges):
        raw_edge_grad = np.mean(np.sqrt(gx_r**2 + gy_r**2)[structural_edges])
        ref_edge_grad = np.mean(np.sqrt(gx_f**2 + gy_f**2)[structural_edges])
        edge_preservation_ratio = float(ref_edge_grad / max(1e-6, raw_edge_grad))
    else:
        edge_preservation_ratio = 1.0

    return {
        "n_pixels": raw.size,
        "n_valid": n_valid,
        "n_changed": n_changed,
        "changed_percent": round(n_changed / max(1, n_valid) * 100.0, 4),
        "n_spikes_detected": n_spikes,
        "n_pixels_replaced": n_replaced,
        "error_metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "median_absolute_error": round(med_ae, 4),
            "max_absolute_error": round(max_ae, 4),
            "p95_absolute_error": round(p95_ae, 4),
            "p99_absolute_error": round(p99_ae, 4),
        },
        "elevation_stats": {
            "before": raw_stats.to_dict(),
            "after": refined_stats.to_dict(),
        },
        "gradient_stats": {
            "before": grad_raw_stats.to_dict(),
            "after": grad_ref_stats.to_dict(),
        },
        "slope_stats": {
            "before": compute_distribution_stats(slope_raw, valid).to_dict(),
            "after": compute_distribution_stats(slope_ref, valid).to_dict(),
        },
        "edge_preservation_ratio": round(edge_preservation_ratio, 4),
    }
