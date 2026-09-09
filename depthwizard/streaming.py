"""Memory-light pooled metric accumulators (streaming evaluation).

The ``eval-baseline`` and ``dummies`` commands evaluate hundreds of
1024x1024 tiles without ever holding a full split in RAM. This module holds
the accumulator trio (new / update / to-metrics) that both commands share,
replacing the two identical private copies that lived inside the old
numbered scripts.

Guarantee: ``PooledStats`` produces EXACTLY the same numbers as
``depthwizard.metrics.pooled_metrics`` on the same pixels (verified by
test_streaming_equivalence in tests/test_metrics_streaming.py). The citable
``evaluate`` command keeps using the frozen ``pooled_metrics`` directly;
this class exists so the baseline reports can stream tile-by-tile.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class PooledStats:
    """Streaming sums for pooled MAE / RMSE / bias / Pearson r / neg-frac."""

    __slots__ = (
        "s_abs",
        "s_sq",
        "s_d",
        "s_x",
        "s_y",
        "s_xx",
        "s_yy",
        "s_xy",
        "neg",
        "n",
    )

    def __init__(self) -> None:
        self.s_abs = 0.0
        self.s_sq = 0.0
        self.s_d = 0.0
        self.s_x = 0.0
        self.s_y = 0.0
        self.s_xx = 0.0
        self.s_yy = 0.0
        self.s_xy = 0.0
        self.neg = 0
        self.n = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one tile's ALREADY-FILTERED valid pixels (1-D arrays)."""
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        d = pred - target
        self.s_abs += float(np.abs(d).sum())
        self.s_sq += float((d**2).sum())
        self.s_d += float(d.sum())
        self.s_x += float(pred.sum())
        self.s_y += float(target.sum())
        self.s_xx += float((pred**2).sum())
        self.s_yy += float((target**2).sum())
        self.s_xy += float((pred * target).sum())
        self.neg += int((pred < 0).sum())
        self.n += int(pred.size)

    def to_metrics(self) -> Dict[str, float]:
        n = self.n
        if n == 0:
            return {
                "n": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "bias": float("nan"),
                "pearson_r": float("nan"),
                "neg_frac_pred": float("nan"),
            }
        mean_x, mean_y = self.s_x / n, self.s_y / n
        cov = self.s_xy / n - mean_x * mean_y
        var_x = self.s_xx / n - mean_x**2
        var_y = self.s_yy / n - mean_y**2
        r = (
            float(cov / np.sqrt(var_x * var_y))
            if var_x > 0 and var_y > 0
            else float("nan")
        )
        return {
            "n": int(n),
            "mae": float(self.s_abs / n),
            "rmse": float(np.sqrt(self.s_sq / n)),
            "bias": float(self.s_d / n),
            "pearson_r": r,
            "neg_frac_pred": float(self.neg / n),
        }


def pooled_stats_from_tiles(
    pixels_pred: Iterable[np.ndarray],
    pixels_target: Iterable[np.ndarray],
    pixels_mask: Optional[Iterable[np.ndarray]] = None,
) -> Dict[str, float]:
    """PooledStats over per-tile [H,W] arrays (masks default: finite target)."""
    stats = PooledStats()
    if pixels_mask is None:
        for p, t in zip(pixels_pred, pixels_target):
            m = np.isfinite(t)
            stats.update(np.asarray(p)[m], np.asarray(t)[m])
    else:
        for p, t, m in zip(pixels_pred, pixels_target, pixels_mask):
            m = np.asarray(m, dtype=bool)
            stats.update(np.asarray(p)[m], np.asarray(t)[m])
    return stats.to_metrics()
