"""PooledStats (streaming) must equal frozen pooled_metrics exactly.

This is the dedup guarantee: eval-baseline / dummies stream tiles through
PooledStats, the citable `evaluate` uses metrics.pooled_metrics — same
pixels must yield the same numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.metrics import pooled_metrics
from depthwizard.streaming import PooledStats, pooled_stats_from_tiles


def _tiles(seed=0):
    rng = np.random.default_rng(seed)
    preds = [rng.normal(5, 3, (16, 16)) for _ in range(3)]
    targets = [rng.normal(4, 3, (16, 16)) for _ in range(3)]
    masks = [rng.random((16, 16)) > 0.1 for _ in range(3)]
    return preds, targets, masks


def test_streaming_equals_frozen_pooled():
    preds, targets, masks = _tiles()
    frozen = pooled_metrics(preds, targets, masks)
    streamed = pooled_stats_from_tiles(preds, targets, masks)
    for k in ("mae", "rmse", "bias", "pearson_r", "neg_frac_pred", "n"):
        assert streamed[k] == pytest.approx(frozen[k], nan_ok=True), k


def test_accumulator_update_matches_direct():
    preds, targets, _masks = _tiles(seed=1)
    stats = PooledStats()
    for p, t in zip(preds, targets):
        stats.update(p.ravel(), t.ravel())           # mask = all valid
    direct = pooled_metrics(preds, targets, None)
    got = stats.to_metrics()
    assert got["n"] == direct["n"]
    assert got["mae"] == pytest.approx(direct["mae"])
    assert got["rmse"] == pytest.approx(direct["rmse"])


def test_empty_stats_nan_safe():
    m = PooledStats().to_metrics()
    assert m["n"] == 0 and np.isnan(m["mae"])


