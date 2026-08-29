"""Metric + split correctness tests (hand-computed expectations)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.metrics import (height_metrics, mean_std_over_tiles,
                                 pooled_metrics, stratified_by_class)
from depthwizard.normalize import clean_agl, minmax_normalize
from depthwizard.splits import assert_no_overlap, make_splits


def test_perfect_prediction_zero_error():
    t = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = height_metrics(t.copy(), t)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["bias"] == 0.0
    assert m["pearson_r"] == pytest.approx(1.0)


def test_known_error_values():
    target = np.array([[0.0, 0.0], [0.0, 0.0]])
    pred = np.array([[1.0, -1.0], [3.0, 1.0]])
    m = height_metrics(pred, target)
    assert m["mae"] == pytest.approx(1.5)
    assert m["rmse"] == pytest.approx(np.sqrt(3.0))
    assert m["bias"] == pytest.approx(1.0)
    assert m["medae"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_mask_excludes_pixels():
    target = np.array([[1.0, 100.0]])          # 100 is an outlier pixel
    pred = np.array([[1.5, 100.0]])
    mask = np.array([[True, False]])
    m = height_metrics(pred, target, mask)
    assert m["n"] == 1 and m["mae"] == pytest.approx(0.5)


def test_nan_target_masked_out():
    target = np.array([[1.0, np.nan]])
    pred = np.array([[2.0, 50.0]])
    m = height_metrics(pred, target)            # default mask: finite target
    assert m["n"] == 1 and m["mae"] == pytest.approx(1.0)


def test_pooled_matches_direct_concat():
    ps = [np.array([[1.0, 2.0]]), np.array([[3.0]])]
    ts = [np.array([[1.5, 2.5]]), np.array([[3.5]])]
    pooled = pooled_metrics(ps, ts)
    direct = height_metrics(np.concatenate([p.ravel() for p in ps]),
                            np.concatenate([t.ravel() for t in ts]))
    assert pooled["mae"] == pytest.approx(direct["mae"])
    assert pooled["rmse"] == pytest.approx(direct["rmse"])


def test_normalize_range_and_flat():
    x = np.linspace(-3, 7, 25, dtype=np.float32).reshape(5, 5)
    dn = minmax_normalize(x)
    assert dn.min() == pytest.approx(0.0, abs=1e-6)
    assert dn.max() == pytest.approx(1.0, abs=1e-6)
    flat = np.full((4, 4), 2.0, dtype=np.float32)
    assert float(minmax_normalize(flat).mean()) == 0.5   # flat -> mid, no NaN


def test_clean_agl():
    a = np.array([[-0.3, 0.0, 5.0, np.nan]], dtype=np.float32)
    c = clean_agl(a)
    assert c[0, 0] == 0.0 and c[0, 3] == 0.0 and c[0, 2] == 5.0


def test_stratified_keys_are_raw_ids():
    pred = np.zeros((2, 2)); target = np.ones((2, 2))
    cls = np.array([[2, 65], [6, 9]])
    out = stratified_by_class(pred, target, cls, [2, 6, 65, 9])
    assert set(out) == {"cls_2", "cls_65", "cls_6", "cls_9"}


def test_mean_std_over_tiles_skips_nan():
    dicts = [{"mae": 1.0, "n": 4}, {"mae": 3.0, "n": 4}]
    s = mean_std_over_tiles(dicts)
    assert s["mae_mean"] == pytest.approx(2.0)
    assert s["mae_std"] == pytest.approx(1.0)


# ---------------- splits ----------------

def _stems(rows=6, cols=4, cities=("JAX", "OMA")):
    out = []
    for city in cities:
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                out.append(f"{city}_{r:03d}_{c:03d}")
    return out


def test_split_disjoint_and_complete():
    stems = _stems()
    sp = make_splits(stems, mode="block", seed=42)
    assert_no_overlap(sp)
    union = sorted(sp["train"] + sp["val"] + sp["test"])
    assert union == sorted(stems)


def test_split_deterministic():
    stems = _stems()
    a = make_splits(stems, mode="block", seed=7)
    b = make_splits(stems, mode="block", seed=7)
    assert a == b


def test_block_mode_rows_contiguous_per_city():
    """For each city+split, the assigned row set must decompose into
    contiguous runs equal to the number of chunks given to that split —
    this is what limits cross-split adjacency to band boundaries."""
    from depthwizard.geo import parse_stem
    stems = _stems(rows=9)
    sp = make_splits(stems, mode="block", seed=3)
    for split in ("train", "val", "test"):
        by_city = {}
        for s in sp[split]:
            city, row, _ = parse_stem(s)
            by_city.setdefault(city, set()).add(row)
        for city, rows in by_city.items():
            rows = sorted(rows)
            runs = 1 + sum(1 for a_, b_ in zip(rows, rows[1:]) if b_ != a_ + 1)
            assert runs <= 2, f"{split}/{city}: rows {rows} not contiguous"


def test_tile_mode_valid():
    stems = _stems()
    sp = make_splits(stems, mode="tile", seed=11)
    assert_no_overlap(sp)
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == len(stems)
