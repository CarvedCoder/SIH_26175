"""Property tests for depthwizard.splits — written AFTER the v1.1 bugfix.

The v1 bug (seed-shuffled chunk→split assignment in block mode) produced
15.6/34.4/50.0 instead of 70/15/15 on real DFC2019 Track-1 data and slipped
through because NO split tests existed. These tests pin the statistical and
structural invariants so this class of bug cannot ship silently again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.splits import (achieved_fractions, assert_no_overlap,
                                make_splits)

# Realistic universe mimicking DFC2019 Track 1: two cities, many rows,
# moderately unequal tiles-per-row (the property v1 ignored).
def _city(city, rows, lo, hi, rng):
    out = []
    for r in rows:
        w = rng.randint(lo, hi)
        out += [f"{city}_{r:03d}_{c:02d}" for c in range(1, w + 1)]
    return out

_r = __import__("random").Random(7)
STEMS = (_city("JAX", list(range(4, 64)), 8, 25, _r)      # 60 rows
         + _city("OMA", list(range(300, 380)), 8, 25, _r))  # 80 rows

# Pathological set: single rows hold ~40% of a city's tiles. Contiguous
# row bands cannot hit 70/15/15 here — used to assert graceful degradation
# (non-empty splits, coverage, guard warning), NOT exact fractions.
HEAVY_STEMS = (
    [f"JAX_{r:03d}_{c:02d}" for r in (4, 17, 18, 20, 22) for c in range(1, 12)]
    + [f"JAX_{r:03d}_{c:02d}" for r in (26,) for c in range(1, 40)]   # heavy row
    + [f"OMA_{r:03d}_{c:02d}" for r in (329, 331, 332, 342) for c in range(1, 25)]
    + [f"OMA_{r:03d}_{c:02d}" for r in (353,) for c in range(1, 45)]  # heavy row
)


def _shares(splits):
    n = sum(len(v) for v in splits.values())
    return {k: len(v) / n for k, v in splits.items()}


def test_block_mode_achieves_target_fractions():
    """THE regression test for the v1.1 bug — v1 gave 15.6/34.4/50.0 here."""
    s = make_splits(STEMS, mode="block", fractions=(0.7, 0.15, 0.15), seed=42)
    sh = _shares(s)
    assert abs(sh["train"] - 0.70) < 0.05, f"train share {sh['train']:.1%}"
    assert abs(sh["val"] - 0.15) < 0.05, f"val share {sh['val']:.1%}"
    assert abs(sh["test"] - 0.15) < 0.05, f"test share {sh['test']:.1%}"


def test_no_overlap_and_full_coverage():
    s = make_splits(STEMS, mode="block", seed=7)
    assert_no_overlap(s)  # raises if any tile is in two splits
    assert sum(len(v) for v in s.values()) == len(set(STEMS))


def test_deterministic_same_seed():
    a = make_splits(STEMS, mode="block", seed=42)
    b = make_splits(STEMS, mode="block", seed=42)
    assert a == b


def test_block_rows_form_contiguous_bands_per_city():
    """Within one split, a city's rows must be a contiguous slice of that
    city's sorted row list — this is the whole point of block mode."""
    s = make_splits(STEMS, mode="block", seed=42)
    for city in ("JAX", "OMA"):
        all_rows = sorted({int(st.split("_")[1])
                           for st in STEMS if st.startswith(city)})
        band_found = False
        for lst in s.values():
            rows = sorted({int(st.split("_")[1])
                           for st in lst if st.startswith(city)})
            if not rows:
                continue
            i0 = all_rows.index(rows[0])
            assert all_rows[i0:i0 + len(rows)] == rows, (
                f"{city}: split rows {rows} are not a contiguous band")
            band_found = True
        assert band_found


def test_tile_mode_hit_fractions_tightly():
    s = make_splits(STEMS, mode="tile", fractions=(0.7, 0.15, 0.15), seed=42)
    sh = _shares(s)
    assert abs(sh["train"] - 0.70) < 0.01
    assert abs(sh["val"] - 0.15) < 0.01
    assert abs(sh["test"] - 0.15) < 0.01


def test_achieved_fractions_helper():
    s = make_splits(STEMS, mode="block", seed=1)
    af = achieved_fractions(s)
    assert set(af) == {"train", "val", "test"}
    assert abs(sum(af.values()) - 1.0) < 1e-9


def test_pathological_heavy_rows_degrade_gracefully(capsys):
    """Contiguous bands can't hit exact fractions when one row dominates;
    the split must still be valid and the drift guard must warn."""
    s = make_splits(HEAVY_STEMS, mode="block", fractions=(0.7, 0.15, 0.15),
                    seed=42)
    assert all(len(s[k]) > 0 for k in ("train", "val", "test"))
    assert_no_overlap(s)
    assert sum(len(v) for v in s.values()) == len(set(HEAVY_STEMS))
    out = capsys.readouterr().out
    assert "[warn]" in out  # the v1.1 drift guard fired


def test_tiny_city_fallback_nonempty():
    """A city with < 3 rows must not crash or produce empty splits."""
    stems = [f"XXX_{r:03d}_{c:02d}" for r in (1, 2) for c in range(1, 6)]
    s = make_splits(stems, mode="block", seed=42)
    assert all(len(s[k]) > 0 for k in ("train", "val", "test"))
    assert_no_overlap(s)

def test_bad_mode_and_fractions_rejected():
    with pytest.raises(ValueError):
        make_splits(STEMS[:10], mode="nope")
    with pytest.raises(ValueError):
        make_splits(STEMS[:10], fractions=(0.5, 0.5, 0.5))
