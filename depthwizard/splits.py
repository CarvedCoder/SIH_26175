"""Scene-level train/val/test splits.

Why NOT random-pixel or naive random-tile splits (spec sec. 7.7):
  * Random pixels leak: neighbouring pixels share the same rooftop/ground
    object, so a random split measures interpolation, not generalization.
  * Even random TILE assignment leaks in US3D-style data: neighbouring tiles
    were cut from a larger orthomosaic and can contain the same buildings.

Therefore two modes are provided:
  tile  — seeded random assignment of whole tiles (minimum acceptable).
  block — tiles grouped by city + contiguous ROW bands; whole bands are
          assigned to a split. Recommended default: it guarantees that any
          two tiles sharing a row-range never straddle two splits except at
          band boundaries. (A full geographic-buffer split comes with the
          Phase 8 geospatial pass, where real CRS bounds are available.)

v1.1 BUGFIX (block mode):
  v1 sized the three chunks by ROW count and then seed-shuffled the
  chunk→split assignment, so the 70% chunk could land on val or test. On
  DFC2019 Track 1 it did exactly that (achieved 15.6/34.4/50.0). Now:
    * chunk boundaries are optimized to match the target fractions in
      TILE counts (rows hold very unequal tile counts),
    * the assignment is deterministic: biggest chunk → train, then val, test,
    * a post-hoc guard warns if any split drifts >5 pp from its target.

The output is a single splits.json consumed by every later stage, so the
split is decided ONCE and frozen — re-splitting between experiments would
make baselines incomparable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from .geo import parse_stem

VALID_SPLITS = ("train", "val", "test")


def _chunk_rows(
    rows: List[int], weights: List[int], fractions: Tuple[float, float, float]
) -> List[List[int]]:
    """Split sorted row indices into 3 contiguous chunks whose *tile counts*
    (weights) best match `fractions` (train, val, test).

    v1.1: exhaustive search over the two cut points, minimizing
    sum_k |share_k − target_k|. O(n^2) with n = rows per city (<= ~100),
    so this is effectively free. Deterministic; no RNG involved.
    """
    n = len(rows)
    if n < 3:
        return [rows]
    total = float(sum(weights))
    best_loss, best_ij = None, (1, n - 1)
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            w_tr = prefix[i] - prefix[0]
            w_va = prefix[j] - prefix[i]
            w_te = prefix[n] - prefix[j]
            loss = (
                abs(w_tr / total - fractions[0])
                + abs(w_va / total - fractions[1])
                + abs(w_te / total - fractions[2])
            )
            if best_loss is None or loss < best_loss - 1e-12:
                best_loss, best_ij = loss, (i, j)
    i, j = best_ij
    return [rows[:i], rows[i:j], rows[j:]]


def make_splits(
    stems: List[str],
    mode: str = "block",
    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]}.

    Deterministic for a given (stems, mode, fractions, seed).
    """
    if mode not in ("tile", "block"):
        raise ValueError(f"Unknown split mode '{mode}' (expected 'tile' or 'block')")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    stems = sorted(set(stems))
    rng = random.Random(seed)

    if mode == "tile":
        order = list(stems)
        rng.shuffle(order)
        n_tr = max(1, round(fractions[0] * len(order)))
        n_va = max(1, round(fractions[1] * len(order)))
        return {
            "train": sorted(order[:n_tr]),
            "val": sorted(order[n_tr : n_tr + n_va]),
            "test": sorted(order[n_tr + n_va :]),
        }

    # ---- block mode ----------------------------------------------------
    by_city_rows: Dict[str, Dict[int, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for s in stems:
        city, row, _col = parse_stem(s)
        by_city_rows[city][row].append(s)

    assignment: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_names = list(VALID_SPLITS)

    for city in sorted(by_city_rows):
        rows = sorted(by_city_rows[city])
        weights = [len(by_city_rows[city][r]) for r in rows]
        chunks = _chunk_rows(rows, weights, fractions)
        # v1.1: deterministic assignment — train, val, test (no shuffle).
        # The seed-shuffle here was the v1 bug: it threw the 70% chunk at a
        # random split (val for JAX, test for OMA on the real data).
        for k, split in enumerate(split_names):
            if k >= len(chunks):
                break
            for r in chunks[k]:
                assignment[split].extend(by_city_rows[city][r])

    # v1.1 guard: report drift between achieved and target tile fractions.
    n_tot = sum(len(v) for v in assignment.values())
    for split, f in zip(VALID_SPLITS, fractions):
        achieved = len(assignment[split]) / n_tot
        if abs(achieved - f) > 0.05:
            print(
                f"[warn] split '{split}' achieved {achieved:.1%} vs target "
                f"{f:.1%} — spatial blocks cannot always hit exact fractions; "
                "consider mode='tile' if exactness matters more than blocking"
            )

    # Guard: every split must be non-empty; degrade to tile mode otherwise.
    if any(len(assignment[s]) == 0 for s in VALID_SPLITS):
        return make_splits(stems, mode="tile", fractions=fractions, seed=seed)

    return {s: sorted(v) for s, v in assignment.items()}


def achieved_fractions(splits: Dict[str, List[str]]) -> Dict[str, float]:
    """Tile-share of each split — call after make_splits for reporting."""
    n = sum(len(v) for v in splits.values())
    if not n:
        return {s: 0.0 for s in VALID_SPLITS}
    return {s: len(splits[s]) / n for s in VALID_SPLITS}


def assert_no_overlap(splits: Dict[str, List[str]]) -> None:
    """Hard invariant — call this before writing splits.json."""
    seen: Dict[str, str] = {}
    for split, lst in splits.items():
        for stem in lst:
            if stem in seen:
                raise AssertionError(
                    f"Tile '{stem}' appears in both '{seen[stem]}' and '{split}' — split leaks!"
                )
            seen[stem] = split


def summarize(splits: Dict[str, List[str]]) -> str:
    lines = ["split   tiles"]
    for s in VALID_SPLITS:
        lines.append(f"{s:6s}  {len(splits[s]):5d}")
    return "\n".join(lines)
