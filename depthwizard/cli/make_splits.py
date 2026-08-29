"""Freeze scene-level train/val/test splits.

Reads the tile inventory directly from disk (same discovery as ``inspect``),
then writes ONE splits.json that every later stage must reuse. Re-splitting
between experiments makes baselines incomparable — don't.

Default mode is 'block' (city + contiguous row bands) to limit leakage from
overlapping neighbouring US3D tiles; see depthwizard/splits.py for the
reasoning and the 'tile' fallback.

Usage:
  python main.py splits \
      --rgb-dir  rgb_data/Train-Track1-RGB/Track1-RGB \
      --truth-dir rgb_data_truth/Train-Track1-Truth/Track1-Truth \
      --out outputs/splits/splits.json \
      --mode block --fractions 0.7 0.15 0.15 --seed 42
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from depthwizard.geo import discover_tiles, dump_json, parse_stem
from depthwizard.splits import (achieved_fractions, assert_no_overlap,
                                make_splits, summarize)

NAME = "splits"
HELP = "freeze scene-level train/val/test splits (block mode default)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb-dir", required=True, type=Path)
    p.add_argument("--truth-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/splits/splits.json"))
    p.add_argument("--mode", choices=("block", "tile"), default="block")
    p.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=42)
    return p


def run(args) -> int:
    tiles, problems = discover_tiles(args.rgb_dir, args.truth_dir)
    if problems:
        print("[warn] orphan files found (fix before training if unexpected):")
        for p in problems:
            print("   -", p)
    stems = [t.stem for t in tiles]
    if len(stems) < 3:
        print("[error] fewer than 3 tiles discovered — nothing to split.")
        return 1

    splits = make_splits(stems, mode=args.mode, fractions=tuple(args.fractions),
                         seed=args.seed)
    assert_no_overlap(splits)   # hard invariant

    # per-split city/row summary so leakage is visible at a glance
    detail = {}
    for split, lst in splits.items():
        rows_by_city: dict = {}
        for s in lst:
            city, row, _ = parse_stem(s)
            rows_by_city.setdefault(city, set()).add(row)
        detail[split] = {c: sorted(r) for c, r in sorted(rows_by_city.items())}

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "generator": "depthwizard-splits v1.1 (weighted block cuts, deterministic assignment)",
        "mode": args.mode, "seed": args.seed,
        "fractions": list(args.fractions),
        "counts": {k: len(v) for k, v in splits.items()},
        "achieved_fractions": {k: round(f, 4) for k, f in achieved_fractions(splits).items()},
        "rows_by_city": detail,
        "splits": splits,
    }
    dump_json(payload, args.out)
    print(summarize(splits))
    print("achieved fractions (tiles):",
          {k: f"{f:.1%}" for k, f in achieved_fractions(splits).items()})
    print("rows per split (leakage check):")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {detail[split]}")
    print(f"-> {args.out}")
    return 0
