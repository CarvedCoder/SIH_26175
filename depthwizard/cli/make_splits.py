"""Freeze scene-level train/val/test splits.

DFC2019 mode (default — UNCHANGED behavior): reads the tile inventory
from disk (same discovery as ``inspect``), then writes ONE splits.json that
every later stage must reuse. Re-splitting between experiments makes
baselines incomparable — don't.

GAMUS mode (--dataset gamus): GAMUS ships OFFICIAL train/val/test splits
(scene-level by construction of the release). We do NOT re-split them —
this command simply freezes the official split inventory into a manifest
JSON (sorted sample ids per split) that ``list_gamus_samples`` can reuse
offline. Deterministic, verbatim, auditable.

Usage (DFC2019):
  python model.py splits \
      --rgb-dir  rgb_data/Train-Track1-RGB/Track1-RGB \
      --truth-dir rgb_data_truth/Train-Track1-Truth/Track1-Truth \
      --out outputs/splits/splits.json \
      --mode block --fractions 0.7 0.15 0.15 --seed 42

Usage (GAMUS):
  python model.py splits --dataset gamus \
      --gamus-source hf --out outputs/splits/gamus_splits.json
  python model.py splits --dataset gamus --gamus-source local \
      --gamus-local-root <dir> --out outputs/splits/gamus_splits.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from depthwizard.geo import discover_tiles, dump_json, parse_stem
from depthwizard.splits import (
    achieved_fractions,
    assert_no_overlap,
    make_splits,
    summarize,
)

NAME = "splits"
HELP = "freeze scene-level train/val/test splits (block mode default)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=("dfc2019", "gamus"),
        default="dfc2019",
        help="dfc2019: our block/tile splits (unchanged); gamus: "
        "freeze the OFFICIAL split inventory into a "
        "manifest (never re-split official splits)",
    )
    p.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="DFC2019 RGB dir (required for --dataset dfc2019)",
    )
    p.add_argument(
        "--truth-dir",
        type=Path,
        default=None,
        help="DFC2019 truth dir (required for --dataset dfc2019)",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/splits/splits.json"))
    p.add_argument("--mode", choices=("block", "tile"), default="block")
    p.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=42)
    # ---- GAMUS source options (ignored for dfc2019) ----
    p.add_argument(
        "--gamus-source",
        choices=("hf", "local"),
        default="hf",
        help="hf: lazy per-file download (primary); local: "
        "offline/pre-downloaded raw-layout directory",
    )
    p.add_argument("--gamus-local-root", type=Path, default=None)
    p.add_argument(
        "--gamus-manifest",
        type=Path,
        default=None,
        help="existing manifest to reuse (skips listing)",
    )
    p.add_argument("--gamus-hf-cache", type=Path, default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="GAMUS: cap ids per split in the manifest (0 = all; "
        "deterministic sorted order)",
    )
    return p


def _run_gamus(args) -> int:
    from depthwizard.datasets.gamus import GAMUSConfig, list_gamus_samples

    cfg = GAMUSConfig(
        source=args.gamus_source,
        local_root=args.gamus_local_root,
        manifest=args.gamus_manifest,
        save_manifest=args.out,
        hf_cache_dir=args.gamus_hf_cache,
        limit=args.limit,
    )
    per_split, problems = list_gamus_samples(cfg)
    if problems:
        print("[warn] GAMUS triple-completeness problems:")
        for pr in problems:
            print("   -", pr)
    print("[gamus] OFFICIAL splits frozen verbatim (no re-splitting):")
    for split, samples in per_split.items():
        print(f"  {split:5s}: {len(samples)} tiles")
    print(f"-> {args.out}")
    print(
        "[note] GAMUS official splits are scene-level by construction of "
        "the release; we reuse them verbatim."
    )
    return 1 if problems else 0


def run(args) -> int:
    if args.dataset == "gamus":
        return _run_gamus(args)

    # ---- DFC2019 path (unchanged behavior) ----
    if args.rgb_dir is None or args.truth_dir is None:
        print("[error] --dataset dfc2019 requires --rgb-dir and --truth-dir.")
        return 1
    tiles, problems = discover_tiles(args.rgb_dir, args.truth_dir)
    if problems:
        print("[warn] orphan files found (fix before training if unexpected):")
        for p in problems:
            print("   -", p)
    stems = [t.stem for t in tiles]
    if len(stems) < 3:
        print("[error] fewer than 3 tiles discovered — nothing to split.")
        return 1

    splits = make_splits(
        stems, mode=args.mode, fractions=tuple(args.fractions), seed=args.seed
    )
    assert_no_overlap(splits)  # hard invariant

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
        "mode": args.mode,
        "seed": args.seed,
        "fractions": list(args.fractions),
        "counts": {k: len(v) for k, v in splits.items()},
        "achieved_fractions": {
            k: round(f, 4) for k, f in achieved_fractions(splits).items()
        },
        "rows_by_city": detail,
        "splits": splits,
    }
    dump_json(payload, args.out)
    print(summarize(splits))
    print(
        "achieved fractions (tiles):",
        {k: f"{f:.1%}" for k, f in achieved_fractions(splits).items()},
    )
    print("rows per split (leakage check):")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {detail[split]}")
    print(f"-> {args.out}")
    return 0
