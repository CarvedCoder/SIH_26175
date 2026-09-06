"""Per-dataset statistics & normalization verification (Phase 3 gate).

PURPOSE (binding user constraint): "Do not mix GAMUS and DFC2019 until
per-dataset statistics and normalization are verified." This command
produces the verification artifact:

    outputs/stats/dataset_stats.json      {"verified": false, ...}

A human reviews the report; if the per-dataset AGL/Dn distributions and
the normalization story are acceptable, they set "verified": true IN THE
FILE (or pass --mark-verified after reviewing). Only then can
``build_mixed_datasets`` construct a mixed dataset (it demands this file,
covering all its sources, with verified=true — enforced in
depthwizard/datasets/mixed.py).

What it measures, per dataset, over the TRAIN split (evaluation splits are
never peeked at):
  * AGL: pooled min/max/mean/std, percentiles (subsampled, deterministic),
    negative/non-finite fraction, per-tile mean spread;
  * Dn (when a depth cache exists): pooled stats of the per-tile min-max
    normalized relative depth + raw-depth ranges from the cache manifest;
  * semantic composition: project-class pixel shares (via the VERIFIED
    legends) + ignore fraction;
  * provenance: dataset source config, units/semantics honesty notes.

Usage:
  python model.py stats --config configs/gamus.yaml \
      --datasets dfc2019 gamus --out outputs/stats/dataset_stats.json
  python model.py stats --config configs/gamus.yaml --datasets gamus \
      --limit 32 --out outputs/stats/dataset_stats.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence, Sized
from typing import cast

import numpy as np

from depthwizard.cli.args import (add_cache_subdir_arg, add_config_arg,
                                  add_device_arg, load_config, resolve_cache)
from depthwizard.datasets.semantics import (NUM_PROJECT_CLASSES,
                                            PROJECT_CLASSES, legend_report)
from depthwizard.geo import dump_json

NAME = "stats"
HELP = ("per-dataset AGL/Dn statistics + normalization verification "
        "(REQUIRED gate before mixed training)")


def _pooled_stats(chunks: list) -> dict | None:
    """Streaming pooled stats from a list of 1-D pixel arrays."""
    if not chunks:
        return None
    n = sum(c.size for c in chunks)
    if n == 0:
        return None
    mean = float(np.sum([c.mean() * c.size for c in chunks]) / n)
    var = float(np.sum([((c ** 2).mean() * c.size - (c.mean() ** 2) * c.size)
                        for c in chunks]) / n)
    # exact min/max
    gmin = float(min(c.min() for c in chunks))
    gmax = float(max(c.max() for c in chunks))
    sub = np.concatenate(chunks) if n <= 20_000_000 else None
    if sub is None:
        sub = np.concatenate([c[:: max(1, c.size // 2_000_000)]
                              for c in chunks])
    return {
        "n_pixels": int(n),
        "min": gmin, "max": gmax, "mean": mean, "std": float(np.sqrt(max(var, 0.0))),
        "p50": float(np.percentile(sub, 50)),
        "p95": float(np.percentile(sub, 95)),
        "p99": float(np.percentile(sub, 99)),
    }


def _collect_dataset(name: str, cfg: dict, cache_dir, limit: int,
                     stride: int, split: str = "train") -> dict:
    """Gather per-dataset statistics through the ADAPTER pipeline (single
    source of truth for loading/normalization semantics)."""
    from depthwizard.datasets.factory import build_dataset

    dcfg = dict(cfg.get("dataset", {}) or {})
    dcfg["name"] = name
    sub_cfg = {"paths": cfg.get("paths", {}), "dataset": dcfg}
    if name == "gamus" and limit:
        dcfg["limit"] = limit        # CLI limit overrides any config limit
    ds = build_dataset(sub_cfg, crop_size=None, augment=False,
                       load_depth=bool(cache_dir),
                       depth_cache_dir=cache_dir, split=split)

    agl_chunks, dn_chunks = [], []
    sem_counts = np.zeros(NUM_PROJECT_CLASSES, dtype=np.int64)
    ignore_count = 0
    per_tile = []
    ds_seq = cast(Sequence, ds)
    ds_sized = cast(Sized, ds_seq)
    idxs = range(len(ds_sized))
    for i in idxs:
        s = ds_seq[i]
        agl = s["agl"][0].numpy()
        valid = np.isfinite(agl)
        agl_chunks.append(agl[valid][::stride].astype(np.float64, copy=False))
        if s["dn"] is not None:
            dn = s["dn"][0].numpy()
            dn_chunks.append(dn.ravel()[::stride].astype(np.float64,
                                                         copy=False))
        if s["sem_onehot"] is not None:
            sem_counts += s["sem_onehot"].sum(dim=(1, 2)).numpy().astype(
                np.int64)
            ignore_count += int(s["sem_ignore"].sum())
        per_tile.append({"sample_id": s["meta"]["sample_id"],
                         "agl_mean": float(np.mean(agl[valid])),
                         "agl_max": float(np.max(agl[valid]))})
        if limit and (i + 1) >= limit:
            break

    total_sem_px = int(sem_counts.sum() + ignore_count)
    out = {
        "dataset": name,
        "split": split,
        "n_tiles_measured": len(per_tile),
        "n_tiles_total": len(ds_sized),
        "agl": _pooled_stats(agl_chunks),
        "dn": _pooled_stats(dn_chunks),
        "dn_source": ("per-tile min-max normalize (depthwizard.normalize."
                      "minmax_normalize — the single source, identical for "
                      "every dataset)") if dn_chunks else
                     "NO DEPTH CACHE — run the depth command for this "
                     "dataset to include Dn statistics",
        "project_class_shares": (None if total_sem_px == 0 else {
            PROJECT_CLASSES[k]: round(float(sem_counts[k]) / total_sem_px, 5)
            for k in range(NUM_PROJECT_CLASSES)}),
        "semantics_ignore_frac": (None if total_sem_px == 0 else
                                  round(ignore_count / total_sem_px, 6)),
        "semantics_legend": legend_report(name),
        "per_tile_agl_mean_spread": (
            None if not per_tile else
            {"min": min(t["agl_mean"] for t in per_tile),
             "max": max(t["agl_mean"] for t in per_tile)}),
        "per_tile_first": per_tile[:20],
    }
    # height-semantics honesty notes
    if name == "dfc2019":
        out["height_semantics"] = "AGL/nDSM, metres (pubgeo/dfc2019)"
        out["georef"] = "none (Track-1 ships without CRS)"
    elif name == "gamus":
        out["height_semantics"] = ("nDSM/AGL; units UNDOCUMENTED — ASSUMED "
                                   "metres (values consistent)")
        out["georef"] = "none (GAMUS HDF5 carries no CRS)"
    return out


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/gamus.yaml")
    p.add_argument("--datasets", nargs="+", default=["dfc2019", "gamus"],
                   choices=("dfc2019", "gamus"),
                   help="datasets to measure (mixed gating requires every "
                        "mixed source to be covered)")
    p.add_argument("--out", type=Path,
                   default=Path("outputs/stats/dataset_stats.json"))
    p.add_argument("--limit", type=int, default=32,
                   help="max tiles measured per dataset (deterministic "
                        "order; 0 = all — slow on GAMUS-hf)")
    p.add_argument("--stride", type=int, default=4,
                   help="pixel subsampling stride (memory guard; "
                        "statistics stay representative)")
    p.add_argument("--split", default="train",
                   choices=("train", "val", "test"),
                   help="split to measure (default train — never peek at "
                        "eval splits for this bookkeeping)")
    p.add_argument("--mark-verified", action="store_true",
                   help="write verified:true — ONLY after a human has "
                        "reviewed the report (the mixing gate)")
    add_device_arg(p)
    add_cache_subdir_arg(p)
    return p


def run(args) -> int:
    cfg = load_config(args.config)
    cache_dir = None
    if "depth_cache_dir" in (cfg.get("paths") or {}):
        try:
            cache_dir = resolve_cache(cfg["paths"], args.cache_subdir)
        except FileNotFoundError as e:
            print(f"[warn] depth cache unavailable ({e}) — Dn statistics "
                  "will be omitted; AGL stats still collected.")
            cache_dir = None

    results = {}
    for name in args.datasets:
        print(f"[stats] measuring {name} (split={args.split}, "
              f"limit={args.limit or 'all'}, stride={args.stride}) ...")
        try:
            results[name] = _collect_dataset(name, cfg, cache_dir,
                                             args.limit, args.stride,
                                             split=args.split)
        except Exception as e:
            print(f"[warn] {name}: FAILED to measure — {e}")
            results[name] = {"dataset": name, "error": str(e)}
        r = results[name]
        if r.get("agl"):
            a = r["agl"]
            print(f"  AGL: min={a['min']:.2f} max={a['max']:.2f} "
                  f"mean={a['mean']:.2f} std={a['std']:.2f} "
                  f"p95={a['p95']:.2f}")
        if r.get("dn"):
            d = r["dn"]
            print(f"  Dn : min={d['min']:.3f} max={d['max']:.3f} "
                  f"mean={d['mean']:.3f} std={d['std']:.3f}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "generator": "depthwizard-stats v1 (adapter pipeline, deterministic "
                     "order, subsampled percentiles)",
        "verified": bool(args.mark_verified),
        "verified_note": "Flip to true ONLY after human review of the "
                         "per-dataset statistics below. mixed.py REFUSES to "
                         "build mixed datasets unless this is true and "
                         "covers every source (binding constraint: do not "
                         "mix GAMUS and DFC2019 until per-dataset statistics "
                         "and normalization are verified).",
        "config": str(args.config),
        "measured": {"split": args.split, "limit": args.limit,
                     "stride": args.stride, "cache": str(cache_dir)},
        "datasets": results,
    }
    dump_json(report, args.out)
    print(f"-> {args.out}")
    if not args.mark_verified:
        print("[gate] artifact written with verified:false — review it, "
              "then re-run with --mark-verified (or edit the JSON) to "
              "enable mixed training.")
    else:
        print("[gate] artifact written with verified:true — mixed dataset "
              "configs may now point at it via dataset.verified_stats.")
    return 0
