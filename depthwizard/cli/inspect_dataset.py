"""Dataset inspection & alignment verification.

What it checks (writes everything to reports/):
  1. Structure   : RGB/AGL/CLS presence per tile, orphan files, directory walk.
  2. Alignment   : grid shape equality for every triple (hard failure if not),
                   plus 3-panel quicklook PNGs for VISUAL confirmation
                   (rooftops in AGL must sit on rooftops in RGB).
  3. Georef state: CRS / transform per file — expected: none for Track-1.
  4. AGL stats   : min/max/mean, fraction of negative pixels.
  5. Class inventory: unique CLS ids + pixel shares ACROSS ALL TILES
                   (meanings are NOT assigned here — they must come from the
                   official DFC2019 documentation).

Usage:
  python main.py inspect \
      --rgb-dir  rgb_data/Train-Track1-RGB/Track1-RGB \
      --truth-dir rgb_data_truth/Train-Track1-Truth/Track1-Truth \
      --report-dir outputs/reports --quicklooks 8

Exit code 0 = healthy; 1 = fatal structure/alignment problems (CI-friendly).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from depthwizard.geo import (discover_tiles, dump_json, is_georeferenced,
                             parse_stem, read_raster, read_tile,
                             save_quicklook)

NAME = "inspect"
HELP = "dataset structure/alignment/class inspection + quicklooks"


def inspect_tile(paths) -> dict:
    rgb_arr, rgb_prof = read_raster(paths.rgb)
    agl_arr, agl_prof = read_raster(paths.agl)
    cls_arr, cls_prof = read_raster(paths.cls)

    agl = agl_arr[0] if agl_arr.ndim == 3 else agl_arr
    cls = cls_arr[0] if cls_arr.ndim == 3 else cls_arr

    geo_ok, geo_reason = is_georeferenced(rgb_prof)
    aligned = (rgb_arr.shape[-2:] == agl.shape == cls.shape)

    neg_frac = float((~np.isfinite(agl) | (agl < 0)).mean())
    return {
        "stem": paths.stem,
        "rgb_shape": list(rgb_arr.shape), "agl_shape": list(agl.shape),
        "cls_shape": list(cls.shape),
        "rgb_dtype": str(rgb_arr.dtype), "agl_dtype": str(agl_arr.dtype),
        "cls_dtype": str(cls_arr.dtype),
        "rgb_bands": int(rgb_arr.shape[0]),
        "aligned": bool(aligned),
        "georeferenced": geo_ok, "georef_reason": geo_reason,
        "agl_min": float(np.nanmin(agl)), "agl_max": float(np.nanmax(agl)),
        "agl_mean": float(np.nanmean(agl)),
        "agl_nonfinite_or_neg_frac": neg_frac,
        "cls_ids": sorted(int(v) for v in np.unique(cls)),
    }


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb-dir", required=True, type=Path)
    p.add_argument("--truth-dir", required=True, type=Path)
    p.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    p.add_argument("--quicklooks", type=int, default=8,
                   help="how many quicklook PNGs to render (spread over the set)")
    return p


def run(args) -> int:
    args.report_dir.mkdir(parents=True, exist_ok=True)

    tiles, problems = discover_tiles(args.rgb_dir, args.truth_dir)
    print(f"[1/4] discovered {len(tiles)} complete triples, "
          f"{len(problems)} orphan-file problems")

    per_tile, fatal = [], []
    class_counter: Counter = Counter()
    total_px = 0
    for i, t in enumerate(tiles):
        try:
            info = inspect_tile(t)
        except Exception as e:  # unreadable raster / band issues
            fatal.append(f"{t.stem}: UNREADABLE — {e}")
            continue
        per_tile.append(info)
        if not info["aligned"]:
            fatal.append(f"{t.stem}: grid mismatch {info['rgb_shape']} vs "
                         f"{info['agl_shape']} vs {info['cls_shape']}")

        cls_arr, _ = read_raster(t.cls)
        cls = cls_arr[0] if cls_arr.ndim == 3 else cls_arr
        ids, counts = np.unique(cls, return_counts=True)
        for v, c in zip(ids, counts):
            class_counter[int(v)] += int(c)
        total_px += cls.size
        if (i + 1) % 25 == 0 or i + 1 == len(tiles):
            print(f"      inspected {i + 1}/{len(tiles)}")

    # ---------------- summary ----------------
    cities = Counter(parse_stem(t.stem)[0] for t in tiles)
    sizes = Counter((info["rgb_shape"][-2], info["rgb_shape"][-1]) for info in per_tile)
    all_cls_ids = sorted(class_counter)
    class_share = {str(cid): {"pixels": class_counter[cid],
                              "share": round(class_counter[cid] / max(total_px, 1), 5)}
                   for cid in all_cls_ids}

    summary = {
        "n_tiles": len(per_tile),
        "cities": dict(cities),
        "tile_sizes": {f"{h}x{w}": n for (h, w), n in sorted(sizes.items())},
        "n_problems": len(problems),
        "problems": problems,
        "fatal": fatal,
        "agl_negative_pixels_frac_across_tiles": None,
        "class_inventory": class_share,
        "class_meanings": "UNVERIFIED — must be read from the official "
                          "DFC2019/US3D documentation.",
    }
    if per_tile:
        summary["agl_negative_pixels_frac_across_tiles"] = float(np.mean(
            [t["agl_nonfinite_or_neg_frac"] for t in per_tile]))
        summary["agl_min_global"] = min(t["agl_min"] for t in per_tile)
        summary["agl_max_global"] = max(t["agl_max"] for t in per_tile)

    dump_json({"summary": summary, "per_tile": per_tile},
              args.report_dir / "dataset_report.json")

    # ---------------- quicklooks ----------------
    if tiles:
        step = max(1, len(tiles) // max(args.quicklooks, 1))
        chosen = tiles[::step][: args.quicklooks]
        for t in chosen:
            save_quicklook(read_tile(t),
                           args.report_dir / "quicklooks" / f"{t.stem}.png",
                           cls_ids=all_cls_ids)
        print(f"[3/4] wrote {len(chosen)} quicklooks -> "
              f"{args.report_dir / 'quicklooks'}")

    # ---------------- console report ----------------
    print("[4/4] SUMMARY")
    print(f"  tiles: {summary['n_tiles']}  cities: {dict(cities)}  sizes: {summary['tile_sizes']}")
    print(f"  orphans: {len(problems)}   fatal: {len(fatal)}")
    if per_tile:
        print(f"  AGL range: [{summary['agl_min_global']:.3f}, {summary['agl_max_global']:.3f}] m"
              f"   neg/invalid px: {100 * summary['agl_negative_pixels_frac_across_tiles']:.2f}%")
        print(f"  CLS ids: {all_cls_ids}")
        print("  georef: " + ("ALL non-georeferenced (expected for Track-1)"
                              if all(not t["georeferenced"] for t in per_tile)
                              else "MIXED — inspect dataset_report.json"))
    for p in problems[:10]:
        print("  [orphan]", p)
    for f in fatal[:10]:
        print("  [FATAL]", f)

    md = ["# Dataset inspection report", "",
          f"- tiles: **{summary['n_tiles']}**  cities: `{dict(cities)}`",
          f"- tile sizes: `{summary['tile_sizes']}`",
          f"- orphan files: **{len(problems)}**, fatal problems: **{len(fatal)}**", "",
          "## Class inventory (raw ids — meanings UNVERIFIED)", "",
          "| class id | pixels | share |", "|---|---|---|"]
    for cid, v in class_share.items():
        md.append(f"| {cid} | {v['pixels']:,} | {v['share']:.4f} |")
    md += ["", "## Per-tile AGL stats (first 20)", "",
           "| stem | AGL min | AGL max | mean | neg px | aligned |", "|---|---|---|---|---|---|"]
    for t in per_tile[:20]:
        md.append(f"| {t['stem']} | {t['agl_min']:.2f} | {t['agl_max']:.2f} "
                  f"| {t['agl_mean']:.2f} | {t['agl_nonfinite_or_neg_frac']:.4f} "
                  f"| {t['aligned']} |")
    (args.report_dir / "dataset_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  report -> {args.report_dir / 'dataset_report.md'}")

    return 1 if (fatal or not per_tile) else 0
