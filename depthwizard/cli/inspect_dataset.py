"""Dataset inspection & alignment verification.

DFC2019 mode (default): Track-1 triple checks (structure/alignment/georef/
AGL stats/class inventory) — output unchanged except the class-meanings
note, which now cites the VERIFIED pubgeo/dfc2019 legend.

GAMUS mode (--dataset gamus): the same checks over the raw HDF5 layout
(images/heights/classes per official split), including the uint8/float32
CLS dtype inventory and the VERIFIED RSI-MMSegmentation legend. In hf mode
each inspected tile's triple is downloaded lazily — use --limit to bound
downloads.

What it checks (writes everything to reports/):
  1. Structure   : RGB/AGL/CLS presence per tile, orphan files, directory walk.
  2. Alignment   : grid shape equality for every triple (hard failure if not),
                   plus 3-panel quicklook PNGs for VISUAL confirmation
                   (rooftops in AGL must sit on rooftops in RGB).
  3. Georef state: CRS / transform per file — expected: none (Track-1, GAMUS).
  4. AGL stats   : min/max/mean, fraction of negative pixels.
  5. Class inventory: unique CLS ids + pixel shares ACROSS ALL TILES + the
                   VERIFIED dataset-specific legend (official sources).

Usage (DFC2019):
  python model.py inspect \
      --rgb-dir  rgb_data/Train-Track1-RGB/Track1-RGB \
      --truth-dir rgb_data_truth/Train-Track1-Truth/Track1-Truth \
      --report-dir outputs/reports --quicklooks 8

Usage (GAMUS):
  python model.py inspect --dataset gamus --gamus-source local \
      --gamus-local-root <dir> --report-dir outputs/reports_gamus
  python model.py inspect --dataset gamus --gamus-source hf --limit 16 \
      --report-dir outputs/reports_gamus

Exit code 0 = healthy; 1 = fatal structure/alignment problems (CI-friendly).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from depthwizard.datasets.semantics import legend_report
from depthwizard.geo import (
    TileData,
    discover_tiles,
    dump_json,
    is_georeferenced,
    parse_stem,
    read_raster,
    read_tile,
    save_quicklook,
)

NAME = "inspect"
HELP = "dataset structure/alignment/class inspection + quicklooks"


def inspect_tile(paths) -> dict:
    rgb_arr, rgb_prof = read_raster(paths.rgb)
    agl_arr, agl_prof = read_raster(paths.agl)
    cls_arr, cls_prof = read_raster(paths.cls)

    agl = agl_arr[0] if agl_arr.ndim == 3 else agl_arr
    cls = cls_arr[0] if cls_arr.ndim == 3 else cls_arr

    geo_ok, geo_reason = is_georeferenced(rgb_prof)
    aligned = rgb_arr.shape[-2:] == agl.shape == cls.shape

    neg_frac = float((~np.isfinite(agl) | (agl < 0)).mean())
    return {
        "stem": paths.stem,
        "rgb_shape": list(rgb_arr.shape),
        "agl_shape": list(agl.shape),
        "cls_shape": list(cls.shape),
        "rgb_dtype": str(rgb_arr.dtype),
        "agl_dtype": str(agl_arr.dtype),
        "cls_dtype": str(cls_arr.dtype),
        "rgb_bands": int(rgb_arr.shape[0]),
        "aligned": bool(aligned),
        "georeferenced": geo_ok,
        "georef_reason": geo_reason,
        "agl_min": float(np.nanmin(agl)),
        "agl_max": float(np.nanmax(agl)),
        "agl_mean": float(np.nanmean(agl)),
        "agl_nonfinite_or_neg_frac": neg_frac,
        "cls_ids": sorted(int(v) for v in np.unique(cls)),
    }


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
        help="which dataset to inspect (GAMUS: raw HDF5 layout, "
        "official splits, VERIFIED legend)",
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
    p.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    p.add_argument(
        "--quicklooks",
        type=int,
        default=8,
        help="how many quicklook PNGs to render (spread over the set)",
    )
    # ---- GAMUS source options (ignored for dfc2019) ----
    p.add_argument("--gamus-source", choices=("hf", "local"), default="hf")
    p.add_argument("--gamus-local-root", type=Path, default=None)
    p.add_argument("--gamus-manifest", type=Path, default=None)
    p.add_argument("--gamus-hf-cache", type=Path, default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="GAMUS: cap tiles per split (deterministic; strongly "
        "recommended in hf mode to bound downloads)",
    )
    return p


# ---------------------------------------------------------------------------
# GAMUS inspection
# ---------------------------------------------------------------------------


def _inspect_gamus(args) -> int:
    from depthwizard.datasets.gamus import (
        GAMUSConfig,
        _resolve_sample_files,
        list_gamus_samples,
        read_gamus_h5,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    cfg = GAMUSConfig(
        source=args.gamus_source,
        local_root=args.gamus_local_root,
        manifest=args.gamus_manifest,
        hf_cache_dir=args.gamus_hf_cache,
        limit=args.limit,
    )
    per_split, problems = list_gamus_samples(cfg)
    total = sum(len(v) for v in per_split.values())
    print(
        f"[1/4] GAMUS: {total} tiles across splits "
        f"({ {k: len(v) for k, v in per_split.items()} })"
    )

    per_tile, fatal = [], []
    class_counter: Counter = Counter()
    dtypes_seen: Counter = Counter()
    total_px = 0
    samples = [s for split in per_split for s in per_split[split]]
    for i, s in enumerate(samples, 1):
        try:
            paths = _resolve_sample_files(cfg, s)
            rgb = read_gamus_h5(paths["rgb"])
            agl = read_gamus_h5(paths["agl"])
            cls = read_gamus_h5(paths["cls"])
        except Exception as e:
            fatal.append(f"{s.split}/{s.sample_id}: UNREADABLE — {e}")
            continue
        dtypes_seen[str(cls.dtype)] += 1
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            fatal.append(f"{s.sample_id}: RGB shape {rgb.shape} (expected H,W,3)")
            continue
        rgb = rgb[:, :, :3]
        agl = np.asarray(agl, dtype=np.float32)
        cls = np.asarray(cls)
        if cls.ndim == 3:
            cls = cls[0]
        h, w = rgb.shape[:2]
        aligned = agl.shape == (h, w) and cls.shape == (h, w)
        if not aligned:
            fatal.append(
                f"{s.sample_id}: grid mismatch rgb{(h, w)} "
                f"agl{agl.shape} cls{cls.shape}"
            )
        ids, counts = np.unique(cls, return_counts=True)
        for v, c in zip(ids, counts):
            class_counter[int(v)] += int(c)
        total_px += cls.size
        per_tile.append(
            {
                "stem": s.sample_id,
                "split": s.split,
                "rgb_shape": list(rgb.shape),
                "agl_shape": list(agl.shape),
                "cls_shape": list(cls.shape),
                "rgb_dtype": str(rgb.dtype),
                "agl_dtype": str(agl.dtype),
                "cls_dtype": str(cls.dtype),
                "aligned": bool(aligned),
                "georeferenced": False,  # GAMUS HDF5 carries no CRS — ever
                "agl_min": float(np.nanmin(agl)),
                "agl_max": float(np.nanmax(agl)),
                "agl_mean": float(np.nanmean(agl)),
                "agl_nonfinite_or_neg_frac": float(
                    (~np.isfinite(agl) | (agl < 0)).mean()
                ),
                "cls_ids": sorted(int(v) for v in np.unique(cls)),
            }
        )
        if i % 25 == 0 or i == len(samples):
            print(f"      inspected {i}/{len(samples)}")

    all_cls_ids = sorted(class_counter)
    class_share = {
        str(cid): {
            "pixels": class_counter[cid],
            "share": round(class_counter[cid] / max(total_px, 1), 5),
        }
        for cid in all_cls_ids
    }
    legend = legend_report("gamus")
    summary = {
        "dataset": "gamus",
        "source": args.gamus_source,
        "n_tiles": len(per_tile),
        "splits": {k: len(v) for k, v in per_split.items()},
        "limit": args.limit,
        "n_problems": len(problems),
        "problems": problems,
        "fatal": fatal,
        "cls_dtypes_seen": dict(dtypes_seen),
        "agl_negative_pixels_frac_across_tiles": None,
        "class_inventory": class_share,
        "class_legend": legend["raw_legend"],
        "class_meanings": "VERIFIED — RSI-MMSegmentation README "
        "(github.com/EarthNets/RSI-MMSegmentation); full "
        "provenance in datasets/semantics.py.",
        "height_semantics": "nDSM/AGL; units UNDOCUMENTED (assumed metres)",
        "georef": "none — GAMUS HDF5 files carry no CRS",
        "gsd_m": 0.33,
    }
    if per_tile:
        summary["agl_negative_pixels_frac_across_tiles"] = float(
            np.mean([t["agl_nonfinite_or_neg_frac"] for t in per_tile])
        )
        summary["agl_min_global"] = min(t["agl_min"] for t in per_tile)
        summary["agl_max_global"] = max(t["agl_max"] for t in per_tile)

    dump_json(
        {"summary": summary, "per_tile": per_tile},
        args.report_dir / "dataset_report_gamus.json",
    )

    # quicklooks reuse the DFC 3-panel renderer (same array shapes)
    if samples:
        step = max(1, len(samples) // max(args.quicklooks, 1))
        chosen = samples[::step][: args.quicklooks]
        for s in chosen:
            try:
                paths = _resolve_sample_files(cfg, s)

                rgb = np.asarray(read_gamus_h5(paths["rgb"]))[:, :, :3]
                agl = np.asarray(read_gamus_h5(paths["agl"]), dtype=np.float32)
                cls = np.asarray(read_gamus_h5(paths["cls"]), dtype=np.int32)

                h, w = rgb.shape[:2]

                tile: TileData = {
                    "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                    "agl": np.ascontiguousarray(agl, dtype=np.float32),
                    "cls": np.ascontiguousarray(cls, dtype=np.int32),
                    "meta": {
                        "stem": s.sample_id,
                        "height": h,
                        "width": w,
                        "rgb_crs": None,
                        "agl_crs": None,
                        "cls_crs": None,
                        "rgb_transform": None,
                        "agl_transform": None,
                    },
                }

                save_quicklook(
                    tile,
                    args.report_dir / "quicklooks" / f"{s.sample_id}.png",
                    cls_ids=all_cls_ids,
                )

            except Exception as e:
                print(f"  [warn] quicklook failed for {s.sample_id}: {e}")
        print(
            f"[3/4] wrote up to {len(chosen)} quicklooks -> "
            f"{args.report_dir / 'quicklooks'}"
        )

    print("[4/4] GAMUS SUMMARY")
    print(f"  tiles: {summary['n_tiles']}  splits: {summary['splits']}")
    print(f"  problems: {len(problems)}   fatal: {len(fatal)}")
    if per_tile:
        print(
            f"  AGL range: [{summary['agl_min_global']:.3f}, "
            f"{summary['agl_max_global']:.3f}] (units ASSUMED metres)   "
            f"neg/invalid px: "
            f"{100 * summary['agl_negative_pixels_frac_across_tiles']:.2f}%"
        )
        print(f"  CLS ids: {all_cls_ids}  dtypes: {dict(dtypes_seen)}")
        print("  georef: none (relative-DSM mode, same as DFC2019 Track-1)")
    for p in problems[:10]:
        print("  [orphan]", p)
    for f in fatal[:10]:
        print("  [FATAL]", f)

    md = [
        "# GAMUS inspection report",
        "",
        f"- tiles: **{summary['n_tiles']}**  splits: `{summary['splits']}`",
        f"- source: `{args.gamus_source}`  limit: {args.limit}",
        f"- problems: **{len(problems)}**, fatal: **{len(fatal)}**",
        "",
        "## Class inventory (raw ids — VERIFIED legend)",
        "",
        "| class id | meaning | pixels | share |",
        "|---|---|---|---|",
    ]
    for cid, v in class_share.items():
        meaning = legend["raw_legend"].get(cid, "NOT IN VERIFIED LEGEND")
        md.append(f"| {cid} | {meaning} | {v['pixels']:,} | {v['share']:.4f} |")
    md += [
        "",
        "## Per-tile AGL stats (first 20)",
        "",
        "| stem | split | AGL min | AGL max | mean | neg px | aligned |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in per_tile[:20]:
        md.append(
            f"| {t['stem']} | {t['split']} | {t['agl_min']:.2f} "
            f"| {t['agl_max']:.2f} | {t['agl_mean']:.2f} "
            f"| {t['agl_nonfinite_or_neg_frac']:.4f} | {t['aligned']} |"
        )
    (args.report_dir / "dataset_report_gamus.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(f"  report -> {args.report_dir / 'dataset_report_gamus.md'}")
    return 1 if (fatal or not per_tile) else 0


def run(args) -> int:
    if args.dataset == "gamus":
        return _inspect_gamus(args)
    if args.rgb_dir is None or args.truth_dir is None:
        print("[error] --dataset dfc2019 requires --rgb-dir and --truth-dir.")
        return 1
    args.report_dir.mkdir(parents=True, exist_ok=True)

    tiles, problems = discover_tiles(args.rgb_dir, args.truth_dir)
    print(
        f"[1/4] discovered {len(tiles)} complete triples, "
        f"{len(problems)} orphan-file problems"
    )

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
            fatal.append(
                f"{t.stem}: grid mismatch {info['rgb_shape']} vs "
                f"{info['agl_shape']} vs {info['cls_shape']}"
            )

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
    class_share = {
        str(cid): {
            "pixels": class_counter[cid],
            "share": round(class_counter[cid] / max(total_px, 1), 5),
        }
        for cid in all_cls_ids
    }

    summary = {
        "n_tiles": len(per_tile),
        "cities": dict(cities),
        "tile_sizes": {f"{h}x{w}": n for (h, w), n in sorted(sizes.items())},
        "n_problems": len(problems),
        "problems": problems,
        "fatal": fatal,
        "agl_negative_pixels_frac_across_tiles": None,
        "class_inventory": class_share,
        "class_legend": {
            "2": "Ground",
            "5": "Trees",
            "6": "Buildings",
            "9": "Water",
            "17": "Bridge/elevated road",
            "65": "Unlabeled (void)",
        },
        "class_meanings": "VERIFIED — pubgeo/dfc2019 data/README.md (one "
        "legend for all tracks); 65 is excluded from the "
        "official evaluation protocol (mapped to IGNORE).",
    }
    if per_tile:
        summary["agl_negative_pixels_frac_across_tiles"] = float(
            np.mean([t["agl_nonfinite_or_neg_frac"] for t in per_tile])
        )
        summary["agl_min_global"] = min(t["agl_min"] for t in per_tile)
        summary["agl_max_global"] = max(t["agl_max"] for t in per_tile)

    dump_json(
        {"summary": summary, "per_tile": per_tile},
        args.report_dir / "dataset_report.json",
    )

    # ---------------- quicklooks ----------------
    if tiles:
        step = max(1, len(tiles) // max(args.quicklooks, 1))
        chosen = tiles[::step][: args.quicklooks]
        for t in chosen:
            save_quicklook(
                read_tile(t),
                args.report_dir / "quicklooks" / f"{t.stem}.png",
                cls_ids=all_cls_ids,
            )
        print(
            f"[3/4] wrote {len(chosen)} quicklooks -> {args.report_dir / 'quicklooks'}"
        )

    # ---------------- console report ----------------
    print("[4/4] SUMMARY")
    print(
        f"  tiles: {summary['n_tiles']}  cities: {dict(cities)}  sizes: {summary['tile_sizes']}"
    )
    print(f"  orphans: {len(problems)}   fatal: {len(fatal)}")
    if per_tile:
        print(
            f"  AGL range: [{summary['agl_min_global']:.3f}, {summary['agl_max_global']:.3f}] m"
            f"   neg/invalid px: {100 * summary['agl_negative_pixels_frac_across_tiles']:.2f}%"
        )
        print(f"  CLS ids: {all_cls_ids}")
        print(
            "  georef: "
            + (
                "ALL non-georeferenced (expected for Track-1)"
                if all(not t["georeferenced"] for t in per_tile)
                else "MIXED — inspect dataset_report.json"
            )
        )
    for p in problems[:10]:
        print("  [orphan]", p)
    for f in fatal[:10]:
        print("  [FATAL]", f)

    md = [
        "# Dataset inspection report",
        "",
        f"- tiles: **{summary['n_tiles']}**  cities: `{dict(cities)}`",
        f"- tile sizes: `{summary['tile_sizes']}`",
        f"- orphan files: **{len(problems)}**, fatal problems: **{len(fatal)}**",
        "",
        "## Class inventory (raw ids — meanings UNVERIFIED)",
        "",
        "| class id | pixels | share |",
        "|---|---|---|",
    ]
    for cid, v in class_share.items():
        md.append(f"| {cid} | {v['pixels']:,} | {v['share']:.4f} |")
    md += [
        "",
        "## Per-tile AGL stats (first 20)",
        "",
        "| stem | AGL min | AGL max | mean | neg px | aligned |",
        "|---|---|---|---|---|---|",
    ]
    for t in per_tile[:20]:
        md.append(
            f"| {t['stem']} | {t['agl_min']:.2f} | {t['agl_max']:.2f} "
            f"| {t['agl_mean']:.2f} | {t['agl_nonfinite_or_neg_frac']:.4f} "
            f"| {t['aligned']} |"
        )
    (args.report_dir / "dataset_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  report -> {args.report_dir / 'dataset_report.md'}")

    return 1 if (fatal or not per_tile) else 0
