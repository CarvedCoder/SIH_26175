"""Evaluate the global affine baseline (correct, masked metrics).

For every split (train/val/test) and every tile:
    pred(tile) = a * minmax_normalize(raw_depth(tile)) + b
    metrics    = height_metrics(pred, clean_agl(AGL))  over valid pixels

Reported (blueprint discipline):
  * pooled metrics per split (the headline numbers)
  * per-tile mean +- std (variance visibility)
  * stratified by city
  * optional --stratify-cls : stratified by RAW class id (keys stay
    'cls_65' etc. until meanings are verified from documentation)
  * error quicklooks for the first --error-maps test tiles

Memory discipline: tiles stream one at a time through PooledStats
accumulators — a full split is never materialized.

Usage:
  python main.py eval-baseline --config configs/phase1.yaml
  python main.py eval-baseline --config configs/phase1.yaml --stratify-cls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from depthwizard.cli.args import add_config_arg, load_config
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.geo import dump_json, load_json, parse_stem, read_tile
from depthwizard.metrics import height_metrics, mean_std_over_tiles
from depthwizard.normalize import clean_agl, minmax_normalize, valid_target_mask
from depthwizard.streaming import PooledStats

NAME = "eval-baseline"
HELP = "masked evaluation of the global affine baseline + error maps"


def evaluate_split(ds, a: float, b: float, cache_dir: Path):
    per_tile = []
    pooled_stats = PooledStats()
    city_stats = {}

    for t in ds.tiles:
        raw = np.load(cache_dir / f"{t.stem}.npy", mmap_mode="r")
        dn = minmax_normalize(raw)
        data = read_tile(t)
        agl = clean_agl(data["agl"])

        pred = (a * dn + b).astype(np.float32)
        mask = valid_target_mask(agl)

        per_tile.append(height_metrics(pred, agl, mask))

        p = np.asarray(pred, dtype=np.float64)[mask]
        y = np.asarray(agl, dtype=np.float64)[mask]
        pooled_stats.update(p, y)

        city = parse_stem(t.stem)[0]
        if city not in city_stats:
            city_stats[city] = PooledStats()
        city_stats[city].update(p, y)

        del raw, dn, data, agl, pred, mask, p, y

    return {
        "per_tile": per_tile,
        "pooled": pooled_stats.to_metrics(),
        "per_tile_summary": mean_std_over_tiles(per_tile),
        "by_city": {city: stats.to_metrics()
                    for city, stats in sorted(city_stats.items())},
    }


def _stratify_test(ds, a, b, cache_dir):
    """Streaming raw-class stratification (keys stay cls_2 / cls_65)."""
    class_stats = {}
    for t in ds.tiles:
        raw = np.load(cache_dir / f"{t.stem}.npy", mmap_mode="r")
        dn = minmax_normalize(raw)
        data = read_tile(t)
        agl = clean_agl(data["agl"])
        cls = data["cls"]
        pred = (a * dn + b).astype(np.float32)

        for cid in np.unique(cls):
            cid = int(cid)
            mask = (cls == cid) & np.isfinite(agl)
            if not np.any(mask):
                continue
            p = np.asarray(pred, dtype=np.float64)[mask]
            y = np.asarray(agl, dtype=np.float64)[mask]
            if cid not in class_stats:
                class_stats[cid] = PooledStats()
            class_stats[cid].update(p, y)

        del raw, dn, data, agl, cls, pred

    return {f"cls_{cid}": stats.to_metrics()
            for cid, stats in sorted(class_stats.items())}


def _quicklook_error(stem, rgb, dn, pred, agl, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = np.abs(pred - agl)
    panels = [
        (rgb, None, "RGB"),
        (dn, "viridis", "Dn (relative)"),
        (pred, "magma", "pred H (m)"),
        (agl, "magma", "AGL truth (m)"),
        (err, "inferno", "|error| (m)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), constrained_layout=True)
    for ax, (img, cmap, title) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(f"{title} {stem}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if cmap:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase1.yaml")
    p.add_argument("--stratify-cls", action="store_true")
    p.add_argument("--error-maps", type=int, default=4,
                   help="error quicklooks rendered for first N test tiles")
    p.add_argument("--baseline-json", type=Path, default=None)
    return p


def run(args) -> int:
    cfg = load_config(args.config)
    paths = cfg["paths"]

    bl_path = args.baseline_json or (
        Path(paths["outputs_dir"]) / "baseline" / "global_affine.json")
    if not bl_path.exists():
        print(f"[error] baseline not found at {bl_path} — run `main.py fit-baseline` first.")
        return 1
    bl = load_json(bl_path)
    a, b = bl["a"], bl["b"]
    print(f"[i] baseline: H = {a:.6f} * Dn + {b:.6f} "
          f"({bl['kind']}, fit on {bl.get('fit_split')})")

    cache_dir = next((d for d in Path(paths["depth_cache_dir"]).iterdir()
                      if d.is_dir()), None)
    if cache_dir is None:
        print("[error] no depth cache found — run `main.py depth` first.")
        return 1

    ds_cfg = DFC2019Config(
        rgb_dir=Path(paths["rgb_dir"]), truth_dir=Path(paths["truth_dir"]),
        depth_cache_dir=cache_dir, load_depth=True, crop_size=None)
    ds = discover_and_split(ds_cfg, Path(paths["splits_json"]))

    report = {"baseline": bl, "splits": {}}
    for split in ("train", "val", "test"):
        res = evaluate_split(ds[split], a, b, cache_dir)
        report["splits"][split] = {
            "pooled": res["pooled"],
            "per_tile_summary": res["per_tile_summary"],
            "by_city": res["by_city"],
        }
        m = res["pooled"]
        print(f"[{split}] pooled: MAE={m['mae']:.3f} m  RMSE={m['rmse']:.3f} m  "
              f"r={m['pearson_r']:.3f}  neg_frac={m['neg_frac_pred']:.3f}  "
              f"n={m['n']:,}")

    if args.stratify_cls:
        report["test_stratified_by_raw_cls"] = _stratify_test(
            ds["test"], a, b, cache_dir)
        print("[i] test stratified by raw CLS ids: "
              f"{list(report['test_stratified_by_raw_cls'])} (meanings UNVERIFIED)")

    out_dir = Path(paths["outputs_dir"]) / "baseline"
    n_maps = min(args.error_maps, len(ds["test"].tiles))
    for t in ds["test"].tiles[:n_maps]:
        raw = np.load(cache_dir / f"{t.stem}.npy", mmap_mode="r")
        dn = minmax_normalize(raw)
        data = read_tile(t)
        pred = (a * dn + b).astype(np.float32)
        agl = clean_agl(data["agl"])
        _quicklook_error(t.stem, data["rgb"], dn, pred, agl,
                         out_dir / "error_maps" / f"{t.stem}.png")
        del raw, dn, data, pred, agl
    if n_maps:
        print(f"[i] {n_maps} error quicklooks -> {out_dir / 'error_maps'}")

    dump_json(report, out_dir / "eval_results.json")

    lines = [
        "# Global affine baseline — evaluation", "",
        (f"- Model: `H = {bl['a']:.6f} · Dn + {bl['b']:.6f}` "
         f"({bl['kind']}, fitted on **{bl.get('fit_split')}** only, "
         f"stride={bl.get('stride')})"),
        f"- Baseline file: `{bl_path}`", "",
        "| split | MAE (m) | RMSE (m) | bias (m) | Pearson r | neg frac | pixels |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        m = report["splits"][split]["pooled"]
        lines.append(f"| {split} | {m['mae']:.3f} | {m['rmse']:.3f} "
                     f"| {m['bias']:+.3f} | {m['pearson_r']:.3f} "
                     f"| {m['neg_frac_pred']:.3f} | {m['n']:,} |")
    lines += ["", "## Per-city (pooled)", "",
              "| split | city | MAE | RMSE | r |", "|---|---|---|---|---|"]
    for split in ("train", "val", "test"):
        for city, m in report["splits"][split].get("by_city", {}).items():
            lines.append(f"| {split} | {city} | {m['mae']:.3f} | {m['rmse']:.3f} "
                         f"| {m['pearson_r']:.3f} |")
    strat = report.get("test_stratified_by_raw_cls")
    if strat:
        lines += ["", "## Test split, stratified by RAW class id (meanings UNVERIFIED)",
                  "", "| class id | MAE | RMSE | pixels |", "|---|---|---|---|"]
        for key, m in strat.items():
            lines.append(f"| {key} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['n']:,} |")
    lines += ["", "_Interpretation guardrails: Dn is relative (not metric); the "
              "affine baseline is expected to oversmooth and to produce negative "
              "predictions on low ground; its purpose is to be the number every "
              "later phase must beat._"]
    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_dir / 'eval_report.md'}")
    return 0
