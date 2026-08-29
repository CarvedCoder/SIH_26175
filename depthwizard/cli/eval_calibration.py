"""Evaluate the calibration net against the FROZEN Phase-1 gates.  [CITABLE]

This command mirrors the certified 09 eval logic EXACTLY — it is the ONLY
source of FINAL / citable numbers in the project. Any refactor must keep the
evaluation code path byte-equivalent in behavior: same checkpoint
construction, same full-tile forward, same pooled/per-tile/per-city metrics
via the frozen depthwizard.metrics module.

Loads best.pt from `train`, runs full-tile inference over val (+test),
computes pooled/per-tile/per-city metrics, compares against
outputs/reference/reference_card.json and prints PASS/FAIL verdicts.

Usage:
  python main.py evaluate --config configs/phase2.yaml
  python main.py evaluate --config configs/phase2.yaml \
      --checkpoint outputs/calib_net/rgb_cos/best.pt --splits val --error-maps 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depthwizard.calibration_net import CalibrationNet
from depthwizard.cli.args import (add_cache_subdir_arg, add_config_arg,
                                  add_device_arg, load_config, resolve_cache,
                                  resolve_device)
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.geo import dump_json, load_json, read_tile
from depthwizard.metrics import (height_metrics, mean_std_over_tiles,
                                 pooled_metrics)
from depthwizard.normalize import clean_agl, minmax_normalize, valid_target_mask

NAME = "evaluate"
HELP = "CITABLE eval of the calibration net vs frozen gates (val [+test])"


def evaluate_split(net, ds, use_rgb: bool, device: str, cache_dir: Path):
    """Certified evaluation path — do not modify without ledger justification."""
    import torch

    pooled_p, pooled_t, pooled_m = [], [], []
    per_tile, by_city = [], defaultdict(lambda: defaultdict(list))
    for t in ds.tiles:
        raw = np.load(cache_dir / f"{t.stem}.npy")
        dn = torch.from_numpy(
            minmax_normalize(raw)[None, None].astype(np.float32)).to(device)
        rgb = None
        if use_rgb:
            from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
            rgb_u8 = read_tile(t)["rgb"]
            rgb_n = ((rgb_u8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            rgb = torch.from_numpy(rgb_n.transpose(2, 0, 1)[None]).to(device)
        with torch.no_grad():
            pred = net(dn, rgb)["pred"][0, 0].cpu().numpy()
        agl = clean_agl(read_tile(t)["agl"])
        m = valid_target_mask(agl)
        pooled_p.append(pred[m].ravel())
        pooled_t.append(agl[m].ravel())
        pooled_m.append(m[m].ravel())
        mt = height_metrics(pred, agl, m)
        per_tile.append(mt)
        city = t.stem.split("_")[0]
        by_city[city]["mae"].append(mt["mae"])
        by_city[city]["rmse"].append(mt["rmse"])
    pooled = pooled_metrics(pooled_p, pooled_t, pooled_m)
    pooled["n_tiles"] = len(ds.tiles)
    summary = mean_std_over_tiles(per_tile)
    city_tab = {c: {k: float(np.mean(v)) for k, v in d.items()}
                for c, d in sorted(by_city.items())}
    return pooled, summary, city_tab, per_tile


def _error_map(stem, rgb, dn, pred, agl, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
    for ax, im, title, cmap in (
            (axes[0], rgb, f"{stem} RGB", None),
            (axes[1], dn, "Dn (relative)", "viridis"),
            (axes[2], pred, "pred H (m)", "magma"),
            (axes[3], agl, "AGL truth (m)", "magma"),
            (axes[4], np.abs(pred - agl), "|error| (m)", "inferno")):
        if cmap:
            im_ = ax.imshow(im, cmap=cmap)
            fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase2.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--error-maps", type=int, default=4)
    add_device_arg(p)
    add_cache_subdir_arg(p)
    p.add_argument("--out-tag", default=None)
    return p


def run(args) -> int:
    import torch

    cfg = load_config(args.config)
    paths, mcfg = cfg["paths"], cfg["model"]
    device = resolve_device(args.device)
    cache_dir = resolve_cache(paths, args.cache_subdir)

    ckpt_path = args.checkpoint or Path(paths["outputs_dir"]) / "calib_net" / "dn_only" / "best.pt"
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path} — run `main.py train` first.")
        return 1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_rgb = bool(ckpt["use_rgb"])
    net = CalibrationNet(in_ch=4 if use_rgb else 1,
                         widths=tuple(ckpt["widths"]),
                         a0=ckpt["affine_init"]["a"], b0=ckpt["affine_init"]["b"],
                         clamp_min=ckpt.get("clamp_min", 0.0)).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"[i] ckpt {ckpt_path} (epoch {ckpt['epoch']}, subset-MAE "
          f"{ckpt.get('val_subset_mae', float('nan')):.3f}, use_rgb={use_rgb})")

    base = DFC2019Config(rgb_dir=Path(paths["rgb_dir"]),
                         truth_dir=Path(paths["truth_dir"]),
                         depth_cache_dir=cache_dir, load_depth=True, crop_size=None)
    ds = discover_and_split(base, Path(paths["splits_json"]))

    out_dir = Path(paths["outputs_dir"]) / "calib_net" / (
        args.out_tag or ("rgb" if use_rgb else "dn_only"))
    results, per_tile_all = {}, {}
    for split in args.splits:
        pooled, summary, city_tab, per_tile = evaluate_split(
            net, ds[split], use_rgb, device, cache_dir)
        results[split] = {"pooled": pooled, "per_tile_summary": summary,
                          "by_city": city_tab}
        per_tile_all[split] = per_tile
        print(f"[{split}] pooled: MAE={pooled['mae']:.3f}  RMSE={pooled['rmse']:.3f}  "
              f"r={pooled['pearson_r']:.3f}  bias={pooled['bias']:+.3f}  "
              f"neg_frac={pooled['neg_frac_pred']:.3f}  tiles={pooled['n_tiles']}")
        print(f"        per-tile MAE {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}")
        for c, tab in city_tab.items():
            print(f"        {c}: MAE {tab['mae']:.3f}  RMSE {tab['rmse']:.3f}")

    # ---- gate verdicts vs frozen reference card -------------------------
    verdicts = {}
    ref_path = Path(paths.get("reference_json", ""))
    if ref_path and Path(ref_path).exists():
        ref = load_json(ref_path)
        g = ref["floors_and_gates"]["val"]
        v = results["val"]["pooled"] if "val" in results else None
        if v is not None:
            verdicts = {
                "must_mae": {"limit": g["gate_must_mae"], "got": v["mae"],
                             "pass": v["mae"] < g["gate_must_mae"]},
                "must_rmse": {"limit": g["gate_must_rmse"], "got": v["rmse"],
                              "pass": v["rmse"] < g["gate_must_rmse"]},
                "target_mae": {"limit": g["gate_target_mae"], "got": v["mae"],
                               "pass": v["mae"] <= g["gate_target_mae"]},
                "target_rmse": {"limit": g["gate_target_rmse"], "got": v["rmse"],
                                "pass": v["rmse"] <= g["gate_target_rmse"]},
            }
            for k, d in verdicts.items():
                print(f"  GATE {k:11s}: {'PASS' if d['pass'] else 'FAIL'}  "
                      f"({d['got']:.3f} vs {d['limit']:.3f})")
    else:
        print(f"[warn] reference card not found at {ref_path!r} — no gate verdicts.")

    # ---- error maps (test split) ----------------------------------------
    if args.error_maps > 0 and "test" in ds:
        for t in ds["test"].tiles[:args.error_maps]:
            raw = np.load(cache_dir / f"{t.stem}.npy")
            dn_np = minmax_normalize(raw)
            dn = torch.from_numpy(dn_np[None, None].astype(np.float32)).to(device)
            rgb_np = read_tile(t)["rgb"]
            data = read_tile(t)
            rgb = None
            if use_rgb:
                from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
                rgb_n = ((rgb_np.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
                rgb = torch.from_numpy(rgb_n.transpose(2, 0, 1)[None]).to(device)
            with torch.no_grad():
                pred = net(dn, rgb)["pred"][0, 0].cpu().numpy()
            agl = clean_agl(data["agl"])
            _error_map(t.stem, rgb_np, dn_np, pred, agl,
                       out_dir / "error_maps" / f"{t.stem}.png")
        print(f"[i] {args.error_maps} error maps -> {out_dir / 'error_maps'}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "calibration_net_eval",
        "checkpoint": {"path": str(ckpt_path), "epoch": ckpt["epoch"],
                       "use_rgb": use_rgb, "loss": ckpt.get("loss")},
        "results": results,
        "gate_verdicts": verdicts,
        "per_tile_mae": {s: [float(x["mae"]) for x in v]
                         for s, v in per_tile_all.items()},
    }
    dump_json(report, out_dir / "eval_calib.json")

    lines = ["# Calibration net — evaluation vs Phase-1 gates", "",
             f"- checkpoint: `{ckpt_path}` (epoch {ckpt['epoch']}, use_rgb={use_rgb})",
             "",
             "| split | MAE (m) | RMSE (m) | r | bias (m) |",
             "|---|---|---|---|---|"]
    for split, r in results.items():
        p = r["pooled"]
        lines.append(f"| {split} | {p['mae']:.3f} | {p['rmse']:.3f} "
                     f"| {p['pearson_r']:.3f} | {p['bias']:+.3f} |")
    if verdicts:
        lines += ["", "## Gates (frozen reference card)", ""]
        for k, d in verdicts.items():
            lines.append(f"- {k}: {'**PASS**' if d['pass'] else '**FAIL**'} "
                         f"— {d['got']:.3f} vs {d['limit']:.3f}")
    lines += ["", "_Floors to quote beside: val MAE 3.186 (median), val RMSE 4.959 "
              "(mean); affine val MAE 4.405. Baselines from reference_card.md._"]
    (out_dir / "eval_calib.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_dir / 'eval_calib.md'}")
    return 0
