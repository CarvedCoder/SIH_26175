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
  python model.py evaluate --config configs/phase2.yaml
  python model.py evaluate --config configs/phase2.yaml \
      --checkpoint outputs/calib_net/rgb_cos/best.pt --splits val --error-maps 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depthwizard.calibration_net import CalibrationNet
from depthwizard.cli.args import (add_cache_subdir_arg, add_config_arg,
                                  add_device_arg, load_config, resolve_cache,
                                  resolve_device)
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.datasets.factory import build_dataset
from depthwizard.geo import (depth_npy_candidates, dump_json, load_json,
                             read_tile)
from depthwizard.metrics import (building_metrics, height_metrics,
                                 mean_std_over_tiles, pooled_metrics,
                                 slope_error, stratified_by_project_class)
from depthwizard.normalize import clean_agl, minmax_normalize, valid_target_mask
from depthwizard.scene_types import classify_scene, stratify_by_scene_type

NAME = "evaluate"
HELP = "CITABLE eval of the calibration net vs frozen gates (val [+test])"


def _extensions_for_tile(pred, agl, cls, dataset, gsd_m):
    """Phase-5 additive per-tile extensions: building-height metrics,
    project-class stratification, slope error, scene-type label.

    Computed from the SAME pred/agl the frozen metrics used; nothing in the
    frozen pooled/per-tile/city computation above is affected.
    """
    from depthwizard.datasets.semantics import semantic_layers
    onehot, _ignore, _unmapped = semantic_layers(cls, dataset)
    ext = {
        "building": building_metrics(pred, agl, onehot[0] > 0.5),
        "by_project_class": stratified_by_project_class(pred, agl, onehot),
        "slope": slope_error(pred, agl, gsd_m=gsd_m),
        "scene_type": classify_scene(agl, cls, dataset),
    }
    return ext


def evaluate_split(net, ds, use_rgb: bool, device: str, cache_dir: Path):
    """Certified evaluation path — pooled metrics byte-equivalent to the
    pre-GAMUS command (see test_evaluate_regression.py); the ONLY changes
    are (a) the depth-cache read now also accepts the namespaced layout
    (fresh `depth --dataset dfc2019` caches would otherwise be invisible)
    and (b) additive Phase-5 extensions returned alongside.
    """
    import torch

    pooled_p, pooled_t, pooled_m = [], [], []
    per_tile, by_city = [], defaultdict(lambda: defaultdict(list))
    ext_tiles = []
    for t in ds.tiles:
        cands = depth_npy_candidates(cache_dir, "dfc2019", t.stem)
        f = next((p for p in cands if p.exists()), None)
        if f is None:
            raise FileNotFoundError(
                f"depth cache miss for {t.stem}: none of {cands} — run the "
                "depth command first.")
        raw = np.load(f)
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
        # Phase 5 extensions (additive; DFC Track-1 has no CRS -> gsd None,
        # slope honestly refused per geo.pixel_size_metres's rule)
        data = read_tile(t)
        ext_tiles.append(_extensions_for_tile(pred, agl, data["cls"],
                                               "dfc2019", gsd_m=None))
    pooled = pooled_metrics(pooled_p, pooled_t, pooled_m)
    pooled["n_tiles"] = len(ds.tiles)
    summary = mean_std_over_tiles(per_tile)
    city_tab = {c: {k: float(np.mean(v)) for k, v in d.items()}
                for c, d in sorted(by_city.items())}
    ext_summary = _summarize_extensions(ext_tiles)
    return pooled, summary, city_tab, per_tile, ext_summary


def _summarize_extensions(ext_tiles):
    """Aggregate the per-tile Phase-5 extensions into a report block."""
    if not ext_tiles:
        return {}
    building = [e["building"] for e in ext_tiles if e["building"]["n"] > 0]
    slopes = [e["slope"] for e in ext_tiles if e["slope"] is not None]
    # scene-type stratification of the FULL per-tile metric dict is done by
    # the caller (it owns per_tile); here we stratify the building metrics.
    scene_tab = stratify_by_scene_type(
        [e["building"] for e in ext_tiles if e["building"]["n"] > 0],
        [e["scene_type"] for e in ext_tiles if e["building"]["n"] > 0])
    out = {
        "building_mae_mean": (float(np.mean([b["mae"] for b in building]))
                              if building else None),
        "building_rmse_mean": (float(np.mean([b["rmse"] for b in building]))
                               if building else None),
        "slope": ({"slope_mae_deg_mean": float(np.mean([s["slope_mae_deg"]
                                                       for s in slopes]))}
                  if slopes else "REFUSED — no GSD (non-georeferenced "
                                  "dataset; geo.pixel_size_metres rule)"),
        "scene_types": scene_tab,
        "n_tiles_with_buildings": len(building),
    }
    return out


def evaluate_split_adapter(model, ds, device: str, dataset: str):
    """Adapter-path evaluation (GAMUS / any BaseDepthDataset; Phase 3/5).

    Mirrors the frozen evaluate_split's math EXACTLY (same clean_agl
    target, same minmax Dn, same pooled/per-tile aggregation, same metric
    functions) but sources samples through the adapter pipeline and feeds
    the net's full input contract (Dn | RGB | SEM). ``model`` is a
    tifops.LoadedModel (single source for checkpoint rebuilding).
    """
    import torch

    net = model.net
    net.eval()
    use_rgb = model.use_rgb
    use_sem = model.use_sem
    gsd = 0.33 if dataset == "gamus" else None      # documented / unknown

    pooled_p, pooled_t, pooled_m = [], [], []
    per_tile, by_city = [], defaultdict(lambda: defaultdict(list))
    ext_tiles = []
    for i in range(len(ds)):
        s = ds[i]
        dn = s["dn"].to(device)
        rgb = s["rgb"].to(device) if use_rgb else None
        sem = (s["sem_onehot"].to(device)
               if (use_sem and s.get("sem_onehot") is not None) else None)
        with torch.no_grad():
            pred = net(dn, rgb, None, sem)["pred"][0, 0].cpu().numpy()
        agl = s["agl"][0].numpy()                  # adapter already clean_agl
        m = valid_target_mask(agl)
        pooled_p.append(pred[m].ravel())
        pooled_t.append(agl[m].ravel())
        pooled_m.append(m[m].ravel())
        mt = height_metrics(pred, agl, m)
        per_tile.append(mt)
        city = s["meta"]["sample_id"].split("_")[0]
        by_city[city]["mae"].append(mt["mae"])
        by_city[city]["rmse"].append(mt["rmse"])
        ext_tiles.append(_extensions_for_tile(pred, agl, s["cls"][0].numpy(),
                                               dataset, gsd_m=gsd))
    pooled = pooled_metrics(pooled_p, pooled_t, pooled_m)
    pooled["n_tiles"] = len(ds)
    summary = mean_std_over_tiles(per_tile)
    city_tab = {c: {k: float(np.mean(v)) for k, v in d.items()}
                for c, d in sorted(by_city.items())}
    scene_tab = stratify_by_scene_type(per_tile,
                                        [e["scene_type"] for e in ext_tiles])
    ext_summary = _summarize_extensions(ext_tiles)
    ext_summary["scene_types_full_metrics"] = scene_tab
    return pooled, summary, city_tab, per_tile, ext_summary


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
    p.add_argument("--dataset", choices=("dfc2019", "gamus"),
                   default="dfc2019",
                   help="evaluate on DFC2019 (frozen path, gates apply) or "
                        "GAMUS (adapter path; cross-dataset = pass a "
                        "DFC-trained --checkpoint). Gates are DFC-only.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--error-maps", type=int, default=4)
    add_device_arg(p)
    add_cache_subdir_arg(p)
    p.add_argument("--out-tag", default=None)
    return p


def _run_gamus(args, cfg, ckpt, net, device, cache_dir, ckpt_dataset) -> int:
    """GAMUS evaluation / cross-dataset evaluation (Phase 5).

    Cross-dataset = a checkpoint trained on the OTHER dataset passed via
    --checkpoint (e.g. the DFC flagship on GAMUS). The net only needs its
    trained input channels (Dn [+RGB [+SEM]]); zero-fill applies when a
    sem-expecting net meets a sample without semantics.
    NO gates: the frozen reference card is DFC-only by construction.
    """
    import torch

    from depthwizard.tifops import LoadedModel

    paths = cfg["paths"]
    model = LoadedModel(
        net=net, use_rgb=bool(ckpt["use_rgb"]),
        widths=tuple(ckpt["widths"]),
        clamp_min=ckpt.get("clamp_min", 0.0),
        affine_init={"a": float(ckpt["affine_init"]["a"]),
                     "b": float(ckpt["affine_init"]["b"])},
        epoch=int(ckpt["epoch"]),
        checkpoint=Path(args.checkpoint) if args.checkpoint else Path("ckpt"),
        val_subset_mae=ckpt.get("val_subset_mae"),
        use_sem=bool(ckpt.get("use_sem", False)),
        sem_classes=int(ckpt.get("sem_classes", 0)),
        sem_aux_head=bool(ckpt.get("sem_aux_head", False)))

    dcfg = dict(cfg.get("dataset") or {})
    dcfg["name"] = "gamus"
    sub_cfg = {"paths": paths, "dataset": dcfg}
    out_dir = Path(paths["outputs_dir"]) / "calib_net" / (
        args.out_tag or "gamus_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    results, per_tile_all = {}, {}
    for split in args.splits:
        if split not in ("train", "val", "test"):
            print(f"[warn] unknown split '{split}' skipped")
            continue
        ds = build_dataset(sub_cfg, split=split, crop_size=None,
                           augment=False, load_depth=True,
                           depth_cache_dir=cache_dir)
        pooled, summary, city_tab, per_tile, ext = evaluate_split_adapter(
            model, ds, device, "gamus")
        results[split] = {"pooled": pooled, "per_tile_summary": summary,
                          "by_city": city_tab, "extensions": ext}
        per_tile_all[split] = per_tile
        print(f"[{split}] pooled: MAE={pooled['mae']:.3f}  RMSE={pooled['rmse']:.3f}  "
              f"r={pooled['pearson_r']:.3f}  bias={pooled['bias']:+.3f}  "
              f"tiles={pooled['n_tiles']}")
        print(f"        per-tile MAE {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}")
        for c, tab in city_tab.items():
            print(f"        {c}: MAE {tab['mae']:.3f}  RMSE {tab['rmse']:.3f}")
        if ext:
            bm = ext.get("building_mae_mean")
            sl = ext.get("slope")
            print(f"        building MAE {bm:.3f}" if isinstance(bm, float) else
                  "        building MAE n/a")
            print(f"        slope MAE "
                  f"{sl['slope_mae_deg_mean']:.3f} deg" if isinstance(sl, dict)
                  else "        slope REFUSED (no GSD) — honest")
            print(f"        scene types: { {k: v['n_tiles'] for k, v in sorted(ext.get('scene_types', {}).items())} }")

    ckpt_path = args.checkpoint or "(config default)"
    cross = (ckpt_dataset != "gamus")
    print(f"[i] {'CROSS-DATASET eval: ' + ckpt_dataset + ' -> gamus' if cross else 'GAMUS in-domain eval'}"
          " — NO GATES (the frozen reference card is DFC-only).")

    # error maps on the test split (adapter path)
    if args.error_maps > 0 and "test" in results:
        ds_test = build_dataset(sub_cfg, split="test", crop_size=None,
                                augment=False, load_depth=True,
                                depth_cache_dir=cache_dir)
        ds_test_any: Any = ds_test
        for i in range(min(args.error_maps, len(ds_test_any))):
            s = ds_test_any[i]
            dn_np = s["dn"][0].numpy()
            dn = s["dn"].to(device)
            rgb = s["rgb"].to(device) if model.use_rgb else None
            sem = (s["sem_onehot"].to(device)
                   if (model.use_sem and s.get("sem_onehot") is not None)
                   else None)
            with torch.no_grad():
                pred = net(dn, rgb, None, sem)["pred"][0, 0].cpu().numpy()
            # de-normalize rgb for display
            from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
            rgb_np = (s["rgb"].permute(1, 2, 0).numpy() * IMAGENET_STD
                      + IMAGENET_MEAN)
            rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
            _error_map(s["meta"]["sample_id"], rgb_np, dn_np, pred,
                       s["agl"][0].numpy(),
                       out_dir / "error_maps" / f"{s['meta']['sample_id']}.png")
        print(f"[i] error maps -> {out_dir / 'error_maps'}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "calibration_net_eval",
        "dataset": "gamus",
        "cross_dataset": cross,
        "trained_on": ckpt_dataset,
        "checkpoint": {"path": str(ckpt_path), "epoch": ckpt["epoch"],
                       "use_rgb": bool(ckpt["use_rgb"]),
                       "use_sem": bool(ckpt.get("use_sem", False)),
                       "loss": ckpt.get("loss")},
        "results": results,
        "gate_verdicts": {},
        "per_tile_mae": {s: [float(x["mae"]) for x in v]
                         for s, v in per_tile_all.items()},
        "notes": ["GAMUS height semantics: nDSM/AGL, units ASSUMED metres "
                  "(undocumented — flagged, not fabricated).",
                  "No gate verdicts: the frozen reference card is DFC-only.",
                  "Slope computed with GSD=0.33 m (documented on the HF card)."],
    }
    dump_json(report, out_dir / "eval_calib_gamus.json")

    lines = ["# Calibration net — GAMUS evaluation", "",
             f"- checkpoint: `{ckpt_path}` (trained on **{ckpt_dataset}**)",
             f"- cross-dataset: **{cross}**", "",
             "| split | MAE (m) | RMSE (m) | r | bias (m) |",
             "|---|---|---|---|---|"]
    for split, r in results.items():
        p = r["pooled"]
        lines.append(f"| {split} | {p['mae']:.3f} | {p['rmse']:.3f} "
                     f"| {p['pearson_r']:.3f} | {p['bias']:+.3f} |")
    ext = results.get("val", {}).get("extensions", {})
    if ext:
        lines += ["", "## Extensions (Phase 5)", "",
                  f"- building MAE: {ext.get('building_mae_mean')}",
                  f"- slope: {ext.get('slope')}"]
        for st, tab in sorted(ext.get("scene_types", {}).items()):
            lines.append(f"- scene {st}: n={tab.get('n_tiles')} "
                         f"bldg-MAE-mean={tab.get('mae_mean')}")
    lines += ["", "_GAMUS heights: nDSM semantics, units ASSUMED metres. "
              "No gates (DFC reference card is DFC-only)._"]
    (out_dir / "eval_calib_gamus.md").write_text("\n".join(lines),
                                                   encoding="utf-8")
    print(f"-> {out_dir / 'eval_calib_gamus.md'}")
    return 0


def run(args) -> int:
    import torch

    cfg = load_config(args.config)
    paths, mcfg = cfg["paths"], cfg["model"]
    device = resolve_device(args.device)
    cache_dir = resolve_cache(paths, args.cache_subdir)

    ckpt_path = args.checkpoint or Path(paths["outputs_dir"]) / "calib_net" / "dn_only" / "best.pt"
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path} — run `model.py train` first.")
        return 1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_rgb = bool(ckpt["use_rgb"])
    # in_ch: prefer the EXPLICIT field stored by train (covers every
    # variant: 1/2/4/5 legacy + 7/8/10/11 sem); fall back to the frozen
    # legacy mapping (identical for the 1/4 flagships).
    in_ch = ckpt.get("in_ch") or (4 if use_rgb else 1)
    net = CalibrationNet(in_ch=int(in_ch),
                         widths=tuple(ckpt["widths"]),
                         a0=ckpt["affine_init"]["a"], b0=ckpt["affine_init"]["b"],
                         clamp_min=ckpt.get("clamp_min", 0.0),
                         sem_classes=int(ckpt.get("sem_classes", 0)),
                         sem_aux_head=bool(ckpt.get("sem_aux_head", False))).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    ckpt_dataset = ckpt.get("dataset", "dfc2019")
    print(f"[i] ckpt {ckpt_path} (epoch {ckpt['epoch']}, subset-MAE "
          f"{ckpt.get('val_subset_mae', float('nan')):.3f}, use_rgb={use_rgb}, "
          f"trained on {ckpt_dataset})")

    if args.dataset == "gamus":
        return _run_gamus(args, cfg, ckpt, net, device, cache_dir,
                          ckpt_dataset)

    base = DFC2019Config(rgb_dir=Path(paths["rgb_dir"]),
                         truth_dir=Path(paths["truth_dir"]),
                         depth_cache_dir=cache_dir, load_depth=True, crop_size=None)
    ds = discover_and_split(base, Path(paths["splits_json"]))

    out_dir = Path(paths["outputs_dir"]) / "calib_net" / (
        args.out_tag or ("rgb" if use_rgb else "dn_only"))
    results, per_tile_all = {}, {}
    for split in args.splits:
        pooled, summary, city_tab, per_tile, ext = evaluate_split(
            net, ds[split], use_rgb, device, cache_dir)
        results[split] = {"pooled": pooled, "per_tile_summary": summary,
                          "by_city": city_tab, "extensions": ext}
        per_tile_all[split] = per_tile
        print(f"[{split}] pooled: MAE={pooled['mae']:.3f}  RMSE={pooled['rmse']:.3f}  "
              f"r={pooled['pearson_r']:.3f}  bias={pooled['bias']:+.3f}  "
              f"neg_frac={pooled['neg_frac_pred']:.3f}  tiles={pooled['n_tiles']}")
        print(f"        per-tile MAE {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}")
        for c, tab in city_tab.items():
            print(f"        {c}: MAE {tab['mae']:.3f}  RMSE {tab['rmse']:.3f}")
        if ext:
            print(f"        buildings: MAE {ext.get('building_mae_mean')}  "
                  f"scene types: {sorted(ext.get('scene_types', {}))}")

    # ---- gate verdicts vs frozen reference card -------------------------
    verdicts = {}
    ref_str = paths.get("reference_json") or ""
    ref_path = Path(ref_str) if ref_str else None
    if ref_path and ref_path.is_file():
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
            cands = depth_npy_candidates(cache_dir, "dfc2019", t.stem)
            f = next((p for p in cands if p.exists()), None)
            if f is None:
                continue
            raw = np.load(f)
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
        "dataset": "dfc2019",
        "checkpoint": {"path": str(ckpt_path), "epoch": ckpt["epoch"],
                       "use_rgb": use_rgb, "loss": ckpt.get("loss"),
                       "trained_on": ckpt_dataset},
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
