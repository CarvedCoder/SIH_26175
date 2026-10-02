"""Evaluate the height model (CalibrationNet | RDAH) against frozen gates.  [CITABLE]

This command mirrors the certified 09 eval logic EXACTLY — it is the ONLY
source of FINAL / citable numbers in the project. Any refactor must keep the
evaluation code path byte-equivalent in behavior: same checkpoint
construction, same full-tile forward, same pooled/per-tile/per-city metrics
via the frozen depthwizard.metrics module.

A/B backend comparison (RDAH integration, task Sec. 16): ``--model``
selects the architecture (calibration_net | rdah | auto-detect). BOTH
backends run through the SAME dataset split, preprocessing, target, masks
and metric functions — the only difference is the height model. The
report additionally includes parameter count, mean inference time per
tile and peak VRAM (CUDA max_memory_allocated; None on CPU).

Usage:
  python model.py evaluate --config configs/phase2.yaml
  python model.py evaluate --config configs/phase2.yaml \
      --checkpoint outputs/calib_net/rgb_cos/best.pt --splits val --error-maps 0
  python model.py evaluate --model rdah --dataset gamus \
      --config configs/gamus.yaml \
      --checkpoint checkpoints/rdah/rdah_track1_best_model.pth
  python model.py evaluate --model calibration_net --dataset gamus \
      --config configs/gamus.yaml --checkpoint outputs/calib_net/rgb/best.pt
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

from depthwizard.calibration_net import CalibrationNet, semantic_mode_from_ckpt
from depthwizard.cli.args import (
    add_cache_subdir_arg,
    add_config_arg,
    add_device_arg,
    load_config,
    resolve_cache,
    resolve_device,
)
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.datasets.factory import build_dataset
from depthwizard.geo import depth_npy_candidates, dump_json, load_json, read_tile
from depthwizard.metrics import (
    building_metrics,
    height_metrics,
    mean_std_over_tiles,
    pooled_metrics,
    slope_error,
    stratified_by_project_class,
)
from depthwizard.normalize import (
    clean_agl,
    dn_tile_stats,
    minmax_normalize,
    valid_target_mask,
)
from depthwizard.scene_types import classify_scene, stratify_by_scene_type

NAME = "evaluate"
HELP = "CITABLE eval of the height model (CalibrationNet | RDAH) vs frozen gates (val [+test])"


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
                "depth command first."
            )
        raw = np.load(f)
        dn = torch.from_numpy(minmax_normalize(raw)[None, None].astype(np.float32)).to(
            device
        )
        # RAW-tile stats at the same granularity dn was normalized at (the
        # full tile) — the FiLM conditioning input (Exp 1); ignored by plain
        # checkpoints (net.film_stats False).
        st_t = torch.from_numpy(dn_tile_stats(raw)[None]).to(device)
        rgb = None
        if use_rgb:
            from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD

            rgb_u8 = read_tile(t)["rgb"]
            rgb_n = ((rgb_u8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            rgb = torch.from_numpy(rgb_n.transpose(2, 0, 1)[None]).to(device)
        with torch.no_grad():
            pred = net(dn, rgb, None, None, st_t)["pred"][0, 0].cpu().numpy()
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
        ext_tiles.append(
            _extensions_for_tile(pred, agl, data["cls"], "dfc2019", gsd_m=None)
        )
    pooled = pooled_metrics(pooled_p, pooled_t, pooled_m)
    pooled["n_tiles"] = len(ds.tiles)
    summary = mean_std_over_tiles(per_tile)
    city_tab = {
        c: {k: float(np.mean(v)) for k, v in d.items()}
        for c, d in sorted(by_city.items())
    }
    ext_summary = _summarize_extensions(ext_tiles)
    return pooled, summary, city_tab, per_tile, ext_summary


def _threshold_metrics(pooled_p, pooled_t) -> dict:
    """MAE / RMSE / Pearson r on the pooled valid pixels where the TARGET
    height exceeds a threshold (the published TerraHeight validation
    bands ">1m" and ">5m"). Same metric implementation as the pooled
    report — the only difference is the pixel band."""
    import numpy as np

    p = np.concatenate([np.asarray(x).ravel() for x in pooled_p])
    t = np.concatenate([np.asarray(x).ravel() for x in pooled_t])
    out = {}
    for name, thr in ((">1m", 1.0), (">5m", 5.0)):
        m = t > thr
        if not m.any():
            out[name] = {"n": 0}
            continue
        d = p[m] - t[m]
        r = float(np.corrcoef(p[m], t[m])[0, 1]) if m.sum() > 1 else None
        out[name] = {
            "n": int(m.sum()),
            "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d**2).mean())),
            "pearson_r": r,
        }
    return out


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
        [e["scene_type"] for e in ext_tiles if e["building"]["n"] > 0],
    )
    out = {
        "building_mae_mean": (
            float(np.mean([b["mae"] for b in building])) if building else None
        ),
        "building_rmse_mean": (
            float(np.mean([b["rmse"] for b in building])) if building else None
        ),
        "slope": (
            {"slope_mae_deg_mean": float(np.mean([s["slope_mae_deg"] for s in slopes]))}
            if slopes
            else "REFUSED — no GSD (non-georeferenced "
            "dataset; geo.pixel_size_metres rule)"
        ),
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
    tifops.LoadedModel (single source for checkpoint rebuilding — works
    for ALL architectures; the RDAH backend additionally consumes the
    dn_stats 4-vector to reconstruct the RAW DAv2 depth scale; TerraHeight
    consumes RGB only — dn/stats stay None and the sample's Dn, when
    present, is simply not fed to the net).
    """
    import time as _time

    import torch

    net = model.net
    net.eval()
    use_rgb = model.use_rgb
    use_sem = model.use_sem
    needs_stats = getattr(model, "needs_stats", False) or model.film_stats
    is_terraheight = model.architecture == "terraheight_s"
    gsd = 0.33 if dataset == "gamus" else None  # documented / unknown

    pooled_p, pooled_t, pooled_m = [], [], []
    per_tile, by_city = [], defaultdict(lambda: defaultdict(list))
    ext_tiles = []
    fwd_secs = []
    for i in range(len(ds)):
        s = ds[i]
        dn = s["dn"].to(device) if s.get("dn") is not None else None
        rgb = s["rgb"].to(device) if use_rgb else None
        sem = (
            s["sem_onehot"].to(device)
            if (use_sem and s.get("sem_onehot") is not None)
            else None
        )
        stats = (
            s["dn_stats"].to(device)
            if (needs_stats and s.get("dn_stats") is not None)
            else None
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fwd = _time.perf_counter()
        with torch.no_grad():
            if is_terraheight:
                # RGB -> AGL through the SAME deployed tiled path
                # (predict_agl_tiled: 630 training-crop windows, seam-free
                # stitching) — the GAMUS tiles are 1024x1024, which the ViT
                # cannot consume directly (patch-14 multiple constraint).
                # The adapter sample's RGB is ImageNet-normalized float;
                # de-normalize back to uint8 for the tiler (exact roundtrip).
                from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
                from depthwizard.terraheight import (
                    TERRAHEIGHT_DEFAULT_OVERLAP,
                    TERRAHEIGHT_TILE_SIZE,
                    predict_agl_tiled,
                )

                rgb_u8 = np.clip(
                    (s["rgb"].permute(1, 2, 0).numpy() * IMAGENET_STD
                     + IMAGENET_MEAN) * 255.0,
                    0, 255,
                ).astype(np.uint8)
                pred, _n_tiles = predict_agl_tiled(
                    model, rgb_u8, device=device,
                    tile_size=TERRAHEIGHT_TILE_SIZE,
                    overlap=TERRAHEIGHT_DEFAULT_OVERLAP, log=lambda *_: None,
                )
            else:
                pred = net(dn, rgb, None, sem, stats)["pred"][0, 0].cpu().numpy()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        fwd_secs.append(_time.perf_counter() - t_fwd)
        agl = s["agl"][0].numpy()  # adapter already clean_agl
        m = valid_target_mask(agl)
        pooled_p.append(pred[m].ravel())
        pooled_t.append(agl[m].ravel())
        pooled_m.append(m[m].ravel())
        mt = height_metrics(pred, agl, m)
        per_tile.append(mt)
        city = s["meta"]["sample_id"].split("_")[0]
        by_city[city]["mae"].append(mt["mae"])
        by_city[city]["rmse"].append(mt["rmse"])
        ext_tiles.append(
            _extensions_for_tile(pred, agl, s["cls"][0].numpy(), dataset, gsd_m=gsd)
        )
    pooled = pooled_metrics(pooled_p, pooled_t, pooled_m)
    pooled["n_tiles"] = len(ds)
    if is_terraheight:
        # TerraHeight published validation protocol reports >1 m / >5 m
        # bands (valid-target pixels where the GT height exceeds the
        # threshold) — computed here with the SAME metric implementation,
        # clearly labelled as our local reproduction.
        pooled["thresholds"] = _threshold_metrics(pooled_p, pooled_t)
    summary = mean_std_over_tiles(per_tile)
    city_tab = {
        c: {k: float(np.mean(v)) for k, v in d.items()}
        for c, d in sorted(by_city.items())
    }
    scene_tab = stratify_by_scene_type(per_tile, [e["scene_type"] for e in ext_tiles])
    ext_summary = _summarize_extensions(ext_tiles)
    ext_summary["scene_types_full_metrics"] = scene_tab
    perf = _perf_block(model, fwd_secs, device)
    ext_summary["perf"] = perf
    return pooled, summary, city_tab, per_tile, ext_summary


def _perf_block(model, fwd_secs, device: str) -> dict:
    """Parameter count / mean inference time / peak VRAM for the A/B
    comparison (task Sec. 16). Peak VRAM is CUDA-only (None on CPU)."""
    import torch

    n_par = sum(p.numel() for p in model.net.parameters())
    peak_vram = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram = float(torch.cuda.max_memory_allocated()) / (1024**2)
    return {
        "architecture": model.architecture,
        "params": int(n_par),
        "mean_fwd_sec_per_tile": (
            float(np.mean(fwd_secs)) if fwd_secs else None
        ),
        "peak_vram_mb": peak_vram,
    }


def _error_map(stem, rgb, dn, pred, agl, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
    if dn is None:  # RGB-only backend (TerraHeight) — no Dn panel data
        dn = np.zeros((1, 1), dtype=np.float32)
        dn_title = "Dn n/a (RGB-only backend)"
    else:
        dn_title = "Dn (relative)"
    for ax, im, title, cmap in (
        (axes[0], rgb, f"{stem} RGB", None),
        (axes[1], dn, dn_title, "viridis"),
        (axes[2], pred, "pred H (m)", "magma"),
        (axes[3], agl, "AGL truth (m)", "magma"),
        (axes[4], np.abs(pred - agl), "|error| (m)", "inferno"),
    ):
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
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(p, "configs/phase2.yaml")
    p.add_argument(
        "--model",
        choices=("calibration_net", "rdah", "terraheight_s", "auto"),
        default="auto",
        help="height-model backend for the A/B comparison (default: "
        "auto-detect from the checkpoint). All backends use the SAME "
        "split, preprocessing, target, masks and metrics.",
    )
    p.add_argument(
        "--dataset",
        choices=("dfc2019", "gamus"),
        default="dfc2019",
        help="evaluate on DFC2019 (frozen path, gates apply) or "
        "GAMUS (adapter path; cross-dataset = pass a "
        "DFC-trained --checkpoint). Gates are DFC-only.",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--error-maps", type=int, default=4)
    add_device_arg(p)
    add_cache_subdir_arg(p)
    p.add_argument("--out-tag", default=None)
    return p


def _run_gamus(args, cfg, model, device, cache_dir, ckpt_dataset) -> int:
    """GAMUS evaluation / cross-dataset evaluation (Phase 5).

    Cross-dataset = a checkpoint trained on the OTHER dataset passed via
    --checkpoint (e.g. the DFC flagship on GAMUS). The net only needs its
    trained input channels (Dn [+RGB [+SEM]]); zero-fill applies when a
    sem-expecting net meets a sample without semantics.
    NO gates: the frozen reference card is DFC-only by construction.
    ``model`` is a tifops.LoadedModel — architecture-agnostic (works for
    CalibrationNet AND the RDAH backend).
    """
    import torch

    paths = cfg["paths"]
    net = model.net
    needs_stats = getattr(model, "needs_stats", False) or model.film_stats

    dcfg = dict(cfg.get("dataset") or {})
    dcfg["name"] = "gamus"
    sub_cfg = {"paths": paths, "dataset": dcfg}
    out_dir = Path(paths["outputs_dir"]) / (
        "rdah_net" if model.architecture == "rdah"
        else "terraheight_s" if model.architecture == "terraheight_s"
        else "calib_net"
    ) / (args.out_tag or "gamus_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # TerraHeight is RGB-only: it consumes NO Dn depth cache (load_depth
    # False leaves dn as None in the samples — the TH branch never touches it).
    needs_depth_cache = model.architecture != "terraheight_s"

    results, per_tile_all = {}, {}
    for split in args.splits:
        if split not in ("train", "val", "test"):
            print(f"[warn] unknown split '{split}' skipped")
            continue
        ds = build_dataset(
            sub_cfg,
            split=split,
            crop_size=None,
            augment=False,
            load_depth=needs_depth_cache,
            depth_cache_dir=cache_dir,
        )
        pooled, summary, city_tab, per_tile, ext = evaluate_split_adapter(
            model, ds, device, "gamus"
        )
        results[split] = {
            "pooled": pooled,
            "per_tile_summary": summary,
            "by_city": city_tab,
            "extensions": ext,
        }
        per_tile_all[split] = per_tile
        print(
            f"[{split}] pooled: MAE={pooled['mae']:.3f}  RMSE={pooled['rmse']:.3f}  "
            f"r={pooled['pearson_r']:.3f}  bias={pooled['bias']:+.3f}  "
            f"tiles={pooled['n_tiles']}"
        )
        thr = pooled.get("thresholds")
        if thr:
            for name, tab in thr.items():
                if tab.get("n"):
                    print(
                        f"        [{name} band] MAE={tab['mae']:.3f}  "
                        f"RMSE={tab['rmse']:.3f}  r={tab['pearson_r']:.3f}  "
                        f"n={tab['n']}"
                    )
        print(
            f"        per-tile MAE {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}"
        )
        for c, tab in city_tab.items():
            print(f"        {c}: MAE {tab['mae']:.3f}  RMSE {tab['rmse']:.3f}")
        if ext:
            bm = ext.get("building_mae_mean")
            sl = ext.get("slope")
            print(
                f"        building MAE {bm:.3f}"
                if isinstance(bm, float)
                else "        building MAE n/a"
            )
            print(
                f"        slope MAE {sl['slope_mae_deg_mean']:.3f} deg"
                if isinstance(sl, dict)
                else "        slope REFUSED (no GSD) — honest"
            )
            print(
                f"        scene types: { {k: v['n_tiles'] for k, v in sorted(ext.get('scene_types', {}).items())} }"
            )
            perf = ext.get("perf")
            if perf:
                vram = (
                    f"{perf['peak_vram_mb']:.0f} MB"
                    if perf.get("peak_vram_mb") is not None
                    else "n/a (CPU)"
                )
                print(
                    f"        perf: architecture={perf['architecture']} "
                    f"params={perf['params']:,}  "
                    f"fwd={perf['mean_fwd_sec_per_tile']:.2f}s/tile  "
                    f"peak VRAM={vram}"
                )

    ckpt_path = args.checkpoint or "(config default)"
    cross = ckpt_dataset != "gamus"
    print(
        f"[i] {'CROSS-DATASET eval: ' + ckpt_dataset + ' -> gamus' if cross else 'GAMUS in-domain eval'}"
        " — NO GATES (the frozen reference card is DFC-only)."
    )

    # error maps on the test split (adapter path)
    if args.error_maps > 0 and "test" in results:
        ds_test = build_dataset(
            sub_cfg,
            split="test",
            crop_size=None,
            augment=False,
            load_depth=True,
            depth_cache_dir=cache_dir,
        )
        ds_test_any: Any = ds_test
        for i in range(min(args.error_maps, len(ds_test_any))):
            s = ds_test_any[i]
            dn_np = s["dn"][0].numpy() if s.get("dn") is not None else None
            dn = s["dn"].to(device) if s.get("dn") is not None else None
            rgb = s["rgb"].to(device) if model.use_rgb else None
            sem = (
                s["sem_onehot"].to(device)
                if (model.use_sem and s.get("sem_onehot") is not None)
                else None
            )
            stats = (
                s["dn_stats"].to(device)
                if (needs_stats and s.get("dn_stats") is not None)
                else None
            )
            with torch.no_grad():
                if model.architecture == "terraheight_s":
                    pred = net(rgb, None, None, None, None)["pred"][0, 0].cpu().numpy()
                else:
                    pred = net(dn, rgb, None, sem, stats)["pred"][0, 0].cpu().numpy()
            # de-normalize rgb for display
            from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD

            rgb_np = s["rgb"].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
            rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
            _error_map(
                s["meta"]["sample_id"],
                rgb_np,
                dn_np,
                pred,
                s["agl"][0].numpy(),
                out_dir / "error_maps" / f"{s['meta']['sample_id']}.png",
            )
        print(f"[i] error maps -> {out_dir / 'error_maps'}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "height_model_eval",
        "architecture": model.architecture,
        "dataset": "gamus",
        "cross_dataset": cross,
        "trained_on": ckpt_dataset,
        "checkpoint": {
            "path": str(ckpt_path),
            "epoch": model.epoch,
            "architecture": model.architecture,
            "use_rgb": model.use_rgb,
            "use_sem": model.use_sem,
            "val_subset_mae": model.val_subset_mae,
        },
        "results": results,
        "gate_verdicts": {},
        "per_tile_mae": {
            s: [float(x["mae"]) for x in v] for s, v in per_tile_all.items()
        },
        "notes": [
            "GAMUS height semantics: nDSM/AGL, units ASSUMED metres "
            "(undocumented — flagged, not fabricated).",
            "No gate verdicts: the frozen reference card is DFC-only.",
            "Slope computed with GSD=0.33 m (documented on the HF card).",
            "Backend: " + (
                "RDAH-Net (official HeightPredTransformer; depth input = "
                "raw DAv2 x depth_scale; output nDSM metres)"
                if model.architecture == "rdah"
                else "TerraHeight-S (external pretrained GAMUS AGL model; "
                "Depth Anything V2 Small backbone; RGB only, no depth cache; "
                "output = raw x scale_m 8.4929 clamped to >= 0, metres AGL)"
                if model.architecture == "terraheight_s"
                else "CalibrationNet (clamp(a*Dn + b, 0), Dn min-max)"
            ),
        ],
    }
    if model.architecture == "terraheight_s":
        report["notes"].append(
            "model.val_subset_mae is the RELEASED checkpoint's PUBLISHED "
            "validation MAE (labelled reference value, not a local metric); "
            "local numbers are the pooled/thresholds blocks above."
        )
    dump_json(report, out_dir / "eval_calib_gamus.json")

    lines = [
        f"# Height model ({model.architecture}) — GAMUS evaluation",
        "",
        f"- checkpoint: `{ckpt_path}` (trained on **{ckpt_dataset}**)",
        f"- architecture: **{model.architecture}**",
        f"- cross-dataset: **{cross}**",
        "",
        "| split | MAE (m) | RMSE (m) | r | bias (m) |",
        "|---|---|---|---|---|",
    ]
    for split, r in results.items():
        p = r["pooled"]
        lines.append(
            f"| {split} | {p['mae']:.3f} | {p['rmse']:.3f} "
            f"| {p['pearson_r']:.3f} | {p['bias']:+.3f} |"
        )
    ext = results.get("val", {}).get("extensions", {})
    if ext:
        lines += [
            "",
            "## Extensions (Phase 5)",
            "",
            f"- building MAE: {ext.get('building_mae_mean')}",
            f"- slope: {ext.get('slope')}",
        ]
        for st, tab in sorted(ext.get("scene_types", {}).items()):
            lines.append(
                f"- scene {st}: n={tab.get('n_tiles')} "
                f"bldg-MAE-mean={tab.get('mae_mean')}"
            )
    lines += [
        "",
        "_GAMUS heights: nDSM semantics, units ASSUMED metres. "
        "No gates (DFC reference card is DFC-only)._",
    ]
    (out_dir / "eval_calib_gamus.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_dir / 'eval_calib_gamus.md'}")
    return 0


def run(args) -> int:
    import torch

    cfg = load_config(args.config)
    paths = cfg["paths"]
    device = resolve_device(args.device)
    cache_dir = resolve_cache(paths, args.cache_subdir)

    # ---- checkpoint + architecture resolution ---------------------------
    # --model (CLI) > auto-detect from the payload. RDAH default checkpoint
    # = the released Track1 release (auto-downloads + MD5-verifies).
    architecture = None if args.model == "auto" else args.model
    mcfg = dict(cfg.get("model") or {})
    if architecture is None and mcfg.get("architecture") in (
        "rdah", "calibration_net", "terraheight_s"
    ):
        architecture = mcfg["architecture"]

    ckpt_path = args.checkpoint
    if ckpt_path is None and architecture == "rdah":
        from depthwizard.rdah import RDAH_CKPT_DIR, RDAH_CHECKPOINTS

        ckpt_path = Path(mcfg.get("checkpoint") or "") or (
            RDAH_CKPT_DIR / RDAH_CHECKPOINTS["track1"]["filename"]
        )
    if ckpt_path is None and architecture == "terraheight_s":
        from depthwizard.terraheight import default_checkpoint_path

        try:
            ckpt_path = Path(mcfg.get("checkpoint") or "") or default_checkpoint_path()
        except FileNotFoundError as e:
            print(f"[error] {e}")
            return 1
    if ckpt_path is None:
        ckpt_path = Path(paths["outputs_dir"]) / "calib_net" / "dn_only" / "best.pt"
    if not ckpt_path.exists() and (
        architecture == "rdah" or "rdah" in str(ckpt_path)
    ):
        from depthwizard.rdah import ensure_rdah_checkpoint

        try:
            ckpt_path = ensure_rdah_checkpoint(ckpt_path)
        except Exception as e:  # noqa: BLE001 — loud, actionable error
            print(f"[error] {e}")
            return 1
    if not ckpt_path.exists():
        print(
            f"[error] checkpoint not found: {ckpt_path} — run `model.py train` first."
        )
        return 1

    # ---- registry load (CalibrationNet | RDAH) — single source ---------
    from depthwizard.tifops import load_height_model

    model = load_height_model(ckpt_path, device, architecture=architecture)
    net = model.net
    net.eval()
    use_rgb = model.use_rgb

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_dataset = str(
        ckpt.get(
            "dataset",
            "gamus" if model.architecture in ("rdah", "terraheight_s") else "dfc2019",
        )
    )
    print(
        f"[i] ckpt {ckpt_path} (epoch {model.epoch}, subset-MAE "
        f"{model.val_subset_mae if model.val_subset_mae is not None else float('nan'):.3f}, "
        f"use_rgb={use_rgb}, architecture={model.architecture}, "
        f"trained on {ckpt_dataset})"
    )

    if args.dataset == "gamus":
        return _run_gamus(args, cfg, model, device, cache_dir, ckpt_dataset)

    base = DFC2019Config(
        rgb_dir=Path(paths["rgb_dir"]),
        truth_dir=Path(paths["truth_dir"]),
        depth_cache_dir=cache_dir,
        load_depth=True,
        crop_size=None,
    )
    ds = discover_and_split(base, Path(paths["splits_json"]))

    out_dir = (
        Path(paths["outputs_dir"])
        / ("rdah_net" if model.architecture == "rdah" else "calib_net")
        / (args.out_tag or ("rgb" if use_rgb else "dn_only"))
    )
    results, per_tile_all = {}, {}
    for split in args.splits:
        pooled, summary, city_tab, per_tile, ext = evaluate_split(
            net, ds[split], use_rgb, device, cache_dir
        )
        results[split] = {
            "pooled": pooled,
            "per_tile_summary": summary,
            "by_city": city_tab,
            "extensions": ext,
        }
        per_tile_all[split] = per_tile
        print(
            f"[{split}] pooled: MAE={pooled['mae']:.3f}  RMSE={pooled['rmse']:.3f}  "
            f"r={pooled['pearson_r']:.3f}  bias={pooled['bias']:+.3f}  "
            f"neg_frac={pooled['neg_frac_pred']:.3f}  tiles={pooled['n_tiles']}"
        )
        print(
            f"        per-tile MAE {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}"
        )
        for c, tab in city_tab.items():
            print(f"        {c}: MAE {tab['mae']:.3f}  RMSE {tab['rmse']:.3f}")
        if ext:
            print(
                f"        buildings: MAE {ext.get('building_mae_mean')}  "
                f"scene types: {sorted(ext.get('scene_types', {}))}"
            )

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
                "must_mae": {
                    "limit": g["gate_must_mae"],
                    "got": v["mae"],
                    "pass": v["mae"] < g["gate_must_mae"],
                },
                "must_rmse": {
                    "limit": g["gate_must_rmse"],
                    "got": v["rmse"],
                    "pass": v["rmse"] < g["gate_must_rmse"],
                },
                "target_mae": {
                    "limit": g["gate_target_mae"],
                    "got": v["mae"],
                    "pass": v["mae"] <= g["gate_target_mae"],
                },
                "target_rmse": {
                    "limit": g["gate_target_rmse"],
                    "got": v["rmse"],
                    "pass": v["rmse"] <= g["gate_target_rmse"],
                },
            }
            for k, d in verdicts.items():
                print(
                    f"  GATE {k:11s}: {'PASS' if d['pass'] else 'FAIL'}  "
                    f"({d['got']:.3f} vs {d['limit']:.3f})"
                )
    else:
        print(f"[warn] reference card not found at {ref_path!r} — no gate verdicts.")

    # ---- error maps (test split) ----------------------------------------
    if args.error_maps > 0 and "test" in ds:
        for t in ds["test"].tiles[: args.error_maps]:
            cands = depth_npy_candidates(cache_dir, "dfc2019", t.stem)
            f = next((p for p in cands if p.exists()), None)
            if f is None:
                continue
            raw = np.load(f)
            dn_np = minmax_normalize(raw)
            dn = torch.from_numpy(dn_np[None, None].astype(np.float32)).to(device)
            st_t = torch.from_numpy(dn_tile_stats(raw)[None]).to(device)
            rgb_np = read_tile(t)["rgb"]
            data = read_tile(t)
            rgb = None
            if use_rgb:
                from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD

                rgb_n = (
                    (rgb_np.astype(np.float32) / 255.0) - IMAGENET_MEAN
                ) / IMAGENET_STD
                rgb = torch.from_numpy(rgb_n.transpose(2, 0, 1)[None]).to(device)
            with torch.no_grad():
                pred = net(dn, rgb, None, None, st_t)["pred"][0, 0].cpu().numpy()
            agl = clean_agl(data["agl"])
            _error_map(
                t.stem,
                rgb_np,
                dn_np,
                pred,
                agl,
                out_dir / "error_maps" / f"{t.stem}.png",
            )
        print(f"[i] {args.error_maps} error maps -> {out_dir / 'error_maps'}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "height_model_eval",
        "architecture": model.architecture,
        "dataset": "dfc2019",
        "checkpoint": {
            "path": str(ckpt_path),
            "epoch": model.epoch,
            "architecture": model.architecture,
            "use_rgb": use_rgb,
            "val_subset_mae": model.val_subset_mae,
            "trained_on": ckpt_dataset,
        },
        "results": results,
        "gate_verdicts": verdicts,
        "per_tile_mae": {
            s: [float(x["mae"]) for x in v] for s, v in per_tile_all.items()
        },
    }
    dump_json(report, out_dir / "eval_calib.json")

    lines = [
        "# Calibration net — evaluation vs Phase-1 gates",
        "",
        f"- checkpoint: `{ckpt_path}` (epoch {ckpt['epoch']}, use_rgb={use_rgb})",
        "",
        "| split | MAE (m) | RMSE (m) | r | bias (m) |",
        "|---|---|---|---|---|",
    ]
    for split, r in results.items():
        p = r["pooled"]
        lines.append(
            f"| {split} | {p['mae']:.3f} | {p['rmse']:.3f} "
            f"| {p['pearson_r']:.3f} | {p['bias']:+.3f} |"
        )
    if verdicts:
        lines += ["", "## Gates (frozen reference card)", ""]
        for k, d in verdicts.items():
            lines.append(
                f"- {k}: {'**PASS**' if d['pass'] else '**FAIL**'} "
                f"— {d['got']:.3f} vs {d['limit']:.3f}"
            )
    lines += [
        "",
        "_Floors to quote beside: val MAE 3.186 (median), val RMSE 4.959 "
        "(mean); affine val MAE 4.405. Baselines from reference_card.md._",
    ]
    (out_dir / "eval_calib.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_dir / 'eval_calib.md'}")
    return 0
