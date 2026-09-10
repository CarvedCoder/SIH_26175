"""Ablation runner + boundary-aware evaluation of AGL post-processing.

Runs the FULL candidate ladder (V0..V8) over the frozen validation split
through the certified checkpoint path, scores every variant with the
post-processing metric suite (height + boundary + gradient + seam +
negative fraction), applies the automatic acceptance criteria against the
raw baseline, and writes machine-readable results (CSV + JSON), a markdown
report, and per-tile visual diagnostics.

CITABLE for post-processing decisions (the task's ablation protocol);
the height metrics are the SAME frozen functions `evaluate` uses.

Variant ladder (V-numbers follow the task's ablation spec):
    v0_raw          raw model output (baseline — no post-processing)
    v1_median       3x3 median + conservative spike removal
    v2_guided       RGB-guided filter (He et al. 2010)
    v3_bilateral    joint bilateral filter
    v3b_wls         edge-aware WLS (RGB-gated smoothness)
    v4_conf_wls     WLS + constructed confidence in the data term
    v5_semantic_wls WLS + predicted-semantic boundary gating
    v6_tta          TTA ensemble only (honest full path per augmentation)
    v7_sem_conf_wls semantic + confidence + WLS (+ spike removal)
    v8_full         v7 + TTA fusion (the composed pipeline)
    v9_full_planar  EXPERIMENTAL: v8 + robust planar building refinement

Usage:
  python model.py eval-postprocess --config configs/gamus.yaml \
      --checkpoint outputs/calib_net/postproc_flagship/best.pt \
      --split val --limit 48 --tta-limit 24 --out-tag ablation_r1

With GT semantic layers available on the eval tiles, boundary/region
metrics use them (EVALUATION ONLY — deployed refinement never sees GT).
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.cli.args import (
    add_cache_subdir_arg,
    add_config_arg,
    add_device_arg,
    load_config,
    resolve_cache,
    resolve_device,
)
from depthwizard.postprocess.acceptance import AcceptanceConfig, evaluate_acceptance
from depthwizard.postprocess.config import PostProcessConfig
from depthwizard.postprocess.metrics import postprocess_tile_metrics, summarize_tile_metrics
from depthwizard.postprocess.refinement import refine_agl

NAME = "eval-postprocess"
HELP = "ablation + boundary-aware eval of AGL post-processing on a frozen split"

# name -> (PostProcessConfig kwargs, needs_confidence)
VARIANT_LADDER = {
    "v0_raw": ({}, False),
    "v1_median": (
        {"enabled": True, "method": "median", "spike_removal": True}, False,
    ),
    "v2_guided": (
        {"enabled": True, "method": "guided", "spike_removal": True}, False,
    ),
    "v3_bilateral": (
        {"enabled": True, "method": "bilateral", "spike_removal": True}, False,
    ),
    "v3b_wls": ({"enabled": True, "method": "wls", "spike_removal": True}, False),
    "v4_conf_wls": (
        {"enabled": True, "method": "conf_wls", "spike_removal": True}, False,
    ),
    "v5_semantic_wls": (
        {"enabled": True, "method": "semantic_wls", "spike_removal": True}, False,
    ),
    "v6_tta": ({"enabled": True, "method": "none", "tta": True}, False),
    "v7_sem_conf_wls": (
        {
            "enabled": True,
            "method": "semantic_wls",
            "spike_removal": True,
        },
        True,  # caller-built confidence supplied
    ),
    "v8_full": (
        {
            "enabled": True,
            "method": "semantic_wls",
            "spike_removal": True,
            "tta": True,
        },
        True,
    ),
    "v9_full_planar": (
        {
            "enabled": True,
            "method": "semantic_wls",
            "spike_removal": True,
            "tta": True,
            "planar": True,
        },
        True,
    ),
}


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME, help=HELP, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(p, "configs/gamus.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val",
                   help="frozen split to evaluate (NEVER the training split)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap the number of tiles (0 = all in the split)")
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="subset of the ladder to run (default: all except v9_full_planar)",
    )
    p.add_argument("--tta-limit", type=int, default=24,
                   help="cap tiles that run TTA variants (0 = all, slow)")
    p.add_argument("--include-planar", action="store_true",
                   help="add the EXPERIMENTAL v9_full_planar variant")
    p.add_argument("--no-visualize", action="store_true",
                   help="skip per-tile diagnostic figures")
    p.add_argument("--visualize-tiles", type=int, default=4)
    p.add_argument("--band-px", type=int, default=2,
                   help="boundary band half-width in px for edge metrics")
    p.add_argument("--out-tag", default="postprocess_ablation")
    add_device_arg(p)
    add_cache_subdir_arg(p)
    # acceptance tolerances (configurable, not hard-coded)
    p.add_argument("--mae-tol", type=float, default=0.01)
    p.add_argument("--rmse-tol", type=float, default=0.05)
    p.add_argument("--building-mae-tol", type=float, default=0.05)
    p.add_argument("--boundary-rel-tol", type=float, default=0.05)
    p.add_argument("--grad-rel-tol", type=float, default=0.10)
    p.add_argument("--seam-tol", type=float, default=0.02)
    p.add_argument("--neg-tol", type=float, default=1e-4)
    p.add_argument("--calib-tol", type=float, default=0.05)
    p.add_argument("--wls-lambda", type=float, default=1.0)
    p.add_argument("--guided-radius", type=int, default=4)
    return p


# ---------------------------------------------------------------------------
# Model forward (the certified tifops path)
# ---------------------------------------------------------------------------


def _forward_tile(model, dn_u8norm, rgb_u8, device):
    """One CalibrationNet forward -> (agl_pred, sem_probs|None, sec).

    Uses the SAME construction as evaluate_split: full-tile, min-max
    normalized Dn, ImageNet-normalized RGB. Semantic probabilities come
    from the PREDICTED auxiliary head (softmax) when the checkpoint has
    one — never from GT layers.
    """
    import torch

    from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD

    t0 = time.perf_counter()
    dn = torch.from_numpy(
        np.ascontiguousarray(dn_u8norm, dtype=np.float32)[None, None]
    ).to(device)
    rgb_n = ((rgb_u8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    rgb = torch.from_numpy(np.ascontiguousarray(rgb_n.transpose(2, 0, 1))[None]).to(
        device
    )
    with torch.no_grad():
        out = model.net(dn, rgb)
    pred = out["pred"][0, 0].cpu().numpy().astype(np.float32)
    sem_probs = None
    if model.sem_aux_head and out.get("sem_logits") is not None:
        probs = torch.softmax(out["sem_logits"][0], dim=0).cpu().numpy()
        sem_probs = probs.astype(np.float32)
    return pred, sem_probs, time.perf_counter() - t0


def _make_tta_predict_fn(model, device, backbone_holder):
    """Honest full-path TTA: flipped RGB -> live DAv2 -> per-tile minmax Dn
    -> CalibrationNet -> AGL (on the transformed grid). The closure resolves
    everything itself so tta.tta_fuse only has to transform RGB."""

    def predict_aug(rgb_aug_u8: np.ndarray) -> np.ndarray:
        import torch

        from depthwizard.inference import TILE, tile_bounds
        from depthwizard.normalize import minmax_normalize

        h, w = rgb_aug_u8.shape[:2]
        if backbone_holder.get("bb") is None:
            from depthwizard.backbone import get_backbone

            backbone_holder["bb"] = get_backbone(
                "depth-anything/Depth-Anything-V2-Base-hf", str(device)
            )
        bb = backbone_holder["bb"]
        ny, nx, hp, wp = tile_bounds(h, w)
        rgb_pad = np.pad(rgb_aug_u8, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge")
        raw = np.empty((hp, wp), dtype=np.float32)
        for i in range(ny):
            for j in range(nx):
                y, x = i * TILE, j * TILE
                raw[y : y + TILE, x : x + TILE] = bb.raw_depth(
                    rgb_pad[y : y + TILE, x : x + TILE]
                )
        raw = raw[:h, :w]
        dn = minmax_normalize(raw)
        pred, _sem, _t = _forward_tile(model, dn, rgb_aug_u8, device)
        return pred

    return predict_aug


# ---------------------------------------------------------------------------
# Diagnostics figure
# ---------------------------------------------------------------------------


def _diagnostic_figure(stem, rgb, agl_raw, agl_ref, agl_gt, onehot, variant,
                       out_png, band_px=2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .postprocess.metrics import boundary_bands

    vmax = float(np.nanpercentile(agl_gt, 99)) if np.isfinite(agl_gt).any() else 1.0
    err_raw = np.abs(agl_raw - agl_gt)
    err_ref = np.abs(agl_ref - agl_gt)
    diff = agl_ref - agl_raw
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    panels = [
        (rgb, "RGB", None),
        (agl_raw, "raw AGL (m)", "magma"),
        (agl_ref, f"refined AGL — {variant} (m)", "magma"),
        (agl_gt, "GT AGL (m)", "magma"),
        (err_raw, "|err| raw (m)", "inferno"),
        (err_ref, "|err| refined (m)", "inferno"),
        (diff, "refined - raw (m)", "coolwarm"),
        (None, "GT building boundary band", None),
    ]
    bmask = None
    if onehot is not None:
        bmask = boundary_bands(onehot, band_px)["building"]
    for ax, (im, title, cmap) in zip(axes.ravel(), panels):
        if im is None:
            if bmask is not None:
                ax.imshow(rgb)
                overlay = np.zeros(rgb.shape[:2] + (4,))
                overlay[bmask] = (1, 0, 0, 0.45)
                ax.imshow(overlay)
            ax.set_title(title)
        else:
            kw = {} if cmap is None else {"cmap": cmap, "vmin": 0.0}
            if title.startswith(("raw AGL", "refined", "GT AGL")):
                kw["vmax"] = vmax
            if title.startswith("refined - raw"):
                kw = {"cmap": cmap, "vmin": -2.0, "vmax": 2.0}
            if title.startswith("|err|"):
                kw["vmax"] = float(np.nanpercentile(err_raw, 99))
            im_ = ax.imshow(im, **kw)
            if cmap:
                fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{stem} — post-processing diagnostics", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(args) -> int:
    import torch

    from depthwizard.datasets.factory import build_dataset
    from depthwizard.geo import dump_json
    from depthwizard.postprocess.confidence import estimate_confidence
    from depthwizard.tifops import load_calib_net

    cfg = load_config(args.config)
    paths = cfg["paths"]
    device = resolve_device(args.device)
    cache_dir = resolve_cache(paths, args.cache_subdir)
    model = load_calib_net(args.checkpoint, device)
    print(
        f"[i] ckpt {args.checkpoint} (tag {model.tag}, sem_aux_head="
        f"{model.sem_aux_head})  device={device}"
    )

    dcfg = dict(cfg.get("dataset") or {})
    dcfg["name"] = dcfg.get("name", "gamus")
    if args.limit:
        dcfg["limit"] = args.limit
    sub_cfg = {"paths": paths, "dataset": dcfg}
    ds = build_dataset(
        sub_cfg, split=args.split, crop_size=None, augment=False,
        load_depth=True, depth_cache_dir=cache_dir,
    )
    n_tiles = len(ds)
    print(f"[i] split={args.split}  tiles={n_tiles}")
    if n_tiles == 0:
        print("[error] empty split")
        return 1

    method_subset = args.methods or [
        v for v in VARIANT_LADDER if v != "v9_full_planar"
    ]
    if args.include_planar:
        method_subset = method_subset + ["v9_full_planar"]
    unknown = [m for m in method_subset if m not in VARIANT_LADDER]
    if unknown:
        print(f"[error] unknown variants: {unknown}")
        return 1

    acc_cfg = AcceptanceConfig(
        mae_tol_m=args.mae_tol,
        rmse_tol_m=args.rmse_tol,
        building_mae_tol_m=args.building_mae_tol,
        boundary_rel_tol=args.boundary_rel_tol,
        grad_rel_tol=args.grad_rel_tol,
        seam_tol_m=args.seam_tol,
        neg_abs_tol=args.neg_tol,
        calib_mean_tol_m=args.calib_tol,
    )

    out_dir = Path(paths["outputs_dir"]) / "postprocess" / args.out_tag
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-variant accumulators
    per_variant_tiles: dict[str, list] = {m: [] for m in method_subset}
    runtimes_ms: dict[str, list] = {m: [] for m in method_subset}
    mean_shifts: dict[str, list] = {m: [] for m in method_subset}
    tta_fn = None
    backbone_holder: dict = {}
    tta_budget = args.tta_limit

    for idx in range(n_tiles):
        s = ds[idx]
        meta = s["meta"]
        stem = meta.get("sample_id", f"tile{idx}")
        dn = s["dn"][0].numpy()
        rgb_n = s["rgb"].permute(1, 2, 0).numpy()
        rgb_u8 = np.clip(rgb_n * np.array([[[0.229, 0.224, 0.225]]]) +
                         np.array([[[0.485, 0.456, 0.406]]]), 0, 1)
        rgb_u8 = (rgb_u8 * 255.0).astype(np.uint8)
        agl_gt = s["agl"][0].numpy()
        onehot_gt = (
            s["sem_onehot"].numpy() if s.get("sem_onehot") is not None else None
        )

        agl_raw, sem_probs, fwd_sec = _forward_tile(model, dn, rgb_u8, device)

        # confidence for variants that want a caller-built map (v7/v8/v9):
        # constructed from local stability + edge consistency — documented,
        # NOT calibrated probabilities.
        conf_map = estimate_confidence(agl_raw, rgb_u8)

        do_tta = tta_fn is not None and tta_budget != 0
        if any("tta" in m or m == "v9_full_planar" for m in method_subset):
            if tta_fn is None:
                tta_fn = _make_tta_predict_fn(model, device, backbone_holder)
            do_tta = tta_budget != 0
            if do_tta and tta_budget > 0:
                tta_budget -= 1

        for name in method_subset:
            kw, wants_conf = VARIANT_LADDER[name]
            if ("tta" in kw and kw.get("tta")) and not do_tta:
                per_variant_tiles[name].append(None)
                runtimes_ms[name].append(float("nan"))
                mean_shifts[name].append(float("nan"))
                continue
            pcfg = PostProcessConfig(**{
                "guided_radius": args.guided_radius,
                "wls_lambda": args.wls_lambda,
                **kw,
            })
            t0 = time.perf_counter()
            refined, report = refine_agl(
                agl_raw,
                rgb_u8,
                pcfg,
                sem_probs=sem_probs,
                confidence=conf_map if wants_conf else None,
                tta_predict_fn=tta_fn if pcfg.tta else None,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            runtimes_ms[name].append(dt_ms)
            if report.calibration:
                mean_shifts[name].append(report.calibration["mean_shift"])
            m = postprocess_tile_metrics(
                refined, agl_gt, onehot_gt, band_px=args.band_px
            )
            per_variant_tiles[name].append(m)
            if (
                not args.no_visualize
                and idx < args.visualize_tiles
                and name != "v0_raw"
            ):
                _diagnostic_figure(
                    f"{stem}_{name}", rgb_u8, agl_raw, refined, agl_gt,
                    onehot_gt, name, viz_dir / f"{stem}_{name}.png",
                    band_px=args.band_px,
                )
        print(
            f"  [{idx + 1}/{n_tiles}] {stem}  fwd {fwd_sec * 1000:.0f} ms"
            f"{'  (+TTA)' if do_tta else ''}"
        )

    # ---- summarize ---------------------------------------------------------
    def flat_summary(name):
        tiles = [t for t in per_variant_tiles[name] if t is not None]
        if not tiles:
            return {}
        g = summarize_tile_metrics(tiles)
        return {
            "mae": g.get("global", {}).get("mae", float("nan")),
            "rmse": g.get("global", {}).get("rmse", float("nan")),
            "bias": g.get("global", {}).get("bias", float("nan")),
            "pearson_r": g.get("global", {}).get("pearson_r", float("nan")),
            "negative_fraction": g.get("global", {}).get("negative_fraction", float("nan")),
            "building_mae": g.get("region_building", {}).get("mae", float("nan")),
            "building_rmse": g.get("region_building", {}).get("rmse", float("nan")),
            "ground_mae": g.get("region_ground", {}).get("mae", float("nan")),
            "vegetation_mae": g.get("region_vegetation", {}).get("mae", float("nan")),
            "boundary_mae": g.get("boundary_building", {}).get("mae", float("nan")),
            "boundary_bg_mae": g.get("boundary_building_ground", {}).get("mae", float("nan")),
            "grad_mae": g.get("gradient", {}).get("grad_mae", float("nan")),
            "seam_mae": g.get("seam", {}).get("seam_mae", float("nan")),
            "jump_ratio": g.get("discontinuity", {}).get("jump_ratio", float("nan")),
            "runtime_ms_per_tile": float(np.nanmean(runtimes_ms[name])),
            "mean_shift_m": float(np.nanmean(mean_shifts[name])),
            "n_tiles": len(tiles),
        }

    summaries = {name: flat_summary(name) for name in method_subset}
    baseline = summaries.get("v0_raw", {})

    csv_path = out_dir / "ablation.csv"
    cols = [
        "variant", "mae", "rmse", "bias", "pearson_r", "negative_fraction",
        "building_mae", "building_rmse", "ground_mae", "vegetation_mae",
        "boundary_mae", "boundary_bg_mae", "grad_mae", "seam_mae",
        "jump_ratio", "runtime_ms_per_tile", "mean_shift_m", "n_tiles",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name in method_subset:
            row = {"variant": name}
            row.update({c: summaries[name].get(c, "") for c in cols[1:]})
            w.writerow(row)

    # acceptance vs baseline
    verdicts = {}
    for name in method_subset:
        if name == "v0_raw":
            verdicts[name] = {"verdict": "BASELINE", "rules": {}}
            continue
        res = evaluate_acceptance(
            baseline, summaries[name],
            summaries[name].get("mean_shift_m"), acc_cfg,
        )
        verdicts[name] = {"verdict": res.verdict, "rules": res.rules}

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "postprocess_ablation",
        "checkpoint": str(args.checkpoint),
        "model_tag": model.tag,
        "split": args.split,
        "n_tiles": n_tiles,
        "tta_tiles_budget": args.tta_limit,
        "band_px": args.band_px,
        "acceptance": acc_cfg.__dict__,
        "semantics_source": (
            "predicted auxiliary head (softmax)" if model.sem_aux_head
            else "UNAVAILABLE — semantic variants degraded to RGB-only"
        ),
        "summaries": summaries,
        "verdicts": verdicts,
        "per_tile_mae": {
            name: [
                t["global"]["mae"] for t in per_variant_tiles[name] if t is not None
            ]
            for name in method_subset
        },
    }
    dump_json(report, out_dir / "ablation.json")

    # markdown
    lines = [
        "# Post-processing ablation",
        "",
        f"- checkpoint: `{args.checkpoint}` ({model.tag})",
        f"- split: **{args.split}** ({n_tiles} tiles), band={args.band_px}px",
        f"- semantics: {report['semantics_source']}",
        "",
        "| variant | MAE | RMSE | bias | bldg MAE | boundary MAE | grad MAE |"
        " seam | neg frac | ms/tile | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in method_subset:
        s = summaries[name]
        v = verdicts[name]["verdict"]
        lines.append(
            f"| {name} | {s.get('mae', float('nan')):.3f} "
            f"| {s.get('rmse', float('nan')):.3f} "
            f"| {s.get('bias', float('nan')):+.3f} "
            f"| {s.get('building_mae', float('nan')):.3f} "
            f"| {s.get('boundary_mae', float('nan')):.3f} "
            f"| {s.get('grad_mae', float('nan')):.3f} "
            f"| {s.get('seam_mae', float('nan')):.3f} "
            f"| {s.get('negative_fraction', float('nan')):.4f} "
            f"| {s.get('runtime_ms_per_tile', float('nan')):.0f} "
            f"| {v} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[done] results -> {out_dir}")
    for name in method_subset:
        s, v = summaries[name], verdicts[name]["verdict"]
        print(
            f"  {name:16s} MAE {s.get('mae', float('nan')):.3f}  "
            f"RMSE {s.get('rmse', float('nan')):.3f}  "
            f"bldg {s.get('building_mae', float('nan')):.3f}  "
            f"boundary {s.get('boundary_mae', float('nan')):.3f}  "
            f"{v}"
        )
    return 0
