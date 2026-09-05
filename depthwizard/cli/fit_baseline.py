"""Global affine calibration baseline  H = a * Dn + b.

Methodology fixes relative to a single-tile fit:
  * Fit uses ONLY the train split (splits.json). The original single-tile
    numbers are a methodology error, not a regression to preserve.
  * least-squares via torch.linalg.lstsq. Optional --fit huber: gradient-
    descent Huber fit, more robust to LiDAR outliers.
  * Diagnostics reported: per-tile Pearson r, residual quantiles, fraction
    of negative predictions on train.

Datasets (Phase 3):
  * default (no --dataset / dfc2019): the frozen DFC2019 path — identical
    behavior, but cache reads now fall back to the namespaced layout.
  * --dataset gamus: fits on the GAMUS train split through the adapter
    (same Dn/AGL semantics); the affine json is written to
    global_affine_gamus.json (NEVER overwrites the DFC baseline file).

Usage:
  python model.py fit-baseline --config configs/phase1.yaml
  python model.py fit-baseline --config configs/phase1.yaml --fit huber
  python model.py fit-baseline --config configs/gamus.yaml --dataset gamus
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.cli.args import add_cache_subdir_arg, add_config_arg, load_config
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.geo import depth_npy_candidates, dump_json, read_tile
from depthwizard.normalize import clean_agl, minmax_normalize

NAME = "fit-baseline"
HELP = "fit the global affine baseline H = a*Dn + b (train split only)"


def collect_pixels(ds, stride: int, cache_tag_dir: Path, max_tiles: int = 0):
    """Stack (Dn, AGL_clean) pixels over the DFC train split (frozen path,
    namespaced-cache-aware) with subsampling."""
    xs, ys = [], []
    tiles = ds.tiles if not max_tiles else ds.tiles[:max_tiles]
    for t in tiles:
        cands = depth_npy_candidates(cache_tag_dir, "dfc2019", t.stem)
        f = next((p for p in cands if p.exists()), None)
        if f is None:
            raise FileNotFoundError(
                f"missing depth cache for {t.stem} (looked in {cands}) — "
                "run `model.py depth` first")
        raw = np.load(f)
        dn = minmax_normalize(raw)[::stride, ::stride].ravel()
        data = read_tile(t)
        agl = clean_agl(data["agl"])[::stride, ::stride].ravel()
        keep = np.isfinite(agl)
        xs.append(dn[keep]); ys.append(agl[keep])
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    return x, y


def collect_pixels_adapter(ds, stride: int, max_tiles: int = 0):
    """Stack (Dn, AGL) pixels over ANY adapter dataset (GAMUS, mixed) via
    the sample contract — identical normalization semantics (minmax +
    clean_agl) as the frozen DFC path."""
    xs, ys = [], []
    n = len(ds) if not max_tiles else min(max_tiles, len(ds))
    for i in range(n):
        s = ds[i]
        if s["dn"] is None:
            raise FileNotFoundError(
                f"sample {s['meta'].get('sample_id')}: no cached Dn — run "
                "`model.py depth --dataset gamus` first")
        dn = s["dn"][0].numpy()[::stride, ::stride].ravel()
        agl = s["agl"][0].numpy()[::stride, ::stride].ravel()
        keep = np.isfinite(agl)
        xs.append(dn[keep].astype(np.float32)); ys.append(agl[keep].astype(np.float32))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    return x, y


def fit_lstsq(x: np.ndarray, y: np.ndarray):
    import torch
    A = torch.stack([torch.from_numpy(x), torch.ones_like(torch.from_numpy(x))], dim=1)
    b = torch.from_numpy(y)[:, None]
    sol = torch.linalg.lstsq(A, b).solution.flatten()
    return float(sol[0]), float(sol[1])


def fit_huber(x: np.ndarray, y: np.ndarray, iters: int = 1500, lr: float = 0.05):
    import torch
    xt = torch.from_numpy(x); yt = torch.from_numpy(y)
    a = torch.tensor(0.15, requires_grad=True)
    b = torch.tensor(-9.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    huber = torch.nn.HuberLoss(delta=1.0)
    for _ in range(iters):
        opt.zero_grad()
        loss = huber(a * xt + b, yt)
        loss.backward()
        opt.step()
    return float(a.detach()), float(b.detach())


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase1.yaml")
    p.add_argument("--dataset", choices=("dfc2019", "gamus"), default="dfc2019",
                   help="fit on DFC2019 (frozen path) or GAMUS (adapter; "
                        "writes global_affine_gamus.json)")
    p.add_argument("--fit", choices=("lstsq", "huber"), default=None)
    p.add_argument("--stride", type=int, default=None,
                   help="pixel subsampling stride (config default: 4)")
    p.add_argument("--max-tiles", type=int, default=0, help="debug: fit on first N train tiles")
    add_cache_subdir_arg(p)
    return p


def _fit_and_write(x, y, fit_kind, stride, n_tiles, out_path: Path,
                   dataset: str, extra: dict | None = None) -> int:
    """Shared fit + diagnostics + json write (single source for both paths)."""
    if fit_kind == "lstsq":
        a, b = fit_lstsq(x, y)
    else:
        a, b = fit_huber(x, y)

    pred = a * x + b
    resid = pred - y
    pear = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
    diagnostics = {
        "train_pearson_r_Dn_vs_AGL": pear,
        "train_neg_pred_frac": float((pred < 0).mean()),
        "resid_q": {q: float(np.quantile(resid, qq))
                    for q, qq in (("p01", .01), ("p50", .5), ("p99", .99))},
        "agl_mean_train": float(y.mean()),
    }
    print(f"[result] H = {a:.6f} * Dn + {b:.6f}")
    print(f"[diag]   {diagnostics}")

    out = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": f"global_affine_{fit_kind}",
        "dataset": dataset,
        "a": a, "b": b,
        "normalization": "per-tile min-max of raw DAv2 depth (see depthwizard.normalize)",
        "fit_split": "train", "stride": stride,
        "n_pixels": int(x.size), "n_tiles": n_tiles,
        "diagnostics": diagnostics,
    }
    if extra:
        out.update(extra)
    dump_json(out, out_path)
    print(f"-> {out_path}")
    return 0


def _run_gamus(args, cfg, fit_kind, stride) -> int:
    """Fit the affine baseline on the GAMUS train split (adapter pipeline)."""
    from depthwizard.cli.args import resolve_cache
    from depthwizard.datasets.factory import build_dataset

    paths = cfg["paths"]
    try:
        cache_dir = resolve_cache(paths, args.cache_subdir)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return 1
    print(f"[i] using depth cache: {cache_dir}")

    dcfg = dict(cfg.get("dataset") or {})
    dcfg["name"] = "gamus"
    if args.max_tiles and not dcfg.get("limit"):
        dcfg["limit"] = args.max_tiles        # deterministic cap for the fit
    sub_cfg = {"paths": paths, "dataset": dcfg}
    ds = build_dataset(sub_cfg, split="train", crop_size=None, augment=False,
                       load_depth=True, depth_cache_dir=cache_dir)
    print(f"[i] GAMUS train tiles: {len(ds)} "
          f"(units: nDSM, ASSUMED metres — flagged per sample in meta)")

    x, y = collect_pixels_adapter(ds, stride, args.max_tiles)
    print(f"[i] fit pixels: {x.size:,} (stride={stride})")

    out_dir = Path(paths["outputs_dir"]) / "baseline"
    # NEVER overwrite the DFC baseline json — dataset-suffixed filename.
    return _fit_and_write(x, y, fit_kind, stride, len(ds),
                          out_dir / "global_affine_gamus.json",
                          dataset="gamus",
                          extra={"height_semantics":
                                 "nDSM/AGL; units UNDOCUMENTED — ASSUMED "
                                 "metres (GAMUS paper/README)"})


def run(args) -> int:
    cfg = load_config(args.config)

    paths = cfg["paths"]
    bl = cfg.get("baseline", {}) or {}
    fit_kind = args.fit or bl.get("fit", "lstsq")
    stride = args.stride or int(bl.get("stride", 4))

    if args.dataset == "gamus":
        return _run_gamus(args, cfg, fit_kind, stride)

    cache_dir = Path(paths["depth_cache_dir"])
    subdirs = sorted(d for d in cache_dir.iterdir() if d.is_dir()) if cache_dir.exists() else []
    if not subdirs:
        print(f"[error] no depth cache under {cache_dir}. Run `model.py depth` first.")
        return 1
    if len(subdirs) > 1:
        if args.cache_subdir:
            cache_tag_dir = cache_dir / args.cache_subdir
            if not cache_tag_dir.is_dir():
                print(f"[error] --cache-subdir '{args.cache_subdir}' not found under {cache_dir}")
                return 1
        else:
            print(f"[error] multiple model caches found: {[d.name for d in subdirs]}. "
                  f"Pass --cache-subdir to disambiguate — never mix depths across backbones.")
            return 1
    else:
        cache_tag_dir = subdirs[0]
    print(f"[i] using depth cache: {cache_tag_dir}")

    ds_cfg = DFC2019Config(
        rgb_dir=Path(paths["rgb_dir"]), truth_dir=Path(paths["truth_dir"]),
        depth_cache_dir=cache_tag_dir, load_depth=True, crop_size=None)
    ds = discover_and_split(ds_cfg, Path(paths["splits_json"]))
    print(f"[i] tiles: {len(ds['train'].tiles)} train / "
          f"{len(ds['val'].tiles)} val / {len(ds['test'].tiles)} test")

    x, y = collect_pixels(ds["train"], stride, cache_tag_dir, args.max_tiles)
    print(f"[i] fit pixels: {x.size:,} (stride={stride})")

    out_dir = Path(paths["outputs_dir"]) / "baseline"
    return _fit_and_write(x, y, fit_kind, stride, len(ds["train"].tiles),
                          out_dir / "global_affine.json", dataset="dfc2019")
