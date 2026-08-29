"""Global affine calibration baseline  H = a * Dn + b.

Methodology fixes relative to a single-tile fit:
  * Fit uses ONLY the train split (splits.json). The original single-tile
    numbers are a methodology error, not a regression to preserve.
  * least-squares via torch.linalg.lstsq. Optional --fit huber: gradient-
    descent Huber fit, more robust to LiDAR outliers.
  * Diagnostics reported: per-tile Pearson r, residual quantiles, fraction
    of negative predictions on train.

Usage:
  python main.py fit-baseline --config configs/phase1.yaml
  python main.py fit-baseline --config configs/phase1.yaml --fit huber
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.cli.args import add_cache_subdir_arg, add_config_arg, load_config
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.geo import dump_json, read_tile
from depthwizard.normalize import clean_agl, minmax_normalize

NAME = "fit-baseline"
HELP = "fit the global affine baseline H = a*Dn + b (train split only)"


def collect_pixels(ds, stride: int, cache_tag_dir: Path, max_tiles: int = 0):
    """Stack (Dn, AGL_clean) pixels over the split with subsampling."""
    xs, ys = [], []
    tiles = ds.tiles if not max_tiles else ds.tiles[:max_tiles]
    for t in tiles:
        f = cache_tag_dir / f"{t.stem}.npy"
        if not f.exists():
            raise FileNotFoundError(
                f"missing depth cache for {t.stem} at {f} — run `main.py depth` first")
        raw = np.load(f)
        dn = minmax_normalize(raw)[::stride, ::stride].ravel()
        data = read_tile(t)
        agl = clean_agl(data["agl"])[::stride, ::stride].ravel()
        keep = np.isfinite(agl)
        xs.append(dn[keep]); ys.append(agl[keep])
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
    p.add_argument("--fit", choices=("lstsq", "huber"), default=None)
    p.add_argument("--stride", type=int, default=None,
                   help="pixel subsampling stride (config default: 4)")
    p.add_argument("--max-tiles", type=int, default=0, help="debug: fit on first N train tiles")
    add_cache_subdir_arg(p)
    return p


def run(args) -> int:
    cfg = load_config(args.config)

    paths = cfg["paths"]
    bl = cfg["baseline"]
    fit_kind = args.fit or bl.get("fit", "lstsq")
    stride = args.stride or int(bl.get("stride", 4))

    cache_dir = Path(paths["depth_cache_dir"])
    subdirs = sorted(d for d in cache_dir.iterdir() if d.is_dir()) if cache_dir.exists() else []
    if not subdirs:
        print(f"[error] no depth cache under {cache_dir}. Run `main.py depth` first.")
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
        "a": a, "b": b,
        "normalization": "per-tile min-max of raw DAv2 depth (see depthwizard.normalize)",
        "fit_split": "train", "stride": stride,
        "n_pixels": int(x.size), "n_tiles": len(ds["train"].tiles),
        "diagnostics": diagnostics,
    }
    out_dir = Path(paths["outputs_dir"]) / "baseline"
    dump_json(out, out_dir / "global_affine.json")
    print(f"-> {out_dir / 'global_affine.json'}")
    return 0
