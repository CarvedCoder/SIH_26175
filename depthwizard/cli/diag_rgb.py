"""Diagnostic for a pinned RGB calibration run — ~2 min, no training.

  A. RGB tensors ImageNet-normalized? (expect min ~ -2.1, max ~ +2.6,
     per-channel mean ~ 0; 0..255 or 0..1 means the loader contract broke)
  B. NaN/Inf anywhere in rgb / dn / agl?
  C. Init exactness: net(dn, rgb) == clamp(a0*Dn + b0)?
  D. Pre-clip gradient norms (total/head) + clip-hit count.

Usage: python model.py diag --config configs/phase2.yaml [--batches 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from depthwizard.calibration_net import (CalibrationNet, load_affine_init,
                                         masked_l1_loss)
from depthwizard.cli.args import (add_config_arg, load_config, resolve_cache,
                                  resolve_device)
from depthwizard.dataset import DFC2019Config, discover_and_split

NAME = "diag"
HELP = "tensor/init/gradient diagnostics for a pinned training run"


def gnorm(params) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.norm()) ** 2
    return s ** 0.5


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase2.yaml")
    p.add_argument("--batches", type=int, default=20)
    return p


def run(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    cfg = load_config(args.config)
    paths, mcfg, tcfg = cfg["paths"], cfg["model"], cfg["train"]
    device = resolve_device(None)
    torch.manual_seed(tcfg.get("seed", 42)); np.random.seed(tcfg.get("seed", 42))

    cache_dir = resolve_cache(paths, None)
    a0, b0 = load_affine_init(paths["affine_json"])

    base = DFC2019Config(
        rgb_dir=Path(paths["rgb_dir"]), truth_dir=Path(paths["truth_dir"]),
        depth_cache_dir=cache_dir, load_depth=True,
        crop_size=tcfg["crop_size"], augment=True,
        clamp_agl_min=cfg["dataset"]["clamp_agl_min"] if "dataset" in cfg else 0.0,
        seed=tcfg.get("seed", 42))
    ds = discover_and_split(base, Path(paths["splits_json"]))
    dl = DataLoader(ds["train"], batch_size=tcfg["batch_size"],
                    shuffle=True, num_workers=0)

    net = CalibrationNet(in_ch=4, widths=tuple(mcfg["widths"]),
                         a0=a0, b0=b0,
                         clamp_min=mcfg.get("clamp_min", 0.0)).to(device)

    print("=" * 72)
    print(f"[A/B] tensor stats over first {args.batches} train batches")
    rgb_min, rgb_max = float("inf"), float("-inf")
    for bi, batch in enumerate(dl):
        if bi >= args.batches:
            break
        rgb, dn, agl = batch["rgb"], batch["dn"], batch["agl"]
        rgb_min = min(rgb_min, float(rgb.min()))
        rgb_max = max(rgb_max, float(rgb.max()))
        for name, t in (("rgb", rgb), ("dn", dn), ("agl", agl)):
            if not torch.isfinite(t).all():
                print(f"  [!!] batch {bi}: {name} contains NaN/Inf")
        if bi == 0:
            print(f"  rgb  shape={tuple(rgb.shape)}  per-ch mean="
                  f"{[round(v, 3) for v in rgb.mean(dim=(0, 2, 3)).tolist()]}")
            print(f"  dn   shape={tuple(dn.shape)}  "
                  f"min={float(dn.min()):.3f} max={float(dn.max()):.3f}")
            print(f"  agl  shape={tuple(agl.shape)}  "
                  f"min={float(agl.min()):.3f} max={float(agl.max()):.3f}")
    print(f"  rgb global min={rgb_min:.3f}  max={rgb_max:.3f}")
    print("  HEALTHY: min ~ -2.1, max ~ +2.6, |per-ch mean| < 0.5 (ImageNet-norm)")
    print("  BROKEN : 0..255 (raw bytes) or 0..1 (scaled but not normalized)")

    print("=" * 72)
    print("[C] init exactness (no optimizer step taken yet)")
    net.eval()
    with torch.no_grad():
        s = ds["val"][0]
        dn1 = s["dn"].to(device); rgb1 = s["rgb"].to(device)
        pred = net(dn1, rgb1)["pred"]
        affine = (a0 * dn1 + b0).clamp(min=mcfg.get("clamp_min", 0.0))
        print(f"  max |pred - clamp(a0*Dn+b0)| = "
              f"{float((pred - affine).abs().max()):.6f}  (expect <= 1e-5)")
    net.train()

    print("=" * 72)
    print(f"[D] pre-clip grad norms ({args.batches} batches, after 1 warmup step)")
    opt = torch.optim.Adam(net.parameters(), lr=tcfg["lr"],
                           weight_decay=tcfg.get("weight_decay", 1e-4))
    clip = float(tcfg.get("grad_clip", 5.0))
    # At exact init the encoder grad is ZERO by construction (zero head), so
    # take one warmup step and measure the steady-state gradient structure.
    for batch in dl:
        loss = masked_l1_loss(net(batch["dn"].to(device),
                                  batch["rgb"].to(device))["pred"],
                              batch["agl"].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        break
    tot, head_n, nb = [], [], 0
    for bi, batch in enumerate(dl):
        if bi >= args.batches:
            break
        pred = net(batch["dn"].to(device), batch["rgb"].to(device))["pred"]
        loss = masked_l1_loss(pred, batch["agl"].to(device))
        opt.zero_grad(set_to_none=True); loss.backward()
        t = gnorm(net.parameters())
        tot.append(t); head_n.append(gnorm([net.head.weight, net.head.bias]))
        nb += 1
    print(f"  total grad norm : mean {np.mean(tot):8.3f}  "
          f"p50 {np.median(tot):8.3f}  max {np.max(tot):8.3f}")
    print(f"  head grad norm  : mean {np.mean(head_n):8.3f}  (must be >> 0)")
    print(f"  clip threshold  : {clip}   batches exceeding it: "
          f"{sum(1 for t in tot if t > clip)}/{nb}")
    return 0
