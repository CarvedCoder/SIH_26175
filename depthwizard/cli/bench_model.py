"""Benchmark the height-model backend: resolution sweep + peak VRAM.

Measures, per input resolution (default 256 / 512 / 1024 — the RDAH
architecture requires multiples of 128 and caps at 1024):

  * wall-clock forward time (batch 1, torch.no_grad, warm-up + repeats)
  * PEAK VRAM (torch.cuda.max_memory_allocated — the number that matters
    on the RTX 4050 6 GB target; None on CPU)
  * parameter count

Optional AMP autocast (fp16) mirroring the training path, so you can check
both settings on your GPU before committing to a tile size.

Usage:
  python model.py bench --architecture rdah
  python model.py bench --architecture rdah --resolutions 256 512 1024 --amp
  python model.py bench --architecture calibration_net \
      --checkpoint outputs/calib_net/rgb_cos/best.pt
  python model.py bench --architecture rdah --device cuda --repeats 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from depthwizard.cli.args import add_config_arg, add_device_arg, resolve_device

NAME = "bench"
HELP = "benchmark a height backend (resolution sweep, params, peak VRAM)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(p, "configs/infer.yaml")
    p.add_argument(
        "--architecture",
        choices=("rdah", "calibration_net", "terraheight_s", "auto"),
        default="rdah",
        help="height-model backend to benchmark (auto = detect from ckpt)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path (rdah default: checkpoints/rdah/... "
        "auto-downloads; terraheight_s default: models/terraheight/"
        "best_model.pth or repo-root best_model.pth — never downloaded; "
        "calibration_net: pass a trained best.pt)",
    )
    p.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="square input sizes to sweep (RDAH: multiples of 128, <= 1024)",
    )
    p.add_argument("--repeats", type=int, default=3, help="timed forwards per size")
    p.add_argument("--warmup", type=int, default=1, help="untimed warm-up forwards")
    p.add_argument(
        "--amp",
        action="store_true",
        help="autocast fp16 on CUDA (mirrors the training path's AMP)",
    )
    add_device_arg(p)
    return p


def run(args) -> int:
    import torch

    from depthwizard.tifops import load_height_model

    device = resolve_device(args.device)
    architecture = None if args.architecture == "auto" else args.architecture

    ckpt = args.checkpoint
    if ckpt is None:
        if architecture == "calibration_net":
            print(
                "[error] --architecture calibration_net needs --checkpoint "
                "(a trained outputs/calib_net/<tag>/best.pt)."
            )
            return 1
        if architecture == "terraheight_s":
            from depthwizard.terraheight import default_checkpoint_path

            try:
                ckpt = str(default_checkpoint_path())
            except FileNotFoundError as e:
                print(f"[error] {e}")
                return 1
        else:
            from depthwizard.rdah import RDAH_CKPT_DIR, RDAH_CHECKPOINTS

            ckpt = str(RDAH_CKPT_DIR / RDAH_CHECKPOINTS["track1"]["filename"])

    model = load_height_model(ckpt, device, architecture=architecture)
    net = model.net
    net.eval()
    n_par = sum(p.numel() for p in net.parameters())

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(
        f"[bench] architecture={model.architecture}  params={n_par:,}  "
        f"device={device}  amp={args.amp and device.startswith('cuda')}  "
        f"repeats={args.repeats}"
    )
    print(f"{'size':>8s} {'fwd ms':>10s} {'peak VRAM MB':>14s}")
    results = []
    for size in args.resolutions:
        if model.architecture == "rdah":
            if size % 128 != 0 or size > 1024:
                print(
                    f"{size:8d}  SKIPPED — RDAH needs multiples of 128 and "
                    "<= 1024 (BlockAttention windows + 64x64 positional "
                    "encoding buffer)"
                )
                continue
        if model.architecture == "terraheight_s" and size % 14 != 0:
            print(
                f"{size:8d}  SKIPPED — TerraHeight needs multiples of 14 "
                "(ViT patch size; the published training crop is 630)"
            )
            continue
        # realistic inputs: raw DAv2-like depth (1..6) and ImageNet-norm RGB
        g = torch.Generator().manual_seed(0)
        raw = 1.0 + 5.0 * torch.rand(1, 1, size, size, generator=g)
        dn, stats = _normalize_like_dataset(raw)
        rgb = torch.randn(1, 3, size, size, generator=g)
        dn, stats, rgb = dn.to(device), stats.to(device), rgb.to(device)
        try:
            with torch.no_grad():
                for _ in range(args.warmup):
                    _forward(net, dn, rgb, stats, device, args.amp)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    _forward(net, dn, rgb, stats, device, args.amp)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / args.repeats
        except torch.cuda.OutOfMemoryError:
            print(f"{size:8d}  OOM — reduce the tile size or enable --amp")
            torch.cuda.empty_cache()
            continue
        peak_mb = None
        if device.startswith("cuda"):
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        results.append((size, dt * 1000.0, peak_mb))
        peak_s = f"{peak_mb:14.0f}" if peak_mb is not None else f"{'n/a (CPU)':>14s}"
        print(f"{size:8d} {dt * 1000.0:10.1f} {peak_s}")

    # summary table the user can paste into the runbook
    if results:
        print("\n[bench] tested resolutions:", [r[0] for r in results])
        ok = [r for r in results if r[2] is None or r[2] < 6000]
        if ok:
            print(
                f"[bench] largest resolution under ~6 GB (RTX 4050 budget): "
                f"{max(ok, key=lambda r: r[0])[0]}"
            )
    return 0


def _normalize_like_dataset(raw):
    """Mirror the dataset contract: min-max [0,1] + dn_tile_stats [4]."""
    import torch

    lo, hi = float(raw.min()), float(raw.max())
    dn = (raw - lo) / (hi - lo)
    stats = torch.log(torch.tensor([lo + 1e-3, hi + 1e-3, hi - lo + 1e-3,
                                    float(raw.mean()) + 1e-3]))
    return dn, stats


def _forward(net, dn, rgb, stats, device: str, amp: bool):
    import torch

    # TerraHeight consumes RGB ONLY (no depth cache / dn / stats); its
    # forward signature is (rgb, dn, dem, sem, stats) — rgb first.
    if getattr(net, "architecture", None) == "terraheight_s":
        with torch.autocast(
            device_type="cuda" if (amp and device.startswith("cuda")) else "cpu",
            dtype=torch.float16 if (amp and device.startswith("cuda")) else torch.float32,
            enabled=bool(amp and device.startswith("cuda")),
        ):
            return net(rgb, None, None, None, None)

    with torch.autocast(
        device_type="cuda" if (amp and device.startswith("cuda")) else "cpu",
        dtype=torch.float16 if (amp and device.startswith("cuda")) else torch.float32,
        enabled=bool(amp and device.startswith("cuda")),
    ):
        return net(dn, rgb, None, None, stats)
