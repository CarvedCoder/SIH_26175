"""TTA-consistency check for CalibrationNet experiments (fixed protocol).

Measures how much a checkpoint's prediction moves under input
transformations that should NOT change the answer, on a FIXED sample of
official-GAMUS val tiles (sorted, first K; default 16 — deterministic, so
before/after numbers are directly comparable across checkpoints):

  A. flip/rotate TTA variance — 8 dihedral orientations of the SAME tile;
     each prediction is mapped back to the identity orientation and the
     per-pixel std across the ensemble is averaged (plus its 95th
     percentile). This is the closest offline equivalent of
     inference.DepthWizardPredictor.make_tta_predict_fn's flip/rotate
     ensemble, run on the cached Dn (no live backbone needed: the Dn input
     is a fixed artifact for these tiles, and Dihedral transforms of the
     RGB/Dn pair are exactly what the live-backbone TTA would feed the net
     up to backbone rerun noise).

  B. window-overlap disagreement — the tiles-mode seam stress test: each
     1024 tile is split into four 512 corner windows, each normalized PER
     WINDOW (the training contract at window granularity), predicted
     independently, and the |pred| disagreement is averaged over the
     overlap strips. Per-window min-max normalization makes the same ground
     pixel a different Dn value in each window — this is the mechanism the
     tile-statistic conditioning (Exp 1) is supposed to fix, so this metric
     is the direct test of that hypothesis.

Both metrics are computed from the SAME certified forward path the evaluate
command uses (net on normalized Dn + ImageNet-normalized RGB). NOT citable
final metrics — this is a consistency diagnostic, like compute_stats.

Usage:
  python tools/exp_tta_check.py --checkpoint outputs/calib_net/<tag>/best.pt \
      [--k 16] [--out-json PATH]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TILE = 1024
CROP = 512
ORIENTATIONS = ("id", "rot90", "rot180", "rot270", "hflip", "vflip", "transp", "transp2")


def _transform(arr: np.ndarray, kind: str) -> np.ndarray:
    """Dihedral transform of [H,W] or [H,W,C] (arr is a view-safe copy)."""
    if kind == "id":
        return arr
    if kind == "rot90":
        return np.rot90(arr, 1).copy()
    if kind == "rot180":
        return np.rot90(arr, 2).copy()
    if kind == "rot270":
        return np.rot90(arr, 3).copy()
    if kind == "hflip":
        return arr[:, ::-1].copy()
    if kind == "vflip":
        return arr[::-1].copy()
    if kind == "transp":
        return np.swapaxes(arr, 0, 1).copy()
    if kind == "transp2":  # anti-transpose
        return np.rot90(np.swapaxes(arr, 0, 1), 2).copy()
    raise ValueError(kind)


def _inverse_transform(arr: np.ndarray, kind: str) -> np.ndarray:
    return _transform(arr, {
        "id": "id", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90",
        "hflip": "hflip", "vflip": "vflip", "transp": "transp",
        "transp2": "transp2",
    }[kind])


def _forward(net, dn_n: np.ndarray, rgb_n: np.ndarray, device: str, stats=None):
    import torch

    from depthwizard.normalize import dn_tile_stats

    dn_t = torch.from_numpy(dn_n[None, None].astype(np.float32)).to(device)
    rgb_t = torch.from_numpy(rgb_n.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    if stats is None:
        stats = dn_tile_stats(dn_n)
    kwargs: dict = {}
    if getattr(net, "film_stats", False):
        st_t = torch.from_numpy(np.asarray(stats, dtype=np.float32)[None]).to(device)
        kwargs["stats"] = st_t
    with torch.no_grad():
        return net(dn_t, rgb_t, None, None, **kwargs)["pred"][0, 0].cpu().numpy()


def check_checkpoint(ckpt_path: str, k: int, device: str) -> dict:
    import torch

    from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
    from depthwizard.normalize import dn_tile_stats, minmax_normalize
    from depthwizard.tifops import load_calib_net
    from depthwizard.datasets.factory import build_dataset
    from depthwizard.geo import depth_npy_candidates

    cfg = {
        "paths": {
            "depth_cache_dir": "outputs/depth_cache",
            "outputs_dir": "outputs",
            "affine_json": "outputs/baseline/global_affine_gamus.json",
        },
        "dataset": {"name": "gamus", "source": "local", "local_root": "gamus_full"},
    }
    model = load_calib_net(ckpt_path, device)
    net = model.net
    net.eval()
    ds = build_dataset(cfg, split="val", crop_size=None, augment=False,
                       load_depth=True, depth_cache_dir=Path("outputs/depth_cache/depth_anything_v2_base_hf"))
    n = min(k, len(ds))
    tta_stds, tta_p95, ov_dis = [], [], []
    for i in range(n):
        s = ds[i]
        sid = s["meta"]["sample_id"]
        cands = depth_npy_candidates(Path("outputs/depth_cache/depth_anything_v2_base_hf"), "gamus", sid)
        raw = np.load(next(p for p in cands if p.exists()))
        dn = minmax_normalize(raw)
        rgb_u8 = (s["rgb"].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN)
        rgb_u8 = np.clip(rgb_u8 * 255.0, 0, 255).astype(np.uint8)
        stats = dn_tile_stats(raw)  # dihedral-invariant; fixed per tile

        # A. flip/rotate TTA variance
        preds = []
        for kind in ORIENTATIONS:
            dn_a = _transform(dn, kind)
            rgb_a = _transform(rgb_u8, kind)
            rgb_n = ((rgb_a.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            p = _forward(net, dn_a, rgb_n, device, stats)
            preds.append(_inverse_transform(p, kind))
        stack = np.stack(preds)  # [8,H,W]
        std = stack.std(axis=0)
        tta_stds.append(float(std.mean()))
        tta_p95.append(float(np.percentile(std, 95)))

        # B. window-overlap disagreement — 3x3 grid of 512 windows,
        # stride 256 -> 256 px overlap between adjacent windows
        offs = [0, 256, TILE - CROP]
        pmaps = {}
        for y0 in offs:
            for x0 in offs:
                w_dn = dn[y0:y0 + CROP, x0:x0 + CROP]
                w_rgb = rgb_u8[y0:y0 + CROP, x0:x0 + CROP]
                w_stats = dn_tile_stats(w_dn)
                rgb_n = ((w_rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
                pmaps[(y0, x0)] = _forward(net, w_dn, rgb_n, device, w_stats)
        ov = CROP - 256  # overlap width between adjacent windows (256)
        dis = []
        for y0 in offs:
            for x0 in offs[:-1]:
                left = pmaps[(y0, x0)][:, CROP - ov:]
                right = pmaps[(y0, x0 + 256)][:, :ov]
                dis.append(np.abs(left - right))
        for x0 in offs:
            for y0 in offs[:-1]:
                top = pmaps[(y0, x0)][CROP - ov:, :]
                bot = pmaps[(y0 + 256, x0)][:ov, :]
                dis.append(np.abs(top - bot))
        ov_dis.append(float(np.mean([d.mean() for d in dis])))
        print(f"  [{sid}] tta_std {tta_stds[-1]:.4f} m  p95 {tta_p95[-1]:.4f}  "
              f"overlap_disagree {ov_dis[-1]:.4f} m")
    return {
        "checkpoint": ckpt_path,
        "k_tiles": n,
        "tta_std_mean_m": float(np.mean(tta_stds)),
        "tta_std_p95_m": float(np.mean(tta_p95)),
        "window_overlap_disagreement_m": float(np.mean(ov_dis)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[i] TTA-consistency check: {args.checkpoint} (k={args.k}, device={device})")
    res = check_checkpoint(args.checkpoint, args.k, device)
    res["created"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(res, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, indent=2))
        print(f"-> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
