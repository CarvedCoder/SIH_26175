"""Precompute Depth Anything V2 relative depth for all RGB tiles.

Design decisions (fixed here so every later stage inherits them):
  * Model   : depth-anything/Depth-Anything-V2-Base-hf by default
              (--model can switch S/B/L variants).
  * Input   : 518x518 (ViT patch 14 * 37), bicubic resize, ImageNet norm —
              the same numbers the HF processor uses, done manually so the
              pipeline does not depend on processor API drift.
  * Output  : ONE raw float32 .npy per tile at NATIVE resolution
              (predicted_depth bilinearly upsampled back to HxW).
              RAW means: exactly what the model head emitted (post-ReLU,
              no sigmoid, no min-max). Normalization to Dn happens ONLY in
              depthwizard.normalize.minmax_normalize at load time, so
              training/eval/inference can never disagree.
  * Manifest: manifest.json records model id, library versions, input size,
              and per-tile raw min/max. A different model writes to a
              different subdir (required — never mix depths across backbones).

Usage:
  python main.py depth \
      --rgb-dir rgb_data/Train-Track1-RGB/Track1-RGB \
      --out-dir outputs/depth_cache \
      --model depth-anything/Depth-Anything-V2-Base-hf \
      --device auto

First run downloads the checkpoint from HuggingFace (ViT-B ~ 390 MB).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.backbone import INPUT_SIZE, DepthAnythingBackbone
from depthwizard.geo import RGB_SUFFIXES, _index_dir, dump_json, read_raster

NAME = "depth"
HELP = "precompute the Depth-Anything-V2 raw depth cache (.npy per tile)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/depth_cache"))
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Base-hf")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--fp16", action="store_true", help="CUDA half precision")
    p.add_argument("--limit", type=int, default=0, help="process only first N tiles")
    p.add_argument("--overwrite", action="store_true")
    return p


def run(args) -> int:
    import torch

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    index = _index_dir(args.rgb_dir, RGB_SUFFIXES)
    stems = sorted(index)
    if args.limit:
        stems = stems[: args.limit]
    print(f"[i] {len(stems)} RGB tiles | model={args.model} | device={device}")

    # Different model => different cache subdir; never mix backbones.
    safe_tag = args.model.split("/")[-1].lower().replace("-", "_")
    cache_dir = args.out_dir / safe_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    backbone = DepthAnythingBackbone(model_id=args.model, device=str(device),
                                     fp16=args.fp16).load()
    manifest = {
        "model": args.model, "input_size": INPUT_SIZE,
        "resize": "bicubic", "resample_back": "bilinear",
        "created": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "raw_definition": "predicted_depth at native HxW (bilinear upsample "
                          "from 518). No normalization applied on disk.",
        "tiles": {},
    }
    if (cache_dir / "manifest.json").exists() and not args.overwrite:
        from depthwizard.geo import load_json
        old = load_json(cache_dir / "manifest.json")
        if old.get("model") != args.model:
            print(f"[warn] existing cache was built with model '{old.get('model')}' "
                  f"but --overwrite was not passed and subdir is shared; continuing "
                  f"(stems are keyed per tile, mismatched entries will be refreshed).")

    skipped = 0
    for i, stem in enumerate(stems, 1):
        out_npy = cache_dir / f"{stem}.npy"
        if out_npy.exists() and not args.overwrite:
            skipped += 1
            continue
        rgb, _prof = read_raster(index[stem])
        rgb = rgb[:3].transpose(1, 2, 0)                       # [H,W,3] uint8
        h, w = rgb.shape[:2]

        pred = backbone.raw_depth(rgb).astype(np.float32)

        np.save(out_npy, pred)
        manifest["tiles"][stem] = {
            "file": out_npy.name, "shape": list(pred.shape),
            "dtype": "float32",
            "raw_min": float(pred.min()), "raw_max": float(pred.max()),
        }
        if i % 10 == 0 or i == len(stems):
            print(f"  [{i}/{len(stems)}] {stem} raw range "
                  f"[{pred.min():.3f}, {pred.max():.3f}]")
        if i % 20 == 0 or i == len(stems):
            dump_json(manifest, cache_dir / "manifest.json")   # crash-safe

    dump_json(manifest, cache_dir / "manifest.json")
    print(f"[done] new: {len(manifest['tiles'])}, skipped(existing): {skipped}")
    print(f"       cache: {cache_dir}  manifest: {cache_dir / 'manifest.json'}")
    print("NOTE: these are RAW relative values — do NOT read them as metres.")
    return 0
