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
  * Cache   : namespaced by DATASET since the GAMUS integration:
              <out_dir>/<model_tag>/{dataset}/{sample_id}.npy
              (legacy flat <tag>/{stem}.npy entries for dfc2019 are still
              recognized and skipped as already-computed — old caches keep
              working, nothing is recomputed or overwritten without
              --overwrite).
  * Manifest: manifest.json records model id, dataset, library versions,
              input size, and per-tile raw min/max. A different model
              writes to a different subdir (required — never mix depths
              across backbones); a different dataset writes to a different
              namespace (never mix depths across datasets).

Datasets:
  dfc2019 (default): --rgb-dir <Track1-RGB dir> (GeoTIFF triples, as before)
  gamus: raw HDF5 tiles — default --gamus-source hf (lazy per-file
      hf_hub_download), or --gamus-source local --gamus-local-root <dir>
      for an offline/pre-downloaded subset. Official splits verbatim;
      --gamus-splits selects which (default train val test).

Usage:
  python model.py depth \
      --rgb-dir rgb_data/Train-Track1-RGB/Track1-RGB \
      --out-dir outputs/depth_cache \
      --model depth-anything/Depth-Anything-V2-Base-hf \
      --device auto

  python model.py depth --dataset gamus --gamus-source hf \
      --out-dir outputs/depth_cache --limit 64

First run downloads the checkpoint from HuggingFace (ViT-B ~ 390 MB).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np

from depthwizard.backbone import INPUT_SIZE, DepthAnythingBackbone
from depthwizard.geo import (RGB_SUFFIXES, _index_dir, depth_npy_candidates,
                             dump_json, read_raster)

NAME = "depth"
HELP = "precompute the Depth-Anything-V2 raw depth cache (.npy per tile)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("dfc2019", "gamus"),
                   default="dfc2019",
                   help="which dataset to precompute depth for (namespaced "
                        "cache layout per dataset)")
    p.add_argument("--rgb-dir", type=Path, default=None,
                   help="DFC2019 RGB directory (required for --dataset "
                        "dfc2019)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/depth_cache"))
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Base-hf")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--fp16", action="store_true", help="CUDA half precision")
    p.add_argument("--limit", type=int, default=0, help="process only first N tiles")
    p.add_argument("--overwrite", action="store_true")
    # ---- GAMUS source options (ignored for dfc2019) ----
    p.add_argument("--gamus-source", choices=("hf", "local"), default="hf",
                   help="GAMUS backend: hf = raw HDF5 + lazy per-file "
                        "download (primary); local = offline/pre-downloaded "
                        "subset directory")
    p.add_argument("--gamus-local-root", type=Path, default=None,
                   help="GAMUS local directory (raw official layout) when "
                        "--gamus-source local")
    p.add_argument("--gamus-manifest", type=Path, default=None,
                   help="reusable GAMUS split manifest (offline index)")
    p.add_argument("--gamus-save-manifest", type=Path, default=None,
                   help="write the GAMUS split manifest after listing")
    p.add_argument("--gamus-hf-cache", type=Path, default=None,
                   help="local cache root for lazy HF downloads")
    p.add_argument("--gamus-splits", nargs="+",
                   default=["train", "val", "test"],
                   help="GAMUS splits to precompute (official, verbatim)")
    return p


def _dfc2019_entries(args) -> List[Tuple[str, object]]:
    """(sample_id, rgb_path) entries for DFC2019 — legacy discovery order."""
    if args.rgb_dir is None:
        raise SystemExit(
            "[error] --dataset dfc2019 requires --rgb-dir "
            "(e.g. rgb_data/Train-Track1-RGB/Track1-RGB)")
    index = _index_dir(args.rgb_dir, RGB_SUFFIXES)
    return [(stem, index[stem]) for stem in sorted(index)]


def _gamus_entries(args) -> List[Tuple[str, object]]:
    """(sample_id, GAMUSSample) entries for GAMUS — official splits."""
    from depthwizard.datasets.gamus import GAMUSConfig, list_gamus_samples
    cfg = GAMUSConfig(
        source=args.gamus_source,
        local_root=args.gamus_local_root,
        manifest=args.gamus_manifest,
        save_manifest=args.gamus_save_manifest,
        hf_cache_dir=args.gamus_hf_cache,
        limit=args.limit,                    # deterministic per-split cap
        splits=tuple(args.gamus_splits))
    per_split, problems = list_gamus_samples(cfg)
    entries: List[Tuple[str, object]] = []
    if problems:
        print("[gamus] discovery problems:")
        for pr in problems:
            print("  -", pr)

    for split in cfg.splits:
        samples = per_split.get(split, [])

        if args.limit > 0:
            samples = samples[:args.limit]

        for s in samples:
            entries.append((s.sample_id, s))
    if not entries:
        raise SystemExit(
            "[error] GAMUS discovery produced no samples — check "
            "--gamus-source/--gamus-local-root/--gamus-manifest.")
    return entries




def run(args) -> int:
    import torch

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    if args.dataset == "dfc2019":
        entries = _dfc2019_entries(args)
    else:
        entries = _gamus_entries(args)
    print(f"[i] dataset={args.dataset}  {len(entries)} tiles | "
          f"model={args.model} | device={device}")

    # Different model => different cache subdir; different dataset =>
    # different namespace. Never mix backbones, never mix datasets.
    safe_tag = args.model.split("/")[-1].lower().replace("-", "_")
    model_dir = args.out_dir / safe_tag
    cache_dir = model_dir / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    backbone = DepthAnythingBackbone(model_id=args.model, device=str(device),
                                     fp16=args.fp16).load()
    manifest = {
        "model": args.model, "dataset": args.dataset,
        "input_size": INPUT_SIZE,
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

    def _rgb_of(sample_id: str, entry: object) -> np.ndarray:
        if args.dataset == "dfc2019":
            rgb_path = cast(Path, entry)
            rgb, _prof = read_raster(rgb_path)
            return rgb[:3].transpose(1, 2, 0)          # [H,W,3] uint8
        from depthwizard.datasets.gamus import (GAMUSSample, read_gamus_h5,
                                                _resolve_sample_files)
        s = cast(GAMUSSample, entry)
        # resolve the rgb .h5 through the sample's own source config
        return read_gamus_h5(_resolve_sample_files(_gamus_cfg_for(args),
                                                   s)["rgb"])

    skipped = 0
    for i, (sample_id, entry) in enumerate(entries, 1):
        out_npy = cache_dir / f"{sample_id}.npy"
        existing = [p for p in depth_npy_candidates(model_dir, args.dataset,
                                                    sample_id) if p.exists()]
        if existing and not args.overwrite:
            skipped += 1
            continue
        rgb = _rgb_of(sample_id, entry)
        h, w = rgb.shape[:2]

        pred = backbone.raw_depth(rgb).astype(np.float32)

        np.save(out_npy, pred)
        manifest["tiles"][sample_id] = {
            "file": out_npy.name, "shape": list(pred.shape),
            "dtype": "float32",
            "raw_min": float(pred.min()), "raw_max": float(pred.max()),
        }
        if i % 10 == 0 or i == len(entries):
            print(f"  [{i}/{len(entries)}] {sample_id} raw range "
                  f"[{pred.min():.3f}, {pred.max():.3f}]")
        if i % 20 == 0 or i == len(entries):
            dump_json(manifest, cache_dir / "manifest.json")   # crash-safe

    dump_json(manifest, cache_dir / "manifest.json")
    print(f"[done] new: {len(manifest['tiles'])}, skipped(existing): {skipped}")
    print(f"       cache: {cache_dir}  manifest: {cache_dir / 'manifest.json'}")
    print("NOTE: these are RAW relative values — do NOT read them as metres.")
    return 0


def _gamus_cfg_for(args):
    from depthwizard.datasets.gamus import GAMUSConfig
    return GAMUSConfig(
        source=args.gamus_source,
        local_root=args.gamus_local_root,
        manifest=args.gamus_manifest,
        save_manifest=args.gamus_save_manifest,
        hf_cache_dir=args.gamus_hf_cache,
        splits=tuple(args.gamus_splits))
