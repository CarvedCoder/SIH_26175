"""Standalone disaster assessment CLI command.

Usage:
    python model.py disaster --input path/to/post.tif [--pre path/to/pre.tif]

Runs:
    post image → building detection → polygonization → damage inference → artifacts

Prints:
    Buildings detected: N
    No damage: N
    Minor: N
    Major: N
    Destroyed: N
    Mode: post_only

Works independently of the web frontend.
"""

from __future__ import annotations

NAME = "disaster"
HELP = "run disaster assessment (building detection + damage) on a post-disaster image"

import argparse
import os
import sys
from pathlib import Path


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP)
    p.add_argument("--input", required=True, help="post-disaster image (GeoTIFF/PNG/JPG)")
    p.add_argument("--pre", default=None, help="optional pre-disaster image")
    p.add_argument("--output", "-o", default=None, help="output directory (default: <input>_disaster/)")
    p.add_argument("--building-model", default=None, help="path to building ONNX model")
    p.add_argument("--damage-model", default=None, help="path to damage ONNX model")
    p.add_argument("--device", default="auto", help="inference device (auto/cpu/cuda)")
    p.add_argument("--threshold", type=float, default=0.5, help="building detection threshold")
    p.add_argument("--tile-size", type=int, default=256, help="building model tile size")
    p.add_argument("--tile-stride", type=int, default=128, help="building model tile stride")
    p.add_argument("--batch-size", type=int, default=1, help="damage model batch size")
    p.add_argument(
        "--no-recover",
        action="store_true",
        help="skip recovering destroyed structures from the damage model "
        "(the building detector cannot see rubble)",
    )
    return p


build_parser = add_parser


def run(args: argparse.Namespace) -> int:
    import numpy as np

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        return 1

    # Determine output directory
    out_dir = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_disaster"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model paths
    building_model = args.building_model or os.environ.get("DW_BUILDING_MODEL_ONNX")
    damage_model = args.damage_model or os.environ.get("DW_DAMAGE_MODEL_ONNX")

    # Auto-discover from project root
    root = Path(__file__).resolve().parents[2]
    if not building_model:
        for fb in ("local_model.onnx", "models/building/model.onnx"):
            p = root / fb
            if p.exists():
                building_model = str(p)
                break
    if not damage_model:
        for fb in ("model.onnx", "models/damage/model.onnx"):
            p = root / fb
            if p.exists():
                damage_model = str(p)
                break

    if not building_model and not damage_model:
        print("[error] No ONNX models found. Set DW_BUILDING_MODEL_ONNX / DW_DAMAGE_MODEL_ONNX")
        return 1

    # Read image
    from depthwizard.inference import read_image
    from depthwizard.pipeline.scene_outputs import georef_state

    print(f"[disaster] Reading {input_path.name}...")
    rgb, profile = read_image(input_path)
    georef, crs, tf = georef_state(profile)
    print(f"[disaster] Image: {rgb.shape[1]}x{rgb.shape[0]}, georeferenced={georef}")

    # Read optional pre-disaster image
    pre_rgb = None
    if args.pre:
        pre_path = Path(args.pre)
        if not pre_path.exists():
            print(f"[error] Pre-disaster image not found: {pre_path}")
            return 1
        pre_rgb, _ = read_image(pre_path)
        print(f"[disaster] Pre-disaster image: {pre_rgb.shape[1]}x{pre_rgb.shape[0]}")

    # Run pipeline
    from depthwizard.disaster.pipeline import run_disaster_pipeline

    result = run_disaster_pipeline(
        rgb,
        out_dir,
        pre_rgb=pre_rgb,
        building_model_path=building_model,
        damage_model_path=damage_model,
        device=args.device,
        threshold=args.threshold,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        batch_size=args.batch_size,
        georeferenced=georef,
        crs_string=str(crs) if crs else None,
        transform=tf,
        recover_destroyed=not args.no_recover,
    )

    # Report
    print()
    print("=" * 50)
    print("  DISASTER ASSESSMENT RESULTS")
    print("=" * 50)
    print(f"  Buildings detected: {result['building_count']}")
    if result['damage_available']:
        for cls, count in result['damage_counts'].items():
            print(f"  {cls}: {count}")
    print(f"  Mode: {result['mode']}")
    print(f"  Elapsed: {result['elapsed_sec']:.1f}s")
    if result['errors']:
        print(f"  Errors: {result['errors']}")
    print(f"  Output: {out_dir}")
    print("=" * 50)

    return 0
