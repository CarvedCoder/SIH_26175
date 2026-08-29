#!/usr/bin/env python3
"""Generate a SYNTHETIC mini-dataset shaped exactly like DFC2019 Track-1.

Purpose: verify the ENTIRE Phase-1 pipeline (inspect -> splits -> depth-cache
-> fit -> eval) in ~30 seconds on any laptop, without downloading 40 GB.
Buildings are drawn as rectangles; AGL = building height field + gentle
terrain; CLS = arbitrary ids mirroring the ids you observed {2,5,6,9,65}.
These fake tiles are for pipeline testing ONLY — never train on them.

Usage:
  python tools/make_fake_dataset.py --root fake_data --n 8 --size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def make_tile(seed: int, size: int):
    rng = np.random.default_rng(seed)
    # gentle terrain + buildings
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    terrain = 2.0 + 1.5 * np.sin(xx / size * np.pi) * np.cos(yy / size * np.pi)

    agl = terrain.copy()
    cls = np.full((size, size), 2, dtype=np.uint8)        # 2 = "ground-ish" here
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = 110 + 30 * np.sin(xx / 9.0).astype(np.uint8) % 40
    rgb[..., 1] = 120 + 25 * np.cos(yy / 11.0).astype(np.uint8) % 30
    rgb[..., 2] = 90

    for _ in range(rng.integers(4, 9)):
        h = int(rng.integers(size // 10, size // 3))
        w = int(rng.integers(size // 10, size // 3))
        y0 = int(rng.integers(0, size - h)); x0 = int(rng.integers(0, size - w))
        height = float(rng.uniform(4.0, 25.0))
        agl[y0:y0 + h, x0:x0 + w] = terrain[y0:y0 + h, x0:x0 + w] + height
        cls[y0:y0 + h, x0:x0 + w] = 6                     # "building-ish" here
        rgb[y0:y0 + h, x0:x0 + w] = (
            np.array([180, 160, 140]) + rng.integers(-25, 25, 3)).astype(np.uint8)

    # a water strip and a road strip with fake ids
    cls[:, : size // 12] = 9
    cls[size // 2, size // 3: 2 * size // 3] = 5
    cls[0, 0] = 65
    agl += rng.normal(0, 0.05, agl.shape).astype(np.float32)   # LiDAR-ish noise
    agl[:2, 0] -= 0.3                                          # tiny negatives
    return rgb, agl.astype(np.float32), cls


def write_tif(path: Path, arr: np.ndarray, count: int, dtype: str):
    """arr: [H,W] with count=1, or [H,W,C] with count=C (rasterio wants
    [bands,H,W])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if count == 1:
        data = arr[None]
    else:
        assert arr.ndim == 3 and arr.shape[2] == count, \
            f"expected [H,W,{count}], got {arr.shape}"
        data = arr.transpose(2, 0, 1)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=count, dtype=dtype) as dst:
        dst.write(data.astype(dtype))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("fake_data"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--rows", type=int, default=2, help="grid rows (block split needs >=3)")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    rgb_dir = args.root / "rgb_data" / "Train-Track1-RGB" / "Track1-RGB"
    truth_dir = args.root / "rgb_data_truth" / "Train-Track1-Truth" / "Track1-Truth"

    k = 0
    for r in range(1, args.rows + 1):
        for c in range(1, args.cols + 1):
            k += 1
            if k > args.n:
                break
            stem = f"JAX_{r:03d}_{c:03d}"
            rgb, agl, cls = make_tile(seed=100 + k, size=args.size)
            write_tif(rgb_dir / f"{stem}_RGB.tif", rgb, 3, "uint8")
            write_tif(truth_dir / f"{stem}_AGL.tif", agl, 1, "float32")
            write_tif(truth_dir / f"{stem}_CLS.tif", cls, 1, "uint8")
    print(f"[done] wrote {min(k, args.n)} tiles under {args.root}")
    print(f"       rgb:   {rgb_dir}")
    print(f"       truth: {truth_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
