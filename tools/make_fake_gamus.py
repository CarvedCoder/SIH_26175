#!/usr/bin/env python3
"""Generate a SYNTHETIC mini-dataset shaped exactly like raw GAMUS HDF5.

Purpose: verify the GAMUS ingestion path (adapter, splits manifest, depth
precompute, training smoke runs) NETWORK-FREE in seconds, matching the
official raw layout verified from the earthflow/GAMUS repo:

    <root>/images/{split}/{ID}_RGB.h5    key "image" (H,W,3) uint8 HWC
    <root>/heights/{split}/{ID}_AGL.h5   key "image" (H,W)   float32
    <root>/classes/{split}/{ID}_CLS.h5   key "image" (H,W)   uint8 OR float32
                                          (the REAL release mixes both —
                                           the fixture reproduces that so
                                           the adapter's dtype handling is
                                           exercised honestly)

IDs use the real stem pattern (city_row_col, e.g. DC_01_25) with cities
DC / NYC / PHL. CLS values 0-6 with the VERIFIED GAMUS legend proportions
(0=others, 1=ground, 2=low vegetation, 3=buildings, 4=water, 5=road,
6=tree) — these are for pipeline testing ONLY, never for real training.

Usage:
  python tools/make_fake_gamus.py --root fake_gamus --per-split 3 --size 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
CITIES = ("DC", "NYC", "PHL")
_H5_KEY = "image"


def make_gamus_tile(seed: int, size: int, cls_dtype: str):
    """(rgb uint8 [H,W,3], agl float32 [H,W], cls [H,W] as cls_dtype)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    agl = np.abs(3.0 * np.sin(xx / size * np.pi)) + \
        2.0 * np.abs(np.cos(yy / size * np.pi)) \
        + rng.normal(0, 0.1, (size, size)).astype(np.float32)

    cls = np.full((size, size), 1, dtype=np.int64)          # ground
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = 100 + (20 * np.sin(xx / 7.0)).astype(np.uint8)
    rgb[..., 1] = 115 + (15 * np.cos(yy / 9.0)).astype(np.uint8)
    rgb[..., 2] = 90

    for _ in range(rng.integers(3, 7)):                     # buildings
        h = int(rng.integers(size // 10, size // 4))
        w = int(rng.integers(size // 10, size // 4))
        y0 = int(rng.integers(0, size - h + 1))
        x0 = int(rng.integers(0, size - w + 1))
        agl[y0:y0 + h, x0:x0 + w] += float(rng.uniform(6.0, 30.0))
        cls[y0:y0 + h, x0:x0 + w] = 3                       # buildings
        rgb[y0:y0 + h, x0:x0 + w] = np.array([170, 150, 135], np.uint8)

    cls[:, : size // 10] = 4                                # water strip
    cls[size // 2, :] = 5                                   # road strip
    cls[0, 0] = 0                                           # others
    cls[1, 1] = 6                                           # tree
    cls[2, 2] = 2                                           # low vegetation

    if cls_dtype == "uint8":
        cls_out = cls.astype(np.uint8)
    else:
        cls_out = cls.astype(np.float32)                    # real-release
        # inconsistency: some files are float32 with integral values
    return rgb, agl.astype(np.float32), cls_out


def write_h5(path: Path, arr: np.ndarray) -> None:
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(_H5_KEY, data=arr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("fake_gamus"))
    ap.add_argument("--per-split", type=int, default=3,
                    help="tiles per split (deterministic ids)")
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    k = 0
    written = 0
    for split in SPLITS:
        for i in range(1, args.per_split + 1):
            k += 1
            city = CITIES[k % len(CITIES)]
            sid = f"{city}_{i:02d}_{k:02d}"
            # dtype inconsistency like the real release: every 3rd CLS file
            # is float32, the rest uint8.
            cls_dtype = "float32" if k % 3 == 0 else "uint8"
            rgb, agl, cls = make_gamus_tile(seed=700 + k, size=args.size,
                                            cls_dtype=cls_dtype)
            write_h5(args.root / "images" / split / f"{sid}_RGB.h5", rgb)
            write_h5(args.root / "heights" / split / f"{sid}_AGL.h5", agl)
            write_h5(args.root / "classes" / split / f"{sid}_CLS.h5", cls)
            written += 1
    print(f"[done] wrote {written} GAMUS-shaped tiles under {args.root}")
    print(f"       images/  heights/  classes/  (splits: {SPLITS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
