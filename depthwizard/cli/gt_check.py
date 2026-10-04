"""GT cross-check for one predicted scene against its AGL truth tile.

Formalized from the ad-hoc gt_check notebook step. Protocol (frozen after
the false-alarm chain in the worklog):
  * compare against <stem>_AGL.tif — NOT an RGB band (wrong-file incident)
  * apply clean_agl + valid_target_mask exactly like `evaluate`
    (raw-GT incident: the AGL min is ~-0.37 m before clamping)

This prints the same error/saturation lines the smoke test used; it is a
diagnostic, never a citable number.

This is AGL evaluation (pred height field vs truth AGL, GAMUS protocol).
It is NOT absolute-DSM evaluation — that lives in `eval-absolute` and
requires a reference ABSOLUTE DSM raster.

Usage:
  python model.py gt-check --pred outputs/infer/JAX_004_014/dsm.npy \
      --truth rgb_data_truth/.../JAX_004_014_AGL.tif
"""

from __future__ import annotations

import argparse

import numpy as np

from depthwizard.geo import read_raster
from depthwizard.metrics import height_metrics
from depthwizard.normalize import clean_agl, valid_target_mask

NAME = "gt-check"
HELP = "cross-check one predicted scene vs its AGL truth (frozen protocol)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pred", required=True, help="predicted DSM (.npy [H,W] or single-band .tif)"
    )
    p.add_argument("--truth", required=True, help="truth <stem>_AGL.tif")
    return p


def run(args) -> int:
    pred_path = args.pred
    if pred_path.endswith(".npy"):
        dsm = np.load(pred_path)
    else:
        arr, _ = read_raster(pred_path)
        dsm = arr[0] if arr.ndim == 3 else arr
    dsm = np.asarray(dsm, dtype=np.float32)

    with __import__("rasterio").open(args.truth) as ds:
        gt_raw = ds.read(1).astype(np.float32)
        print(f"[raw] dtype={ds.dtypes[0]}  nodata={ds.nodata}")

    gt = clean_agl(gt_raw)  # frozen protocol
    m = valid_target_mask(gt_raw)

    if dsm.shape != gt.shape:
        raise ValueError(f"pred {dsm.shape} vs truth {gt.shape} grid mismatch")

    print(
        f"[gt]   valid {m.mean():.1%}  min {np.nanmin(gt_raw):.2f}  "
        f"mean {np.nanmean(gt_raw):.2f}  median {np.nanmedian(gt_raw):.2f}  "
        f"max {np.nanmax(gt_raw):.2f}"
    )
    err = (dsm - gt)[m]
    print(
        f"[err]  MAE {np.abs(err).mean():.3f}  medae {np.median(np.abs(err)):.3f}  "
        f"bias {err.mean():.3f}  RMSE {np.sqrt((err**2).mean()):.3f}"
    )
    met = height_metrics(dsm, gt, m)
    print(f"[r]    pearson {met['pearson_r']:.3f}")
    print(
        f"[sat]  pred zero-frac {(dsm <= 1e-6).mean():.3f}   "
        f"gt AGL<=0.5m frac {(gt[m] <= 0.5).mean():.3f}"
    )
    return 0
