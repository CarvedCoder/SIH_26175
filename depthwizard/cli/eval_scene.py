"""Evaluate a PREDICTED DSM raster against a truth AGL raster.  [NOT citable]

This is the real-data acceptance tool (single scenes, GeoTIFF in / GeoTIFF
out), NOT the citable benchmark: it inherits the exact cleaning convention
of the certified `evaluate` command —
    truth = clean_agl(<stem>_AGL.tif), mask = valid_target_mask(truth)
(the gt_check false-alarm chain in the worklog is why) — but it runs on
hand-picked scenes, so its numbers are diagnostics, never leaderboard rows.

AGL evaluation only: the comparison target is a truth AGL raster. This is
NOT absolute-DSM evaluation — see `eval-absolute` for predicted absolute
DSM vs reference absolute DSM.

Usage:
  python model.py eval-scene --pred outputs/infer/JAX_004_014/dsm.tif \
      --truth rgb_data_truth/Train-Track1-Truth/Track1-Truth/JAX_004_014_AGL.tif
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.cli.args import add_device_arg
from depthwizard.geo import dump_json, read_raster
from depthwizard.metrics import height_metrics
from depthwizard.normalize import clean_agl, valid_target_mask

NAME = "eval-scene"
HELP = "predicted DSM vs truth AGL for one scene (diagnostic, NOT citable)"


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pred", required=True, help="predicted DSM (.tif single band, or .npy [H,W])"
    )
    p.add_argument(
        "--truth",
        required=True,
        help="truth AGL GeoTIFF (clean_agl convention applied)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report json path (default: alongside --pred)",
    )
    add_device_arg(p)  # unused; kept out of muscle-memory typos
    return p


def run(args) -> int:
    pred_path = Path(args.pred)
    truth_path = Path(args.truth)

    if pred_path.suffix == ".npy":
        pred = np.load(pred_path)
    else:
        arr, _prof = read_raster(pred_path)
        pred = arr[0] if arr.ndim == 3 else arr
    pred = np.asarray(pred, dtype=np.float64)

    tarr, _tprof = read_raster(truth_path)
    truth_raw = tarr[0] if tarr.ndim == 3 else tarr
    if truth_raw.shape != pred.shape:
        raise ValueError(
            f"grid mismatch pred{pred.shape} vs truth{truth_raw.shape} — "
            "resample the prediction onto the truth grid first; do NOT "
            "broadcast."
        )
    truth = clean_agl(truth_raw)  # same convention as `evaluate`
    mask = valid_target_mask(truth_raw)

    m = height_metrics(pred, truth, mask)
    sat = {
        "pred_zero_frac": float((pred <= 1e-6).mean()),
        "truth_le_0.5m_frac": float((truth[mask] <= 0.5).mean()),
    }
    print(f"[scene] {pred_path.name} vs {truth_path.name}")
    print(
        f"[gt]    valid {mask.mean():.1%}  min {np.nanmin(truth_raw):.2f}  "
        f"mean {np.nanmean(truth_raw):.2f}  median {np.nanmedian(truth_raw):.2f}  "
        f"max {np.nanmax(truth_raw):.2f}"
    )
    print(
        f"[err]   MAE {m['mae']:.3f}  medae {m['medae']:.3f}  "
        f"bias {m['bias']:.3f}  RMSE {m['rmse']:.3f}  r {m['pearson_r']:.3f}"
    )
    print(
        f"[sat]   pred zero-frac {sat['pred_zero_frac']:.3f}   "
        f"gt AGL<=0.5m frac {sat['truth_le_0.5m_frac']:.3f}"
    )
    print("[note]  diagnostic only — citable numbers come from `evaluate`.")

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "scene_diagnostic_eval",
        "pred": str(pred_path),
        "truth": str(truth_path),
        "metrics": m,
        "saturation": sat,
        "citable": False,
    }
    out = args.out or (pred_path.parent / f"{pred_path.stem}_scene_eval.json")
    dump_json(report, out)
    print(f"-> {out}")
    return 0
