"""Dummy baselines: the honest floor every real model must beat.

Why: a global affine fit optimizes MSE and is dragged upward by tall
structures. A *constant* predictor (train-set mean or median AGL) can be
embarrassingly competitive on MAE in height-skewed data (most pixels are
near ground). If the calibration net cannot beat these numbers, it has
learned nothing — so we measure them ONCE and freeze them in the report.

Predictors (constants fitted on TRAIN split only, like the affine):
  zero            H = 0                       (ground prior)
  train_mean      H = mean(AGL_train)
  train_median    H = median(AGL_train)

Usage:
  python model.py dummies --config configs/phase1.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.cli.args import add_config_arg, load_config
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.geo import dump_json, read_tile
from depthwizard.metrics import height_metrics
from depthwizard.normalize import clean_agl, valid_target_mask
from depthwizard.streaming import PooledStats

NAME = "dummies"
HELP = "constant-predictor floors (zero / train-mean / train-median)"

STRIDE = 4   # subsampling for the constant fit (exactness is irrelevant here)


def sweep_train_constants(ds_train, cfg_stride: int = STRIDE):
    """One streaming pass over train AGL -> mean / median (subsampled, masked)."""
    total_sum = 0.0
    total_n = 0
    median_chunks = []

    for t in ds_train.tiles:
        data = read_tile(t)
        agl = clean_agl(data["agl"])[::cfg_stride, ::cfg_stride]
        m = valid_target_mask(agl)
        if m.any():
            vals = agl[m].astype(np.float32, copy=False)
            total_sum += float(vals.astype(np.float64).sum())
            total_n += int(vals.size)
            median_chunks.append(vals)
        del data, agl, m

    if total_n == 0:
        raise RuntimeError("No valid AGL pixels found in TRAIN split.")

    mean_value = total_sum / total_n
    median_values = np.concatenate(median_chunks)     # 4x-subsampled only
    median_value = float(np.median(median_values))
    return float(mean_value), median_value


def eval_constant(ds, c: float, name: str):
    """Evaluate a constant predictor without retaining an entire split."""
    pooled_stats = PooledStats()
    per_tile = []
    for t in ds.tiles:
        data = read_tile(t)
        agl = clean_agl(data["agl"])
        m = valid_target_mask(agl)
        pred = np.full_like(agl, c, dtype=np.float32)
        per_tile.append(height_metrics(pred, agl, m))
        pooled_stats.update(pred[m], agl[m])
        del data, agl, m, pred

    pooled = pooled_stats.to_metrics()
    pooled["n_tiles"] = len(ds.tiles)
    print(f"  {name:12s}  MAE {pooled['mae']:.3f}  RMSE {pooled['rmse']:.3f}  "
          f"bias {pooled['bias']:+.3f}  r {pooled['pearson_r']:.3f}")
    return pooled, per_tile


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase1.yaml")
    return p


def run(args) -> int:
    full_cfg = load_config(args.config)
    paths = full_cfg["paths"]
    clamp = full_cfg["dataset"]["clamp_agl_min"]

    datasets = discover_and_split(
        DFC2019Config(rgb_dir=Path(paths["rgb_dir"]),
                      truth_dir=Path(paths["truth_dir"]),
                      load_depth=False, clamp_agl_min=clamp),
        Path(paths["splits_json"]))

    print("[1/2] fitting constants on TRAIN only ...")
    mean_c, med_c = sweep_train_constants(datasets["train"])
    constants = {"zero": 0.0, "train_mean": mean_c, "train_median": med_c}
    print(f"      train_mean={mean_c:.3f} m  train_median={med_c:.3f} m")

    print("[2/2] evaluating constants on every split ...")
    results = {}
    for name, c in constants.items():
        results[name] = {}
        for split in ("train", "val", "test"):
            print(f"  [{name} | {split}]")
            pooled, _per_tile = eval_constant(datasets[split], c, name)
            results[name][split] = {"pooled": pooled}

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "dummy_constant_baselines",
        "constants": constants,
        "splits_json": str(paths["splits_json"]),
        "results": results,
        "note": "Reference floor. A model that cannot beat train_median MAE "
                "has learned nothing useful.",
    }
    out_dir = Path(paths["outputs_dir"]) / "dummy_baselines"
    dump_json(payload, out_dir / "dummy_baselines.json")

    lines = [
        "# Dummy baselines (constant predictors) — the floor", "",
        (f"- constants (fitted on train): zero = 0.0, mean = {mean_c:.3f} m, "
         f"median = {med_c:.3f} m"),
        "",
        "| predictor | split | MAE (m) | RMSE (m) | bias (m) | r |",
    ]
    for name in constants:
        for split in ("train", "val", "test"):
            m = results[name][split]["pooled"]
            lines.append(f"| {name} | {split} | {m['mae']:.3f} | {m['rmse']:.3f} "
                         f"| {m['bias']:+.3f} | {m['pearson_r']:.3f} |")
    lines += ["", "_These require no depth input at all. Quote them next to the "
              "global-affine baseline; any learned model must beat both._"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dummy_baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_dir / 'dummy_baselines.md'}")
    return 0
