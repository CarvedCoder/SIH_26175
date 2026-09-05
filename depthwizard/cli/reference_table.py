"""Reference card: merge baseline + dummy results into ONE quotable table.

Why: after Phase 1 there are THREE reference predictors (constant zero /
train-mean / train-median, and the global affine). Every later phase must be
compared against the right floor for the right metric — and reports must
never mix numbers across different splits.json versions. This command reads
the three result JSONs, computes the per-split floors, checks whether the
affine actually beats them, and writes a frozen reference card used by:
  * the SIH technical document (quotable table),
  * training gates (success criteria, consumed by `evaluate`).

Usage:
  python model.py reference --config configs/phase1.yaml
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from depthwizard.cli.args import add_config_arg, load_config
from depthwizard.geo import dump_json, load_json

NAME = "reference"
HELP = "merge baseline + dummies into the frozen reference card / gates"

DOC_HINTS = {
    "missing": "run the producing command first "
               "(fit-baseline/eval-baseline for baseline, dummies for dummies)."
}


def _finite(x: float) -> float:
    """Map NaN -> None for strict-JSON output (NaN is not valid JSON)."""
    return None if (isinstance(x, float) and math.isnan(x)) else x


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase1.yaml")
    return p


def run(args) -> int:
    paths = load_config(args.config)["paths"]
    out_root = Path(paths["outputs_dir"])

    affine_json = out_root / "baseline" / "global_affine.json"
    eval_json = out_root / "baseline" / "eval_results.json"
    dummy_json = out_root / "dummy_baselines" / "dummy_baselines.json"
    for p in (affine_json, eval_json, dummy_json):
        if not p.exists():
            print(f"[error] missing {p} — {DOC_HINTS['missing']}")
            return 1

    bl = load_json(affine_json)
    ev = load_json(eval_json)
    du = load_json(dummy_json)

    # ---- collect per-split numbers --------------------------------------
    rows = []            # (predictor, split, mae, rmse, r)
    for split in ("train", "val", "test"):
        p = ev["splits"][split]["pooled"]
        rows.append(("global_affine", split, p["mae"], p["rmse"], p["pearson_r"]))
    for name in du["results"]:
        for split in ("train", "val", "test"):
            p = du["results"][name][split]["pooled"]
            rows.append((name, split, p["mae"], p["rmse"], p["pearson_r"]))

    floors = {}
    for split in ("train", "val", "test"):
        dummies = [r for r in rows if r[0] != "global_affine" and r[1] == split]
        best_mae = min(dummies, key=lambda r: r[2])
        best_rmse = min(dummies, key=lambda r: r[3])
        affine = next(r for r in rows if r[0] == "global_affine" and r[1] == split)
        floors[split] = {
            "mae_predictor": best_mae[0], "mae": best_mae[2],
            "rmse_predictor": best_rmse[0], "rmse": best_rmse[3],
            "affine_mae": affine[2], "affine_rmse": affine[3],
            "affine_beats_floor_mae": affine[2] < best_mae[2],
            "affine_beats_floor_rmse": affine[3] < best_rmse[3],
            # Phase-2 gates (project decisions, derived from measured floors):
            "gate_must_mae": best_mae[2],            # any improvement counts
            "gate_target_mae": round(0.8 * best_mae[2], 3),   # decisive win
            "gate_must_rmse": best_rmse[3],
            "gate_target_rmse": round(0.9 * best_rmse[3], 3),
        }

    # ---- markdown card ---------------------------------------------------
    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

    lines = [
        "# Reference card — Phase 1 (FROZEN)", "",
        f"- splits.json: `{paths['splits_json']}`",
        f"- affine: `H = {bl['a']:.6f} · Dn + {bl['b']:.6f}` (lstsq, train-only fit)",
        f"- constants (train-only fit): mean = {du['constants']['train_mean']:.3f} m, "
        f"median = {du['constants']['train_median']:.3f} m",
        "",
        "| predictor | split | MAE (m) | RMSE (m) | Pearson r |",
        "|---|---|---|---|---|",
    ]
    for name, split, mae, rmse, r in rows:
        star = ""
        f = floors[split]
        if name == "global_affine":
            pass
        elif name == f["mae_predictor"] and f["mae"] == mae:
            star = " ← MAE floor"
        elif name == f["rmse_predictor"] and f["rmse"] == rmse:
            star = " ← RMSE floor"
        lines.append(f"| {name} | {split} | {fmt(mae)} | {fmt(rmse)} | {fmt(r)} |{star}")

    lines += ["", "## Affine vs the floor (out-of-sample honesty check)", "",
              "| split | affine MAE vs floor | affine RMSE vs floor |", "|---|---|---|"]
    for split in ("train", "val", "test"):
        f = floors[split]
        lines.append(
            f"| {split} | {f['affine_mae']:.3f} vs {f['mae']:.3f} ({f['mae_predictor']}) "
            f"{'OK' if f['affine_beats_floor_mae'] else '**LOSES**'} "
            f"| {f['affine_rmse']:.3f} vs {f['rmse']:.3f} ({f['rmse_predictor']}) "
            f"{'OK' if f['affine_beats_floor_rmse'] else '**LOSES**'} |")

    f = floors["val"]
    lines += [
        "",
        "## Phase-2 gates (binding, derived from measured floors)", "",
        f"- MUST: val MAE < {f['gate_must_mae']:.3f} m AND val RMSE < {f['gate_must_rmse']:.3f} m",
        f"- TARGET (decisive): val MAE <= {f['gate_target_mae']:.3f} m, val RMSE <= {f['gate_target_rmse']:.3f} m",
        f"- Test-split reference at gate time: MAE {floors['test']['mae']:.3f}, RMSE {floors['test']['rmse']:.3f}",
        "",
        "_All numbers produced from the SAME splits.json and the SAME clean_agl"
        "(clamp>=0) convention. Never mix with numbers from other split versions._",
    ]
    out_dir = out_root / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_card.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "kind": "phase1_reference_card",
        "splits_json": str(paths["splits_json"]),
        "affine": {"a": bl["a"], "b": bl["b"]},
        "constants": du["constants"],
        "rows": [
            {"predictor": n, "split": s, "mae": _finite(m), "rmse": _finite(r),
             "pearson_r": _finite(rr)}
            for n, s, m, r, rr in rows],
        "floors_and_gates": floors,
    }
    dump_json(payload, out_dir / "reference_card.json")
    print(f"-> {out_dir / 'reference_card.md'}")
    print("\n".join(lines[-14:]))
    return 0
