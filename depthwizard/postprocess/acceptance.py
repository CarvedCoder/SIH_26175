"""Automatic acceptance criteria for post-processing candidates.

A refinement variant is compared against the RAW baseline (V0) on the SAME
validation tiles and ACCEPTED only when every rule below holds (tolerances
are configuration, never hard-coded):

    mae            improved, or worse by <= mae_tol_m
    rmse           improved, or worse by <= rmse_tol_m
    building_mae   improved, or worse by <= building_mae_tol_m
    bias           |bias| not worse by more than bias_tol_m
    boundary_mae   (band around GT building boundaries) not worse by more
                   than boundary_rel_tol * baseline value
    grad_mae       not worse by more than grad_rel_tol * baseline value
    seam_mae       not worse (absolute) by more than seam_tol_m
    negative_frac  not increased by more than neg_abs_tol
    calibration    |mean_shift| <= calib_mean_tol_m  (absolute-height guard)

Every rule returns an evidence dict; the overall verdict is PASS only when
all rules pass. Visual quality is deliberately NOT a rule (task rule:
never accept on looks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class AcceptanceConfig:
    mae_tol_m: float = 0.01
    rmse_tol_m: float = 0.05
    building_mae_tol_m: float = 0.05
    bias_tol_m: float = 0.05
    boundary_rel_tol: float = 0.05
    grad_rel_tol: float = 0.10
    seam_tol_m: float = 0.02
    neg_abs_tol: float = 1e-4
    calib_mean_tol_m: float = 0.05


@dataclass
class AcceptanceResult:
    verdict: str  # "PASS" | "FAIL"
    rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def failed_rules(self):
        return [k for k, v in self.rules.items() if not v["pass"]]


def _rule(name: str, got: float, limit: float, ok: bool) -> tuple:
    return (
        name,
        {
            "got": None if got is None else float(got),
            "limit": None if limit is None else float(limit),
            "pass": bool(ok),
        },
    )


def evaluate_acceptance(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    calib_mean_shift: float | None,
    cfg: AcceptanceConfig | None = None,
) -> AcceptanceResult:
    """Compare a candidate variant's summary metrics against the baseline.

    ``baseline``/``candidate`` are flat metric dicts as produced by the
    ablation runner (pooled mae/rmse/bias/negative_fraction + per-tile-mean
    building_mae / boundary_mae / grad_mae / seam_mae keys). Missing keys
    produce a skipped rule recorded as passing with limit None — a variant
    cannot be failed by a metric nobody could compute, but the omission is
    visible in the evidence.
    """
    cfg = cfg or AcceptanceConfig()
    rules: Dict[str, Dict[str, Any]] = {}

    def get(d: Dict[str, float], k: str):
        v = d.get(k)
        return None if v is None or not np.isfinite(v) else float(v)

    pairs = [
        ("mae", get(baseline, "mae"), get(candidate, "mae"), cfg.mae_tol_m),
        ("rmse", get(baseline, "rmse"), get(candidate, "rmse"), cfg.rmse_tol_m),
        (
            "building_mae",
            get(baseline, "building_mae"),
            get(candidate, "building_mae"),
            cfg.building_mae_tol_m,
        ),
    ]
    for name, base_v, cand_v, tol in pairs:
        if base_v is None or cand_v is None:
            rules[name] = {"got": cand_v, "limit": tol, "pass": True, "skipped": True}
            continue
        rules[name] = {
            "got": cand_v,
            "baseline": base_v,
            "limit": tol,
            "pass": bool(cand_v <= base_v + tol),
        }

    # bias: |bias| must not grow beyond the baseline |bias| + tol
    b_b, b_c = get(baseline, "bias"), get(candidate, "bias")
    if b_b is None or b_c is None:
        rules["bias"] = {"got": b_c, "limit": cfg.bias_tol_m, "pass": True, "skipped": True}
    else:
        rules["bias"] = {
            "got": b_c,
            "baseline": b_b,
            "limit": cfg.bias_tol_m,
            "pass": bool(abs(b_c) <= abs(b_b) + cfg.bias_tol_m),
        }

    # relative-tolerance rules (boundary / gradient)
    for name, rel in (
        ("boundary_mae", cfg.boundary_rel_tol),
        ("grad_mae", cfg.grad_rel_tol),
    ):
        base_v, cand_v = get(baseline, name), get(candidate, name)
        if base_v is None or cand_v is None or base_v <= 0:
            rules[name] = {"got": cand_v, "limit": rel, "pass": True, "skipped": True}
            continue
        rules[name] = {
            "got": cand_v,
            "baseline": base_v,
            "limit": rel,
            "pass": bool(cand_v <= base_v * (1.0 + rel)),
        }

    # seam: absolute tolerance
    s_b, s_c = get(baseline, "seam_mae"), get(candidate, "seam_mae")
    if s_b is None or s_c is None:
        rules["seam_mae"] = {"got": s_c, "limit": cfg.seam_tol_m, "pass": True, "skipped": True}
    else:
        rules["seam_mae"] = {
            "got": s_c,
            "baseline": s_b,
            "limit": cfg.seam_tol_m,
            "pass": bool(s_c <= s_b + cfg.seam_tol_m),
        }

    # negative fraction must not increase
    n_b, n_c = get(baseline, "negative_fraction"), get(candidate, "negative_fraction")
    if n_b is None or n_c is None:
        rules["negative_fraction"] = {
            "got": n_c, "limit": cfg.neg_abs_tol, "pass": True, "skipped": True,
        }
    else:
        rules["negative_fraction"] = {
            "got": n_c,
            "baseline": n_b,
            "limit": cfg.neg_abs_tol,
            "pass": bool(n_c <= n_b + cfg.neg_abs_tol),
        }

    # calibration guard: the refinement may not shift the global mean
    if calib_mean_shift is None or not np.isfinite(calib_mean_shift):
        rules["calibration_mean_shift"] = {
            "got": calib_mean_shift, "limit": cfg.calib_mean_tol_m,
            "pass": True, "skipped": True,
        }
    else:
        rules["calibration_mean_shift"] = {
            "got": float(calib_mean_shift),
            "limit": cfg.calib_mean_tol_m,
            "pass": bool(abs(calib_mean_shift) <= cfg.calib_mean_tol_m),
        }

    verdict = "PASS" if all(r["pass"] for r in rules.values()) else "FAIL"
    return AcceptanceResult(verdict=verdict, rules=rules)
