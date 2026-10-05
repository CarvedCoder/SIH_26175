"""eval-absolute: predicted ABSOLUTE DSM vs reference absolute DSM (Part H).

Distinct from gt-check / eval-scene (AGL evaluation, GAMUS protocol).
This command evaluates a predicted ABSOLUTE DSM raster against a
reference ABSOLUTE DSM raster:

  python model.py eval-absolute \
      --pred predicted_absolute_dsm.tif \
      --truth reference_absolute_dsm.tif \
      [--json-out report.json] [--resampling bilinear]

The reference must be a genuinely external absolute DSM (organizer
reference data, LiDAR-derived DSM, or another authoritative source).
This command does NOT manufacture reference data and does NOT mix its
numbers with the AGL benchmark.
"""

from __future__ import annotations

import argparse

NAME = "eval-absolute"
HELP = (
    "evaluate a predicted ABSOLUTE DSM vs a reference absolute DSM "
    "(NOT the AGL benchmark)"
)


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        NAME,
        help=HELP,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pred", required=True, help="predicted absolute-DSM GeoTIFF")
    p.add_argument(
        "--truth", required=True,
        help="reference absolute-DSM GeoTIFF (external/organizer reference)",
    )
    p.add_argument(
        "--json-out", default=None,
        help="write the machine-readable evaluation report to this path",
    )
    p.add_argument(
        "--resampling", default="bilinear",
        help="resampling used if the prediction must be aligned to the "
        "reference grid (default bilinear)",
    )
    p.add_argument(
        "--no-slope", action="store_true",
        help="skip the slope-MAE metric (auto-skipped when GSD is unknown)",
    )
    return p


def run(args) -> int:
    from depthwizard.eval_absolute import evaluate_absolute_dsm
    from depthwizard.statuses import DWStatusError

    try:
        result = evaluate_absolute_dsm(
            args.pred,
            args.truth,
            resampling=args.resampling,
            compute_slope=not args.no_slope,
            out_json=args.json_out,
        )
    except DWStatusError as exc:
        print(f"[error] {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"[error] [REFERENCE_DSM_REQUIRED_FOR_EVALUATION] {exc}")
        return 1

    r = result.report
    m = r["metrics"]
    px = r["pixels"]
    print("=" * 72)
    print("ABSOLUTE DSM EVALUATION (not the GAMUS AGL benchmark)")
    print("=" * 72)
    print(f"  prediction : {r['prediction']['path']}")
    print(f"               crs={r['prediction']['crs']}  gsd={r['prediction']['gsd_m']}")
    print(f"  reference  : {r['reference']['path']}")
    print(f"               crs={r['reference']['crs']}  gsd={r['reference']['gsd_m']}")
    print(
        f"  alignment  : reprojected={r['alignment']['reprojected']} "
        f"resampling={r['alignment']['resampling']} "
        f"overlap={r['alignment']['overlap_fraction']:.1%}"
    )
    print(
        f"  pixels     : valid {px['valid']:,}/{px['total']:,} "
        f"({px['valid_coverage_pct']:.1f}% coverage)"
    )
    print("-" * 72)
    print(f"  MAE        : {m['mae']:.3f} m")
    print(f"  RMSE       : {m['rmse']:.3f} m")
    print(f"  bias       : {m['bias']:+.3f} m")
    print(f"  medAE      : {m['medae']:.3f} m")
    print(f"  Pearson r  : {m['pearson_r']:.3f}")
    if r.get("slope"):
        print(f"  slope MAE  : {r['slope']['slope_mae_deg']:.2f} deg")
    else:
        print("  slope MAE  : n/a (GSD unknown or skipped — reported as null)")
    if result.out_json:
        print(f"[out] {result.out_json}")
    return 0
