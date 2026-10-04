"""Absolute-DSM evaluation (Part H) — prediction vs REFERENCE absolute DSM.

This is a DISTINCT evaluator from `gt-check` / `eval-scene` (which compare
predictions against *_AGL.tif — AGL evaluation, the GAMUS protocol). Do
NOT mix the two:

    AGL evaluation        : pred height field vs truth AGL  (gt-check,
                            eval-scene, evaluate)
    ABSOLUTE DSM evaluation: pred absolute DSM vs reference absolute
                            DSM (this module — `eval-absolute`)

Grid handling:
  * Both rasters must carry a CRS (CRS_REQUIRED otherwise).
  * Same CRS + transform + shape -> evaluated directly.
  * Otherwise the PREDICTION is reprojected onto the REFERENCE grid
    (bilinear) — the reference is the authority. The overlap fraction is
    reported; zero overlap raises GRID_MISMATCH.
  * Arrays are never broadcast against each other; a residual shape
    mismatch is an error, not a silent zip.

Nodata: pixels invalid in either raster (file nodata, NaN, or +-inf) are
excluded; valid-pixel coverage is reported explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .statuses import CRS_REQUIRED, GRID_MISMATCH, DWStatusError
from .metrics import height_metrics

EVAL_KIND = "absolute_dsm_evaluation"
DEFAULT_RESAMPLING = "bilinear"


@dataclass
class AbsoluteEvalResult:
    report: Dict
    out_json: Optional[Path]

    @property
    def metrics(self) -> Dict:
        return self.report["metrics"]


def _read_single_band(path: Path, what: str):
    import rasterio

    if not Path(path).exists():
        raise DWStatusError(
            "REFERENCE_DSM_REQUIRED_FOR_EVALUATION"
            if what == "reference"
            else "INVALID_GEOREFERENCE",
            f"{what} raster not found: {path}",
        )
    with rasterio.open(path) as src:
        if src.count < 1:
            raise DWStatusError(
                "INVALID_GEOREFERENCE", f"{what} raster has no bands: {path}"
            )
        arr = src.read(1).astype(np.float64)
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }
    return arr, profile


def _valid_mask(arr: np.ndarray, nodata, other_mask: np.ndarray) -> np.ndarray:
    m = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        m &= arr != float(nodata)
    return m & other_mask


def evaluate_absolute_dsm(
    pred_path: Path | str,
    reference_path: Path | str,
    *,
    resampling: str = DEFAULT_RESAMPLING,
    compute_slope: bool = True,
    out_json: Path | str | None = None,
) -> AbsoluteEvalResult:
    """Evaluate a predicted absolute DSM against a reference absolute DSM.

    Returns a JSON-serializable report (kind: absolute_dsm_evaluation);
    when ``out_json`` is given the report is also written there.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import array_bounds

    pred_path, reference_path = Path(pred_path), Path(reference_path)
    pred, pred_prof = _read_single_band(pred_path, "prediction")
    ref, ref_prof = _read_single_band(reference_path, "reference")

    if pred_prof["crs"] is None:
        raise DWStatusError(
            CRS_REQUIRED,
            f"prediction raster has no CRS ({pred_path.name}) — absolute-DSM "
            "evaluation requires georeferenced rasters.",
        )
    if ref_prof["crs"] is None:
        raise DWStatusError(
            CRS_REQUIRED,
            f"reference raster has no CRS ({reference_path.name}) — absolute-DSM "
            "evaluation requires georeferenced rasters.",
        )

    same_grid = (
        pred_prof["crs"] == ref_prof["crs"]
        and pred_prof["transform"] == ref_prof["transform"]
        and pred.shape == ref.shape
    )

    reprojected = False
    if same_grid:
        pred_grid = pred
    else:
        # align the PREDICTION onto the REFERENCE grid (reference is authority)
        pred_grid = np.full(ref.shape, np.nan, dtype=np.float64)
        try:
            # rasterio needs a source dataset for band-based reprojection;
            # use the in-memory route with explicit src/dst georeferencing.
            reproject(
                source=pred,
                destination=pred_grid,
                src_transform=pred_prof["transform"],
                src_crs=pred_prof["crs"],
                src_nodata=pred_prof["nodata"],
                dst_transform=ref_prof["transform"],
                dst_crs=ref_prof["crs"],
                dst_nodata=np.nan,
                resampling=getattr(Resampling, resampling, Resampling.bilinear),
            )
        except Exception as exc:  # noqa: BLE001
            raise DWStatusError(
                GRID_MISMATCH,
                f"prediction->reference reprojection failed: {exc}",
            ) from exc
        reprojected = True

    total_px = int(ref.size)
    ref_valid = _valid_mask(ref, ref_prof["nodata"], np.ones(ref.shape, dtype=bool))
    pred_valid = _valid_mask(pred_grid, np.nan, ref_valid)  # NaN nodata
    overlap_px = int((ref_valid & np.isfinite(pred_grid)).sum())
    overlap_fraction = overlap_px / total_px if total_px else 0.0
    if overlap_px == 0:
        raise DWStatusError(
            GRID_MISMATCH,
            "prediction and reference rasters do not overlap after "
            f"alignment (overlap_fraction=0) — wrong CRS/extent? "
            f"pred crs={pred_prof['crs']} ref crs={ref_prof['crs']}",
        )

    metrics = height_metrics(pred_grid, ref, valid_mask=pred_valid)

    slope = None
    if compute_slope:
        from .geo import pixel_size_metres
        from .metrics import slope_error

        ps = pixel_size_metres(ref_prof["crs"], ref_prof["transform"])
        if ps is not None:
            slope = slope_error(
                pred_grid, ref, gsd_m=float(max(abs(ps[0]), abs(ps[1]))),
                valid_mask=pred_valid,
            )
        # else: GSD unknown -> slope honestly None (honesty contract)

    ref_vals = ref[pred_valid]
    pred_vals = pred_grid[pred_valid]

    report: Dict = {
        "kind": EVAL_KIND,
        "note": (
            "ABSOLUTE DSM evaluation (predicted absolute DSM vs reference "
            "absolute DSM). This is NOT the GAMUS AGL benchmark — AGL "
            "evaluation lives in gt-check / eval-scene / evaluate."
        ),
        "prediction": {
            "path": str(pred_path),
            "crs": str(pred_prof["crs"]),
            "gsd_m": None,
            "shape": [int(pred.shape[0]), int(pred.shape[1])],
            "nodata": None if pred_prof["nodata"] is None else float(pred_prof["nodata"]),
            "range": [float(np.nanmin(pred_vals)), float(np.nanmax(pred_vals))],
        },
        "reference": {
            "path": str(reference_path),
            "crs": str(ref_prof["crs"]),
            "gsd_m": None,
            "shape": [int(ref.shape[0]), int(ref.shape[1])],
            "nodata": None if ref_prof["nodata"] is None else float(ref_prof["nodata"]),
            "range": [float(ref_vals.min()), float(ref_vals.max())],
        },
        "alignment": {
            "reprojected": reprojected,
            "reprojection_target": "reference grid (prediction reprojected)",
            "resampling": resampling if reprojected else None,
            "overlap_fraction": float(overlap_fraction),
        },
        "pixels": {
            "total": total_px,
            "valid": overlap_px,
            "valid_coverage_pct": float(100.0 * overlap_px / total_px) if total_px else 0.0,
        },
        "metrics": {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "bias": metrics["bias"],
            "medae": metrics["medae"],
            "pearson_r": metrics["pearson_r"],
            "n": metrics["n"],
        },
        "slope": slope,
    }

    # per-raster GSDs (reference authority grid + prediction native grid)
    from .geo import pixel_size_metres

    ps_ref = pixel_size_metres(ref_prof["crs"], ref_prof["transform"])
    ps_pred = pixel_size_metres(pred_prof["crs"], pred_prof["transform"])
    report["reference"]["gsd_m"] = (
        round(float(max(abs(ps_ref[0]), abs(ps_ref[1]))), 6) if ps_ref else None
    )
    report["prediction"]["gsd_m"] = (
        round(float(max(abs(ps_pred[0]), abs(ps_pred[1]))), 6) if ps_pred else None
    )

    out: Optional[Path] = None
    if out_json is not None:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["written_to"] = str(out)

    return AbsoluteEvalResult(report=report, out_json=out)
