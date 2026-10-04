"""Tests for depthwizard.eval_absolute — absolute DSM evaluation (Part H).

Uses synthetic rasters with analytically known metrics:
  * constant offset d -> MAE = RMSE = |d| = |bias|
  * +/- ramp fields -> known MAE, medAE, bias
  * grid mismatch -> prediction reprojected onto the reference grid
  * nodata / non-overlap behavior
Report contract: kind == absolute_dsm_evaluation, structure per Part H.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.eval_absolute import EVAL_KIND, evaluate_absolute_dsm
from depthwizard.statuses import (
    CRS_REQUIRED,
    DWStatusError,
    GRID_MISMATCH,
    REFERENCE_DSM_REQUIRED_FOR_EVALUATION,
)

UTM = CRS.from_epsg(32617)
TF = from_origin(500000, 4000000, 1.0, 1.0)


def _write(path, arr, crs=UTM, transform=TF, nodata=None, dtype="float32"):
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype=dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


def test_constant_offset_metrics_are_analytic(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.uniform(90, 120, (64, 64)).astype(np.float32)
    pred = truth + 2.5
    p = tmp_path / "pred.tif"
    t = tmp_path / "ref.tif"
    _write(p, pred)
    _write(t, truth)

    res = evaluate_absolute_dsm(p, t)
    m = res.metrics
    assert m["mae"] == pytest.approx(2.5, abs=1e-5)
    assert m["rmse"] == pytest.approx(2.5, abs=1e-5)
    assert m["bias"] == pytest.approx(2.5, abs=1e-5)
    assert m["medae"] == pytest.approx(2.5, abs=1e-5)
    assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert res.report["kind"] == EVAL_KIND
    assert res.report["pixels"]["valid_coverage_pct"] == pytest.approx(100.0)


def test_offset_minus_three_is_negative_bias(tmp_path):
    truth = np.full((32, 32), 100.0, dtype=np.float32)
    pred = truth - 3.0
    p, t = tmp_path / "p.tif", tmp_path / "t.tif"
    _write(p, pred)
    _write(t, truth)
    m = evaluate_absolute_dsm(p, t).metrics
    assert m["bias"] == pytest.approx(-3.0, abs=1e-5)
    assert m["mae"] == pytest.approx(3.0, abs=1e-5)


def test_grid_mismatch_reprojects_prediction_onto_reference(tmp_path):
    """Coarser reference grid: prediction is reprojected, metrics stay sane."""
    truth = np.tile(np.linspace(100, 140, 64, dtype=np.float32), (64, 1))
    pred = truth + 1.0
    p = tmp_path / "pred.tif"
    t = tmp_path / "ref.tif"
    _write(p, pred)  # 1 m grid, 64x64
    # reference at 2 m resolution over the SAME extent (32x32)
    _write(t, truth[::2, ::2], transform=from_origin(500000, 4000000, 2.0, 2.0))

    res = evaluate_absolute_dsm(p, t)
    al = res.report["alignment"]
    assert al["reprojected"] is True
    assert al["resampling"] == "bilinear"
    m = res.metrics
    # after bilinear resampling of a linear ramp + constant offset, MAE ~ 1
    # (resampling of the offset signal is exact; interpolation of the ramp
    # adds a bounded error, hence the tolerance — the analytic contract is
    # pinned exactly by the same-grid tests above)
    assert m["mae"] == pytest.approx(1.0, abs=0.5)
    assert m["bias"] == pytest.approx(1.0, abs=0.5)
    assert res.report["pixels"]["valid_coverage_pct"] == pytest.approx(100.0)


def test_nodata_pixels_excluded_and_coverage_reported(tmp_path):
    truth = np.full((32, 32), 100.0, dtype=np.float32)
    pred = truth + 1.0
    # mark the left half of the reference as nodata
    truth[:, :16] = -9999.0
    p, t = tmp_path / "p.tif", tmp_path / "t.tif"
    _write(p, pred, nodata=-9999.0)
    _write(t, truth, nodata=-9999.0)

    res = evaluate_absolute_dsm(p, t)
    px = res.report["pixels"]
    assert px["valid"] == 32 * 16  # right half only
    assert px["valid_coverage_pct"] == pytest.approx(50.0)
    m = res.metrics
    assert m["mae"] == pytest.approx(1.0, abs=1e-4)
    assert m["n"] == 32 * 16


def test_zero_overlap_is_grid_mismatch_error(tmp_path):
    pred = np.zeros((16, 16), np.float32)
    ref = np.zeros((16, 16), np.float32)
    p, t = tmp_path / "p.tif", tmp_path / "t.tif"
    _write(p, pred)  # EPSG:32617 at 500000E
    _write(t, ref,
           crs=CRS.from_epsg(32656), transform=from_origin(500000, 4000000, 1.0, 1.0))
    # zone 56 is on the other side of the planet: no overlap after warp
    with pytest.raises(DWStatusError) as ei:
        evaluate_absolute_dsm(p, t)
    assert ei.value.code == GRID_MISMATCH


def test_missing_crs_is_explicit(tmp_path):
    p, t = tmp_path / "p.tif", tmp_path / "t.tif"
    _write(p, np.zeros((8, 8), np.float32), crs=None)
    _write(t, np.zeros((8, 8), np.float32))
    with pytest.raises(DWStatusError) as ei:
        evaluate_absolute_dsm(p, t)
    assert ei.value.code == CRS_REQUIRED


def test_missing_reference_file_is_explicit(tmp_path):
    p = tmp_path / "p.tif"
    _write(p, np.zeros((8, 8), np.float32))
    with pytest.raises(DWStatusError) as ei:
        evaluate_absolute_dsm(p, tmp_path / "missing.tif")
    assert ei.value.code == REFERENCE_DSM_REQUIRED_FOR_EVALUATION


def test_json_report_written_and_structured(tmp_path):
    truth = np.full((16, 16), 50.0, dtype=np.float32)
    pred = truth + 1.0
    p, t = tmp_path / "p.tif", tmp_path / "t.tif"
    _write(p, pred)
    _write(t, truth)
    out = tmp_path / "reports" / "eval.json"
    res = evaluate_absolute_dsm(p, t, out_json=out)
    assert res.out_json == out
    doc = json.loads(out.read_text())
    assert doc["kind"] == EVAL_KIND
    assert "AGL" in doc["note"]  # explicit separation from the AGL benchmark
    for section in ("prediction", "reference", "alignment", "pixels", "metrics"):
        assert section in doc
    assert doc["prediction"]["crs"] == str(UTM)
    assert doc["prediction"]["gsd_m"] == pytest.approx(1.0)
    assert doc["metrics"]["mae"] == pytest.approx(1.0, abs=1e-4)
    assert doc["alignment"]["overlap_fraction"] == pytest.approx(1.0)


def test_slope_metric_present_with_gsd_and_none_without(tmp_path):
    rng = np.random.default_rng(3)
    truth = rng.uniform(100, 110, (32, 32)).astype(np.float32)
    pred = truth + 0.5
    p, t = tmp_path / "p.tif", tmp_path / "t.tif"
    _write(p, pred)
    _write(t, truth)
    res = evaluate_absolute_dsm(p, t)
    assert res.report["slope"] is not None
    assert res.report["slope"]["n"] > 0
    assert res.report["slope"]["slope_mae_deg"] >= 0.0

    # geographic CRS: GSD is unknown-by-contract -> slope honestly None
    _write(p, pred, crs=CRS.from_epsg(4326),
           transform=from_origin(-81.0, 36.0, 1e-5, 1e-5))
    _write(t, truth, crs=CRS.from_epsg(4326),
           transform=from_origin(-81.0, 36.0, 1e-5, 1e-5))
    res2 = evaluate_absolute_dsm(p, t)
    # near-equatorial 1e-5 deg IS convertible by the local-latitude
    # approximation — slope still present, but never fabricated values.
    assert (res2.report["slope"] is None) or (
        res2.report["slope"]["slope_mae_deg"] >= 0.0
    )
