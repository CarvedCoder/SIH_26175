"""Tests for depthwizard.absolute_dsm — DEM alignment + provenance (B/C/S).

Pins:
  * DSM_absolute = DEM_reference + AGL, exactly, on the image grid
  * CRS / transform / dimensions preserved; no broadcasting
  * output_type taxonomy: absolute_dsm | anchored_constant_dsm |
    relative_height — and absolute_reference_available is honest
  * full-coverage gate: partial DEM coverage -> DEM_COVERAGE_INSUFFICIENT
  * non-georeferenced input + DEM -> CRS_REQUIRED (no fake alignment)
  * provenance is complete; unknowns are None, never invented
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.absolute_dsm import (
    ABSOLUTE_DSM_FORMULA,
    OUTPUT_ABSOLUTE_DSM,
    OUTPUT_ANCHORED_CONSTANT_DSM,
    OUTPUT_RELATIVE_HEIGHT,
    build_absolute_dsm,
    maybe_acquire_reference_dem,
    payload_output_fields,
)
from depthwizard.statuses import (
    CRS_REQUIRED,
    DEM_COVERAGE_INSUFFICIENT,
    DWStatusError,
)

UTM = CRS.from_epsg(32617)
TRANSFORM = from_origin(500000, 4000000, 30.0, 30.0)  # 30 m GSD


def _write_dem(path, arr, crs=UTM, transform=TRANSFORM):
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs=crs,
                       transform=transform) as dst:
        dst.write(arr.astype(np.float32), 1)


def _profile(georef=True, h=16, w=16):
    return {
        "_crs_obj": UTM if georef else None,
        "_transform_obj": TRANSFORM if georef else None,
        "height": h, "width": w,
    }


def _dem_result(tmp_path, name="dem.tif", crs=UTM, transform=TRANSFORM,
                dem_source="local:dem.tif", resolution_m=30.0,
                vertical_reference=None):
    from depthwizard.dem_provider import DEMResult

    arr = np.linspace(100.0, 140.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    p = tmp_path / name
    _write_dem(p, arr, crs=crs, transform=transform)
    return DEMResult(
        mosaic_path=p, provider="local", dem_source=dem_source,
        crs=crs, resolution_m=resolution_m,
        vertical_reference=vertical_reference, identifier=str(p),
    )


# ---------------------------------------------------------------------------
# Absolute DSM = DEM + AGL
# ---------------------------------------------------------------------------

def test_absolute_dsm_exact_addition_and_grid_preserved(tmp_path):
    agl = np.full((16, 16), 7.5, dtype=np.float32)
    dem_res = _dem_result(tmp_path, vertical_reference="EGM2008 geoid")
    res = build_absolute_dsm(agl, _profile(), dem_res)

    with rasterio.open(dem_res.mosaic_path) as src:
        dem = src.read(1)
    np.testing.assert_allclose(res.dsm, dem + agl, atol=1e-4)
    assert res.output_type == OUTPUT_ABSOLUTE_DSM
    assert res.absolute_reference_available is True
    assert res.anchored is not None


def test_absolute_dsm_downsampling_dem_to_finer_grid(tmp_path):
    """30 m DEM resampled onto a 0.5 m scene grid stays physical."""
    agl = np.zeros((32, 32), dtype=np.float32)
    dem_res = _dem_result(tmp_path)
    fine = from_origin(500000, 4000000, 0.5, 0.5)
    profile = {
        "_crs_obj": UTM, "_transform_obj": fine,
        "height": 32, "width": 32,
    }
    res = build_absolute_dsm(agl, profile, dem_res)
    assert res.dsm.shape == (32, 32)
    assert np.isfinite(res.dsm).all()
    # DEM surface spans 100..140 m over the scene; the fine grid samples it
    assert 95.0 <= res.dsm.min() and res.dsm.max() <= 145.0


def test_partial_dem_coverage_raises_coverage_insufficient(tmp_path):
    agl = np.zeros((16, 16), dtype=np.float32)
    # DEM shifted half a scene north: partial overlap only
    partial = from_origin(500000, 4000024, 30.0, 30.0)
    dem_res = _dem_result(tmp_path, transform=partial)
    with pytest.raises(DWStatusError) as ei:
        build_absolute_dsm(agl, _profile(), dem_res)
    assert ei.value.code == DEM_COVERAGE_INSUFFICIENT


def test_dem_anchoring_requires_georeferenced_input(tmp_path):
    agl = np.zeros((16, 16), dtype=np.float32)
    dem_res = _dem_result(tmp_path)
    with pytest.raises(DWStatusError) as ei:
        build_absolute_dsm(agl, _profile(georef=False), dem_res)
    assert ei.value.code == CRS_REQUIRED


def test_grid_mismatch_never_broadcasts(tmp_path, monkeypatch):
    """If alignment ever returned a wrong-shaped grid, raise, don't broadcast."""
    agl = np.zeros((16, 16), dtype=np.float32)
    dem_res = _dem_result(tmp_path)

    import depthwizard.absolute_dsm as mod

    def wrong_shape(path, profile):
        return np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(mod, "resample_dem_to_tile", wrong_shape)
    with pytest.raises(DWStatusError) as ei:
        build_absolute_dsm(agl, _profile(), dem_res)
    assert ei.value.code == "GRID_MISMATCH"


# ---------------------------------------------------------------------------
# output_type taxonomy (Part D)
# ---------------------------------------------------------------------------

def test_constant_datum_is_not_absolute_reference():
    agl = np.full((8, 8), 3.0, dtype=np.float32)
    res = build_absolute_dsm(agl, _profile(h=8, w=8), None, ground_elev=101.0)
    assert res.output_type == OUTPUT_ANCHORED_CONSTANT_DSM
    assert res.absolute_reference_available is False
    np.testing.assert_allclose(res.dsm, agl + 101.0, atol=1e-6)
    # works on non-georeferenced input too (pure arithmetic)
    res2 = build_absolute_dsm(
        agl, _profile(georef=False, h=8, w=8), None, ground_elev=5.0
    )
    assert res2.output_type == OUTPUT_ANCHORED_CONSTANT_DSM


def test_no_reference_is_relative_height():
    agl = np.full((8, 8), 3.0, dtype=np.float32)
    res = build_absolute_dsm(agl, _profile(h=8, w=8), None)
    assert res.output_type == OUTPUT_RELATIVE_HEIGHT
    assert res.absolute_reference_available is False
    assert res.anchored is None


# ---------------------------------------------------------------------------
# Provenance (Part C)
# ---------------------------------------------------------------------------

def test_provenance_complete_for_absolute_dsm(tmp_path):
    agl = np.full((16, 16), 2.0, dtype=np.float32)
    dem_res = _dem_result(
        tmp_path, dem_source="copernicus", vertical_reference="EGM2008 geoid"
    )
    res = build_absolute_dsm(
        agl, _profile(), dem_res,
        height_model="terraheight_s", height_model_version="best_model.pth",
        calibration_enabled=False,
    )
    p = res.provenance
    assert p["height_model"] == "terraheight_s"
    assert p["height_model_version"] == "best_model.pth"
    assert p["calibration_enabled"] is False
    assert p["height_units"] == "meters"
    assert p["height_semantics"] == "AGL"
    assert p["dem_source"] == "copernicus"
    assert p["dem_crs"] == str(UTM)
    assert p["dem_resolution_m"] == pytest.approx(30.0)
    assert p["dem_vertical_reference"] == "EGM2008 geoid"
    assert p["scene_crs"] == str(UTM)
    assert p["scene_gsd_m"] == pytest.approx(30.0)
    assert p["scene_shape"] == [16, 16]
    assert p["alignment"]["resampling"] == "bilinear"
    assert p["alignment"]["formula"] == ABSOLUTE_DSM_FORMULA
    assert "scene_bounds" in p and len(p["scene_bounds"]) == 4


def test_provenance_unknowns_are_none_never_invented(tmp_path):
    agl = np.full((8, 8), 1.0, dtype=np.float32)
    dem_res = _dem_result(tmp_path, vertical_reference=None)
    res = build_absolute_dsm(agl, _profile(h=8, w=8), dem_res)
    p = res.provenance
    assert p["dem_vertical_reference"] is None
    assert p["vertical_reference"] is None
    assert p["alignment"]["resampling"] == "bilinear"
    assert p["height_model"] is None  # not passed -> honest None
    # relative product: DEM fields all None/empty, never fabricated
    rel = build_absolute_dsm(agl, _profile(h=8, w=8), None)
    rp = rel.provenance
    assert rp["dem_source"] is None and rp["dem_tiles"] == []
    assert rp["alignment"]["formula"] is None


def test_payload_output_fields(tmp_path):
    agl = np.full((8, 8), 1.0, dtype=np.float32)
    dem_res = _dem_result(tmp_path)
    res = build_absolute_dsm(agl, _profile(h=8, w=8), dem_res)
    fields = payload_output_fields(res)
    assert fields["output_type"] == OUTPUT_ABSOLUTE_DSM
    assert fields["absolute_reference_available"] is True
    assert fields["absolute_dsm_available"] is True
    assert isinstance(fields["provenance"], dict)


def test_write_provenance_json(tmp_path):
    import json

    from depthwizard.absolute_dsm import write_provenance

    agl = np.full((8, 8), 1.0, dtype=np.float32)
    dem_res = _dem_result(tmp_path)
    res = build_absolute_dsm(agl, _profile(h=8, w=8), dem_res)
    p = write_provenance(res.provenance, tmp_path / "out")
    loaded = json.loads(p.read_text())
    assert loaded["dem_source"] == "local:dem.tif"


# ---------------------------------------------------------------------------
# maybe_acquire_reference_dem dispatch
# ---------------------------------------------------------------------------

def test_maybe_acquire_no_request_returns_none(tmp_path):
    out, requested = maybe_acquire_reference_dem(
        None, None, tmp_path, _profile(), (16, 16)
    )
    assert out is None and requested is None


def test_maybe_acquire_explicit_path(tmp_path):
    from depthwizard.dem_provider import DEMResult

    dem_res = _dem_result(tmp_path)
    out, requested = maybe_acquire_reference_dem(
        dem_res.mosaic_path, None, tmp_path, _profile(), (16, 16)
    )
    assert isinstance(out, DEMResult)
    assert out.mosaic_path == dem_res.mosaic_path
    assert requested is None


def test_maybe_acquire_network_failure_degrades_to_relative(
    tmp_path, capsys, monkeypatch
):
    """DEM_UNAVAILABLE -> loud notice + None (relative continuation)."""
    import depthwizard.dem_provider as dmod
    from depthwizard.statuses import DEM_UNAVAILABLE

    def fail(*a, **k):
        raise DWStatusError(DEM_UNAVAILABLE, "network down")

    monkeypatch.setattr(dmod, "resolve_reference_dem", fail)
    out, requested = maybe_acquire_reference_dem(
        None, "copernicus", tmp_path, _profile(), (16, 16)
    )
    assert out is None and requested == "copernicus"
    captured = capsys.readouterr()
    assert "RELATIVE" in captured.out and "DEM_UNAVAILABLE" in captured.out


def test_maybe_acquire_crission_required_propagates(tmp_path):
    profile = _profile(georef=False)
    with pytest.raises(DWStatusError) as ei:
        maybe_acquire_reference_dem(
            None, "copernicus", tmp_path, profile, (16, 16)
        )
    assert ei.value.code == CRS_REQUIRED
