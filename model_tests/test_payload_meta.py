"""Tests for the GSD honesty contract pinned by worklog Section 4
("GSD honesty regression").

What these tests pin:

  * ``depthwizard.geo.pixel_size_metres`` — the single CRS-unit-to-metres
    helper. It MUST return ``None`` when CRS is absent (Track-1 honest null)
    and the exact hand-computed metres-per-pixel for a synthetic projected
    CRS+transform (EPSG:32617 with 30 m pixels -> [30.0, 30.0]). The
    fallback-to-1.0 bug from the GSD honesty regression is forbidden here.

  * ``depthwizard.inference.build_scene_payload`` — the payload contract
    that ``webapp/src/lib/dw.ts`` mirrors. The new
    ``meta.pixel_size_m`` field MUST be present, MUST be ``[float, float]``
    for georeferenced input, and MUST be ``None`` (JSON null) for
    non-georeferenced input. Breaking either side breaks the webapp.

The synthetic-GeoTIFF / no-CRS patterns mirror ``tests/test_anchoring.py`` —
the same CRS gate is exercised here from the payload side.

Run:  pytest tests/test_payload_meta.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.geo import pixel_size_metres
from depthwizard.inference import build_scene_payload


# ---------------------------------------------------------------------------
# pixel_size_metres — pure CRS/transform helper
# ---------------------------------------------------------------------------

def test_pixel_size_metres_none_for_missing_crs():
    """Track-1 (no CRS) MUST return None. Fabricating a default 1.0 here is
    the silent-wrong-number bug the worklog Section 4 incident describes."""
    assert pixel_size_metres(None, None) is None
    # transform present but CRS absent — still None (we cannot derive metres
    # from a transform alone; the unit machinery lives on the CRS).
    from rasterio.transform import Affine
    assert pixel_size_metres(None, Affine.translation(0, 0)) is None


def test_pixel_size_metres_none_for_zero_pixel_size():
    """A degenerate transform (a=0 or e=0) cannot express a real GSD."""
    from rasterio.transform import Affine
    zero_a = Affine(0.0, 0.0, 0.0, 0.0, -30.0, 0.0)
    # The CRS here is irrelevant — degenerate transform short-circuits.
    crs = rasterio.crs.CRS.from_epsg(32617)
    assert pixel_size_metres(crs, zero_a) is None


def test_pixel_size_metres_projected_metre_units_exact():
    """EPSG:32617 (UTM 17N, metre) + 30 m pixels -> [30.0, 30.0].

    Hand computation: ``transform.a = 30.0`` (pixel width east, metres),
    ``transform.e = -30.0`` (pixel height, negative because north-up),
    factor (metre) = 1.0 -> ``[30.0 * 1.0, 30.0 * 1.0]``.
    """
    crs = rasterio.crs.CRS.from_epsg(32617)
    tf = from_origin(500000.0, 4_000_000.0, 30.0, 30.0)   # 30 m at UTM origin
    out = pixel_size_metres(crs, tf)
    assert out is not None
    assert out == pytest.approx((30.0, 30.0), abs=1e-9)


def test_pixel_size_metres_projected_foot_units_converted():
    """EPSG:2240 (Georgia West, US survey foot) -> metres via the CRS factor.

    Hand computation: 1 US survey foot = 0.3048006096012192 m (exact per
    NIST). Pixel size 100 ft -> 30.48006096012192 m.
    """
    crs = rasterio.crs.CRS.from_epsg(2240)
    tf = from_origin(0.0, 0.0, 100.0, 100.0)              # 100 ft pixels
    out = pixel_size_metres(crs, tf)
    assert out is not None
    assert out[0] == pytest.approx(30.48006096012192, abs=1e-6)
    assert out[1] == pytest.approx(30.48006096012192, abs=1e-6)


def test_pixel_size_metres_geographic_degrees_to_metres():
    """EPSG:4326 (WGS84, degrees) -> local-latitude approximation at y-origin.

    Hand computation: tile centered at lat 0 -> cos_lat = 1, so:
      pixel_w_m = a * 111_320 * cos(0) = a * 111_320
      pixel_h_m = |e| * 111_320
    With a = 0.0001 degrees -> 11.132 m.
    """
    crs = rasterio.crs.CRS.from_epsg(4326)
    # 0.0001 degree pixels centered at the equator, lat 0
    tf = from_origin(0.0, 0.0, 0.0001, 0.0001)
    out = pixel_size_metres(crs, tf)
    assert out is not None
    assert out[0] == pytest.approx(11.132, abs=1e-3)   # lon at equator
    assert out[1] == pytest.approx(11.132, abs=1e-3)   # lat is constant


# ---------------------------------------------------------------------------
# build_scene_payload — pixel_size_m threading through the contract
# ---------------------------------------------------------------------------

def _synthetic_georef_payload(crs, transform):
    """Build a ScenePayload with a real CRS+transform through the public
    build_scene_payload entry — same surface the FastAPI service and the
    `infer` CLI use."""
    rng = np.random.default_rng(1)
    dsm = rng.random((16, 16)).astype(np.float32) * 10
    rgb = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    return build_scene_payload(
        dsm, rgb, stem="X_001_001", mode="tiles", dn_source="explicit",
        model_tag="calib_test", device="cpu",
        profile={"_crs_obj": crs, "_transform_obj": transform},
        anchored=None, outputs={}, elapsed_sec=0.01)


def test_payload_pixel_size_m_present_for_georeferenced_input():
    """Georeferenced payload MUST carry meta.pixel_size_m as [float, float]."""
    crs = rasterio.crs.CRS.from_epsg(32617)
    tf = from_origin(500_000.0, 4_000_000.0, 30.0, 30.0)
    payload = _synthetic_georef_payload(crs, tf)
    assert "pixel_size_m" in payload["meta"]
    psm = payload["meta"]["pixel_size_m"]
    assert psm is not None
    assert len(psm) == 2
    assert psm[0] == pytest.approx(30.0, abs=1e-6)
    assert psm[1] == pytest.approx(30.0, abs=1e-6)


def test_payload_pixel_size_m_null_for_non_georeferenced_input():
    """Non-georeferenced (Track-1) payload MUST carry meta.pixel_size_m = null.

    This is the honest-null contract: the GSD honesty regression silently
    assumed 1 source px = 1 m when the value was missing. The fix is to
    surface ``null`` so consumers (Viewer3D, demprior) can branch on the
    absence and refuse metric claims. NEVER a fabricated default.
    """
    payload = _synthetic_georef_payload(None, None)
    assert "pixel_size_m" in payload["meta"]
    assert payload["meta"]["pixel_size_m"] is None


def test_payload_pixel_size_m_round_trips_through_json():
    """The payload (and thus the field) MUST survive a JSON round trip — the
    FastAPI service serializes through JSON; null has to stay null, not
    coerce to []."""
    import json
    crs = rasterio.crs.CRS.from_epsg(32617)
    tf = from_origin(500_000.0, 4_000_000.0, 30.0, 30.0)
    payload = _synthetic_georef_payload(crs, tf)
    reloaded = json.loads(json.dumps(payload))
    assert reloaded["meta"]["pixel_size_m"] == [30.0, 30.0]

    payload_noref = _synthetic_georef_payload(None, None)
    reloaded_noref = json.loads(json.dumps(payload_noref))
    assert reloaded_noref["meta"]["pixel_size_m"] is None
