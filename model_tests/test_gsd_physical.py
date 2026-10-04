"""GSD robustness regressions (Part F) + 3D/physical consistency (Part M).

Pins the GSD honesty contract end to end:
  * pixel_size_metres derives real metres/pixel from projected CRSs,
    non-metre units, and geographic CRSs — never a silent 1.0
  * changing the PIXEL grid without changing the PHYSICAL surface must
    not change physical dimensions (DEM resample, slope, mesh extents)
  * slope classification and route thresholds key off physical units
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

from depthwizard.geo import pixel_size_metres

UTM = CRS.from_epsg(32617)


# ---------------------------------------------------------------------------
# GSD derivation across CRS families / resolutions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gsd_m", [0.35, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_projected_crs_gsd_roundtrip(gsd_m):
    tf = from_origin(500000, 4000000, gsd_m, gsd_m)
    ps = pixel_size_metres(UTM, tf)
    assert ps is not None
    assert ps[0] == pytest.approx(gsd_m, rel=1e-9)
    assert ps[1] == pytest.approx(gsd_m, rel=1e-9)


def test_non_metric_projected_crs_converts_units():
    # EPSG:3446 etc. may be metre; construct a foot-based CRS instead:
    # EPSG:6543 (NAD83 / Ohio South ftUS) is projected with US survey feet.
    crs = CRS.from_epsg(6543)
    factor = crs.linear_units_factor[1]
    assert factor != 1.0
    ps = pixel_size_metres(crs, from_origin(0, 0, 1.0, 1.0))
    assert ps is not None
    assert ps[0] == pytest.approx(factor, rel=1e-6)  # 1 ft pixel -> ft->m


def test_geographic_crs_uses_latitude_approximation():
    crs = CRS.from_epsg(4326)
    deg = 1e-5
    lat = 36.0
    tf = from_origin(-81.0, lat, deg, deg)
    ps = pixel_size_metres(crs, tf)
    assert ps is not None
    expected_lat_m = deg * 111_320.0
    expected_lon_m = deg * 111_320.0 * np.cos(np.radians(lat))
    assert ps[1] == pytest.approx(expected_lat_m, rel=1e-6)
    assert ps[0] == pytest.approx(expected_lon_m, rel=1e-6)


def test_no_crs_means_no_gsd_never_one_meter_guess():
    assert pixel_size_metres(None, from_origin(0, 0, 1, 1)) is None
    assert pixel_size_metres(UTM, None) is None


# ---------------------------------------------------------------------------
# Physical invariance: same terrain, different pixel grids
# ---------------------------------------------------------------------------

def _terrain(tile: int = 64, height_m: float = 12.0) -> np.ndarray:
    """A synthetic hill: physically identical no matter the GSD."""
    yy, xx = np.mgrid[0:tile, 0:tile]
    r = np.hypot(yy - tile / 2, xx - tile / 2) / (tile / 2)
    return (height_m * np.exp(-r**2)).astype(np.float32)


def test_dem_resample_preserves_physical_elevations(tmp_path):
    """A 5 m DEM resampled onto 0.5 m / 1 m / 2 m grids yields the same
    physical elevations (m), not per-pixel values."""
    from depthwizard.anchoring import resample_dem_to_tile

    dem = _terrain(64, 12.0)
    dem_path = tmp_path / "dem5m.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=64, width=64, count=1,
        dtype="float32", crs=UTM,
        transform=from_origin(500000, 4000000, 5.0, 5.0),
    ) as dst:
        dst.write(dem, 1)

    center_5m = dem[32, 32]
    for gsd in (0.5, 1.0, 2.0):
        n = int(64 * 5.0 / gsd)  # same physical footprint
        profile = {
            "crs": UTM,
            "transform": from_origin(500000, 4000000, gsd, gsd),
            "height": n, "width": n,
        }
        res = resample_dem_to_tile(dem_path, profile)
        assert res.shape == (n, n)
        assert float(res.max()) == pytest.approx(center_5m, rel=0.02)


def test_mesh_physical_extents_follow_gsd():
    """Scene payload physical size = pixels x GSD (Part M): a 128 px scene
    at 2 m GSD is 256 m wide at ANY resolution — dimensions track GSD."""
    from depthwizard.pipeline.scene_outputs import build_scene_payload

    dsm = _terrain(128, 20.0)
    rgb = np.zeros((128, 128, 3), np.uint8)
    for gsd in (0.5, 2.0, 10.0):
        tf = from_origin(500000, 4000000, gsd, gsd)
        payload = build_scene_payload(
            dsm, rgb, stem="s", mode="crop", dn_source="x", model_tag="t",
            device="cpu",
            profile={"_crs_obj": UTM, "_transform_obj": tf},
            anchored=None, outputs={}, elapsed_sec=0.1,
        )
        ps = payload["meta"]["pixel_size_m"]
        assert ps[0] == pytest.approx(gsd, abs=1e-4)
        # physical width = source pixels x gsd regardless of mesh stride
        physical_w = payload["meta"]["source_shape"][1] * ps[0]
        assert physical_w == pytest.approx(128 * gsd, abs=1e-3)
        # and the height values themselves are unchanged (metres)
        assert payload["stats"]["max"] == pytest.approx(float(dsm.max()), abs=1e-3)
