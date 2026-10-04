"""Tests for depthwizard.dem_provider — pluggable DEM acquisition (Part A).

Pins:
  * provider registry / dispatch (explicit path wins; named providers)
  * the mock provider is test-only and refuses production paths
  * Copernicus footprint geometry is pure (no network) and conservative
  * network failures are EXPLICIT (DEM_UNAVAILABLE), never silent fakes
  * caching: a repeat request is served from disk (no second download)
  * provenance records the source and labels synthetic data
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

from depthwizard.dem_provider import (
    PROVIDER_MOCK,
    CopernicusDem30mProvider,
    LocalFileProvider,
    MockDEMProvider,
    _tile_name,
    copernicus_tile_url,
    required_copernicus_tiles,
    resolve_reference_dem,
)
from depthwizard.statuses import CRS_REQUIRED, DEM_UNAVAILABLE, DWStatusError

UTM = CRS.from_epsg(32617)
BOUNDS = (500000.0, 3999000.0, 500300.0, 4000000.0)  # 300m x 1km, EPSG:32617


def _request(crs=UTM, bounds=BOUNDS):
    from depthwizard.dem_provider import DEMRequest

    return DEMRequest(
        west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3],
        scene_crs=crs,
    )


# ---------------------------------------------------------------------------
# Copernicus footprint geometry (pure, offline)
# ---------------------------------------------------------------------------

def test_tile_naming_matches_aws_layout():
    assert _tile_name(32, 74) == "Copernicus_DSM_COG_10_N32_00_E074_00_DEM"
    assert _tile_name(-33, -118) == "Copernicus_DSM_COG_10_S33_00_W118_00_DEM"
    assert copernicus_tile_url(_tile_name(32, 74)).startswith(
        "https://copernicus-dem-30m.s3.amazonaws.com/"
    )


def test_required_tiles_conservative_and_offline():
    # ~300 m x 1 km near lon -81.5, lat 36.1 (EPSG:32617 around NC, USA)
    tiles = required_copernicus_tiles(_request())
    assert len(tiles) >= 1 and len(tiles) <= 4  # 1 km footprint: few cells
    for lat, lon in tiles:
        assert isinstance(lat, int) and isinstance(lon, int)


def test_required_tiles_spans_lon_antimeridian_style_bounds():
    # a footprint crossing a whole-degree boundary pulls BOTH columns
    r = _request(
        crs=CRS.from_epsg(4326),
        bounds=(10.2, 40.2, 11.4, 40.8),
    )
    tiles = required_copernicus_tiles(r)
    assert (40, 10) in tiles and (40, 11) in tiles


# ---------------------------------------------------------------------------
# Local + mock providers
# ---------------------------------------------------------------------------

def test_local_provider_reads_provenance_from_raster(tmp_path):
    dem = np.full((16, 16), 42.5, dtype=np.float32)
    p = tmp_path / "dtm.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=16, width=16, count=1,
        dtype="float32", crs=UTM,
        transform=from_origin(500000, 4000000, 30.0, 30.0),
    ) as dst:
        dst.write(dem, 1)

    res = LocalFileProvider(p).fetch(_request(), tmp_path / "cache")
    assert res.mosaic_path == p
    assert res.dem_source == "local:dtm.tif"
    assert res.resolution_m == pytest.approx(30.0)
    assert res.synthetic is False
    prov = res.provenance()
    assert prov["dem_source"] == "local:dtm.tif"
    assert prov["dem_synthetic"] is False


def test_local_provider_missing_file_is_explicit(tmp_path):
    with pytest.raises(DWStatusError) as ei:
        LocalFileProvider(tmp_path / "nope.tif").fetch(_request(), tmp_path)
    assert ei.value.code == DEM_UNAVAILABLE


def test_mock_provider_labels_synthetic_and_never_runs_in_production(tmp_path):
    res = MockDEMProvider().fetch(_request(), tmp_path)
    assert res.synthetic is True
    prov = res.provenance()
    assert "synthetic" in prov["dem_source"]
    assert prov["dem_synthetic"] is True
    assert rasterio.open(res.mosaic_path).read(1).max() > 0

    # production paths refuse the mock provider
    with pytest.raises(DWStatusError):
        resolve_reference_dem(
            None, PROVIDER_MOCK, UTM, BOUNDS, tmp_path, allow_mock=False
        )
    # tests may opt in
    res2 = resolve_reference_dem(
        None, PROVIDER_MOCK, UTM, BOUNDS, tmp_path, allow_mock=True
    )
    assert res2.synthetic is True


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_resolve_none_returns_nothing(tmp_path):
    assert resolve_reference_dem(None, "none", UTM, BOUNDS, tmp_path) is None
    assert resolve_reference_dem(None, None, UTM, BOUNDS, tmp_path) is None


def test_resolve_explicit_path_wins(tmp_path):
    p = tmp_path / "manual.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=8, width=8, count=1,
        dtype="float32", crs=UTM,
        transform=from_origin(500000, 4000000, 30.0, 30.0),
    ) as dst:
        dst.write(np.zeros((8, 8), np.float32), 1)
    res = resolve_reference_dem(p, "copernicus", UTM, BOUNDS, tmp_path)
    assert res.mosaic_path == p  # the manual DEM wins over any provider


def test_resolve_unknown_provider_is_explicit(tmp_path):
    with pytest.raises(DWStatusError) as ei:
        resolve_reference_dem(None, "magical-dem", UTM, BOUNDS, tmp_path)
    assert ei.value.code == DEM_UNAVAILABLE


def test_resolve_auto_requires_crs(tmp_path):
    with pytest.raises(DWStatusError) as ei:
        resolve_reference_dem(
            None, "copernicus", None, (0, 0, 1, 1), tmp_path, allow_mock=True
        )
    assert ei.value.code == CRS_REQUIRED


# ---------------------------------------------------------------------------
# Copernicus provider: network failure is explicit; cache avoids re-download
# ---------------------------------------------------------------------------

def test_copernicus_network_failure_is_explicit_not_fake(tmp_path, monkeypatch):
    """A network failure raises DEM_UNAVAILABLE — no synthetic substitution."""
    import rasterio.errors

    def boom(*a, **k):
        raise rasterio.errors.RasterioIOError("connection refused (offline)")

    monkeypatch.setattr(rasterio, "open", boom)
    with pytest.raises(DWStatusError) as ei:
        CopernicusDem30mProvider().fetch(_request(), tmp_path)
    assert ei.value.code == DEM_UNAVAILABLE
    assert "copernicus" in ei.value.detail.lower()


def test_copernicus_provider_downloads_caches_and_mosaics(tmp_path, monkeypatch):
    """Windowed reads against a stubbed remote COG -> cached tiles + mosaic.

    The stub simulates exactly what the real COG offers (windowed reads,
    EPSG:4326, 30 m nominal resolution) without touching the network.
    """
    calls = {"open": 0}

    class FakeRemote:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __init__(self, url):
            assert url.endswith(".tif")
            calls["open"] += 1
            # real GLO-30 geometry: 3600x3600 px per 1-degree tile
            self.width = 3600
            self.height = 3600
            self.count = 1
            self.crs = CRS.from_epsg(4326)
            # tile origin derived from the URL (Copernicus naming)
            import re
            m = re.search(r"_([NS])(\d+)_00_([EW])(\d+)_00_DEM", url)
            lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
            lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
            self.transform = from_origin(float(lon), float(lat + 1),
                                         1.0 / self.width, 1.0 / self.height)

        def read(self, band, window=None):
            return np.full(
                (int(window.height), int(window.width)), 77.0, dtype=np.float32
            )

        def window_transform(self, window):
            from rasterio.transform import from_origin as _fo

            # top-left of the window: origin + offset * pixel size
            a, e = self.transform.a, self.transform.e
            return _fo(
                self.transform.c + window.col_off * a,
                self.transform.f + window.row_off * e,
                a, e,
            )

    real_open = rasterio.open

    def fake_open(path, *a, **k):
        if str(path).startswith("https://"):
            return FakeRemote(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(rasterio, "open", fake_open)

    provider = CopernicusDem30mProvider()
    res1 = provider.fetch(_request(), tmp_path)
    assert res1.provider == "copernicus"
    assert res1.resolution_m == pytest.approx(30.0)
    assert res1.synthetic is False
    assert res1.vertical_reference is not None  # EGM2008, from the spec
    with rasterio.open(res1.mosaic_path) as src:
        assert src.crs == CRS.from_epsg(4326)
        data = src.read(1)
        assert np.isfinite(data).all()
    n_first = calls["open"]

    # second fetch over the same footprint: mosaic cache hit, no network
    res2 = provider.fetch(_request(), tmp_path)
    assert res2.mosaic_path == res1.mosaic_path
    assert calls["open"] == n_first  # zero additional remote opens
    assert res2.extra.get("cached") is True
