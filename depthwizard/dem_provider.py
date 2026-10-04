"""Pluggable DEM acquisition subsystem (absolute-DSM Part A).

Architecture:
    DEMProvider (abstract)
        |-- LocalFileProvider        user-supplied DEM raster (manual path)
        |-- CopernicusDem30mProvider Copernicus GLO-30 public COGs on AWS
        |-- MockDEMProvider          TEST-ONLY synthetic DEM (never real data)

The provider's job is ACQUISITION ONLY: given the scene's geographic
footprint it downloads/copies the required DEM tiles into a local cache,
mosaics them, clips to the requested extent, and returns
:class:`DEMResult` — the mosaic path plus a complete provenance record.

REPROJECT/RESAMPLE onto the scene grid is deliberately NOT done here: it
happens once, in :mod:`depthwizard.absolute_dsm` /
:mod:`depthwizard.anchoring`, on the exact image grid, with a hard
full-coverage gate. Keeping acquisition and grid-alignment separate means
the alignment contract (CRS/transform/dimensions preserved, no
broadcasting) has exactly one implementation.

Honesty rules (frozen):
  * Metadata is NEVER fabricated. Fields that cannot be verified are
    recorded as None/"unknown", never guessed.
  * Network failures raise DWStatusError(DEM_UNAVAILABLE) — the caller
    then continues with the RELATIVE product after printing an explicit
    notice. A fake DEM is never substituted.
  * The mock provider writes ``dem_source: "mock (synthetic)"`` into its
    provenance and is rejected by resolve_reference_dem() unless the
    caller explicitly passes ``allow_mock=True`` (tests only).
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .statuses import (
    CRS_REQUIRED,
    DEM_UNAVAILABLE,
    DWStatusError,
)

# Copernicus GLO-30 public COGs on the AWS Open Data registry (no account,
# no key; HTTPS reads). Vertical datum per the ESA GLO-30 product
# specification: EGM2008 geoid.
COPERNICUS_AWS_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
COPERNICUS_NOMINAL_RESOLUTION_M = 30.0
COPERNICUS_VERTICAL_REFERENCE = "EGM2008 geoid (per ESA GLO-30 product specification)"

PROVIDER_COPERNICUS = "copernicus"
PROVIDER_LOCAL = "local"
PROVIDER_MOCK = "mock"


@dataclass
class DEMRequest:
    """Scene footprint a provider must cover.

    bounds are (west, south, east, north) in ``scene_crs``.
    """

    west: float
    south: float
    east: float
    north: float
    scene_crs: object  # rasterio CRS
    scene_gsd_m: Optional[float] = None  # nominal scene resolution if known

    @property
    def key(self) -> str:
        """Stable cache key for this footprint (8 hex chars)."""
        raw = f"{self.west:.6f},{self.south:.6f},{self.east:.6f},{self.north:.6f},{self.scene_crs}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


@dataclass
class DEMResult:
    """One provider's acquisition outcome + full provenance."""

    mosaic_path: Path  # clipped, mosaiced DEM raster on disk (source CRS)
    provider: str  # "copernicus" | "local" | "mock"
    dem_source: str  # provenance: human label recorded downstream
    crs: object  # CRS of the mosaic (usually the provider's native CRS)
    resolution_m: Optional[float]  # nominal resolution in metres, if known
    vertical_reference: Optional[str]  # e.g. EGM2008 — None when unknown
    identifier: Optional[str] = None  # product/tile identifier(s)
    acquisition_timestamp: Optional[str] = None  # UTC ISO, when known
    tile_ids: List[str] = field(default_factory=list)
    synthetic: bool = False  # True ONLY for the test-only mock provider
    extra: Dict = field(default_factory=dict)

    def provenance(self) -> Dict:
        """Machine-readable provenance block (Part C). Unknown -> None."""
        return {
            "dem_source": self.dem_source,
            "dem_identifier": self.identifier,
            "dem_crs": str(self.crs) if self.crs is not None else None,
            "dem_resolution_m": self.resolution_m,
            "dem_vertical_reference": self.vertical_reference,
            "dem_provider": self.provider,
            "dem_tiles": list(self.tile_ids),
            "dem_acquisition_timestamp_utc": self.acquisition_timestamp,
            "dem_synthetic": self.synthetic,
        }


class DEMProvider(ABC):
    """Interface every DEM source implements. Pluggable by design."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, request: DEMRequest, cache_dir: Path) -> DEMResult:
        """Acquire + mosaic the DEM for the requested footprint.

        Implementations must either succeed with a mosaic that fully
        covers the footprint or raise DWStatusError — never return a
        partial cover and never fabricate missing data.
        """


# ---------------------------------------------------------------------------
# Copernicus GLO-30 on AWS (real public terrain source)
# ---------------------------------------------------------------------------


def _tile_name(lat_deg: int, lon_deg: int) -> str:
    """Copernicus GLO-30 tile identifier for a 1x1 degree cell origin.

    e.g. lat=32, lon=74 -> 'Copernicus_DSM_COG_10_N32_00_E074_00_DEM'
    """
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_"
        f"{ew}{abs(lon_deg):03d}_00_DEM"
    )


def copernicus_tile_url(tile_name: str) -> str:
    return f"{COPERNICUS_AWS_URL}/{tile_name}/{tile_name}.tif"


def required_copernicus_tiles(request: DEMRequest) -> List[tuple]:
    """1x1-degree cell origins (lat, lon) covering the scene footprint.

    Pure geometry — no network. The footprint is reprojected to EPSG:4326
    first; every intersecting whole-degree cell is returned (conservative:
    border-straddling footprints pull the neighbouring tiles too).
    """
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds(request.scene_crs, "EPSG:4326",
                                  request.west, request.south,
                                  request.east, request.north,
                                  densify_pts=21)
    lat0, lat1 = math.floor(s), math.ceil(n)
    lon0, lon1 = math.floor(w), math.ceil(e)
    cells = []
    for lat in range(lat0, lat1):
        for lon in range(lon0, lon1):
            cells.append((lat, lon))
    return cells


class CopernicusDem30mProvider(DEMProvider):
    """Copernicus GLO-30 DEM from the AWS Open Data registry.

    Per-tile clipped windows are cached under ``<cache_dir>/copernicus30m/``;
    the per-request mosaic under ``<cache_dir>/copernicus30m/mosaics/``.
    A repeat request over the same footprint is served entirely from disk
    (no network) — the DEM download adds latency only on the first run.
    """

    name = PROVIDER_COPERNICUS

    def fetch(self, request: DEMRequest, cache_dir: Path) -> DEMResult:
        import datetime as _dt

        import rasterio
        from rasterio.merge import merge as merge_datasets
        from rasterio.windows import from_bounds as window_from_bounds
        from rasterio.warp import transform_bounds

        cache = Path(cache_dir) / "copernicus30m"
        tile_cache = cache / "tiles"
        mosaic_dir = cache / "mosaics"
        mosaic_dir.mkdir(parents=True, exist_ok=True)

        mosaic_path = mosaic_dir / f"cop30_{request.key}.tif"
        if mosaic_path.exists():
            with rasterio.open(mosaic_path) as src:
                return DEMResult(
                    mosaic_path=mosaic_path,
                    provider=self.name,
                    dem_source="copernicus",
                    crs=src.crs,
                    resolution_m=COPERNICUS_NOMINAL_RESOLUTION_M,
                    vertical_reference=COPERNICUS_VERTICAL_REFERENCE,
                    identifier="Copernicus GLO-30 (AWS Open Data)",
                    tile_ids=list(src.tags(1).get("TILE_IDS", "").split(",")),
                    extra={"cached": True},
                )

        w, s, e, n = transform_bounds(request.scene_crs, "EPSG:4326",
                                      request.west, request.south,
                                      request.east, request.north,
                                      densify_pts=21)
        tile_paths: List[Path] = []
        tile_ids: List[str] = []
        all_cached = True
        for lat, lon in required_copernicus_tiles(request):
            name = _tile_name(lat, lon)
            tile_ids.append(name)
            local = tile_cache / f"{name}_{request.key}.tif"
            if not local.exists():
                all_cached = False
                tile_cache.mkdir(parents=True, exist_ok=True)
                url = copernicus_tile_url(name)
                # cell bounds in EPSG:4326, clipped to the scene footprint
                cw = max(w, float(lon))
                ce = min(e, float(lon + 1))
                cs = max(s, float(lat))
                cn = min(n, float(lat + 1))
                if cw >= ce or cs >= cn:
                    continue
                try:
                    with rasterio.open(url) as src:
                        win = window_from_bounds(cw, cs, ce, cn, src.transform)
                        # Snap to the pixel grid, then EXPAND by a 1-px
                        # margin on every side (clamped to the raster):
                        # pixel snapping must never truncate the requested
                        # footprint — a truncated window would fail the
                        # downstream full-coverage gate by border pixels.
                        # Extra margin is harmless (alignment re-crops).
                        from rasterio.windows import Window

                        c0 = max(0, math.floor(win.col_off) - 1)
                        r0 = max(0, math.floor(win.row_off) - 1)
                        c1 = min(src.width, math.ceil(win.col_off + win.width) + 1)
                        r1 = min(src.height, math.ceil(win.row_off + win.height) + 1)
                        win = Window(
                            col_off=c0, row_off=r0,
                            width=c1 - c0, height=r1 - r0,
                        )
                        data = src.read(1, window=win)
                        transform = src.window_transform(win)
                        profile = {
                            "driver": "GTiff", "height": data.shape[0],
                            "width": data.shape[1], "count": 1,
                            "dtype": data.dtype, "crs": src.crs,
                            "transform": transform, "compress": "deflate",
                        }
                    with rasterio.open(local, "w", **profile) as dst:
                        dst.write(data, 1)
                except Exception as exc:  # noqa: BLE001 — network/IO all explicit
                    raise DWStatusError(
                        DEM_UNAVAILABLE,
                        f"Copernicus GLO-30 tile '{name}' could not be "
                        f"retrieved ({type(exc).__name__}: {exc}). Absolute "
                        "DSM requires an externally referenced DEM — pass "
                        "--anchor-dem <DEM.tif> for manual input; continuing "
                        "with the relative (AGL) product otherwise.",
                    ) from exc
            tile_paths.append(local)

        if not tile_paths:
            raise DWStatusError(
                DEM_UNAVAILABLE,
                "no Copernicus GLO-30 tiles intersect the scene footprint — "
                "the footprint may be invalid; refusing to fabricate a DEM.",
            )

        cached = all_cached

        if len(tile_paths) == 1:
            mosaic_path = tile_paths[0]
        else:
            datasets = [rasterio.open(p) for p in tile_paths]
            try:
                mosaic, mosaic_transform = merge_datasets(datasets)
            finally:
                for ds in datasets:
                    ds.close()
            profile = {
                "driver": "GTiff", "height": mosaic.shape[1],
                "width": mosaic.shape[2], "count": 1,
                "dtype": mosaic.dtype, "crs": datasets[0].crs,
                "transform": mosaic_transform, "compress": "deflate",
            }
            with rasterio.open(mosaic_path, "w", **profile) as dst:
                dst.write(mosaic, 1)
                dst.update_tags(1, TILE_IDS=",".join(tile_ids))

        return DEMResult(
            mosaic_path=mosaic_path,
            provider=self.name,
            dem_source="copernicus",
            crs="EPSG:4326",
            resolution_m=COPERNICUS_NOMINAL_RESOLUTION_M,
            vertical_reference=COPERNICUS_VERTICAL_REFERENCE,
            identifier="Copernicus GLO-30 (AWS Open Data)",
            acquisition_timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            tile_ids=tile_ids,
            extra={"cached": bool(cached)},
        )


# ---------------------------------------------------------------------------
# Local file (manual --anchor-dem path) + test-only mock
# ---------------------------------------------------------------------------


class LocalFileProvider(DEMProvider):
    """Wraps a user-supplied DEM raster (the --anchor-dem manual path).

    Reads (never writes) the file; provenance comes from the raster itself.
    """

    name = PROVIDER_LOCAL

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def fetch(self, request: DEMRequest, cache_dir: Path) -> DEMResult:
        import rasterio

        if not self.path.exists():
            raise DWStatusError(
                DEM_UNAVAILABLE, f"DEM file not found: {self.path}"
            )
        with rasterio.open(self.path) as src:
            gsd = None
            if src.transform is not None:
                gsd = float(abs(src.transform.a))
            # geographic CRS: pixel size is degrees, not metres — report the
            # nominal 1-arc-second family value only when recognisable,
            # otherwise None (unknown, not guessed).
            if src.crs is not None and getattr(src.crs, "is_geographic", False):
                gsd = None
            return DEMResult(
                mosaic_path=self.path,
                provider=self.name,
                dem_source=f"local:{self.path.name}",
                crs=src.crs,
                resolution_m=gsd,
                vertical_reference=src.tags(1).get("VERTICAL_DATUM") or None,
                identifier=str(self.path),
            )


class MockDEMProvider(DEMProvider):
    """TEST-ONLY provider: writes a deterministic synthetic DEM.

    The output is labelled ``dem_source: "mock (synthetic)"`` and
    ``synthetic: True`` everywhere it appears. It exists so the
    acquisition->alignment->absolute-DSM machinery can be exercised in
    CI without network access. Production code paths must never select
    it (resolve_reference_dem refuses it without allow_mock=True).
    """

    name = PROVIDER_MOCK

    def __init__(self, value=lambda x, y: 50.0 + 10.0 * x + 5.0 * y):
        self._value = value

    def fetch(self, request: DEMRequest, cache_dir: Path) -> DEMResult:
        import rasterio
        from rasterio.transform import from_bounds

        import numpy as np

        # 128 px across the footprint, in the scene CRS (exercises the
        # reprojection path downstream exactly like a real remote DEM).
        n = 128
        transform = from_bounds(
            request.west, request.south, request.east, request.north, n, n
        )
        xs = np.linspace(0.0, 1.0, n, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, n, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        dem = self._value(gx, gy).astype(np.float32)

        out = Path(cache_dir) / "mock" / f"mock_dem_{request.key}.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out, "w", driver="GTiff", height=n, width=n, count=1,
            dtype="float32", crs=request.scene_crs, transform=transform,
        ) as dst:
            dst.write(dem, 1)
        return DEMResult(
            mosaic_path=out,
            provider=self.name,
            dem_source="mock (synthetic)",
            crs=request.scene_crs,
            resolution_m=float(abs(transform.a)),
            vertical_reference=None,
            identifier="synthetic test DEM",
            synthetic=True,
        )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

PROVIDERS = {
    PROVIDER_COPERNICUS: CopernicusDem30mProvider,
}


def resolve_reference_dem(
    dem_path: Path | str | None,
    provider: str | None,
    scene_crs,
    scene_bounds: tuple,
    cache_dir: Path | str,
    scene_gsd_m: float | None = None,
    allow_mock: bool = False,
) -> Optional[DEMResult]:
    """Resolve the reference-DEM acquisition request for one scene.

    ``dem_path`` wins (manual input, LocalFileProvider). Otherwise a named
    automatic provider is used. Returns None when neither is requested —
    the caller then stays with the RELATIVE product. Raises DWStatusError
    on acquisition failure (never returns fake data).
    """
    if dem_path is not None:
        return LocalFileProvider(dem_path).fetch(
            DEMRequest(
                west=scene_bounds[0], south=scene_bounds[1],
                east=scene_bounds[2], north=scene_bounds[3],
                scene_crs=scene_crs, scene_gsd_m=scene_gsd_m,
            ),
            Path(cache_dir),
        )
    if provider in (None, "", "none"):
        return None
    if provider == PROVIDER_MOCK:
        if not allow_mock:
            raise DWStatusError(
                DEM_UNAVAILABLE,
                "the 'mock' DEM provider is test-only and refuses to run in "
                "production paths — pass a real provider or --anchor-dem.",
            )
        prov: DEMProvider = MockDEMProvider()
    elif provider in PROVIDERS:
        prov = PROVIDERS[provider]()
    else:
        raise DWStatusError(
            DEM_UNAVAILABLE,
            f"unknown DEM provider '{provider}' — available: "
            f"{sorted(PROVIDERS)} (plus 'mock' for tests).",
        )
    request = DEMRequest(
        west=scene_bounds[0], south=scene_bounds[1],
        east=scene_bounds[2], north=scene_bounds[3],
        scene_crs=scene_crs, scene_gsd_m=scene_gsd_m,
    )
    if request.scene_crs is None:
        raise DWStatusError(
            CRS_REQUIRED,
            "automatic DEM acquisition requires a georeferenced input "
            "(the scene footprint is unknown without a CRS).",
        )
    return prov.fetch(request, Path(cache_dir))
