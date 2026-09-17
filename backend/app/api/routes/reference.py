"""Reference elevation route — honest availability only.

A scene has reference elevation data ONLY when a reference raster was
produced for it (e.g. by validation against a reference DEM). The backend
never fabricates a reference source: scenes without one report
``available: false`` with null URLs.
"""

from __future__ import annotations

import rasterio
from fastapi import APIRouter

from backend.app.api.routes.scenes import _require_scene
from backend.app.schemas.reference import ReferenceResponse
from backend.app.services.result_service import result_service

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Reference"],
)

_REFERENCE_CANDIDATES = ("reference.tif", "reference_dem.tif", "ref_dem.tif")
_REFERENCE_ARRAYS = ("reference.npy", "reference_dem.npy", "ref_dem.npy")


@router.get("/{scene_id}/reference", response_model=ReferenceResponse)
async def get_reference(scene_id: str):
    """Return the reference elevation metadata for a scene."""
    _require_scene(scene_id)

    output_dir = result_service.get_output_dir(scene_id)

    raster_ref = next(
        (output_dir / name for name in _REFERENCE_CANDIDATES if (output_dir / name).is_file()),
        None,
    )
    array_ref = next(
        (output_dir / name for name in _REFERENCE_ARRAYS if (output_dir / name).is_file()),
        None,
    )

    if raster_ref is None and array_ref is None:
        return ReferenceResponse(scene_id=scene_id, available=False)

    crs: str | None = None
    if raster_ref is not None:
        with rasterio.open(raster_ref) as ds:
            crs = ds.crs.to_string() if ds.crs else None
        download_url = f"/api/v1/scenes/{scene_id}/results/reference"
        # Browser-renderable greyscale texture (the raw .tif is not
        # decodable by a WebGL texture loader).
        visualization_url = f"/api/v1/scenes/{scene_id}/results/reference-preview"
        name = raster_ref.name
    else:
        download_url = None
        visualization_url = f"/api/v1/scenes/{scene_id}/results/reference-preview"
        name = array_ref.name if array_ref else None

    return ReferenceResponse(
        scene_id=scene_id,
        available=True,
        name=name,
        download_url=download_url,
        visualization_url=visualization_url,
        crs=crs,
        units="meters",
    )
