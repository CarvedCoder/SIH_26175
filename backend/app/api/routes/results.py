from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.schemas.result import (
    DSMResult,
    DepthResult,
    ElevationStats,
    ResultAsset,
    ResultCapabilities,
    SceneResultsResponse,
)
from backend.app.services.result_service import result_service
from backend.app.jobs.manager import job_manager


router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Results"],
)


def _asset(
    name: str,
    scene_id: str,
    path: Path,
) -> ResultAsset:
    """Build a browser-accessible result asset."""

    url = f"/api/v1/scenes/{scene_id}/results/{name}"

    return ResultAsset(
        name=name,
        url=url,
        format=path.suffix.lstrip("."),
        size_bytes=path.stat().st_size,
    )


@router.get(
    "/{scene_id}/results",
    response_model=SceneResultsResponse,
)
async def get_scene_results(scene_id: str):
    """Return structured results for a processed scene."""

    files = result_service.get_result_files(scene_id)

    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for scene '{scene_id}'.",
        )

    depth = DepthResult()
    dsm = DSMResult()

    preview_path = files.get("preview")

    job = job_manager.get_latest_job_for_scene(scene_id)

    if preview_path is not None:
        depth.preview = _asset(
            "preview",
            scene_id,
            preview_path,
        )

        dsm.preview = _asset(
            "preview",
            scene_id,
            preview_path,
        )

    depth_path = files.get("depth")

    if depth_path is not None:
        stats = result_service._load_array_stats(depth_path)

        depth.available = True
        depth.raw = _asset(
            "depth",
            scene_id,
            depth_path,
        )
        depth.width = stats["width"]
        depth.height = stats["height"]
        depth.statistics = ElevationStats(
            minimum=stats["minimum"],
            maximum=stats["maximum"],
            mean=stats["mean"],
            median=stats["median"],
            relief=stats["relief"],
            units=stats["units"],
        )

    dsm_path = files.get("dsm")

    if dsm_path is not None:
        metadata = result_service._read_dsm_metadata(dsm_path)

        dsm.available = True
        dsm.raster = _asset(
            "dsm",
            scene_id,
            dsm_path,
        )
        dsm.width = metadata["width"]
        dsm.height = metadata["height"]
        dsm.crs = metadata["crs"]
        dsm.bounds = metadata["bounds"]

        if depth.statistics is not None:
            dsm.statistics = depth.statistics

    capabilities = ResultCapabilities(
        depth=depth.available,
        dsm=dsm.available,
        terrain=False,
        validation=False,
        reference_comparison=False,
        export=True,
    )

    return SceneResultsResponse(
        scene_id=scene_id,
        job_id=job.job_id if job is not None else "unknown",
        status=job.status if job is not None else "completed",
        depth=depth,
        dsm=dsm,
        capabilities=capabilities,
        assets=[
            _asset(name, scene_id, path)
            for name, path in files.items()
        ],
    )