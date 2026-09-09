"""Processed-scene result routes — honest state, typed responses.

The audit flagged fabricated result fields (``job_id: "unknown"`` and
``status: "completed"`` when no job existed). The rule now: the response
reports the ACTUAL state —
    * a job record exists  -> its real status and id;
    * result files exist but no job record (e.g. jobs were evicted or the
      process restarted) -> ``status: "completed"`` (the artifacts ARE the
      completed products) with ``job_id: None`` — never an invented id;
    * nothing exists       -> 404.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from backend.app.core.errors import AppError
from backend.app.jobs.manager import job_manager
from backend.app.schemas.result import (
    DSMResult,
    DepthMetaResponse,
    DepthResult,
    DsmMetaResponse,
    ElevationStats,
    ResultAsset,
    ResultCapabilities,
    SceneResultsResponse,
)
from backend.app.services.result_service import result_service

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Results"],
)


def _asset(name: str, scene_id: str, path: Path) -> ResultAsset:
    """Build a browser-accessible result asset."""
    url = f"/api/v1/scenes/{scene_id}/results/{name}"
    return ResultAsset(
        name=name,
        url=url,
        format=path.suffix.lstrip("."),
        size_bytes=path.stat().st_size,
    )


@router.get("/{scene_id}/results", response_model=SceneResultsResponse)
async def get_scene_results(scene_id: str):
    """Return structured results for a processed scene."""

    files = result_service.get_result_files(scene_id)
    if not files:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No results exist for this scene yet.",
            details={"scene_id": scene_id},
            recoverable=True,
        )

    job = job_manager.get_latest_job_for_scene(scene_id)

    if job is not None:
        job_id: str | None = job.job_id
        status = job.status
    else:
        # No job record (evicted/restart) but the artifacts exist: report
        # the real state — completed products, unknown provenance. The
        # null job_id is honest: we do not know which run produced them.
        job_id = None
        status = "completed"

    depth = DepthResult()
    dsm = DSMResult()

    preview_path = files.get("preview")
    if preview_path is not None:
        depth.preview = _asset("preview", scene_id, preview_path)
        dsm.preview = _asset("preview", scene_id, preview_path)

    depth_path = files.get("depth")
    if depth_path is not None:
        stats = result_service._load_array_stats(depth_path)
        depth.available = True
        depth.raw = _asset("depth", scene_id, depth_path)
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
        dsm.raster = _asset("dsm", scene_id, dsm_path)
        dsm.width = metadata["width"]
        dsm.height = metadata["height"]
        dsm.crs = metadata["crs"]
        dsm.bounds = metadata["bounds"]
        if depth.statistics is not None:
            dsm.statistics = depth.statistics

    validation_available = any(
        (result_service.get_output_dir(scene_id) / name).is_file()
        for name in ("validation.json", "validation_metrics.json", "error_map.png")
    )

    capabilities = ResultCapabilities(
        depth=depth.available,
        dsm=dsm.available,
        terrain=files.get("dsm") is not None,
        validation=validation_available,
        reference_comparison=validation_available,
        export=True,
    )

    return SceneResultsResponse(
        scene_id=scene_id,
        job_id=job_id,
        status=status,
        depth=depth,
        dsm=dsm,
        capabilities=capabilities,
        assets=[
            _asset(name, scene_id, path) for name, path in files.items()
        ],
    )


@router.get("/{scene_id}/depth", response_model=DepthMetaResponse)
async def get_depth_meta(scene_id: str):
    """Depth-layer metadata: preview (visual) + raw array download URLs."""
    files = result_service.get_result_files(scene_id)
    depth_path = files.get("depth")
    if depth_path is None:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No depth results exist for this scene yet — process it first.",
            details={"scene_id": scene_id},
            recoverable=True,
        )

    stats = result_service._load_array_stats(depth_path)
    preview_path = files.get("preview")
    return DepthMetaResponse(
        scene_id=scene_id,
        available=True,
        url=(
            f"/api/v1/scenes/{scene_id}/results/preview"
            if preview_path is not None
            else None
        ),
        download_url=f"/api/v1/scenes/{scene_id}/results/depth",
        format="npy",
        width=stats["width"],
        height=stats["height"],
        statistics=ElevationStats(
            minimum=stats["minimum"],
            maximum=stats["maximum"],
            mean=stats["mean"],
            median=stats["median"],
            relief=stats["relief"],
            units=stats["units"],
        ),
    )


@router.get("/{scene_id}/dsm", response_model=DsmMetaResponse)
async def get_dsm_meta(scene_id: str):
    """DSM-layer metadata: preview (visual) + GeoTIFF download URLs."""
    files = result_service.get_result_files(scene_id)
    dsm_path = files.get("dsm")
    if dsm_path is None:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No DSM results exist for this scene yet — process it first.",
            details={"scene_id": scene_id},
            recoverable=True,
        )

    metadata = result_service._read_dsm_metadata(dsm_path)
    preview_path = files.get("preview")
    return DsmMetaResponse(
        scene_id=scene_id,
        available=True,
        url=(
            f"/api/v1/scenes/{scene_id}/results/preview"
            if preview_path is not None
            else None
        ),
        download_url=f"/api/v1/scenes/{scene_id}/results/dsm",
        format="tif",
        width=metadata["width"],
        height=metadata["height"],
        crs=metadata["crs"],
    )
