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

from backend.app.api.routes._deps import require_scene as _require_scene
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
    """Build a browser-accessible result asset: an absolute short-lived
    presigned URL when the object store is active (loads fine from <img>/
    fetch without headers), else the legacy API-relative path."""
    from backend.app.storage.service import storage_service

    url = storage_service.presign_artifact(scene_id, path) or (
        f"/api/v1/scenes/{scene_id}/results/{name}"
    )
    return ResultAsset(
        name=name,
        url=url,
        format=path.suffix.lstrip("."),
        size_bytes=path.stat().st_size,
    )


@router.get("/{scene_id}/results", response_model=SceneResultsResponse)
async def get_scene_results(scene_id: str):
    _require_scene(scene_id)

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
    _require_scene(scene_id)

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
    from backend.app.storage.service import storage_service

    preview_url = (
        storage_service.presign_artifact(scene_id, preview_path)
        if preview_path is not None
        else None
    ) or (
        f"/api/v1/scenes/{scene_id}/results/preview"
        if preview_path is not None
        else None
    )
    depth_url = (
        storage_service.presign_artifact(scene_id, depth_path)
        or f"/api/v1/scenes/{scene_id}/results/depth"
    )
    return DepthMetaResponse(
        scene_id=scene_id,
        available=True,
        url=preview_url,
        download_url=depth_url,
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
    _require_scene(scene_id)

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
    from backend.app.storage.service import storage_service

    return DsmMetaResponse(
        scene_id=scene_id,
        available=True,
        url=(
            # Greyscale data texture — the viewer's shader colormaps the red
            # channel, so the matplotlib preview would render wrong colours.
            # With the object store this is a presigned URL, generated for
            # the texture the terrain service materializes on demand.
            storage_service.presign_artifact(
                scene_id, result_service.get_output_dir(scene_id) / "dsm_layer.png"
            )
            or f"/api/v1/scenes/{scene_id}/results/dsm-texture"
        ),
        download_url=(
            storage_service.presign_artifact(scene_id, dsm_path)
            or f"/api/v1/scenes/{scene_id}/results/dsm"
        ),
        format="tif" if dsm_path.suffix == ".tif" else "npy",
        width=metadata["width"],
        height=metadata["height"],
        crs=metadata["crs"],
    )
