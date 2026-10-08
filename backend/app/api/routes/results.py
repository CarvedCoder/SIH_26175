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
        analysis_status=_analysis_status(job),
        depth=depth,
        dsm=dsm,
        capabilities=capabilities,
        assets=[
            _asset(name, scene_id, path) for name, path in files.items()
        ],
        # Processing provenance from the job payload meta (height model,
        # output_type, absolute-reference availability, DEM provenance,
        # GSD). Empty when no job record exists — never reconstructed
        # from guesses.
        metadata=_provenance_meta(job),
    )


def _provenance_meta(job) -> dict:
    """The payload meta block of the latest job, if a job record exists."""
    if job is None:
        return {}
    result = getattr(job, "result", None)
    if not isinstance(result, dict):
        return {}
    meta = result.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _analysis_status(job) -> str:
    """The latest job's deferred-analysis state; legacy payloads (no flag)
    and missing job records report "complete" — never a fake pending."""
    if job is None:
        return "complete"
    result = getattr(job, "result", None)
    if isinstance(result, dict):
        status = result.get("analysis_status")
        if isinstance(status, str) and status:
            return status
    return "complete"


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
    from backend.app.storage.service import storage_service

    # ``url`` is the BROWSER-RENDERABLE texture: a clean Pillow-generated
    # greyscale PNG of the depth array (no axes/colourbar/border). The
    # Matplotlib dsm_preview.png is a diagnostic/download artifact and is
    # never used as an interactive WebGL texture.
    depth_layer_path = None
    try:
        from backend.app.services.terrain_service import terrain_service

        depth_layer_path = terrain_service.get_depth_layer_path(scene_id)
    except (FileNotFoundError, ValueError):
        depth_layer_path = None

    texture_url = (
        storage_service.presign_artifact(scene_id, depth_layer_path)
        if depth_layer_path is not None
        else None
    ) or (
        f"/api/v1/scenes/{scene_id}/results/depth-texture"
        if depth_layer_path is not None
        else None
    )
    depth_url = (
        storage_service.presign_artifact(scene_id, depth_path)
        or f"/api/v1/scenes/{scene_id}/results/depth"
    )
    return DepthMetaResponse(
        scene_id=scene_id,
        available=True,
        url=texture_url,
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
    depth_path = files.get("depth")
    if dsm_path is None and depth_path is None:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No DSM results exist for this scene yet — process it first.",
            details={"scene_id": scene_id},
            recoverable=True,
        )

    # Non-georeferenced scenes have no GeoTIFF twin; the predicted surface
    # array (dsm.npy) is the DSM product and metadata falls back to it.
    metadata = (
        result_service._read_dsm_metadata(dsm_path)
        if dsm_path is not None
        else result_service._read_dsm_metadata(depth_path)
    )
    from backend.app.storage.service import storage_service

    # Greyscale data texture — the viewer's shader colormaps the red
    # channel, so the matplotlib preview would render wrong colours. The
    # texture the terrain service materializes on demand covers BOTH
    # georeferenced (dsm.tif) and array-only scenes (dsm.npy).
    texture_path = None
    try:
        from backend.app.services.terrain_service import terrain_service

        texture_path = terrain_service.get_dsm_layer_path(scene_id)
    except (FileNotFoundError, ValueError):
        texture_path = None
    return DsmMetaResponse(
        scene_id=scene_id,
        available=True,
        url=(
            storage_service.presign_artifact(scene_id, texture_path)
            if texture_path is not None
            else None
        )
        or (
            f"/api/v1/scenes/{scene_id}/results/dsm-texture"
            if texture_path is not None
            else None
        ),
        download_url=(
            (
                storage_service.presign_artifact(scene_id, dsm_path)
                or f"/api/v1/scenes/{scene_id}/results/dsm"
            )
            if dsm_path is not None
            else (
                storage_service.presign_artifact(scene_id, depth_path)
                or f"/api/v1/scenes/{scene_id}/results/depth"
            )
        ),
        format="tif" if (dsm_path is not None and dsm_path.suffix == ".tif") else "npy",
        width=metadata["width"],
        height=metadata["height"],
        crs=metadata["crs"],
    )
