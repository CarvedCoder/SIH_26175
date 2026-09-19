"""Job orchestration routes: process, poll, cancel.

Contract fixes from the audit (H2 + C5):
    * POST /process validates the scene EXISTS before creating a job —
      unknown scenes get a typed 404, never a doomed background job;
    * a scene with an already-active job gets 409 SCENE_BUSY instead of a
      silent duplicate;
    * the request body is a strict Pydantic model (unsupported options are
      422s, not silent ignores);
    * job responses are typed (JobResponse), including cancellation state.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from backend.app.api.routes.scenes import _require_scene
from backend.app.core.auth import current_user, ensure_owner
from backend.app.core.errors import AppError, SceneBusy
from backend.app.jobs.manager import job_manager
from backend.app.schemas.job import (
    CancelResponse,
    JobResponse,
    JobStage,
    JobStatus,
)
from backend.app.schemas.processing import (
    ProcessAccepted,
    ProcessRequest,
    RefineAccepted,
    RefineRequest,
)
from backend.app.services.processing_service import processing_service

router = APIRouter(
    tags=["Jobs"],
)


def _run_processing(job_id: str, scene_id: str, request: ProcessRequest) -> None:
    """Background entry point (threadpool): run the certified pipeline."""
    try:
        processing_service.process_scene(
            job_id=job_id,
            scene_id=scene_id,
            mode=request.mode.value,
            ground_elev=request.ground_elev,
        )
    except Exception as exc:
        # record the typed job error; full detail is logged server-side
        processing_service.record_failure(job_id, exc)


def _run_refinement(job_id: str, scene_id: str, request: RefineRequest) -> None:
    try:
        processing_service.refine_region(
            job_id=job_id,
            scene_id=scene_id,
            bbox=(
                request.bbox.x_min,
                request.bbox.y_min,
                request.bbox.x_max,
                request.bbox.y_max,
            ),
        )
    except Exception as exc:
        processing_service.record_failure(job_id, exc)


@router.post("/api/v1/scenes/{scene_id}/process", response_model=ProcessAccepted)
async def process_scene(
    scene_id: str,
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """Create an asynchronous DepthWizard processing job for a scene."""
    _require_scene(scene_id)

    active = job_manager.get_active_job_for_scene(scene_id)
    if active is not None:
        raise SceneBusy(scene_id, active.job_id)

    job = job_manager.create_job(scene_id, owner_id=current_user().user_id)

    background_tasks.add_task(_run_processing, job.job_id, scene_id, request)

    return ProcessAccepted(
        job_id=job.job_id,
        scene_id=job.scene_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
    )


@router.post("/api/v1/scenes/{scene_id}/refine", response_model=RefineAccepted)
async def refine_scene(
    scene_id: str,
    request: RefineRequest,
    background_tasks: BackgroundTasks,
):
    """Create a local refinement job re-processing a bbox of the scene."""
    _require_scene(scene_id)

    active = job_manager.get_active_job_for_scene(scene_id)
    if active is not None:
        raise SceneBusy(scene_id, active.job_id)

    job = job_manager.create_job(scene_id, owner_id=current_user().user_id)
    background_tasks.add_task(_run_refinement, job.job_id, scene_id, request)

    return RefineAccepted(
        job_id=job.job_id,
        scene_id=job.scene_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
    )


@router.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Return the current status of a processing job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise AppError(
            status_code=404,
            code="JOB_NOT_FOUND",
            message="Job does not exist or has expired.",
            recoverable=False,
        )
    ensure_owner(job.owner_id, f"job:{job.job_id}")

    return JobResponse(
        job_id=job.job_id,
        scene_id=job.scene_id,
        status=JobStatus(job.status),
        stage=JobStage(job.stage),
        progress=job.progress,
        message=job.message,
        cancel_requested=job.cancel_requested,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/api/v1/jobs/{job_id}/cancel", response_model=CancelResponse)
async def cancel_job(job_id: str):
    """Cancel a job. Queued jobs are cancelled immediately; running jobs
    are flagged and end cancelled at the next cooperative checkpoint."""
    job = job_manager.request_cancel(job_id)
    if job is None:
        raise AppError(
            status_code=404,
            code="JOB_NOT_FOUND",
            message="Job does not exist or has expired.",
            recoverable=False,
        )
    ensure_owner(job.owner_id, f"job:{job.job_id}")

    return CancelResponse(
        job_id=job.job_id,
        scene_id=job.scene_id,
        status=JobStatus(job.status),
        cancel_requested=job.cancel_requested,
    )
