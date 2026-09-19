"""Job orchestration routes: process, poll, cancel.

Contract fixes from the audit (H2 + C5):
    * POST /process validates the scene EXISTS before creating a job —
      unknown scenes get a typed 404, never a doomed background job;
    * a scene with an already-active job gets 409 SCENE_BUSY instead of a
      silent duplicate;
    * the request body is a strict Pydantic model (unsupported options are
      422s, not silent ignores);
    * job responses are typed (JobResponse), including cancellation state.

Tranche 2 (execution split): the validated request parameters are
PERSISTED on the job record (`Job.request`), so the API process never
needs to survive for the work to happen:
    * DW_WORKER_MODE=inline (default): the InlineTaskQueue dispatches
      execute_job_record on a worker thread (dev parity with the old
      BackgroundTasks behavior);
    * DW_WORKER_MODE=external: the API ONLY records the job — the durable
      record is the queue entry and `python -m backend.app.worker` claims
      and executes it. Any worker, any instance, any time.

Authorization: jobs are owned by their SCENE's owner — the scene SQL row
is the ownership record, and polling/cancelling enforces it (403 for a
non-owner; the Supabase-era contract).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Request

from backend.app.api.routes._deps import require_scene as _require_scene
from backend.app.appstate import task_queue_for
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


def _job_request_payload(request: ProcessRequest | RefineRequest) -> dict:
    """Validated request parameters as a JSON-safe dict (persisted on the
    job record so any worker can execute it)."""
    if isinstance(request, RefineRequest):
        return {
            "kind": "refine",
            "bbox": {
                "x_min": request.bbox.x_min,
                "y_min": request.bbox.y_min,
                "x_max": request.bbox.x_max,
                "y_max": request.bbox.y_max,
            },
        }
    return {
        "kind": "process",
        "mode": request.mode.value,
        "ground_elev": request.ground_elev,
    }


def _dispatch(job_id: str, http_request: Request, background_tasks) -> None:
    """Hand the durable job to the execution layer (inline dispatch in dev
    mode; record-only when DW_WORKER_MODE=external)."""
    queue = task_queue_for(http_request, background_tasks)
    if queue is not None:
        queue.add_task(processing_service.execute_job_record, job_id)
    # external mode: nothing to do — the record IS the queue entry.


def _ensure_job_owner(job) -> None:
    """A job is owned by its scene's owner: the scene SQL row is the
    ownership record; a scene without a row (pre-migration/local) is the
    local owner's."""
    from backend.app.db.database import session_scope
    from backend.app.db.models import SceneRow

    with session_scope() as session:
        row = session.get(SceneRow, job.scene_id)
        owner_id = row.owner_id if row is not None else current_user().user_id
    ensure_owner(owner_id, f"job:{job.job_id}")


@router.post("/api/v1/scenes/{scene_id}/process", response_model=ProcessAccepted)
async def process_scene(
    scene_id: str,
    request: ProcessRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
):
    """Create an asynchronous DepthWizard processing job for a scene."""
    _require_scene(scene_id)

    active = job_manager.get_active_job_for_scene(scene_id)
    if active is not None:
        raise SceneBusy(scene_id, active.job_id)

    job = job_manager.create_job(scene_id)
    job_manager.update_job(job.job_id, request=_job_request_payload(request))
    _dispatch(job.job_id, http_request, background_tasks)

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
    http_request: Request,
    background_tasks: BackgroundTasks,
):
    """Create a local refinement job re-processing a bbox of the scene."""
    _require_scene(scene_id)

    active = job_manager.get_active_job_for_scene(scene_id)
    if active is not None:
        raise SceneBusy(scene_id, active.job_id)

    job = job_manager.create_job(scene_id)
    job_manager.update_job(job.job_id, request=_job_request_payload(request))
    _dispatch(job.job_id, http_request, background_tasks)

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
    _ensure_job_owner(job)

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
    _ensure_job_owner(job)

    return CancelResponse(
        job_id=job.job_id,
        scene_id=job.scene_id,
        status=JobStatus(job.status),
        cancel_requested=job.cancel_requested,
    )
