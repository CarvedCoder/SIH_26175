from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.app.jobs.manager import job_manager
from backend.app.services.processing_service import processing_service


router = APIRouter(
    tags=["Jobs"],
)


def _run_processing(
    job_id: str,
    scene_id: str,
    mode: str,
    ground_elev: Optional[float],
) -> None:
    """Background entry point for DepthWizard processing."""
    try:
        processing_service.process_scene(
            job_id=job_id,
            scene_id=scene_id,
            mode=mode,
            ground_elev=ground_elev,
        )
    except Exception:
        # ProcessingService already records the detailed job error.
        pass


@router.post("/api/v1/scenes/{scene_id}/process")
async def process_scene(
    scene_id: str,
    background_tasks: BackgroundTasks,
    mode: str = "auto",
    ground_elev: Optional[float] = None,
):
    """Create an asynchronous DepthWizard processing job."""

    if mode not in {"auto", "crop", "resize", "tiles"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported processing mode: {mode}",
        )

    job = job_manager.create_job(scene_id)

    background_tasks.add_task(
        _run_processing,
        job.job_id,
        scene_id,
        mode,
        ground_elev,
    )

    return {
        "job_id": job.job_id,
        "scene_id": job.scene_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
    }


@router.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Return the current status of a processing job."""

    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    return {
        "job_id": job.job_id,
        "scene_id": job.scene_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }