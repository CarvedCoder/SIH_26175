from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Overall state of a processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


class JobStage(str, Enum):
    """Current stage of the DepthWizard processing pipeline."""

    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    DEPTH_INFERENCE = "depth_inference"
    CALIBRATION = "calibration"
    REFINEMENT = "refinement"
    DSM_GENERATION = "dsm_generation"
    TERRAIN_GENERATION = "terrain_generation"
    VALIDATION = "validation"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreateResponse(BaseModel):
    """Returned when processing is started for a scene."""

    job_id: str
    scene_id: str
    status: JobStatus = JobStatus.QUEUED


class JobError(BaseModel):
    """Typed error embedded in a failed job record."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = False


class JobResponse(BaseModel):
    """Current state of a processing job (GET /jobs/{id} contract)."""

    job_id: str
    scene_id: str

    status: JobStatus
    stage: JobStage = JobStage.QUEUED

    progress: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    message: str | None = None
    cancel_requested: bool = False

    result: dict[str, Any] | None = None
    error: JobError | None = None

    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class CancelResponse(BaseModel):
    """Returned by POST /jobs/{id}/cancel."""

    job_id: str
    scene_id: str
    status: JobStatus
    cancel_requested: bool = True
