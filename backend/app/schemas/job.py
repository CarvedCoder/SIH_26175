from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Overall state of a processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    """Current stage of the DepthWizard processing pipeline."""

    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    DEPTH_INFERENCE = "depth_inference"
    CALIBRATION = "calibration"
    DSM_GENERATION = "dsm_generation"
    TERRAIN_GENERATION = "terrain_generation"
    VALIDATION = "validation"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


class JobCreateResponse(BaseModel):
    """Returned when processing is started for a scene."""

    job_id: str
    scene_id: str
    status: JobStatus = JobStatus.QUEUED


class JobResponse(BaseModel):
    """Current state of a processing job."""

    job_id: str
    scene_id: str

    status: JobStatus
    stage: JobStage = JobStage.QUEUED

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    message: str | None = None

    error_code: str | None = None
    error_message: str | None = None

    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None