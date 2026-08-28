from enum import Enum

from pydantic import BaseModel


class ProcessingStatus(str, Enum):
    uploaded = "uploaded"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ProcessingResponse(BaseModel):
    job_id: str
    status: ProcessingStatus


class ProcessingStatusResponse(BaseModel):
    job_id: str
    status: ProcessingStatus
    progress: int
    stage: str


class ProcessingResult(BaseModel):
    job_id: str
    status: ProcessingStatus
    input_type: str | None = None
    calibration_method: str | None = None
    depth_map: str | None = None
    dsm: str | None = None