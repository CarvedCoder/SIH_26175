from fastapi import APIRouter

from app.schemas.processing import (
    ProcessingResponse,
    ProcessingResult,
    ProcessingStatus,
    ProcessingStatusResponse,
)
from app.services.processing_service import process_image


router = APIRouter(
    prefix="/api/processing",
    tags=["Processing"],
)


@router.post(
    "/{job_id}",
    response_model=ProcessingResponse,
)
async def start_processing(job_id: str):
    result = process_image(job_id)

    return ProcessingResponse(
        job_id=result.job_id,
        status=result.status,
    )


@router.get(
    "/{job_id}/status",
    response_model=ProcessingStatusResponse,
)
async def get_processing_status(job_id: str):
    return ProcessingStatusResponse(
        job_id=job_id,
        status=ProcessingStatus.running,
        progress=50,
        stage="Processing imagery",
    )


@router.get(
    "/{job_id}/result",
    response_model=ProcessingResult,
)
async def get_processing_result(job_id: str):
    return ProcessingResult(
        job_id=job_id,
        status=ProcessingStatus.completed,
        input_type="image",
        calibration_method="relative",
        depth_map=None,
        dsm=None,
    )