from uuid import uuid4

from fastapi import APIRouter, File, UploadFile

from app.schemas.processing import ProcessingResponse, ProcessingStatus

router = APIRouter(prefix="/api/imagery", tags=["Imagery"])


@router.post("/upload", response_model=ProcessingResponse)
async def upload_image(file: UploadFile = File(...)):
    job_id = str(uuid4())

    return ProcessingResponse(
        job_id=job_id,
        status=ProcessingStatus.uploaded,
    )