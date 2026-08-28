from io import BytesIO
from uuid import uuid4

import rasterio
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.imagery import ImageMetadata
from app.schemas.processing import ProcessingResponse, ProcessingStatus


router = APIRouter(
    prefix="/api/imagery",
    tags=["Imagery"],
)


@router.post(
    "/upload",
    response_model=ProcessingResponse,
)
async def upload_image(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/geotiff",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG, and GeoTIFF files are supported.",
        )

    file_data = await file.read()

    if not file_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        with rasterio.open(BytesIO(file_data)) as dataset:
            metadata = ImageMetadata(
                filename=file.filename,
                file_type=file.content_type,
                width=dataset.width,
                height=dataset.height,
                channels=dataset.count,
                has_geospatial_metadata=dataset.crs is not None,
                crs=str(dataset.crs) if dataset.crs else None,
            )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read raster image: {exc}",
        ) from exc

    job_id = str(uuid4())

    return ProcessingResponse(
        job_id=job_id,
        status=ProcessingStatus.uploaded,
    )