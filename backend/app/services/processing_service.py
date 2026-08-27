from app.schemas.processing import ProcessingResult, ProcessingStatus


def process_image(job_id: str) -> ProcessingResult:
    # Temporary implementation.
    # Real ML processing will be connected here later.

    return ProcessingResult(
        job_id=job_id,
        status=ProcessingStatus.completed,
        input_type="image",
        calibration_method="relative",
    )