from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.core.paths import ensure_directories


app = FastAPI(
    title="DepthWizard API",
    description="Backend API for the DepthWizard processing pipeline",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    ensure_directories()


app.include_router(api_router)