from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.imagery import router as imagery_router
from app.api.routes.processing import router as processing_router

app = FastAPI(
    title="DepthWizard API",
    description="Backend API for the DepthWizard prototype",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(imagery_router)
app.include_router(processing_router)