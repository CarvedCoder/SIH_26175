"""DepthWizard API — the CANONICAL production/development backend.

Serves the frontend's scene/job-oriented ``/api/v1`` contract. The legacy
``service/api.py`` stateless predictor is NOT this app (kept only as a
manual smoke-test door); Docker deploys THIS application.

    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import api_router
from backend.app.api.routes.health import _health_payload, router as health_router
from backend.app.core.config import settings
from backend.app.core.errors import register_error_handlers
from backend.app.core.logging import logger
from backend.app.core.middleware import AccessLogMiddleware, RequestIdMiddleware
from backend.app.core.paths import ensure_directories
from backend.app.core.security import auth_enabled
from backend.app.schemas.health import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    logger.info(
        "DepthWizard API %s starting; auth=%s; cors_origins=%s",
        settings.version,
        "enabled" if auth_enabled() else "DISABLED (local development)",
        ",".join(settings.cors_origins),
    )
    yield
    logger.info("DepthWizard API shutting down")


app = FastAPI(
    title="DepthWizard API",
    description="Backend API for the DepthWizard processing pipeline",
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(AccessLogMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

register_error_handlers(app)

# Public: health only. Everything under api_router requires the API key
# when DW_API_KEY is configured.
app.include_router(health_router)
app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def platform_health() -> HealthResponse:
    """Unversioned health alias for Docker/platform probes."""
    return _health_payload()
