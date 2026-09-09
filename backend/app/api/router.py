from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.imagery import router as imagery_router
from backend.app.api.routes.jobs import router as jobs_router
from backend.app.api.routes.results import router as results_router
from backend.app.api.routes.scenes import router as scenes_router
from backend.app.api.routes.results_files import router as results_files_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(imagery_router)
api_router.include_router(jobs_router)
api_router.include_router(results_router)
api_router.include_router(scenes_router)
api_router.include_router(results_files_router)