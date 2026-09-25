"""API router wiring.

``api_router`` is the PROTECTED surface (API-key dependency applied); the
health router is public and included separately by main.py so Docker and
platform healthchecks work unauthenticated.
"""

from fastapi import APIRouter, Depends

from backend.app.api.routes.export import router as export_router
from backend.app.api.routes.jobs import router as jobs_router
from backend.app.api.routes.reference import router as reference_router
from backend.app.api.routes.results import router as results_router
from backend.app.api.routes.route import router as route_router
from backend.app.api.routes.scenes import router as scenes_router
from backend.app.api.routes.results_files import router as results_files_router
from backend.app.api.routes.semantic import router as semantic_router
from backend.app.api.routes.terrain import router as terrain_router
from backend.app.api.routes.validation import router as validation_router
from backend.app.core.auth import get_current_user

api_router = APIRouter(dependencies=[Depends(get_current_user)])

api_router.include_router(jobs_router)
api_router.include_router(results_router)
api_router.include_router(scenes_router)
api_router.include_router(terrain_router)
api_router.include_router(results_files_router)
api_router.include_router(reference_router)
api_router.include_router(export_router)
api_router.include_router(validation_router)
api_router.include_router(route_router)
api_router.include_router(semantic_router)
