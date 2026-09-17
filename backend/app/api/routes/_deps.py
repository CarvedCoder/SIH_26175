"""Shared route dependencies — the single home for cross-route guards.

Previously these lived in routes/scenes.py and other route modules
imported from it (route-to-route imports: an anti-pattern). The guard
logic itself is in the application layer; this module is the FastAPI-side
alias so route modules have one clean import.
"""

from __future__ import annotations

from backend.app.application.scenes.service import scene_service


def require_scene(scene_id: str) -> None:
    """Raise typed 404/400 unless the scene exists (routes/jobs, terrain,
    results, export, reference, validation all use this)."""
    scene_service.require_scene(scene_id)
