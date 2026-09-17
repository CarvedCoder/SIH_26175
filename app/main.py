"""Convenience alias so `uvicorn app.main:app` works from the repo root.

The real application lives at ``backend.app.main:app`` (its internals use
``backend.app.*`` absolute imports, which require the repo root as the
working directory anyway). This shim simply re-exports it so the short
module path works too:

    uvicorn app.main:app --reload --port 8000
"""
from backend.app.main import app  # noqa: F401
