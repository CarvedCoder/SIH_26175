from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Canonical health payload (frontend contract in client.js).

    ``model_loaded`` reports whether the serving stack can currently serve
    inference: the checkpoint is present and resolvable. It is NOT a claim
    that a DAv2 backbone is resident — the backbone lazy-loads on demand.
    """

    status: str = "ok"
    version: str
    model_loaded: bool
