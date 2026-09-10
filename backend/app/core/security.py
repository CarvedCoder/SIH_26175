"""API-key authentication boundary (audit finding C1).

Contract:
    * ``DW_API_KEY`` set     -> every protected route requires the
      ``X-API-Key`` header to match; missing/wrong keys get a 401 envelope.
    * ``DW_API_KEY`` unset   -> auth is DISABLED for local development.
      This is an explicit, configuration-driven state (logged at startup),
      never a silent bypass.

The key is never logged and never echoed in error details. Health
endpoints stay public so Docker/platform healthchecks work unauthenticated.
"""

from __future__ import annotations

import hmac

from fastapi import Request

from backend.app.core.config import get_settings
from backend.app.core.errors import Unauthorized
from backend.app.core.logging import logger

API_KEY_HEADER = "X-API-Key"


def auth_enabled() -> bool:
    return get_settings().api_key is not None


async def require_api_key(request: Request) -> None:
    """FastAPI dependency enforcing the API-key boundary."""
    api_key = get_settings().api_key
    if api_key is None:
        return

    provided = request.headers.get(API_KEY_HEADER)
    if provided is None or not hmac.compare_digest(provided, api_key):
        logger.info("rejected request: missing/invalid API key path=%s", request.url.path)
        raise Unauthorized()
