"""Supabase authentication boundary.

AUTHORITY: Supabase owns authentication. This module verifies Supabase-
issued JWTs (HS256 via SUPABASE_JWT_SECRET, or asymmetric keys via the
project's JWKS endpoint) and exposes the verified identity as an
``AuthUser`` to every protected route.

Modes (first match wins):
    1. SUPABASE_JWT_SECRET or SUPABASE_URL configured -> Bearer JWT
       required; the verified ``sub`` is the owner id used for all
       authorization checks.
    2. DW_API_KEY set          -> legacy X-API-Key mode (CI/tooling);
       owner id "api-key".
    3. neither                 -> auth DISABLED (local development);
       owner id "local".

The verified identity is stored in a contextvar set by the router-level
``get_current_user`` dependency; the scene guard reads it for ownership
checks. A request's user id is NEVER taken from payloads.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

import jwt
from fastapi import Request

from backend.app.core.config import get_settings
from backend.app.core.errors import Forbidden, Unauthorized
from backend.app.core.logging import logger

LOCAL_USER_ID = "local"
API_KEY_USER_ID = "api-key"

_current_user: contextvars.ContextVar["AuthUser | None"] = contextvars.ContextVar(
    "current_user", default=None
)


@dataclass(frozen=True)
class AuthUser:
    """Verified identity — the ONLY trusted source of a user id."""

    user_id: str
    email: str | None
    mode: str  # "supabase" | "api_key" | "disabled"


def current_user() -> AuthUser:
    """The verified user for this request (set by the router dependency)."""
    user = _current_user.get()
    if user is None:  # pragma: no cover — router dependency always sets it
        user = AuthUser(user_id=LOCAL_USER_ID, email=None, mode="disabled")
        _current_user.set(user)
    return user


def auth_mode() -> str:
    s = get_settings()
    if s.supabase_jwt_secret or s.supabase_url:
        return "supabase"
    if s.api_key:
        return "api_key"
    return "disabled"


_JWKS_CLIENTS: dict[str, jwt.PyJWKClient] = {}


def _jwks_client(supabase_url: str) -> jwt.PyJWKClient:
    client = _JWKS_CLIENTS.get(supabase_url)
    if client is None:
        url = supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        client = jwt.PyJWKClient(url)
        _JWKS_CLIENTS[supabase_url] = client
    return client


def _verify_supabase_jwt(token: str) -> AuthUser:
    s = get_settings()
    audiences = s.supabase_jwt_audience or None
    try:
        if s.supabase_jwt_secret:
            payload = jwt.decode(
                token,
                s.supabase_jwt_secret,
                algorithms=["HS256"],
                audience=audiences if s.supabase_jwt_audience else None,
                options={"verify_aud": bool(s.supabase_jwt_audience)},
            )
        else:
            signing_key = _jwks_client(s.supabase_url or "").get_signing_key_from_jwt(
                token
            )
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                audience=audiences if s.supabase_jwt_audience else None,
                options={"verify_aud": bool(s.supabase_jwt_audience)},
            )
    except jwt.PyJWTError as exc:
        logger.info("rejected Supabase JWT: %s", type(exc).__name__)
        raise Unauthorized() from exc

    sub = payload.get("sub")
    if not sub:
        raise Unauthorized()
    return AuthUser(
        user_id=str(sub),
        email=payload.get("email"),
        mode="supabase",
    )


async def get_current_user(request: Request) -> AuthUser:
    """FastAPI dependency (applied once at the api_router level)."""
    mode = auth_mode()

    if mode == "supabase":
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise Unauthorized()
        user = _verify_supabase_jwt(header[len("Bearer "):].strip())
        _current_user.set(user)
        return user

    if mode == "api_key":
        from backend.app.core.security import require_api_key

        await require_api_key(request)
        user = AuthUser(user_id=API_KEY_USER_ID, email=None, mode="api_key")
        _current_user.set(user)
        return user

    # explicit local-development mode
    user = AuthUser(user_id=LOCAL_USER_ID, email=None, mode="disabled")
    _current_user.set(user)
    return user


def ensure_owner(owner_id: str, resource: str) -> None:
    """Ownership check: the verified user must own the resource.

    Raises 403 when the resource exists but belongs to another user (a
    404 would leak nothing, but 403 matches the spec's error contract).
    """
    user = current_user()
    if user.user_id != owner_id:
        logger.info(
            "ownership denied: user=%s does not own %s", user.user_id, resource
        )
        raise Forbidden()
