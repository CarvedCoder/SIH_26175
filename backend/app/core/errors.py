"""Typed error contract: {error: {code, message, details, recoverable}}.

The frontend client (frontend/src/api/client.js) parses exactly this shape.
Every error leaving the API passes through one of the handlers registered
in register_error_handlers(); raw exception text, filesystem paths and
tracebacks NEVER reach the client — they are logged server-side with the
request id instead.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.logging import logger, request_id_var


class AppError(Exception):
    """An application error with a stable code and client-safe message."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}
        self.recoverable = recoverable
        super().__init__(message)


class SceneNotFound(AppError):
    def __init__(self, scene_id: str) -> None:
        super().__init__(
            status_code=404,
            code="SCENE_NOT_FOUND",
            message=f"Scene '{scene_id}' does not exist.",
            details={"scene_id": scene_id},
            recoverable=False,
        )


class JobNotFound(AppError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            status_code=404,
            code="JOB_NOT_FOUND",
            message=f"Job '{job_id}' does not exist.",
            details={"job_id": job_id},
            recoverable=False,
        )


class InvalidSceneId(AppError):
    def __init__(self, scene_id: str) -> None:
        super().__init__(
            status_code=400,
            code="INVALID_SCENE_ID",
            message="Scene identifiers must match 'scene_' followed by 12 hex characters.",
            details={"scene_id": scene_id},
            recoverable=False,
        )


class Unauthorized(AppError):
    def __init__(self, message: str = "Missing or invalid credentials.") -> None:
        super().__init__(
            status_code=401,
            code="UNAUTHORIZED",
            message=message,
            recoverable=False,
        )


class Forbidden(AppError):
    def __init__(self, message: str = "You do not have access to this resource.") -> None:
        super().__init__(
            status_code=403,
            code="FORBIDDEN",
            message=message,
            recoverable=False,
        )


class SceneBusy(AppError):
    def __init__(self, scene_id: str, job_id: str) -> None:
        super().__init__(
            status_code=409,
            code="SCENE_BUSY",
            message="A processing job is already running for this scene.",
            details={"scene_id": scene_id, "job_id": job_id},
            recoverable=True,
        )


def error_envelope(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    recoverable: bool = False,
) -> JSONResponse:
    """Build the canonical error response (client contract in client.js)."""
    body = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "recoverable": recoverable,
        }
    }
    response = JSONResponse(status_code=status_code, content=body)
    response.headers["X-Request-ID"] = request_id_var.get()
    return response


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.info(
            "app error: code=%s status=%s path=%s: %s",
            exc.code,
            exc.status_code,
            request.url.path,
            exc.message,
        )
        return error_envelope(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            recoverable=exc.recoverable,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("validation error path=%s: %s", request.url.path, exc.errors())
        return error_envelope(
            status_code=422,
            code="VALIDATION_ERROR",
            message="Request payload failed validation.",
            details={"errors": exc.errors()},
            recoverable=True,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # Map framework HTTPExceptions into the envelope. Route-raised
        # HTTPExceptions carry curated, client-safe detail text (this code
        # never puts raw exception/path strings into detail); framework-
        # raised ones (routing 404, method 405) get generic messages.
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            409: "CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            422: "VALIDATION_ERROR",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        generic_messages = {
            404: "The requested resource does not exist.",
            405: "HTTP method not allowed for this resource.",
            401: "Missing or invalid API key.",
        }
        message = (
            str(exc.detail)
            if exc.detail
            else generic_messages.get(exc.status_code, "Request failed.")
        )
        logger.info(
            "http error: status=%s path=%s", exc.status_code, request.url.path
        )
        return error_envelope(
            status_code=exc.status_code,
            code=code_map.get(exc.status_code, "HTTP_ERROR"),
            message=message,
            recoverable=False,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Stack trace stays server-side, keyed by the request id the client
        # already has; the client gets a stable sanitized message.
        logger.exception("unhandled error path=%s", request.url.path)
        return error_envelope(
            status_code=500,
            code="INTERNAL_ERROR",
            message="An unexpected server error occurred.",
            recoverable=True,
        )
