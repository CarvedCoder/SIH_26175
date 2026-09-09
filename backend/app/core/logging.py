"""Logging + request-ID context — replaces print() operational logging.

Every request gets a short correlation id (returned to the client in the
``X-Request-ID`` response header and embedded in error envelopes) so a
frontend-reported failure can be matched to exactly one server-side log
line with its stack trace.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_HANDLER_INSTALLED = False


class RequestIdLogFormatter(logging.Formatter):
    """Formatter that threads the current request id into every record."""

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_var.get()
        return super().format(record)


def configure_logging() -> None:
    """Idempotent root logging setup for the backend process."""
    global _HANDLER_INSTALLED
    if _HANDLER_INSTALLED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        RequestIdLogFormatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
        )
    )

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _HANDLER_INSTALLED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


logger = get_logger("depthwizard.backend")
