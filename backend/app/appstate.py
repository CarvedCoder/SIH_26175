"""Application-singleton wiring: the TaskQueue is chosen from config and
attached to app.state at startup (lifespan). Routes reach it through
task_queue_for(); the function falls back to a module-level inline queue
so route handlers stay testable without a full app lifespan.

This is CONFIGURATION wiring, not application state: the queue holds no
per-job or per-user data, and correctness never depends on the same
process dispatching and executing a job (DW_WORKER_MODE=external runs
with NO API-side dispatch at all).
"""

from __future__ import annotations

from typing import Any, Optional

from backend.app.core.config import get_settings
from backend.app.infrastructure.queue.inline_queue import (
    BackgroundTaskQueue,
    InlineTaskQueue,
)

# Fallback for callers outside a running app (tests, tools).
_module_inline_queue: Optional[InlineTaskQueue] = None


def build_task_queue():
    """Build the standalone dispatch queue from configuration (used only
    outside a request — the request-scoped transport is
    BackgroundTaskQueue, which preserves the exact pre-refactor timing)."""
    settings = get_settings()
    if settings.worker_mode == "external":
        return None  # record-only: the durable job record is the queue
    return InlineTaskQueue(max_workers=settings.max_concurrent_jobs)


def task_queue_for(http_request: Any, background_tasks: Any) -> Optional[Any]:
    """The dispatch transport for this request, or None in external-worker
    mode (the durable job record is the queue; backend.app.worker claims)."""
    settings = get_settings()
    if settings.worker_mode == "external":
        return None
    return BackgroundTaskQueue(background_tasks)
