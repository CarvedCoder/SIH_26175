"""Inline TaskQueue — development dispatch.

Runs the handler in a daemon worker thread immediately. Semantically
equivalent to FastAPI BackgroundTasks (async jobs run off the request
path) but decoupled from the request/response lifecycle: the handler
survives the HTTP connection closing and the queue object carries no
per-job state.

Production mode ("external"): the API enqueues NOTHING — the durable job
record is the queue entry and backend.app.worker claims it. See
core/config.py DW_WORKER_MODE.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from backend.app.core.logging import logger


class InlineTaskQueue:
    """Minimal TaskQueue over daemon threads. No state per task."""

    def __init__(self, max_workers: int = 1) -> None:
        self._semaphore = threading.Semaphore(max(1, max_workers))

    def add_task(self, fn: Callable, /, *args: Any, **kwargs: Any) -> None:
        def _run() -> None:
            with self._semaphore:
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001 — handler owns its errors;
                    # a queue must never crash the API process
                    logger.exception("queued task %r crashed", getattr(fn, "__name__", fn))

        threading.Thread(
            target=_run, name=f"inline-queue-{getattr(fn, '__name__', 'task')}",
            daemon=True,
        ).start()


class BackgroundTaskQueue:
    """TaskQueue over FastAPI's BackgroundTasks (the request-scoped inline
    transport). Behaviorally identical to the pre-refactor dispatch: tasks
    run off the response path, after the response is sent. Used for
    DW_WORKER_MODE=inline; external mode uses no API-side queue at all."""

    def __init__(self, background_tasks) -> None:
        self._background_tasks = background_tasks

    def add_task(self, fn: Callable, /, *args: Any, **kwargs: Any) -> None:
        self._background_tasks.add_task(fn, *args, **kwargs)
