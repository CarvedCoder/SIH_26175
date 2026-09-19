"""Domain entities — pure Python, NO framework imports.

The FastAPI layer, persistence and storage adapt around these types;
nothing in this package may import fastapi, torch, rasterio, or any
infrastructure module (refactor brief §37: dependency direction).
"""

from .entities import Job, TERMINAL_STATUSES, utc_now
from .protocols import (
    ArtifactMetadata,
    ArtifactRef,
    ArtifactStore,
    JobRepository,
    TaskQueue,
)

__all__ = [
    "Job",
    "TERMINAL_STATUSES",
    "utc_now",
    "ArtifactRef",
    "ArtifactMetadata",
    "ArtifactStore",
    "JobRepository",
    "TaskQueue",
]
