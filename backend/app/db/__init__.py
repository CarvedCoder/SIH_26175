"""Persistence layer (SQLAlchemy 2.x over Supabase/PostgreSQL).

The relational database is the single source of truth for scene, job and
result-artifact metadata. Object bytes live in MinIO (S3 API) and are
referenced here by object KEY, never by local filesystem paths.
"""

from backend.app.db.database import get_db, init_db, session_scope
from backend.app.db.models import Base, JobRow, SceneRow

__all__ = [
    "Base",
    "JobRow",
    "SceneRow",
    "get_db",
    "init_db",
    "session_scope",
]
