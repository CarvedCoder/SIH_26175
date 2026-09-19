"""Database engine/session wiring — one configuration source.

``DATABASE_URL`` selects the database (e.g.
``postgresql+psycopg://user:pass@host:5432/depthwizard``). There is NO
silent fallback to SQLite: when the configured database is unreachable the
backend fails fast at startup with a clear message.

Tests override ``DATABASE_URL`` with a per-test SQLite file before the
engine is first used (the engine is created lazily).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings
from backend.app.core.logging import logger

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_database_url() -> str:
    """The single source of truth for the database connection string."""
    return get_settings().database_url


def get_engine():
    """Lazily create the shared engine (laziness keeps tests overridable)."""
    global _engine, _SessionLocal
    if _engine is None:
        url = get_database_url()
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
            if url.startswith("sqlite")
            else {},
        )
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def _session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    session = _session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for non-request contexts (background jobs).

    Commits on success, rolls back on any exception — a job worker never
    leaves a half-written row behind.
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create schema objects and verify connectivity (fail fast)."""
    engine = get_engine()
    from backend.app.db.models import Base

    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("database ready: %s", engine.url.render_as_string(hide_password=True))


def reset_engine_for_tests() -> None:
    """Dispose and forget the engine (test isolation only)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
