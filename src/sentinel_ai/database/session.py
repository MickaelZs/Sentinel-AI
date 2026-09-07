"""Synchronous SQLAlchemy engine and session management."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from sentinel_ai.core.config import settings


def _database_url(database_url: str | None = None) -> str:
    """Resolve an explicit or environment-provided database URL."""
    url = database_url or settings.database_url
    if url is None:
        raise RuntimeError("DATABASE_URL must be configured to use the database.")
    return url


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    """Create and cache an engine for the configured database URL."""
    return create_engine(_database_url(database_url))


@lru_cache
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Create and cache a synchronous session factory."""
    return sessionmaker(
        bind=get_engine(database_url), autoflush=False, expire_on_commit=False
    )


def get_db() -> Generator[Session]:
    """Yield a database session and always close it afterwards."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
