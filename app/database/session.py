"""Async database engine and session management for Adit-Agent.

This module owns the SQLAlchemy *async* engine and session factory and exposes
ergonomic helpers for acquiring sessions:

>>> from app.database.session import sessionmanager
>>> sessionmanager.init_from_settings()
>>> async with sessionmanager.session() as session:
...     user = await session.get(User, 1)

Design
------
* :class:`DatabaseSessionManager` encapsulates the engine + ``async_sessionmaker``
  so the rest of the app never touches global SQLAlchemy state directly. This
  keeps tests isolated (each test can spin up its own manager) and makes the
  lifecycle explicit (``init`` / ``close``).
* A process-wide :data:`sessionmanager` instance is provided for the common
  case; :func:`get_session` is a FastAPI/Telegram-friendly dependency that
  yields a session from it.
* :meth:`DatabaseSessionManager.session` is transactional: it commits on clean
  exit, rolls back on exception, and always closes the session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.utils.logger import get_logger

__all__ = [
    "DatabaseSessionManager",
    "DatabaseNotInitializedError",
    "sessionmanager",
    "get_session",
]

log = get_logger(__name__)


class DatabaseNotInitializedError(RuntimeError):
    """Raised when a session is requested before the manager is initialized."""


class DatabaseSessionManager:
    """Owns the async engine and session factory for one database.

    Call :meth:`init` (or :meth:`init_from_settings`) exactly once at startup,
    and :meth:`close` at shutdown. Accessing sessions before init raises
    :class:`DatabaseNotInitializedError`.
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def init(self, url: str, *, echo: bool = False) -> None:
        """Create the engine and session factory for ``url``.

        Idempotent-ish: re-initializing while already initialized is treated as
        a programming error and logged, because it usually means a leaked
        engine. Call :meth:`close` first if you genuinely need to re-init.
        """
        if self._engine is not None:
            log.warning(
                "DatabaseSessionManager already initialized; ignoring re-init. "
                "Call close() before re-initializing."
            )
            return

        try:
            self._engine = create_async_engine(
                url,
                echo=echo,
                pool_pre_ping=True,    # transparently recycle dead connections
            )
        except SQLAlchemyError as exc:
            raise DatabaseNotInitializedError(
                f"Failed to create database engine for {url!r}: {exc}"
            ) from exc

        self._sessionmaker = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,    # objects stay usable after commit
            autoflush=False,
        )
        log.info("Database session manager initialized ({}).", url)

    def init_from_settings(self, settings: Settings | None = None) -> None:
        """Initialize using the application :class:`Settings`."""
        settings = settings or get_settings()
        self.init(settings.database_url, echo=settings.database_echo)

    async def close(self) -> None:
        """Dispose the engine and drop references. Safe to call repeatedly."""
        if self._engine is None:
            return
        try:
            await self._engine.dispose()
        except SQLAlchemyError as exc:  # best-effort during shutdown
            log.warning("Error disposing database engine: {}", exc)
        finally:
            self._engine = None
            self._sessionmaker = None
            log.info("Database session manager closed.")

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> AsyncEngine:
        """Return the live engine or raise if not initialized."""
        if self._engine is None:
            raise DatabaseNotInitializedError(
                "Database engine not initialized. Call init()/init_from_settings() first."
            )
        return self._engine

    @property
    def is_initialized(self) -> bool:
        return self._engine is not None

    # ------------------------------------------------------------------ #
    # Context managers
    # ------------------------------------------------------------------ #
    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional session.

        Commits on clean exit, rolls back on any exception, and always closes
        the session. Use for unit-of-work blocks::

            async with sessionmanager.session() as s:
                s.add(obj)
        """
        if self._sessionmaker is None:
            raise DatabaseNotInitializedError(
                "Session factory not initialized. Call init()/init_from_settings() first."
            )

        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Yield a raw transactional connection (for DDL / migrations)."""
        async with self.engine.begin() as conn:
            yield conn


# Process-wide default manager. Initialize it once during app startup
# (see app/dependencies.py) and reuse everywhere via get_session().
sessionmanager = DatabaseSessionManager()


async def get_session() -> AsyncIterator[AsyncSession]:
    """Dependency that yields a transactional session from :data:`sessionmanager`.

    Usable as a FastAPI/Telegram-handler dependency::

        async for session in get_session():
            ...
    """
    async with sessionmanager.session() as session:
        yield session
