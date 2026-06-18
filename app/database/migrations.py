"""Schema creation and migration helpers for Adit-Agent.

Two strategies are supported, chosen automatically by :func:`run_migrations`:

1. **Alembic** — if an ``alembic.ini`` is present at the project root, schema
   changes are applied via versioned migrations (the production path).
2. **create_all fallback** — otherwise the ORM metadata is materialized
   directly with ``Base.metadata.create_all``. This is convenient for local
   development, tests, and first-run bootstrapping before Alembic is set up.

The async-safe primitives (:func:`create_all`, :func:`drop_all`) run the
synchronous SQLAlchemy DDL inside ``connection.run_sync`` so they work with the
async engine. Alembic itself runs synchronously and is therefore executed via
:func:`asyncio.to_thread` to avoid blocking the event loop.

CLI usage::

    python -m app.database.migrations create     # create all tables
    python -m app.database.migrations drop        # drop all tables
    python -m app.database.migrations reset       # drop + create
    python -m app.database.migrations upgrade      # alembic upgrade head (or create_all)
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import PROJECT_ROOT, get_settings
from app.database.models import Base
from app.database.session import DatabaseSessionManager
from app.utils.logger import get_logger

__all__ = [
    "create_all",
    "drop_all",
    "init_db",
    "run_migrations",
    "alembic_config_path",
]

log = get_logger(__name__)

# Alembic config is considered "available" only if this file exists.
_ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def alembic_config_path() -> Path | None:
    """Return the alembic.ini path if it exists, else ``None``."""
    return _ALEMBIC_INI if _ALEMBIC_INI.is_file() else None


# --------------------------------------------------------------------------- #
# Low-level DDL (async-safe)
# --------------------------------------------------------------------------- #
async def create_all(engine: AsyncEngine) -> None:
    """Create all tables defined on :data:`Base.metadata` (idempotent)."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.success("Database schema created ({} tables).", len(Base.metadata.tables))
    except SQLAlchemyError as exc:
        log.error("Failed to create schema: {}", exc)
        raise


async def drop_all(engine: AsyncEngine) -> None:
    """Drop all tables defined on :data:`Base.metadata`. **Destructive.**"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        log.warning("Database schema dropped ({} tables).", len(Base.metadata.tables))
    except SQLAlchemyError as exc:
        log.error("Failed to drop schema: {}", exc)
        raise


async def init_db(engine: AsyncEngine, *, drop: bool = False) -> None:
    """Bring the schema up. If ``drop`` is True, drop everything first."""
    if drop:
        await drop_all(engine)
    await create_all(engine)


# --------------------------------------------------------------------------- #
# Alembic integration
# --------------------------------------------------------------------------- #
def _run_alembic_upgrade_sync(revision: str = "head") -> None:
    """Run ``alembic upgrade <revision>`` programmatically (blocking)."""
    # Imported lazily so alembic is only required when actually used.
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    command.upgrade(cfg, revision)


async def run_migrations(engine: AsyncEngine, *, revision: str = "head") -> None:
    """Apply migrations using Alembic if configured, else ``create_all``.

    This is the entry point intended for :meth:`app.dependencies.AppContainer`
    startup. It never silently does nothing: it logs which strategy ran.
    """
    if alembic_config_path() is not None:
        log.info("Applying Alembic migrations (revision={}).", revision)
        try:
            # Alembic is synchronous; keep the event loop responsive.
            await asyncio.to_thread(_run_alembic_upgrade_sync, revision)
            log.success("Alembic migrations applied.")
            return
        except Exception as exc:  # noqa: BLE001 - surface a clear message
            log.error("Alembic migration failed: {}", exc)
            raise
    else:
        log.info(
            "No alembic.ini found — creating schema directly via metadata "
            "(development fallback)."
        )
        await create_all(engine)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
async def _amain(action: str, *, yes: bool) -> int:
    settings = get_settings()
    manager = DatabaseSessionManager()
    manager.init_from_settings(settings)
    engine = manager.engine

    destructive = action in {"drop", "reset"}
    if destructive and not yes:
        prompt = f"This will DROP all tables in {settings.database_url!r}. Continue? [y/N] "
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            log.info("Aborted.")
            await manager.close()
            return 1

    try:
        if action == "create":
            await create_all(engine)
        elif action == "drop":
            await drop_all(engine)
        elif action == "reset":
            await init_db(engine, drop=True)
        elif action == "upgrade":
            await run_migrations(engine)
        else:  # pragma: no cover - argparse restricts choices
            log.error("Unknown action: {}", action)
            return 2
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        log.error("Migration command {!r} failed: {}", action, exc)
        return 1
    finally:
        await manager.close()


def main() -> None:
    """Console entry point for schema management."""
    parser = argparse.ArgumentParser(description="Adit-Agent database schema management.")
    parser.add_argument(
        "action",
        choices=("create", "drop", "reset", "upgrade"),
        help="create=make tables, drop=remove tables, reset=drop+create, "
        "upgrade=alembic upgrade head (or create_all fallback).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for destructive actions.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args.action, yes=args.yes)))


if __name__ == "__main__":
    main()
