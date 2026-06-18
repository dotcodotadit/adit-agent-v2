"""Centralized logging configuration for Adit-Agent.

This module wires :mod:`loguru` as the single logging backend and routes the
standard library :mod:`logging` (used by third-party packages such as
``python-telegram-bot``, ``httpx`` and ``sqlalchemy``) through it, so the whole
application emits a single, consistent, structured log stream.

Usage
-----
>>> from app.utils.logger import setup_logging, get_logger
>>> setup_logging(level="info")          # call once at startup
>>> log = get_logger(__name__)
>>> log.info("agent started")

Design notes
------------
* ``setup_logging`` is idempotent — calling it twice will not duplicate sinks.
* Console output is colorized; file output is plain and rotated.
* ``InterceptHandler`` bridges stdlib logging into loguru without losing the
  original module name or call site.
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

__all__ = ["setup_logging", "get_logger"]

# Guard so repeated calls (e.g. in tests) don't stack duplicate sinks.
_CONFIGURED: bool = False

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)


class InterceptHandler(logging.Handler):
    """Redirect stdlib ``logging`` records into loguru.

    Installed as the root handler so libraries that use the standard logging
    module are captured with correct level mapping and call-site depth.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        # Map stdlib level number to a loguru level name when possible.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back to the frame that originated the log call so file/line are
        # reported correctly instead of pointing at the logging machinery.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: str = "info",
    *,
    log_dir: str | Path | None = None,
    json_logs: bool = False,
    retention: str = "10 days",
    rotation: str = "20 MB",
) -> None:
    """Configure the global logger. Safe to call multiple times.

    Parameters
    ----------
    level:
        Minimum level to emit (case-insensitive), e.g. ``"info"``, ``"debug"``.
    log_dir:
        Directory for rotating file logs. If ``None``, only console logging is
        enabled (useful for containers that scrape stdout).
    json_logs:
        When ``True``, file sink serializes records as JSON for ingestion by
        log aggregators (Loki, ELK, etc.).
    retention:
        How long rotated files are kept (loguru duration string).
    rotation:
        Size or time threshold that triggers rotation.
    """
    global _CONFIGURED

    level = level.upper()
    logger.remove()  # drop loguru's default stderr sink

    # ---- Console sink --------------------------------------------------------
    logger.add(
        sys.stderr,
        level=level,
        format=_CONSOLE_FORMAT,
        colorize=True,
        backtrace=False,   # avoid leaking variable values in prod tracebacks
        diagnose=False,
        enqueue=True,      # async-safe across asyncio tasks / threads
    )

    # ---- File sink (optional) ------------------------------------------------
    # A failing file sink (unwritable path, full disk, permissions) must not
    # take down the application — degrade to console-only logging instead.
    if log_dir is not None:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_path / "adit-agent.log",
                level=level,
                format=_FILE_FORMAT,
                rotation=rotation,
                retention=retention,
                compression="zip",
                serialize=json_logs,
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )
        except OSError as exc:
            logger.warning(
                "Could not set up file logging in {} ({}); continuing with "
                "console logging only.",
                log_dir,
                exc,
            )

    # ---- Bridge stdlib logging ----------------------------------------------
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for noisy in ("httpx", "httpcore", "telegram", "apscheduler", "chromadb"):
        logging.getLogger(noisy).handlers = [InterceptHandler()]
        logging.getLogger(noisy).propagate = False

    _CONFIGURED = True
    logger.debug("Logging configured (level={}, file={})", level, bool(log_dir))


def get_logger(name: str | None = None) -> "Logger":
    """Return a logger bound with the given ``name`` context.

    The returned logger behaves like the global loguru logger but records carry
    a ``name`` field so per-module filtering remains possible.
    """
    if not _CONFIGURED:
        # Lazily fall back to sane defaults if startup forgot to configure.
        setup_logging()
    return logger.bind(name=name or "adit-agent")
