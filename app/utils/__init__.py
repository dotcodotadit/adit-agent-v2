"""Shared utilities for Adit-Agent.

Currently exposes the logging subsystem. Call :func:`setup_logging` once at
startup; then use :func:`get_logger` anywhere::

    from app.utils import setup_logging, get_logger
    setup_logging(level="info", log_dir=Path("data/logs"))
    log = get_logger(__name__)
    log.info("System ready.")
"""

from __future__ import annotations

from app.utils.logger import get_logger, setup_logging

__all__ = ["setup_logging", "get_logger"]
