"""Telegram bot layer for Adit-Agent.

Exposes :func:`~app.bot.telegram_bot.build_application`, which assembles a fully
wired ``telegram.ext.Application`` from a started application container.
"""

from __future__ import annotations

from app.bot.telegram_bot import build_application

__all__ = ["build_application"]
