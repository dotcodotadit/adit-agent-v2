"""Telegram application assembly for Adit-Agent.

:func:`build_application` turns a started :class:`~app.dependencies.AppContainer`
into a fully-wired ``telegram.ext.Application``: it constructs the bot-layer
services, publishes them into ``bot_data``, and registers every handler in the
right order (access gate first, then commands, message handling, and the
confirmation callback), plus a global error handler.

The caller (``app.main``) owns the application lifecycle (initialize / start /
polling / shutdown); this module only builds and wires it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from app.bot.commands import publish_command_menu, register_commands
from app.bot.dev_mode import DevModeStore
from app.bot.handlers import confirmation_callback, message_handler
from app.bot.middlewares import (
    RateLimiter,
    access_gate,
    error_handler,
    rate_limit_gate,
)
from app.bot.safety import ConfirmationManager
from app.bot.services import SERVICES_KEY, BotServices
from app.multimodal import MediaPipeline
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from telegram.ext import Application

    from app.dependencies import AppContainer

log = get_logger(__name__)

__all__ = ["build_application"]

# Messages we treat as agent input (anything but bare commands / service msgs).
_CONTENT_FILTER = (
    (filters.TEXT & ~filters.COMMAND)
    | filters.PHOTO
    | filters.Document.ALL
    | filters.VOICE
    | filters.AUDIO
    | filters.VIDEO
    | filters.VIDEO_NOTE
)


def build_application(container: "AppContainer") -> "Application":
    """Build and wire the Telegram application from a started container."""
    settings = container.settings
    token = settings.telegram_bot_token.get_secret_value()
    if not token:
        raise RuntimeError("Cannot build the bot: TELEGRAM_BOT_TOKEN is empty.")

    # Inbound per-user rate limiter (None when disabled in config).
    rate_limiter = (
        RateLimiter(
            max_events=settings.rate_limit_max_messages,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if settings.rate_limit_enabled
        else None
    )

    # Bot-layer services (orchestrator/memory come from the container).
    services = BotServices(
        settings=settings,
        orchestrator=container.orchestrator,
        memory=container.memory_manager,
        media=MediaPipeline(
            provider_router=container.provider_router, settings=settings
        ),
        dev=DevModeStore(),
        confirm=ConfirmationManager(),
        rate_limiter=rate_limiter,
    )

    builder = ApplicationBuilder().token(token).post_init(publish_command_menu)
    # The rate limiter is an optional extra; use it when available, otherwise
    # continue without outgoing throttling rather than failing to start.
    try:
        from telegram.ext import AIORateLimiter

        builder = builder.rate_limiter(AIORateLimiter())
    except (ImportError, RuntimeError) as exc:
        log.warning("AIORateLimiter unavailable ({}); running without it.", exc)

    application = builder.build()
    application.bot_data[SERVICES_KEY] = services

    # Early gates, each in its own group so both run (PTB executes only the
    # first matching handler *per group*). Lower group numbers run first, and
    # either gate may halt the update via ApplicationHandlerStop.
    #   group -2: access gate (authorization)
    #   group -1: rate-limit gate (flood control)
    application.add_handler(TypeHandler(Update, access_gate), group=-2)
    application.add_handler(TypeHandler(Update, rate_limit_gate), group=-1)

    # Group 0: commands, content messages, and the confirmation callback.
    register_commands(application)
    application.add_handler(MessageHandler(_CONTENT_FILTER, message_handler))
    application.add_handler(CallbackQueryHandler(confirmation_callback, pattern=r"^cf:"))

    application.add_error_handler(error_handler)

    log.info("Telegram application built and wired.")
    return application
