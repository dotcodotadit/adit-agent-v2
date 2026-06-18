"""Adit-Agent application entry point.

Boots the whole system:

1. Loads configuration (``app.config``).
2. Configures logging (``app.utils.logger``).
3. Builds and starts the dependency container (``app.dependencies``).
4. Constructs and runs the Telegram bot until interrupted.
5. Performs graceful shutdown on SIGINT/SIGTERM.

Run with::

    python -m app.main
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import suppress

from app.config import ConfigError, get_settings
from app.dependencies import AppContainer, StartupError, get_container
from app.utils.logger import get_logger, setup_logging


async def _run() -> None:
    """Async application lifecycle: startup, serve, graceful shutdown."""
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.data_dir / "logs")
    log = get_logger("app.main")

    log.info("Booting {} (env={})", settings.app_name, settings.environment)

    if not settings.telegram_bot_token.get_secret_value().strip():
        log.error("TELEGRAM_BOT_TOKEN is not set — cannot start the bot.")
        raise SystemExit(1)

    container: AppContainer = get_container()
    try:
        await container.startup()
    except StartupError as exc:
        # startup() has already rolled back any partially-initialized
        # subsystems, so there is nothing left to clean up here.
        log.error("Cannot start: {}", exc)
        raise SystemExit(1) from exc

    # Build the Telegram application lazily to avoid import cost when the
    # container fails to start.
    from app.bot import build_application

    application = build_application(container)
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    log.success("Bot is online and polling for updates.")

    # Cross-platform shutdown signaling via an asyncio.Event.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop(*_: object) -> None:
        log.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):  # Windows lacks add_signal_handler
            loop.add_signal_handler(sig, _request_stop)

    try:
        await stop_event.wait()
    finally:
        log.info("Beginning graceful shutdown.")
        # Stop the bot before the container so in-flight handlers can finish.
        with suppress(Exception):
            if application.updater is not None:
                await application.updater.stop()
            await application.stop()
            await application.shutdown()
        await container.shutdown()
        log.success("{} stopped cleanly.", settings.app_name)


def main() -> None:
    """Synchronous wrapper used as the console / module entry point."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Ctrl-C before the async signal handler is installed; exit quietly.
        pass
    except ConfigError as exc:
        # Configuration errors happen before logging is set up, so write a
        # concise message to stderr rather than emitting a full traceback.
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
