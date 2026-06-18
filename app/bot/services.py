"""Shared bot-layer services container for Adit-Agent.

A single :class:`BotServices` bundle is stashed in the Telegram
``application.bot_data`` at startup so every handler can reach the orchestrator,
memory manager, media pipeline and bot-only helpers without globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

    from app.agent.orchestrator import Orchestrator
    from app.agent.memory_manager import MemoryManager
    from app.bot.dev_mode import DevModeStore
    from app.bot.middlewares import RateLimiter
    from app.bot.safety import ConfirmationManager
    from app.config import Settings
    from app.multimodal import MediaPipeline

__all__ = ["BotServices", "get_services"]

# Key under which the services bundle lives in application.bot_data.
SERVICES_KEY = "services"


@dataclass(slots=True)
class BotServices:
    """Long-lived services every handler needs."""

    settings: "Settings"
    orchestrator: "Orchestrator | None"
    memory: "MemoryManager"
    media: "MediaPipeline"
    dev: "DevModeStore"
    confirm: "ConfirmationManager"
    rate_limiter: "RateLimiter | None" = None

    def is_admin(self, user_id: int | None) -> bool:
        """True when ``user_id`` is configured as an admin."""
        return user_id is not None and user_id in self.settings.telegram_admin_user_ids

    def is_allowed(self, user_id: int | None) -> bool:
        """True when ``user_id`` may use the bot (empty allow-list = everyone)."""
        if user_id is None:
            return False
        allowed = self.settings.telegram_allowed_user_ids
        if not allowed:
            return True
        return user_id in allowed or self.is_admin(user_id)


def get_services(context: "ContextTypes.DEFAULT_TYPE") -> BotServices:
    """Fetch the :class:`BotServices` bundle from a handler context."""
    services: Any = context.application.bot_data.get(SERVICES_KEY)
    if services is None:  # pragma: no cover - indicates a wiring bug
        raise RuntimeError("BotServices not initialized in application.bot_data.")
    return services
