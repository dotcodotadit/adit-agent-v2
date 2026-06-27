"""Dangerous-tool confirmation via Telegram inline buttons.

When the executor wants to run a tool marked *dangerous* (shell, write_file,
browser, process), it calls the orchestrator's ``confirmer`` — and that confirmer
is bound here to a Telegram chat. :class:`ConfirmationManager` sends an inline
keyboard ("Allow" / "Deny"), then *suspends* the agent loop on an
:class:`asyncio.Future` until the user taps a button (or a timeout elapses, which
denies by default).

Because the message handler task and the callback-query handler run on the same
event loop, the future created in :meth:`make_confirmer` is simply resolved by
:meth:`handle_callback` when the button press arrives.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from telegram import Bot
    from telegram.ext import ContextTypes

    from app.tools.base import Tool

log = get_logger(__name__)

__all__ = ["ConfirmationManager"]

# Callback-data prefix identifying confirmation button presses.
_PREFIX = "cf"


class ConfirmationManager:
    """Issues confirmation prompts and resolves them from button callbacks.

    Parameters
    ----------
    timeout:
        Seconds to wait for a decision before auto-denying.
    """

    def __init__(self, *, timeout: float = 120.0) -> None:
        self._timeout = timeout
        self._pending: dict[str, asyncio.Future[bool]] = {}

    # ------------------------------------------------------------------ #
    # Confirmer factory (passed to the orchestrator per request)
    # ------------------------------------------------------------------ #
    def make_confirmer(self, bot: "Bot", chat_id: int):
        """Return an async ``confirmer(tool, arguments) -> bool`` for one chat."""

        async def confirm(tool: "Tool", arguments: dict[str, Any]) -> bool:
            token = uuid.uuid4().hex[:12]
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._pending[token] = future

            log.debug(
                "Requesting confirmation for tool {!r} with token={} (pending={})",
                tool.name, token, len(self._pending),
            )

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("✅ Allow", callback_data=f"{_PREFIX}:{token}:allow"),
                        InlineKeyboardButton("⛔ Deny", callback_data=f"{_PREFIX}:{token}:deny"),
                    ]
                ]
            )
            try:
                await bot.send_message(
                    chat_id,
                    self._prompt_text(tool, arguments),
                    reply_markup=keyboard,
                )
            except Exception as exc:  # noqa: BLE001 - can't ask → deny safely
                log.warning("Failed to send confirmation prompt: {}", exc)
                self._pending.pop(token, None)
                return False

            try:
                result = await asyncio.wait_for(future, timeout=self._timeout)
                log.debug("Confirmation for tool {!r}: approved={}", tool.name, result)
                return result
            except asyncio.TimeoutError:
                log.info("Confirmation for {!r} timed out; denying.", tool.name)
                return False
            finally:
                self._pending.pop(token, None)

        return confirm

    # ------------------------------------------------------------------ #
    # Callback handling
    # ------------------------------------------------------------------ #
    async def handle_callback(
        self, update: Any, context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Resolve the pending confirmation referenced by a button press."""
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        parts = (query.data or "").split(":")
        if len(parts) != 3 or parts[0] != _PREFIX:
            log.debug("Ignoring malformed callback data: {}", query.data)
            return
        _, token, decision = parts
        approved = decision == "allow"

        log.debug(
            "Confirmation callback: token={} decision={} approved={} pending_tokens={}",
            token, decision, approved, list(self._pending.keys()),
        )

        future = self._pending.get(token)
        if future is not None and not future.done():
            future.set_result(approved)
            log.debug("Future resolved with approved={}", approved)
        else:
            log.warning(
                "Confirmation callback for token {!r} had no matching pending future "
                "(found={}, done={}). This may indicate a race condition or timeout.",
                token, future is not None, future.done() if future else None,
            )

        with _suppress_telegram_errors():
            await query.edit_message_text(
                "✅ Approved — running the tool." if approved else "⛔ Denied."
            )

    @staticmethod
    def _prompt_text(tool: "Tool", arguments: dict[str, Any]) -> str:
        """Compose the confirmation message body."""
        preview = "\n".join(f"  • {k}: {_short(v)}" for k, v in (arguments or {}).items())
        preview = preview or "  (no arguments)"
        return (
            f"⚠️ Adit wants to run a sensitive tool:\n\n"
            f"Tool: {tool.name}\n"
            f"{tool.description}\n\n"
            f"Arguments:\n{preview}\n\n"
            f"Allow this action?"
        )


def _short(value: Any, limit: int = 200) -> str:
    """Render a short, single-line preview of an argument value."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class _suppress_telegram_errors:
    """Context manager that swallows Telegram edit errors (best-effort UI)."""

    def __enter__(self) -> "_suppress_telegram_errors":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            log.debug("Ignored Telegram UI error: {}", exc)
        return True
