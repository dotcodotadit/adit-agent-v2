"""Cross-cutting handlers ("middleware") for the Adit-Agent Telegram bot.

python-telegram-bot has no formal middleware concept, so the same effect is
achieved with handlers registered in an early group (``-1``) that run before the
real handlers and can stop propagation via :class:`ApplicationHandlerStop`:

* an **access gate** — blocks users not on the allow-list;
* a **rate-limit gate** — throttles how fast a single user may send messages,
  protecting the bot and your LLM credits from floods/abuse;
* a global **error handler** — classifies and logs any uncaught failure and
  sends the user a helpful, internals-free apology.

The inbound rate limiter here is distinct from PTB's ``AIORateLimiter`` (wired
in :mod:`app.bot.telegram_bot`), which throttles *outgoing* calls to Telegram.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from telegram.ext import ApplicationHandlerStop

from app.bot.services import get_services
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

log = get_logger(__name__)

__all__ = [
    "RateLimiter",
    "RateLimitDecision",
    "access_gate",
    "rate_limit_gate",
    "error_handler",
]


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RateLimitDecision:
    """The outcome of a rate-limit check for one event."""

    allowed: bool
    retry_after: float = 0.0  # seconds until the next event would be allowed


class RateLimiter:
    """In-memory, per-key sliding-window rate limiter.

    Tracks the timestamps of recent events per key (here, a Telegram user id)
    in a bounded deque and allows at most ``max_events`` within any window of
    ``window_seconds``. Old timestamps are evicted lazily on each check, so
    memory stays proportional to the number of *recently active* users.

    State is process-local and intentionally not persisted: a restart simply
    resets everyone's window, which is the desired behavior for abuse control.
    Single-threaded asyncio access means no locking is required.
    """

    def __init__(self, *, max_events: int, window_seconds: float) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._max = max_events
        self._window = float(window_seconds)
        self._events: dict[int, deque[float]] = defaultdict(deque)

    def check(self, key: int, *, now: float | None = None) -> RateLimitDecision:
        """Record an event for ``key`` and decide whether it is allowed.

        On an *allowed* event the timestamp is recorded; on a *denied* event
        nothing is recorded (so a flood does not keep pushing the window
        forward) and ``retry_after`` estimates the wait until a slot frees up.
        """
        current = time.monotonic() if now is None else now
        cutoff = current - self._window
        bucket = self._events[key]

        # Evict timestamps that have aged out of the window.
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self._max:
            # Oldest event leaves the window at bucket[0] + window.
            retry_after = max(0.0, bucket[0] + self._window - current)
            return RateLimitDecision(allowed=False, retry_after=retry_after)

        bucket.append(current)
        return RateLimitDecision(allowed=True)

    def reset(self, key: int) -> None:
        """Clear all recorded events for ``key`` (e.g. on manual unblock)."""
        self._events.pop(key, None)


# --------------------------------------------------------------------------- #
# Gates (handler group -1)
# --------------------------------------------------------------------------- #
async def access_gate(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Block users not on the allow-list (when one is configured).

    Registered in group ``-1`` so it runs first. Raises
    :class:`ApplicationHandlerStop` to prevent any downstream handler from
    processing the update for an unauthorized user.
    """
    user = update.effective_user
    if user is None:
        return  # service messages without a user — let them pass

    services = get_services(context)
    if services.is_allowed(user.id):
        return

    log.warning("Blocked unauthorized user id={} (@{}).", user.id, user.username)
    message = update.effective_message
    if message is not None:
        with _ignore_errors():
            await message.reply_text(
                "⛔ Sorry, you're not authorized to use this bot. "
                "Ask the administrator to add your Telegram ID."
            )
    raise ApplicationHandlerStop


# Minimum seconds between "slow down" notices, so a flood doesn't get one
# rejection reply per message (which would itself be spam).
_RL_NOTICE_INTERVAL = 5.0


async def rate_limit_gate(
    update: "Update", context: "ContextTypes.DEFAULT_TYPE"
) -> None:
    """Throttle inbound messages per user via a sliding window.

    Registered in group ``-1`` after :func:`access_gate`. Admins and disabled
    configuration bypass the check. When a user exceeds their allowance the
    update is dropped (:class:`ApplicationHandlerStop`) and — at most once per
    :data:`_RL_NOTICE_INTERVAL` — they are told how long to wait.
    """
    user = update.effective_user
    if user is None:
        return

    services = get_services(context)
    if not services.settings.rate_limit_enabled or services.rate_limiter is None:
        return
    if services.is_admin(user.id):
        return  # admins are never throttled

    decision = services.rate_limiter.check(user.id)
    if decision.allowed:
        return

    log.info(
        "Rate-limited user id={} (@{}); retry in {:.1f}s.",
        user.id,
        user.username,
        decision.retry_after,
    )
    await _maybe_notify_throttled(update, context, decision.retry_after)
    raise ApplicationHandlerStop


async def _maybe_notify_throttled(
    update: "Update",
    context: "ContextTypes.DEFAULT_TYPE",
    retry_after: float,
) -> None:
    """Tell the user to slow down, but no more than once per interval."""
    message = update.effective_message
    if message is None:
        return

    last = context.user_data.get("_rl_last_notice", 0.0)
    now = time.monotonic()
    if now - last < _RL_NOTICE_INTERVAL:
        return
    context.user_data["_rl_last_notice"] = now

    wait = max(1, round(retry_after))
    with _ignore_errors():
        await message.reply_text(
            f"🚦 You're sending messages a bit fast. Please wait ~{wait}s and try again."
        )


# --------------------------------------------------------------------------- #
# Global error handler
# --------------------------------------------------------------------------- #
async def error_handler(update: object, context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Log any uncaught handler error and send a tailored, safe apology."""
    error = context.error

    # ApplicationHandlerStop is control flow, not a failure — ignore if it ever
    # reaches here.
    if isinstance(error, ApplicationHandlerStop):
        return

    log.opt(exception=error).error(
        "Unhandled error while processing update: {}", _describe(error)
    )

    user_message = _user_message_for(error)

    # Best-effort apology to the originating chat.
    from telegram import Update

    if isinstance(update, Update) and update.effective_message is not None:
        with _ignore_errors():
            await update.effective_message.reply_text(user_message)


def _user_message_for(error: BaseException | None) -> str:
    """Map an exception to a helpful, internals-free user-facing message.

    Classification is done by class name / module so this module does not need
    to import optional heavy dependencies (openai, telegram.error) at the top
    level just to branch on their exception types.
    """
    if error is None:
        return "⚠️ Something went wrong on my end. Please try again in a moment."

    name = type(error).__name__
    module = type(error).__module__
    text = str(error).lower()

    # --- LLM provider problems (app.providers.base.ProviderError) ----------- #
    if name == "ProviderError" or "providers" in module:
        # Surface the common, actionable cases without leaking the raw error.
        if "402" in text or "payment" in text or "quota" in text or "insufficient" in text:
            return (
                "💳 My AI provider rejected the request (out of credit or quota). "
                "The administrator needs to top up or switch providers — please try later."
            )
        if "401" in text or "403" in text or "api key" in text or "unauthor" in text:
            return (
                "🔑 My AI provider rejected my credentials. "
                "The administrator needs to check the API key configuration."
            )
        if "429" in text or "rate limit" in text:
            return "🐢 My AI provider is rate-limiting me right now. Please try again shortly."
        return (
            "🤖 I couldn't reach my AI backend just now. "
            "All configured providers failed — please try again in a moment."
        )

    # --- Telegram transport hiccups (telegram.error.*) ---------------------- #
    if module.startswith("telegram"):
        if name in ("TimedOut", "NetworkError"):
            return "🌐 The network was slow just now. Please try again."
        if name == "RetryAfter":
            return "🐢 I'm being rate-limited by Telegram. Please try again shortly."
        # Other Telegram errors are usually transient display issues.
        return "⚠️ A messaging error occurred. Please try again in a moment."

    # --- Timeouts / cancellations from asyncio ------------------------------ #
    if name in ("TimeoutError", "CancelledError"):
        return "⏱️ That took too long and timed out. Please try again."

    # --- Fallback ----------------------------------------------------------- #
    return "⚠️ Something went wrong on my end. Please try again in a moment."


def _describe(error: BaseException | None) -> str:
    """Short, log-friendly description of an error (type + message)."""
    if error is None:
        return "<no error on context>"
    return f"{type(error).__name__}: {error}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _ignore_errors:
    """Swallow secondary errors raised while reporting a primary one."""

    def __enter__(self) -> "_ignore_errors":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            log.debug("Suppressed secondary error: {}", exc)
        return True
