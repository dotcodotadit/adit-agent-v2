"""Message handling for the Adit-Agent Telegram bot.

This is where a user's message becomes an agent turn. The flow:

1. Guard against overlapping requests from the same user.
2. Download and process any attachments (image / document / audio / video) into
   prompt-ready text via the :class:`~app.multimodal.MediaPipeline`.
3. Build an :class:`~app.agent.orchestrator.AgentRequest` (carrying the per-user
   mode, active conversation, and a Telegram-bound dangerous-tool confirmer).
4. Stream the orchestrator's events into a single, live-updating Telegram message
   via :class:`StreamRenderer`, honoring the user's developer-mode flags.

The :class:`StreamRenderer` is deliberately tolerant: Telegram edit failures
("message is not modified", rate limits) are swallowed so a cosmetic UI hiccup
never breaks the actual answer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.agent.executor import EventType
from app.agent.orchestrator import AgentRequest
from app.bot.services import BotServices, get_services
from app.database.models import AttachmentType
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from telegram import Bot, Update
    from telegram.ext import ContextTypes

    from app.bot.dev_mode import DevFlags

log = get_logger(__name__)

__all__ = ["message_handler", "confirmation_callback"]

# Telegram's hard message-length cap.
_TELEGRAM_LIMIT = 4096
# Minimum seconds between live edits to respect rate limits.
_EDIT_INTERVAL = 1.1


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
async def message_handler(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Handle an inbound text/media message by running an agent turn."""
    services = get_services(context)
    user = update.effective_user
    message = update.effective_message
    if user is None or message is None:
        return

    if services.orchestrator is None:
        await message.reply_text(
            "⚠️ I'm not fully configured yet — no LLM provider is available. "
            "Please set a provider API key and restart me."
        )
        return

    # Prevent overlapping turns from the same user (keeps streaming coherent).
    if context.user_data.get("busy"):
        await message.reply_text("⏳ I'm still working on your previous message…")
        return
    context.user_data["busy"] = True
    try:
        await _process_turn(update, context, services)
    finally:
        context.user_data["busy"] = False


async def _process_turn(
    update: "Update",
    context: "ContextTypes.DEFAULT_TYPE",
    services: BotServices,
) -> None:
    user = update.effective_user
    message = update.effective_message
    chat_id = update.effective_chat.id

    await _safe_typing(context.bot, chat_id)

    # 1) Process attachments into text context.
    media_blocks, had_attachments = await _process_attachments(context, services, message)

    user_text = (message.text or message.caption or "").strip()
    combined = _combine(user_text, media_blocks)
    if not combined.strip():
        await message.reply_text("Send me a message, question, or a file to work with.")
        return

    # 2) Build the agent request.
    request = AgentRequest(
        telegram_id=user.id,
        text=combined,
        username=user.username,
        first_name=user.first_name,
        mode=context.user_data.get("mode_override"),
        conversation_id=context.user_data.get("conversation_id"),
        allow_tools=True,
        confirmer=services.confirm.make_confirmer(context.bot, chat_id),
    )

    # 3) Stream the run into a live message.
    flags = services.dev.flags(user.id)
    renderer = StreamRenderer(context.bot, chat_id, flags=flags)
    await renderer.start()

    response = None
    async for event in services.orchestrator.stream(request):
        if event.type is EventType.FINAL:
            response = event.data.get("response")
        await renderer.handle(event)

    await renderer.finalize(response)

    # Remember the active conversation so threading continues without /new.
    if response is not None and response.conversation_id:
        context.user_data["conversation_id"] = response.conversation_id


# --------------------------------------------------------------------------- #
# Attachment processing
# --------------------------------------------------------------------------- #
async def _process_attachments(
    context: "ContextTypes.DEFAULT_TYPE",
    services: BotServices,
    message: Any,
) -> tuple[list[str], bool]:
    """Download and process any attachments; return context blocks."""
    targets = _attachment_targets(message)
    if not targets:
        return [], False

    blocks: list[str] = []
    upload_dir = services.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file_id, file_name, kind in targets:
        try:
            tg_file = await context.bot.get_file(file_id)
            dest = upload_dir / _safe_name(file_name)
            await tg_file.download_to_drive(custom_path=str(dest))
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the turn
            log.warning("Failed to download attachment {}: {}", file_name, exc)
            blocks.append(f"[ATTACHMENT {file_name}: download failed]")
            continue

        processed = await services.media.process(dest, kind=kind)
        blocks.append(processed.as_context(file_name=file_name))

    return blocks, True


def _attachment_targets(message: Any) -> list[tuple[str, str, AttachmentType]]:
    """Extract ``(file_id, file_name, kind)`` tuples from a message."""
    targets: list[tuple[str, str, AttachmentType]] = []

    if message.photo:
        photo = message.photo[-1]  # largest rendition
        targets.append((photo.file_id, f"{photo.file_unique_id}.jpg", AttachmentType.IMAGE))
    if message.document:
        doc = message.document
        targets.append(
            (doc.file_id, doc.file_name or f"{doc.file_unique_id}", AttachmentType.DOCUMENT)
        )
    if message.voice:
        voice = message.voice
        targets.append((voice.file_id, f"{voice.file_unique_id}.ogg", AttachmentType.AUDIO))
    if message.audio:
        audio = message.audio
        targets.append(
            (audio.file_id, audio.file_name or f"{audio.file_unique_id}.mp3", AttachmentType.AUDIO)
        )
    if message.video:
        video = message.video
        targets.append(
            (video.file_id, video.file_name or f"{video.file_unique_id}.mp4", AttachmentType.VIDEO)
        )
    if getattr(message, "video_note", None):
        vn = message.video_note
        targets.append((vn.file_id, f"{vn.file_unique_id}.mp4", AttachmentType.VIDEO))

    return targets


def _safe_name(name: str) -> str:
    """Reduce a user-supplied filename to a safe basename."""
    base = Path(name).name.replace("\x00", "")
    return base or "upload.bin"


def _combine(user_text: str, media_blocks: list[str]) -> str:
    """Merge the user's text with extracted attachment context."""
    if not media_blocks:
        return user_text
    media = "\n\n".join(media_blocks)
    if user_text:
        return f"{user_text}\n\n{media}"
    return f"(The user sent the following without a caption.)\n\n{media}"


# --------------------------------------------------------------------------- #
# Streaming renderer
# --------------------------------------------------------------------------- #
class StreamRenderer:
    """Renders the orchestrator's event stream into one live Telegram message.

    Maintains separate regions — status (tools/plan), interim thoughts, raw
    events, and the answer — and recomposes them on each update. Edits are
    throttled and de-duplicated to stay within Telegram's rate limits.
    """

    def __init__(self, bot: "Bot", chat_id: int, *, flags: "DevFlags") -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._flags = flags

        self._answer = ""           # committed final-answer text
        self._live: list[str] = []  # tokens of the in-progress turn
        self._status: list[str] = []
        self._thoughts: list[str] = []
        self._raw: list[str] = []

        self._message: Any = None
        self._last_render = 0.0
        self._last_text = ""

    async def start(self) -> None:
        """Send the initial placeholder message."""
        self._message = await self._bot.send_message(self._chat_id, "🤔 Thinking…")

    # -- event handling ------------------------------------------------ #
    async def handle(self, event: Any) -> None:
        if self._flags.raw:
            self._raw.append(_raw_line(event))

        t = event.type
        if t is EventType.TOKEN:
            self._live.append(event.text)
            await self._render()
        elif t is EventType.TURN_END:
            if event.data.get("final"):
                self._answer += "".join(self._live)
            else:
                thought = "".join(self._live).strip()
                if thought:
                    self._thoughts.append(thought)
            self._live = []
            await self._render()
        elif t is EventType.TOOL_START:
            line = f"🔧 {event.tool_name}…"
            if self._flags.dev and event.arguments:
                line += f"  {_short(event.arguments)}"
            self._status.append(line)
            await self._render(force=True)
            await _safe_typing(self._bot, self._chat_id)
        elif t is EventType.TOOL_RESULT:
            self._status.append(f"{'✅' if event.success else '❌'} {event.tool_name}")
            await self._render(force=True)
        elif t is EventType.TOOL_DENIED:
            self._status.append(f"🚫 {event.tool_name} (denied)")
            await self._render(force=True)
        elif t is EventType.PLAN:
            if self._flags.dev or self._flags.thoughts:
                self._status.append(f"📋 Plan:\n{event.text}")
                await self._render(force=True)
        elif t is EventType.REFLECTION:
            if event.data.get("revised"):
                self._answer = event.text  # adopt the improved answer
            if self._flags.dev:
                verdict = "revised" if event.data.get("revised") else "verified"
                self._thoughts.append(f"↻ reflection: {verdict}")
            await self._render(force=True)
        elif t is EventType.ERROR:
            self._status.append(f"⚠️ {event.text}")
            await self._render(force=True)

    async def finalize(self, response: Any) -> None:
        """Render the final state, collapsing transient UI in normal mode."""
        self._live = []
        if response is not None and response.text:
            self._answer = response.text

        if not self._flags.any_on():
            # Clean result: just the answer.
            self._status = []
            self._thoughts = []
            self._raw = []

        final_text = self._answer.strip() or "(no response)"
        await self._render(force=True, override=self._compose_with_answer(final_text))

        # Spill any overflow beyond Telegram's limit into follow-up messages.
        if len(final_text) > _TELEGRAM_LIMIT:
            for chunk in _chunk(final_text, _TELEGRAM_LIMIT)[1:]:
                with _ignore():
                    await self._bot.send_message(self._chat_id, chunk)

    # -- rendering ----------------------------------------------------- #
    def _compose(self) -> str:
        return self._compose_with_answer(self._answer + "".join(self._live))

    def _compose_with_answer(self, answer: str) -> str:
        parts: list[str] = []
        if self._flags.raw and self._raw:
            parts.append("```\n" + "\n".join(self._raw[-15:]) + "\n```")
        if self._status:
            parts.append("\n".join(self._status))
        if (self._flags.thoughts or self._flags.dev) and self._thoughts:
            parts.append("💭 " + "\n\n".join(self._thoughts[-6:]))
        if answer.strip():
            parts.append(answer)
        text = "\n\n".join(p for p in parts if p.strip()) or "🤔 Thinking…"
        return text[:_TELEGRAM_LIMIT]

    async def _render(self, *, force: bool = False, override: str | None = None) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < _EDIT_INTERVAL:
            return
        text = override if override is not None else self._compose()
        if text == self._last_text or self._message is None:
            return
        self._last_text = text
        self._last_render = now
        with _ignore():
            await self._bot.edit_message_text(
                text, chat_id=self._chat_id, message_id=self._message.message_id
            )


# --------------------------------------------------------------------------- #
# Callback handler (dangerous-tool confirmations)
# --------------------------------------------------------------------------- #
async def confirmation_callback(
    update: "Update", context: "ContextTypes.DEFAULT_TYPE"
) -> None:
    """Route a confirmation button press to the :class:`ConfirmationManager`."""
    log.debug("Confirmation callback received: data={}", update.callback_query.data if update.callback_query else None)
    services = get_services(context)
    await services.confirm.handle_callback(update, context)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _safe_typing(bot: "Bot", chat_id: int) -> None:
    """Send a 'typing' chat action, ignoring failures."""
    with _ignore():
        await bot.send_chat_action(chat_id, action="typing")


def _raw_line(event: Any) -> str:
    """One-line repr of an event for the raw stream view."""
    bits = [event.type.value]
    if event.tool_name:
        bits.append(event.tool_name)
    if event.text:
        bits.append(_short(event.text, 60))
    return " | ".join(bits)


def _short(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class _ignore:
    """Context manager that swallows (and debug-logs) Telegram UI errors."""

    def __enter__(self) -> "_ignore":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            log.debug("Ignored Telegram UI error: {}", exc)
        return True
