"""Slash-command handlers for the Adit-Agent Telegram bot.

Covers conversation control (``/start``, ``/help``, ``/new``, ``/mode``),
developer mode (``/dev``, ``/thoughts``, ``/raw``), and account/memory utilities
(``/whoami``, ``/forget``, ``/stats``). :func:`register_commands` wires them onto
the application and publishes the command menu shown in Telegram clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import BotCommand
from telegram.ext import CommandHandler

from app.agent.context_builder import AgentMode
from app.bot.services import get_services
from app.database.models import ConversationMode
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import Application, ContextTypes

log = get_logger(__name__)

__all__ = ["register_commands"]


_MODE_MAP: dict[str, AgentMode] = {
    "chat": AgentMode.CHAT,
    "agent": AgentMode.AGENT,
    "deep": AgentMode.DEEP,
}
# How per-turn agent modes map onto the persisted conversation default.
_MODE_TO_CONVERSATION = {
    AgentMode.CHAT: ConversationMode.CHAT,
    AgentMode.AGENT: ConversationMode.AGENT,
    AgentMode.DEEP: ConversationMode.AGENT,
}

_WELCOME = (
    "👋 Hi, I'm *Adit* — your AI assistant.\n\n"
    "Just talk to me naturally. I can chat, search the web, work with files, and "
    "understand images, documents, audio and video you send.\n\n"
    "Useful commands:\n"
    "/new — start a fresh conversation\n"
    "/mode chat|agent|deep — set how hard I think\n"
    "/help — full command list"
)

_HELP = (
    "*Adit — command reference*\n\n"
    "*Conversation*\n"
    "/new — start a new conversation (clears working context)\n"
    "/mode chat|agent|deep — chat = fast replies, agent = tool-using, "
    "deep = careful reasoning\n\n"
    "*Developer mode*\n"
    "/dev — toggle developer mode (plans, tool details, timings)\n"
    "/thoughts — toggle showing my interim reasoning\n"
    "/raw — toggle the raw event stream\n\n"
    "*Account & memory*\n"
    "/whoami — show your status and current settings\n"
    "/forget — erase everything I remember about you\n"
    "/stats — bot statistics (admins only)\n\n"
    "Tip: send a photo, PDF, voice note or video and I'll read it."
)


# --------------------------------------------------------------------------- #
# Conversation commands
# --------------------------------------------------------------------------- #
async def start_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Greet the user and register them."""
    services = get_services(context)
    user = update.effective_user
    if user is not None:
        await services.memory.get_or_create_user(
            user.id, username=user.username, first_name=user.first_name
        )
    await update.effective_message.reply_text(_WELCOME, parse_mode="Markdown")


async def help_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Show the command reference."""
    await update.effective_message.reply_text(_HELP, parse_mode="Markdown")


async def new_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Open a fresh conversation and make it active for this user."""
    services = get_services(context)
    user = update.effective_user
    db_user = await services.memory.get_or_create_user(
        user.id, username=user.username, first_name=user.first_name
    )
    conversation = await services.memory.start_new_conversation(db_user.id)
    context.user_data["conversation_id"] = conversation.id
    await update.effective_message.reply_text(
        "🆕 Started a new conversation. Previous context is set aside (I still "
        "remember durable facts about you)."
    )


async def mode_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Set the agent mode for subsequent messages."""
    services = get_services(context)
    args = context.args or []
    current = context.user_data.get("mode_override")
    current_label = current.value if current else "auto"

    if not args or args[0].lower() not in _MODE_MAP:
        await update.effective_message.reply_text(
            f"Current mode: *{current_label}*.\n\n"
            "Usage: /mode chat|agent|deep\n"
            "• chat — quick conversational replies\n"
            "• agent — full tool-using problem solving\n"
            "• deep — deliberate, step-by-step reasoning",
            parse_mode="Markdown",
        )
        return

    mode = _MODE_MAP[args[0].lower()]
    context.user_data["mode_override"] = mode

    # Persist onto the active conversation when there is one.
    conversation_id = context.user_data.get("conversation_id")
    if conversation_id is not None:
        await services.memory.set_conversation_mode(
            conversation_id, _MODE_TO_CONVERSATION[mode]
        )
    await update.effective_message.reply_text(f"✅ Mode set to *{mode.value}*.",
                                              parse_mode="Markdown")


# --------------------------------------------------------------------------- #
# Developer-mode commands
# --------------------------------------------------------------------------- #
async def dev_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Toggle developer mode."""
    await _toggle(update, context, "dev", "🛠 Developer mode")


async def thoughts_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Toggle showing the agent's interim reasoning."""
    await _toggle(update, context, "thoughts", "💭 Thoughts")


async def raw_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Toggle the raw event stream."""
    await _toggle(update, context, "raw", "🔬 Raw event stream")


async def _toggle(
    update: "Update",
    context: "ContextTypes.DEFAULT_TYPE",
    flag: str,
    label: str,
) -> None:
    services = get_services(context)
    user = update.effective_user
    new_value = services.dev.toggle(user.id, flag)
    flags = services.dev.flags(user.id)
    await update.effective_message.reply_text(
        f"{label}: *{'ON' if new_value else 'OFF'}*\n`{flags.summary()}`",
        parse_mode="Markdown",
    )


# --------------------------------------------------------------------------- #
# Account & memory commands
# --------------------------------------------------------------------------- #
async def whoami_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Report the user's status and current settings."""
    services = get_services(context)
    user = update.effective_user
    db_user = await services.memory.get_or_create_user(
        user.id, username=user.username, first_name=user.first_name
    )
    memory_count = await services.memory.count_memories(db_user.id)
    mode = context.user_data.get("mode_override")
    flags = services.dev.flags(user.id)
    await update.effective_message.reply_text(
        "👤 *Who you are to Adit*\n"
        f"Telegram ID: `{user.id}`\n"
        f"Admin: {'yes' if services.is_admin(user.id) else 'no'}\n"
        f"Mode: {mode.value if mode else 'auto'}\n"
        f"Developer: `{flags.summary()}`\n"
        f"Stored memories: {memory_count}",
        parse_mode="Markdown",
    )


async def forget_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Erase everything stored about the user."""
    services = get_services(context)
    user = update.effective_user
    db_user = await services.memory.get_or_create_user(user.id)
    removed = await services.memory.forget_user(db_user.id)
    await update.effective_message.reply_text(
        f"🧹 Done — I forgot {removed} stored memory item(s) about you."
    )


async def stats_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Report basic statistics (admins only)."""
    services = get_services(context)
    user = update.effective_user
    if not services.is_admin(user.id):
        await update.effective_message.reply_text("⛔ This command is for admins only.")
        return
    has_llm = services.orchestrator is not None
    await update.effective_message.reply_text(
        "📊 *Adit status*\n"
        f"LLM online: {'yes' if has_llm else 'no'}\n"
        f"Environment: {services.settings.environment}\n"
        f"Default model: {services.settings.llm_default_model}",
        parse_mode="Markdown",
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
_COMMANDS: list[tuple[str, str, object]] = [
    ("start", "Start / show the welcome message", start_cmd),
    ("help", "Show the command reference", help_cmd),
    ("new", "Start a new conversation", new_cmd),
    ("mode", "Set mode: chat | agent | deep", mode_cmd),
    ("dev", "Toggle developer mode", dev_cmd),
    ("thoughts", "Toggle showing interim reasoning", thoughts_cmd),
    ("raw", "Toggle the raw event stream", raw_cmd),
    ("whoami", "Show your status and settings", whoami_cmd),
    ("forget", "Erase what I remember about you", forget_cmd),
    ("stats", "Bot statistics (admins)", stats_cmd),
]


def register_commands(application: "Application") -> None:
    """Add every command handler to ``application``."""
    for name, _desc, callback in _COMMANDS:
        application.add_handler(CommandHandler(name, callback))
    log.info("Registered {} command handlers.", len(_COMMANDS))


async def publish_command_menu(application: "Application") -> None:
    """Publish the slash-command menu shown in Telegram clients."""
    commands = [BotCommand(name, desc) for name, desc, _cb in _COMMANDS]
    await application.bot.set_my_commands(commands)
