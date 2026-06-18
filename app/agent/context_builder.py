"""Prompt assembly and token-budgeted context construction for Adit-Agent.

This module turns the raw materials of a turn — the user's message, recent
conversation history, recalled long-term memories, an optional plan — into the
concrete list of OpenAI-style chat messages that is sent to the LLM.

It owns three concerns that the rest of the agent depends on:

* **The provider contract.** :class:`LLMRouter`, :class:`LLMResponse` and
  :class:`StreamChunk` describe the (duck-typed) interface the planner and
  executor expect from ``app.providers``. Defining it here — at the bottom of
  the dependency graph — lets every sibling import a single, documented shape
  instead of passing ``Any`` around.
* **Token accounting.** :class:`TokenCounter` provides best-effort token counts
  (via ``tiktoken`` when available, a fast heuristic otherwise) so the builder
  can fit context inside a model's window while reserving room for the reply.
* **Persona & system-prompt engineering.** The "Adit" persona and the
  mode-specific behavioral contracts live in :data:`PERSONA` and
  :func:`system_prompt`, kept in one place so tone stays consistent across chat,
  agentic and deep-reasoning modes.

The public entry point is :meth:`ContextBuilder.build`, which returns a
:class:`BuiltContext` carrying the assembled messages plus a token breakdown for
observability and budget enforcement upstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.agent.memory_manager import RecalledMemory
    from app.agent.planner import Plan

log = get_logger(__name__)

__all__ = [
    "AgentMode",
    "ChatMessage",
    "ToolCallRequest",
    "LLMResponse",
    "StreamChunk",
    "LLMRouter",
    "TokenCounter",
    "PERSONA",
    "system_prompt",
    "BuiltContext",
    "ContextBuilder",
]


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
class AgentMode(str, Enum):
    """How hard the agent should think and act on a turn.

    Distinct from :class:`app.database.models.ConversationMode` (which records a
    conversation's *default* behavior). The orchestrator resolves a concrete
    :class:`AgentMode` per request, possibly overriding the stored default.
    """

    CHAT = "chat"      # fast conversational reply, tools allowed opportunistically
    AGENT = "agent"    # full ReAct loop, planning for complex tasks
    DEEP = "deep"      # deliberate, multi-step reasoning with explicit reflection

    @property
    def is_deep(self) -> bool:
        return self is AgentMode.DEEP


# --------------------------------------------------------------------------- #
# Message + provider contract
# --------------------------------------------------------------------------- #
# A chat message is an OpenAI-compatible mapping. Kept as a plain dict (rather
# than a model) so it can be handed straight to the provider SDK and persisted
# without translation. Recognized keys: ``role``, ``content``, ``name``,
# ``tool_calls`` (assistant), ``tool_call_id`` (tool replies).
ChatMessage = dict[str, Any]


@dataclass(slots=True)
class ToolCallRequest:
    """A single tool invocation the model asked for, with parsed arguments.

    ``id`` is the provider-assigned call id that must be echoed back on the
    corresponding ``tool`` message so the model can correlate request/result.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_assistant_tool_call(self) -> dict[str, Any]:
        """Render back into the OpenAI ``assistant.tool_calls[*]`` wire shape."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(slots=True)
class LLMResponse:
    """A completed model response, normalized across providers.

    ``tool_calls`` is empty for a plain textual answer; ``content`` is empty
    when the model only requested tools. ``finish_reason`` follows the OpenAI
    vocabulary (``stop`` / ``tool_calls`` / ``length`` / ...).
    """

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        """True when the model is asking to call one or more tools."""
        return bool(self.tool_calls)


@dataclass(slots=True)
class StreamChunk:
    """An incremental event from a streaming completion.

    The router emits ``text`` chunks as content tokens arrive and exactly one
    terminal ``done`` chunk carrying the fully-assembled :class:`LLMResponse`
    (including any tool calls accumulated from the stream). This keeps wire-level
    delta reassembly inside the provider layer, where the format is known.
    """

    type: str               # "text" | "done"
    text: str = ""
    response: LLMResponse | None = None


@runtime_checkable
class LLMRouter(Protocol):
    """The provider interface the agent core depends on.

    Implemented by ``app.providers`` (the failover router across the configured
    providers). Declared here as a ``Protocol`` so the agent is decoupled from
    any concrete client and can be unit-tested with a fake.
    """

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a single, fully-formed completion (non-streaming)."""
        ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk` deltas, ending with a ``done`` chunk."""
        ...

    async def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        """Return one embedding vector per input string."""
        ...


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
class TokenCounter:
    """Best-effort token counter with a graceful fallback.

    Uses ``tiktoken`` when installed (accurate for OpenAI-family tokenizers);
    otherwise falls back to a conservative characters-per-token heuristic so the
    agent still degrades safely on platforms where ``tiktoken`` is unavailable.

    Instances are cheap and stateless apart from the cached encoder, so a single
    shared instance can be reused across the process.
    """

    # Per-message structural overhead (role tokens, delimiters) used by the
    # heuristic path and as a small fixed add-on for the accurate path.
    _PER_MESSAGE_OVERHEAD = 4
    # Average bytes-per-token for English-ish text; deliberately low so the
    # heuristic over- rather than under-estimates and we stay under the window.
    _HEURISTIC_CHARS_PER_TOKEN = 3.5

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._encoder: Any = None
        self._init_encoder(model)

    def _init_encoder(self, model: str) -> None:
        try:
            import tiktoken

            try:
                self._encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                # Unknown model name → use the modern default encoding.
                self._encoder = tiktoken.get_encoding("o200k_base")
        except Exception as exc:  # noqa: BLE001 - any import/runtime issue → heuristic
            log.debug("tiktoken unavailable ({}); using heuristic token counts.", exc)
            self._encoder = None

    def count_text(self, text: str) -> int:
        """Approximate the number of tokens in ``text``."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, int(len(text) / self._HEURISTIC_CHARS_PER_TOKEN) + 1)

    def count_message(self, message: ChatMessage) -> int:
        """Approximate tokens for one chat message, including tool-call payloads."""
        total = self._PER_MESSAGE_OVERHEAD
        content = message.get("content")
        if isinstance(content, str):
            total += self.count_text(content)
        elif isinstance(content, list):
            # Multimodal content parts: count any text fragments, ignore binaries.
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += self.count_text(part["text"])
        if name := message.get("name"):
            total += self.count_text(str(name))
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function", {})
            total += self.count_text(str(fn.get("name", "")))
            total += self.count_text(str(fn.get("arguments", "")))
        return total

    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Approximate total tokens for a list of chat messages."""
        return sum(self.count_message(m) for m in messages)


# --------------------------------------------------------------------------- #
# Persona & system prompts
# --------------------------------------------------------------------------- #
# The stable identity shared by every mode. Keep this tight: behavior that
# varies by mode lives in the per-mode contracts below, not here.
PERSONA = """\
You are Adit — a sharp, capable, and personable AI assistant operating inside a \
Telegram bot. You were built to be genuinely useful: you think clearly, act \
decisively, and communicate like a trusted, knowledgeable colleague rather than \
a corporate help desk.

Identity and voice:
- Your name is Adit. Refer to yourself as Adit when it is natural to.
- Warm, direct, and concise. You respect the user's time and attention.
- Confident but honest: you say what you know, flag what you don't, and never \
fabricate facts, citations, file contents, or tool results.
- You match the user's language and tone, and you have a light, dry wit you use \
sparingly — never at the expense of clarity.

Core principles:
- Lead with the answer, then support it. Don't bury the point under preamble.
- Prefer specifics over generalities. Show, with examples, when it helps.
- When a request is ambiguous and the cost of guessing wrong is high, ask one \
crisp clarifying question; otherwise make a reasonable assumption and state it.
- Think before you act. Reason privately; share conclusions, not raw \
deliberation, unless the user asks to see your working.
- You are running on a real system with real tools. When a task needs current \
information or an action in the world, use your tools rather than guessing."""


# Mode-specific behavioral contracts appended after the persona.
_MODE_CONTRACTS: dict[AgentMode, str] = {
    AgentMode.CHAT: """\
Mode: CHAT.
Respond conversationally and efficiently. Answer directly from what you know. \
Reach for a tool only when the question genuinely requires fresh data, a \
calculation you shouldn't eyeball, or an action on the user's files or system — \
not for things you can answer well from your own knowledge. Keep replies as \
short as the question deserves; expand only when depth is clearly wanted.""",
    AgentMode.AGENT: """\
Mode: AGENT.
You are operating as an autonomous, tool-using agent. Work the problem in \
steps: decide what you need, call the appropriate tool, read the result, and \
continue until the task is genuinely done. Use one tool call at a time and let \
each result inform the next. Do not claim an action succeeded unless a tool \
result confirms it. When you have everything required, stop calling tools and \
give the user a clear, final answer that states what you did and what you found.""",
    AgentMode.DEEP: """\
Mode: DEEP REASONING.
This task warrants careful, deliberate thought. Before answering: clarify the \
real objective, break the problem into parts, consider more than one approach, \
and weigh trade-offs and edge cases. Use tools to ground your reasoning in facts \
rather than assumptions. Be rigorous about correctness — check your logic, your \
arithmetic, and your sources. Then deliver a well-structured, decisive answer: \
the conclusion first, the reasoning that supports it second, and any important \
caveats or assumptions made explicit. Depth and correctness matter more here \
than brevity, but stay organized and never pad.""",
}

# Guidance injected only when tools are actually advertised for the turn.
_TOOL_GUIDANCE = """\
Tool use:
- Choose the single most appropriate tool for the immediate sub-goal; provide \
arguments that exactly match each tool's schema.
- Treat tool output as the ground truth for this turn — prefer it over your \
prior assumptions, and quote concrete details from it when relevant.
- If a tool fails, read the error, adjust your arguments or approach, and retry \
sensibly; after repeated failure, explain the obstacle instead of pretending.
- Some tools change files or the system and require user confirmation; request \
them only when needed and explain why."""


# Output-channel contract. Always included: the bot renders replies as PLAIN
# Telegram text (no Markdown/HTML parsing), so heavy markup would show up as
# literal characters. Keep formatting light and chat-native.
_OUTPUT_GUIDANCE = """\
Output format (Telegram chat):
- Your replies are shown as plain text. Do NOT use Markdown or HTML markup: \
no **bold**, *italics*, `backticks`, #headings, or [text](links). Such symbols \
appear literally to the user.
- Structure with plain text only: short paragraphs, blank lines between them, \
and simple hyphen bullets ("- ") or numbered lists ("1.") when they aid \
scanning. Write URLs in full.
- Favor brevity — a phone screen is small. Lead with the answer; add detail only \
if it earns its place. Use a relevant emoji occasionally for warmth, never to \
decorate.
- If you must show code or a command, put it on its own lines and tell the user \
plainly that it is code."""


def system_prompt(
    mode: AgentMode,
    *,
    with_tools: bool = False,
    extra_sections: list[str] | None = None,
) -> str:
    """Compose the full system prompt for ``mode``.

    Section order: persona → mode contract → output-format contract →
    (tool guidance, if any) → extra sections (memories, preferences, plan).

    Parameters
    ----------
    mode:
        The resolved per-turn :class:`AgentMode`.
    with_tools:
        Append tool-use guidance (only meaningful when tools are advertised).
    extra_sections:
        Optional additional blocks (e.g. recalled memories, user preferences,
        the current plan) appended after the contract, in order.
    """
    parts: list[str] = [PERSONA, _MODE_CONTRACTS[mode], _OUTPUT_GUIDANCE]
    if with_tools:
        parts.append(_TOOL_GUIDANCE)
    if extra_sections:
        parts.extend(s.strip() for s in extra_sections if s and s.strip())
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Built context
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BuiltContext:
    """The assembled prompt plus accounting metadata.

    Attributes
    ----------
    messages:
        The final, ordered list ready to send to the LLM (system first).
    prompt_tokens:
        Approximate token count of :attr:`messages`.
    response_budget:
        Tokens reserved for the model's reply (``max_tokens`` to pass through).
    dropped_messages:
        How many history messages were trimmed to honor the budget — surfaced
        so the orchestrator can decide whether to summarize older history.
    """

    messages: list[ChatMessage]
    prompt_tokens: int
    response_budget: int
    dropped_messages: int = 0


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #
class ContextBuilder:
    """Assembles a token-budgeted prompt from a turn's raw materials.

    The builder never mutates its inputs. It reserves space for the system
    prompt, recalled memories and the response, then greedily fills the
    remaining budget with the most recent conversation history (newest first),
    dropping the oldest turns that do not fit.
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        max_context_tokens: int = 16_000,
        response_reserve_tokens: int = 1_024,
    ) -> None:
        """
        Parameters
        ----------
        token_counter:
            Shared :class:`TokenCounter`; one is created if omitted.
        max_context_tokens:
            Upper bound on the *prompt* (system + memories + history + input).
            Should be the model's context window minus the response reserve;
            defaults conservatively so unknown windows still behave.
        response_reserve_tokens:
            Floor of tokens always kept free for the model's reply.
        """
        self.tokens = token_counter or TokenCounter()
        self.max_context_tokens = max_context_tokens
        self.response_reserve_tokens = response_reserve_tokens

    # ------------------------------------------------------------------ #
    # Section rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_memories(memories: list["RecalledMemory"]) -> str | None:
        """Render recalled long-term memories as a system sub-section."""
        if not memories:
            return None
        lines = [
            "Relevant things you remember about this user and past conversations "
            "(use them naturally; do not announce that you are recalling them):"
        ]
        for mem in memories:
            label = getattr(mem, "memory_type", "note")
            content = getattr(mem, "content", str(mem)).strip()
            if content:
                lines.append(f"- ({label}) {content}")
        return "\n".join(lines) if len(lines) > 1 else None

    @staticmethod
    def _render_preferences(preferences: dict[str, Any] | None) -> str | None:
        """Render stored user preferences as a system sub-section."""
        if not preferences:
            return None
        rendered = ", ".join(f"{k}={v}" for k, v in preferences.items() if v not in (None, ""))
        if not rendered:
            return None
        return f"Known user preferences (honor these unless overridden): {rendered}."

    @staticmethod
    def _render_plan(plan: "Plan | None") -> str | None:
        """Render the active plan so the model executes against it."""
        if plan is None:
            return None
        steps = getattr(plan, "steps", None)
        if not steps:
            return None
        lines = [
            "You are executing the following plan. Work through the open steps in "
            "order; you do not need to mention the plan unless it helps the user:"
        ]
        for i, step in enumerate(steps, start=1):
            desc = getattr(step, "description", str(step))
            done = getattr(step, "done", False)
            mark = "x" if done else " "
            lines.append(f"  [{mark}] {i}. {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(
        self,
        *,
        mode: AgentMode,
        user_input: ChatMessage | str,
        history: list[ChatMessage] | None = None,
        memories: list["RecalledMemory"] | None = None,
        preferences: dict[str, Any] | None = None,
        plan: "Plan | None" = None,
        with_tools: bool = False,
        response_budget: int | None = None,
    ) -> BuiltContext:
        """Assemble the final prompt for one turn.

        The ordering is: system prompt → trimmed history (oldest kept that fit)
        → the current user input. History is filled newest-first against the
        remaining budget, then re-ordered chronologically for the model.

        Parameters
        ----------
        mode:
            Resolved per-turn mode driving the system prompt.
        user_input:
            The current user message (raw text or a prebuilt chat message).
        history:
            Prior messages in chronological order (oldest first), excluding the
            current input.
        memories / preferences / plan:
            Optional context blocks folded into the system prompt.
        with_tools:
            Whether tools are advertised (adds tool-use guidance).
        response_budget:
            Tokens to reserve for the reply; defaults to the configured reserve.
        """
        history = history or []
        reserve = max(response_budget or self.response_reserve_tokens, 1)

        # 1) System prompt with all available context sections.
        extra = [
            section
            for section in (
                self._render_preferences(preferences),
                self._render_memories(memories or []),
                self._render_plan(plan),
            )
            if section
        ]
        system_text = system_prompt(mode, with_tools=with_tools, extra_sections=extra)
        system_msg: ChatMessage = {"role": "system", "content": system_text}

        # 2) Normalize the current user input.
        input_msg: ChatMessage = (
            {"role": "user", "content": user_input}
            if isinstance(user_input, str)
            else dict(user_input)
        )

        # 3) Compute the budget left for history.
        fixed_tokens = (
            self.tokens.count_message(system_msg)
            + self.tokens.count_message(input_msg)
        )
        history_budget = self.max_context_tokens - fixed_tokens
        if history_budget < 0:
            log.warning(
                "System prompt + input ({} tok) exceed the context budget ({} tok); "
                "history will be dropped entirely.",
                fixed_tokens,
                self.max_context_tokens,
            )
            history_budget = 0

        # 4) Greedily keep the most recent history that fits (newest first).
        kept_reversed: list[ChatMessage] = []
        used = 0
        dropped = 0
        for msg in reversed(history):
            cost = self.tokens.count_message(msg)
            if used + cost <= history_budget:
                kept_reversed.append(msg)
                used += cost
            else:
                dropped += 1
        kept = list(reversed(kept_reversed))

        # Avoid leading orphan tool/assistant-tool messages that would reference
        # a request now trimmed away — drop from the front until we hit a clean
        # user/assistant boundary the model can interpret on its own.
        kept = self._trim_dangling_tool_prefix(kept)

        messages = [system_msg, *kept, input_msg]
        prompt_tokens = fixed_tokens + self.tokens.count_messages(kept)

        log.debug(
            "Built context: mode={}, messages={}, prompt_tokens={}, dropped={}.",
            mode.value,
            len(messages),
            prompt_tokens,
            dropped,
        )
        return BuiltContext(
            messages=messages,
            prompt_tokens=prompt_tokens,
            response_budget=reserve,
            dropped_messages=dropped,
        )

    @staticmethod
    def _trim_dangling_tool_prefix(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Drop leading ``tool`` replies (and tool-only assistant turns) whose
        originating request was trimmed, which providers reject."""
        start = 0
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                start += 1
                continue
            if role == "assistant" and msg.get("tool_calls") and not msg.get("content"):
                start += 1
                continue
            break
        return messages[start:] if start else messages
