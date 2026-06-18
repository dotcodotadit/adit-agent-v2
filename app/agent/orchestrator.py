"""The top-level agent coordinator for Adit-Agent.

The :class:`Orchestrator` is the single entry point the bot layer talks to. It
turns a raw user request into a streamed answer by wiring together every other
piece of the agent core, in order:

    resolve user + conversation   (memory_manager)
      → choose a mode             (chat / agent / deep reasoning)
      → recall memories + history (memory_manager)
      → optionally plan           (planner)
      → build the prompt          (context_builder, token-budgeted)
      → run the ReAct loop        (executor, streaming)
      → optionally self-reflect   (deep / agentic modes)
      → persist + consolidate     (memory_manager)

It exposes two surfaces: :meth:`Orchestrator.stream`, an async generator of
:class:`AgentEvent` objects for live UIs, and :meth:`Orchestrator.run`, a
convenience wrapper that drains the stream into a single :class:`AgentResponse`.

The orchestrator owns *policy* (which mode, whether to plan, whether to reflect,
which tools to advertise) and delegates *mechanism* to its collaborators, so the
behavioral knobs live in one readable place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from app.agent.context_builder import AgentMode, ContextBuilder, TokenCounter
from app.agent.executor import AgentEvent, EventType, ExecutionResult, Executor
from app.agent.planner import Plan, Planner, _extract_json_object
from app.database.models import ConversationMode, MemoryType, MessageRole
from app.tools.base import ToolContext
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.agent.context_builder import ChatMessage, LLMRouter
    from app.agent.executor import Confirmer
    from app.agent.memory_manager import MemoryManager
    from app.config import Settings
    from app.tools.base import Tool
    from app.tools.registry import ToolRegistry

log = get_logger(__name__)

__all__ = ["AgentRequest", "AgentResponse", "Orchestrator"]


# Phrases that nudge an otherwise-ordinary request into deep-reasoning mode.
_DEEP_TRIGGERS: tuple[str, ...] = (
    "think hard", "think carefully", "deep dive", "deeply", "step by step",
    "step-by-step", "reason through", "carefully analyze", "be thorough",
    "rigorous", "work it out",
)
# Conversation defaults that imply tool-using (agentic) behavior.
_AGENTIC_MODES = {ConversationMode.AGENT, ConversationMode.PLANNER}


# --------------------------------------------------------------------------- #
# Request / response DTOs
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AgentRequest:
    """A single inbound turn from a user.

    Attributes
    ----------
    telegram_id:
        The Telegram user id (natural key); the orchestrator resolves it to a
        :class:`~app.database.models.User`.
    text:
        The user's message text.
    username / first_name:
        Optional profile fields kept fresh on the user record.
    mode:
        Explicit mode override; when ``None`` the orchestrator infers one.
    conversation_id:
        Continue a specific conversation; when ``None`` the user's most recent
        conversation is used (or a new one opened).
    allow_tools:
        Whether tools may be advertised this turn (e.g. disabled for a quick
        "/ask" command).
    confirmer:
        Async predicate to approve dangerous tools; without it, dangerous tools
        are refused.
    """

    telegram_id: int
    text: str
    username: str | None = None
    first_name: str | None = None
    mode: AgentMode | None = None
    conversation_id: int | None = None
    allow_tools: bool = True
    confirmer: "Confirmer | None" = None


@dataclass(slots=True)
class AgentResponse:
    """The fully-resolved outcome of a turn (non-streaming convenience)."""

    text: str
    mode: AgentMode
    conversation_id: int
    iterations: int = 0
    stop_reason: str = "completed"
    plan: Plan | None = None
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    reflected: bool = False


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Coordinates memory, planning, execution and reflection for each turn."""

    def __init__(
        self,
        *,
        provider_router: "LLMRouter",
        registry: "ToolRegistry",
        memory_manager: "MemoryManager",
        settings: "Settings",
        context_builder: ContextBuilder | None = None,
        planner: Planner | None = None,
        executor: Executor | None = None,
        max_context_tokens: int = 16_000,
        max_chat_tools: int = 6,
        reflect_in_agent_mode: bool = False,
        auto_memory: bool = True,
    ) -> None:
        self._provider = provider_router
        self._registry = registry
        self._memory = memory_manager
        self._settings = settings

        token_counter = TokenCounter(settings.llm_default_model)
        self._context = context_builder or ContextBuilder(
            token_counter=token_counter,
            max_context_tokens=max_context_tokens,
            response_reserve_tokens=settings.agent_max_tokens,
        )
        self._planner = planner or Planner(
            provider_router,
            max_steps=settings.agent_max_steps,
            default_model=settings.llm_default_model,
        )
        self._executor = executor or Executor(
            provider_router=provider_router,
            registry=registry,
            settings=settings,
            default_model=settings.llm_default_model,
        )
        self._tokens = token_counter
        self._max_chat_tools = max_chat_tools
        self._reflect_in_agent_mode = reflect_in_agent_mode
        self._auto_memory = auto_memory

    # ================================================================== #
    # Public API
    # ================================================================== #
    async def run(self, request: AgentRequest) -> AgentResponse:
        """Process a turn end-to-end and return the final :class:`AgentResponse`.

        Convenience wrapper around :meth:`stream` that ignores intermediate
        events and returns only the resolved result.
        """
        response: AgentResponse | None = None
        async for event in self.stream(request):
            if event.type is EventType.FINAL:
                response = event.data.get("response")
        if response is None:  # pragma: no cover - stream always ends with FINAL
            raise RuntimeError("Execution stream ended without a final response.")
        return response

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        """Process a turn, yielding live :class:`AgentEvent` objects.

        The stream always ends with a ``FINAL`` event whose ``data["response"]``
        is the :class:`AgentResponse`. The executor's own internal ``FINAL`` is
        intercepted so reflection and persistence can run before the public
        ``FINAL`` is emitted.
        """
        try:
            async for event in self._run(request):
                yield event
        except Exception as exc:  # noqa: BLE001 - never leak a raw error to the bot
            log.exception("Orchestrator failed for telegram_id={}.", request.telegram_id)
            yield AgentEvent.error(
                "Something went wrong while I was working on that. Please try again."
            )
            yield AgentEvent(
                EventType.FINAL,
                text="",
                data={
                    "response": AgentResponse(
                        text="",
                        mode=request.mode or AgentMode.CHAT,
                        conversation_id=request.conversation_id or 0,
                        stop_reason="error",
                    )
                },
            )

    # ================================================================== #
    # Core pipeline
    # ================================================================== #
    async def _run(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        text = (request.text or "").strip()

        # 1) Identity + conversation.
        user = await self._memory.get_or_create_user(
            request.telegram_id,
            username=request.username,
            first_name=request.first_name,
        )
        conversation = None
        if request.conversation_id is not None:
            conversation = await self._memory.get_conversation(request.conversation_id)
        if conversation is None:
            conversation = await self._memory.get_or_create_conversation(user.id)
        conversation_id = conversation.id

        # 2) Resolve the per-turn mode.
        mode = self._resolve_mode(request, default=conversation.mode)
        log.info(
            "Turn: user={} conv={} mode={} tools={}.",
            user.id, conversation_id, mode.value, request.allow_tools,
        )

        # 3) Gather context: long-term memories + short-term history.
        memories = await self._memory.recall(user.id, text, limit=5)
        history = await self._memory.load_history(conversation_id)

        # 4) Dynamic tool selection.
        include_dangerous = request.confirmer is not None
        tool_schemas = (
            self._select_tools(text, mode=mode, include_dangerous=include_dangerous)
            if request.allow_tools
            else []
        )

        # 5) Plan complex tasks before execution.
        plan: Plan | None = None
        if self._settings.enable_planner and self._planner.should_plan(
            text, mode=mode, tool_count=len(tool_schemas)
        ):
            plan = await self._planner.make_plan(
                text,
                mode=mode,
                tool_names=[s["function"]["name"] for s in tool_schemas],
                history=history,
            )
            if plan:
                yield AgentEvent(
                    EventType.PLAN, text=plan.render(), data={"plan": plan}
                )

        # 6) Build the token-budgeted prompt.
        built = self._context.build(
            mode=mode,
            user_input=text,
            history=history,
            memories=memories,
            preferences=dict(user.preferences or {}),
            plan=plan,
            with_tools=bool(tool_schemas),
            response_budget=self._settings.agent_max_tokens,
        )

        # 7) Run the streaming ReAct loop, intercepting its terminal result.
        tool_context = ToolContext(
            settings=self._settings,
            provider_router=self._provider,
            user_id=user.id,
        )
        result: ExecutionResult | None = None
        async for event in self._executor.run(
            messages=built.messages,
            mode=mode,
            tool_schemas=tool_schemas or None,
            tool_context=tool_context,
            max_iterations=self._settings.agent_max_steps,
            model=self._settings.llm_default_model,
            response_budget=self._settings.agent_max_tokens,
            confirmer=request.confirmer,
        ):
            if event.type is EventType.FINAL:
                result = event.data.get("result")
            else:
                yield event
        if result is None:  # pragma: no cover - executor always emits FINAL
            result = ExecutionResult(stop_reason="error")

        # 8) Optional self-reflection (deep mode, or agent mode if enabled).
        reflected = False
        if self._should_reflect(mode, result):
            async for event, revised in self._reflect(text, result.final_text, mode):
                yield event
                if revised is not None:
                    result.final_text = revised
                    reflected = True

        # 9) Persist the turn and consolidate memory (best-effort).
        await self._persist_turn(
            conversation_id=conversation_id,
            user_id=user.id,
            user_text=text,
            result=result,
        )
        if self._auto_memory and mode is not AgentMode.CHAT:
            await self._extract_memories(user.id, conversation_id, text, result.final_text)
        await self._memory.summarize_if_needed(conversation_id, user.id)

        # 10) Emit the public terminal event.
        yield AgentEvent(
            EventType.FINAL,
            text=result.final_text,
            data={
                "response": AgentResponse(
                    text=result.final_text,
                    mode=mode,
                    conversation_id=conversation_id,
                    iterations=result.iterations,
                    stop_reason=result.stop_reason,
                    plan=plan,
                    tool_invocations=result.tool_invocations,
                    reflected=reflected,
                )
            },
        )

    # ================================================================== #
    # Mode resolution
    # ================================================================== #
    def _resolve_mode(
        self, request: AgentRequest, *, default: ConversationMode | None
    ) -> AgentMode:
        """Pick the per-turn :class:`AgentMode`.

        Precedence: explicit request override → deep-reasoning trigger phrases →
        the conversation's stored default → plain chat.
        """
        if request.mode is not None:
            return request.mode

        text = (request.text or "").lower()
        if any(trigger in text for trigger in _DEEP_TRIGGERS):
            return AgentMode.DEEP
        if default in _AGENTIC_MODES:
            return AgentMode.AGENT
        return AgentMode.CHAT

    # ================================================================== #
    # Dynamic tool selection
    # ================================================================== #
    def _select_tools(
        self, user_text: str, *, mode: AgentMode, include_dangerous: bool
    ) -> list[dict[str, Any]]:
        """Choose which tools to advertise to the model this turn.

        Agent and deep-reasoning turns receive the full toolset, since the loop
        may need any capability as it unfolds. Chat turns receive only the tools
        whose name/description/category overlaps the request (capped), so casual
        conversation isn't padded with irrelevant schemas — while still letting
        the model reach for a tool when the message clearly calls for one.
        """
        candidates: list["Tool"] = [
            t for t in self._registry.all() if include_dangerous or not t.dangerous
        ]
        if not candidates:
            return []

        if mode in (AgentMode.AGENT, AgentMode.DEEP):
            return [t.openai_schema() for t in candidates]

        scored = [
            (self._tool_relevance(tool, user_text), tool) for tool in candidates
        ]
        relevant = sorted(
            (pair for pair in scored if pair[0] > 0),
            key=lambda pair: pair[0],
            reverse=True,
        )
        chosen = [tool for _, tool in relevant[: self._max_chat_tools]]
        return [tool.openai_schema() for tool in chosen]

    @staticmethod
    def _tool_relevance(tool: "Tool", user_text: str) -> int:
        """Score a tool's relevance to ``user_text`` by keyword overlap."""
        words = set(re.findall(r"[a-z0-9]+", user_text.lower()))
        if not words:
            return 0
        haystack = set(
            re.findall(
                r"[a-z0-9]+",
                f"{tool.name} {tool.category} {tool.description}".lower(),
            )
        )
        # Name/category matches are worth more than description matches.
        name_words = set(re.findall(r"[a-z0-9]+", f"{tool.name} {tool.category}".lower()))
        score = 0
        for word in words:
            if word in name_words:
                score += 3
            elif word in haystack:
                score += 1
        return score

    # ================================================================== #
    # Reflection
    # ================================================================== #
    def _should_reflect(self, mode: AgentMode, result: ExecutionResult) -> bool:
        """Decide whether to run a post-hoc self-review."""
        if result.stop_reason == "error" or not result.final_text.strip():
            return False
        if mode is AgentMode.DEEP:
            return True
        if mode is AgentMode.AGENT and self._reflect_in_agent_mode:
            return True
        return False

    async def _reflect(
        self, user_text: str, draft: str, mode: AgentMode
    ) -> AsyncIterator[tuple[AgentEvent, str | None]]:
        """Critique the draft answer, optionally producing a revision.

        Yields a single ``(event, revised_text_or_None)`` pair: ``revised`` is
        ``None`` when the draft passed review, otherwise the improved answer.
        Best-effort — provider failures pass the draft through unchanged.
        """
        system = (
            "You are Adit performing a rigorous self-review. Given the user's "
            "request and your draft answer, judge whether the draft is correct, "
            "complete, and directly responsive. If it is already good, reply with "
            "exactly the token OK and nothing else. Otherwise, reply ONLY with an "
            "improved final answer (no preamble, no mention of this review)."
        )
        try:
            response = await self._provider.complete(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"User request:\n{user_text}\n\nDraft answer:\n{draft}",
                    },
                ],
                model=self._settings.llm_default_model,
                temperature=0.2,
                max_tokens=self._settings.agent_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - reflection is optional
            log.warning("Reflection failed; keeping the draft answer: {}", exc)
            yield (
                AgentEvent(EventType.REFLECTION, text="review skipped", data={"revised": False}),
                None,
            )
            return

        verdict = (response.content or "").strip()
        if not verdict or _is_approval(verdict):
            yield (
                AgentEvent(
                    EventType.REFLECTION, text="verified", data={"revised": False}
                ),
                None,
            )
            return

        log.info("Reflection revised the draft answer.")
        yield (
            AgentEvent(EventType.REFLECTION, text=verdict, data={"revised": True}),
            verdict,
        )

    # ================================================================== #
    # Persistence + memory consolidation
    # ================================================================== #
    async def _persist_turn(
        self,
        *,
        conversation_id: int,
        user_id: int,
        user_text: str,
        result: ExecutionResult,
    ) -> None:
        """Write the user message and the assistant message (with tool calls)."""
        try:
            await self._memory.record_message(
                conversation_id,
                MessageRole.USER,
                user_text,
                token_count=self._tokens.count_text(user_text),
            )
            await self._memory.record_message(
                conversation_id,
                MessageRole.ASSISTANT,
                result.final_text,
                token_count=self._tokens.count_text(result.final_text),
                tool_calls=result.tool_invocations or None,
            )
        except Exception as exc:  # noqa: BLE001 - persistence must not break the reply
            log.warning("Failed to persist turn for conv={}: {}", conversation_id, exc)

    async def _extract_memories(
        self, user_id: int, conversation_id: int, user_text: str, answer: str
    ) -> None:
        """Mine the turn for durable user facts/preferences worth remembering.

        Best-effort and quiet: any failure (or an empty extraction) is a no-op.
        """
        system = (
            "Extract any durable, user-specific facts or stated preferences from "
            "this exchange that would help in future conversations (names, goals, "
            "settings, recurring context). Ignore one-off task details and "
            "anything ephemeral. Respond with STRICT JSON: "
            '{"memories": [{"content": "<concise fact>", '
            '"type": "fact|preference", "importance": 0.0-1.0}]}. '
            "Return an empty list if nothing is worth keeping."
        )
        try:
            response = await self._provider.complete(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"User: {user_text}\n\nAssistant: {answer}",
                    },
                ],
                model=self._settings.llm_default_model,
                temperature=0.1,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Memory extraction call failed: {}", exc)
            return

        payload = _extract_json_object(response.content or "")
        if not payload:
            return
        for item in payload.get("memories", []) or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            mem_type = (
                MemoryType.PREFERENCE
                if str(item.get("type", "fact")).lower() == "preference"
                else MemoryType.FACT
            )
            importance = item.get("importance", 0.5)
            try:
                importance = float(importance)
            except (TypeError, ValueError):
                importance = 0.5
            await self._memory.remember(
                user_id,
                content,
                memory_type=mem_type,
                importance=importance,
                conversation_id=conversation_id,
            )

    # ================================================================== #
    # Lifecycle
    # ================================================================== #
    async def aclose(self) -> None:
        """Release orchestrator-owned resources (none beyond shared ones).

        Present so the application container can call it uniformly during
        shutdown; the provider router and database are closed by the container.
        """
        log.debug("Orchestrator closed.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_approval(verdict: str) -> bool:
    """True when a reflection verdict is an 'OK' approval rather than a rewrite."""
    head = verdict.strip().strip("\"'`.").upper()
    return head == "OK" or head.startswith("OK ") or head.startswith("OK\n")
