"""The ReAct execution loop for Adit-Agent.

The :class:`Executor` is the agent's hands. Given a fully-built prompt and a set
of advertised tools, it runs the classic *Reason + Act* loop: ask the model what
to do, run any tools it requests, feed the results back, and repeat until the
model produces a final answer or a hard iteration limit is reached.

It is built around two design commitments:

* **Everything streams.** :meth:`Executor.run` is an async generator of
  :class:`AgentEvent` objects — text deltas, tool start/result markers, turn
  boundaries — so the bot layer can surface live progress (typing the answer,
  showing "🔧 searching the web…") instead of waiting for the whole loop.
* **Failures never escape.** Unknown tools, tool errors, denied dangerous
  actions and provider hiccups are all turned into events and fed back to the
  model as context so it can adapt, rather than raised. The loop is bounded, so
  it always terminates with a ``FINAL`` event.

Token deltas are emitted live as they stream. Because a turn that *prefaces a
tool call with reasoning* cannot be told apart from a final answer until the
stream ends, every model turn is closed by a ``TURN_END`` event whose ``final``
flag tells the consumer whether the tokens it just saw were the answer or
interim thinking. In practice tool-calling turns carry no text, so live tokens
are almost always the final answer.

Dangerous tools (file writes, shell, browser, ...) are gated behind a caller-
supplied confirmation callback, honoring ``settings.require_tool_confirmation``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from app.agent.context_builder import LLMResponse
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.agent.context_builder import AgentMode, ChatMessage, LLMRouter
    from app.config import Settings
    from app.tools.base import Tool

log = get_logger(__name__)

__all__ = [
    "EventType",
    "AgentEvent",
    "ExecutionResult",
    "Confirmer",
    "Executor",
]

# An async predicate the orchestrator supplies to approve a dangerous tool call.
# Receives the resolved tool and its arguments; returns True to allow.
Confirmer = Callable[["Tool", dict[str, Any]], Awaitable[bool]]

# Tool output fed back to the model is capped so a single chatty tool cannot
# blow the context window. The model still sees success/error plus a useful slice.
_MAX_TOOL_RESULT_CHARS = 8_000

# Sentinel iteration value used for the post-limit forced final answer.
_FORCED_ITERATION = -1


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    """Kinds of streamed events emitted during execution."""

    TOKEN = "token"              # incremental text delta from the current turn
    TURN_END = "turn_end"        # a model turn finished (data: final, iteration)
    TOOL_START = "tool_start"    # about to invoke a tool (tool_name, arguments)
    TOOL_RESULT = "tool_result"  # tool finished (tool_name, success, text)
    TOOL_DENIED = "tool_denied"  # dangerous tool not confirmed (tool_name)
    PLAN = "plan"                # orchestrator built a plan (data: plan -> Plan)
    REFLECTION = "reflection"    # post-hoc self-review (data: revised: bool)
    FINAL = "final"              # loop finished (data: result -> ExecutionResult)
    ERROR = "error"              # unrecoverable error (text)


@dataclass(slots=True)
class AgentEvent:
    """A single streamed event from the executor.

    Only the fields relevant to :attr:`type` are populated; the rest keep their
    defaults. ``data`` carries type-specific extras (e.g. the terminal
    :class:`ExecutionResult` on a ``FINAL`` event, or ``final``/``iteration`` on
    a ``TURN_END``).
    """

    type: EventType
    text: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    success: bool | None = None
    data: dict[str, Any] = field(default_factory=dict)

    # Convenience constructors ----------------------------------------- #
    @classmethod
    def token(cls, text: str, *, iteration: int) -> "AgentEvent":
        return cls(EventType.TOKEN, text=text, data={"iteration": iteration})

    @classmethod
    def turn_end(cls, *, final: bool, text: str, iteration: int) -> "AgentEvent":
        return cls(
            EventType.TURN_END,
            text=text,
            data={"final": final, "iteration": iteration},
        )

    @classmethod
    def tool_start(cls, name: str, arguments: dict[str, Any]) -> "AgentEvent":
        return cls(EventType.TOOL_START, tool_name=name, arguments=arguments)

    @classmethod
    def tool_result(cls, name: str, *, success: bool, text: str) -> "AgentEvent":
        return cls(EventType.TOOL_RESULT, tool_name=name, success=success, text=text)

    @classmethod
    def tool_denied(cls, name: str, *, reason: str) -> "AgentEvent":
        return cls(EventType.TOOL_DENIED, tool_name=name, text=reason)

    @classmethod
    def error(cls, message: str) -> "AgentEvent":
        return cls(EventType.ERROR, text=message)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ExecutionResult:
    """The terminal outcome of an execution loop.

    Attributes
    ----------
    final_text:
        The assistant's final answer.
    tool_invocations:
        One record per executed tool call, shaped for persistence by the memory
        manager (``name``, ``arguments``, ``result``, ``success``,
        ``execution_time``).
    iterations:
        Number of model turns taken.
    stop_reason:
        ``"completed"`` (model answered), ``"max_iterations"`` (forced answer),
        or ``"error"``.
    """

    final_text: str = ""
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = "completed"


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
class Executor:
    """Runs the streaming ReAct loop against a provider and tool registry."""

    def __init__(
        self,
        *,
        provider_router: "LLMRouter",
        registry: ToolRegistry,
        settings: "Settings",
        default_model: str | None = None,
    ) -> None:
        self._provider = provider_router
        self._registry = registry
        self._settings = settings
        self._model = default_model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def run(
        self,
        *,
        messages: list["ChatMessage"],
        mode: "AgentMode",
        tool_schemas: list[dict[str, Any]] | None = None,
        tool_context: ToolContext | None = None,
        max_iterations: int | None = None,
        model: str | None = None,
        response_budget: int | None = None,
        confirmer: Confirmer | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the ReAct loop, yielding :class:`AgentEvent` objects.

        The final event is always a ``FINAL`` carrying an
        :class:`ExecutionResult`. Tools are advertised only when ``tool_schemas``
        is non-empty; otherwise the loop reduces to a single streamed answer.

        Parameters
        ----------
        messages:
            The fully-built prompt (system + history + current input).
        mode:
            The active agent mode (kept for logging / future per-mode tuning).
        tool_schemas:
            OpenAI tool definitions to advertise (e.g. from
            ``registry.openai_schemas()``).
        tool_context:
            Shared :class:`ToolContext` passed to every tool invocation.
        max_iterations:
            Loop cap; defaults to ``settings.agent_max_steps``.
        model / response_budget:
            Forwarded to the provider as the model id and ``max_tokens``.
        confirmer:
            Async predicate gating dangerous tools; see :data:`Confirmer`.
        """
        working: list["ChatMessage"] = list(messages)
        tools = tool_schemas or None
        max_iter = max(1, max_iterations or self._settings.agent_max_steps)
        invocations: list[dict[str, Any]] = []

        for iteration in range(1, max_iter + 1):
            holder: dict[str, Any] = {}

            # 1) Stream one model turn, emitting text deltas live as they arrive.
            try:
                async for ev in self._stream_turn(
                    working,
                    tools=tools,
                    model=model,
                    response_budget=response_budget,
                    iteration=iteration,
                    holder=holder,
                ):
                    yield ev
            except Exception as exc:  # noqa: BLE001 - provider failure → clean stop
                log.exception("Provider error during execution turn {}.", iteration)
                yield AgentEvent.error(f"The model provider failed: {exc}")
                yield _final_event(
                    ExecutionResult(
                        final_text=(
                            "I hit a problem reaching my reasoning engine and "
                            "couldn't complete that. Please try again."
                        ),
                        tool_invocations=invocations,
                        iterations=iteration,
                        stop_reason="error",
                    )
                )
                return

            response: "LLMResponse" = holder["response"]
            streamed_text: str = holder.get("text", "")

            # 2a) No tool calls → this turn is the final answer.
            if not response.wants_tools:
                final_text = response.content or streamed_text
                working.append({"role": "assistant", "content": final_text})
                yield AgentEvent.turn_end(
                    final=True, text=final_text, iteration=iteration
                )
                yield _final_event(
                    ExecutionResult(
                        final_text=final_text,
                        tool_invocations=invocations,
                        iterations=iteration,
                        stop_reason="completed",
                    )
                )
                return

            # 2b) Tool calls requested → this was a reasoning/acting turn.
            yield AgentEvent.turn_end(
                final=False, text=response.content, iteration=iteration
            )
            working.append(self._assistant_tool_message(response))

            # 3) Execute each requested tool, appending results for the next turn.
            for call in response.tool_calls:
                async for event, tool_message, record in self._run_tool_call(
                    call, tool_context=tool_context, confirmer=confirmer
                ):
                    if event is not None:
                        yield event
                    if tool_message is not None:
                        working.append(tool_message)
                    if record is not None:
                        invocations.append(record)

        # 4) Iteration budget exhausted — force a final answer without tools.
        holder = {}
        async for ev in self._stream_turn(
            self._with_limit_nudge(working),
            tools=None,
            model=model,
            response_budget=response_budget,
            iteration=_FORCED_ITERATION,
            holder=holder,
        ):
            yield ev
        forced = holder.get("response")
        final_text = (
            (forced.content if forced else "")
            or holder.get("text", "")
            or (
                "I reached my step limit before fully finishing. Here is what I "
                "have so far — tell me if you'd like me to keep going."
            )
        )
        yield AgentEvent.turn_end(
            final=True, text=final_text, iteration=_FORCED_ITERATION
        )
        yield _final_event(
            ExecutionResult(
                final_text=final_text,
                tool_invocations=invocations,
                iterations=max_iter,
                stop_reason="max_iterations",
            )
        )

    # ------------------------------------------------------------------ #
    # Turn streaming
    # ------------------------------------------------------------------ #
    async def _stream_turn(
        self,
        messages: list["ChatMessage"],
        *,
        tools: list[dict[str, Any]] | None,
        model: str | None,
        response_budget: int | None,
        iteration: int,
        holder: dict[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        """Stream one completion, yielding ``TOKEN`` events as text arrives.

        The resolved :class:`LLMResponse` and the concatenated text are stashed
        into ``holder`` (keys ``"response"`` and ``"text"``) for the caller,
        since an async generator cannot return a value directly.
        """
        response: "LLMResponse | None" = None
        parts: list[str] = []

        async for chunk in self._provider.stream(
            messages,
            model=model or self._model,
            tools=tools,
            temperature=self._settings.llm_temperature,
            max_tokens=response_budget or self._settings.agent_max_tokens,
        ):
            if chunk.type == "text" and chunk.text:
                parts.append(chunk.text)
                yield AgentEvent.token(chunk.text, iteration=iteration)
            elif chunk.type == "done":
                response = chunk.response

        if response is None:
            # Defensive: a stream that never produced a terminal chunk. Synthesize
            # a response from whatever text we buffered so the loop can finish.
            response = LLMResponse(content="".join(parts), finish_reason="stop")

        holder["response"] = response
        holder["text"] = "".join(parts)

    def _with_limit_nudge(
        self, messages: list["ChatMessage"]
    ) -> list["ChatMessage"]:
        """Append the system nudge that forces a final answer after the cap."""
        return [
            *messages,
            {
                "role": "system",
                "content": (
                    "You have reached the maximum number of tool-using steps. "
                    "Stop calling tools and give your best final answer now using "
                    "what you have already gathered. If something remains "
                    "uncertain, say so plainly."
                ),
            },
        ]

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    async def _run_tool_call(
        self,
        call: Any,
        *,
        tool_context: ToolContext | None,
        confirmer: Confirmer | None,
    ) -> AsyncIterator[
        tuple[AgentEvent | None, "ChatMessage | None", dict[str, Any] | None]
    ]:
        """Resolve, gate, and execute a single tool call.

        Yields ``(event, tool_message, record)`` triples so the caller can relay
        the event to the user, append the tool reply to the working transcript,
        and record the invocation for persistence — any of which may be ``None``.
        """
        name = call.name
        arguments = call.arguments or {}

        # Resolve the tool; an unknown name is reported back so the model recovers.
        tool = self._registry.try_get(name)
        if tool is None:
            msg = (
                f"Unknown tool {name!r}. Available tools: "
                f"{', '.join(self._registry.names()) or 'none'}."
            )
            yield (
                AgentEvent.tool_result(name, success=False, text=msg),
                self._tool_reply(call.id, name, msg),
                None,
            )
            return

        yield (AgentEvent.tool_start(name, arguments), None, None)

        # Gate dangerous tools behind confirmation.
        allowed, deny_reason = await self._authorize(tool, arguments, confirmer)
        if not allowed:
            yield (
                AgentEvent.tool_denied(name, reason=deny_reason),
                self._tool_reply(
                    call.id, name, f"Action not permitted: {deny_reason}"
                ),
                None,
            )
            return

        # Execute, timing the call. ``Tool.invoke`` never raises.
        ctx = self._context_for(tool_context, confirmed=True)
        started = time.perf_counter()
        result = await tool.invoke(arguments, ctx)
        elapsed = time.perf_counter() - started

        result_text = self._render_result(result)
        record = {
            "name": name,
            "arguments": arguments,
            "result": result_text,
            "success": result.success,
            "execution_time": round(elapsed, 4),
        }
        yield (
            AgentEvent.tool_result(name, success=result.success, text=result_text),
            self._tool_reply(call.id, name, result_text),
            record,
        )

    async def _authorize(
        self,
        tool: "Tool",
        arguments: dict[str, Any],
        confirmer: Confirmer | None,
    ) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for a (possibly dangerous) tool call."""
        if not tool.dangerous or not self._settings.require_tool_confirmation:
            return True, ""
        if confirmer is None:
            # No way to ask → refuse by default (safe), telling the model why.
            return (
                False,
                f"the tool {tool.name!r} can modify the system and requires "
                "explicit user confirmation, which is unavailable here",
            )
        try:
            approved = await confirmer(tool, arguments)
        except Exception as exc:  # noqa: BLE001 - a failing confirmer denies safely
            log.warning("Confirmer raised for tool {}: {}", tool.name, exc)
            return False, "confirmation could not be obtained"
        if approved:
            return True, ""
        return False, "the user declined to confirm this action"

    # ------------------------------------------------------------------ #
    # Message / result helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _assistant_tool_message(response: Any) -> "ChatMessage":
        """Build the assistant message that records the requested tool calls."""
        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [c.to_assistant_tool_call() for c in response.tool_calls],
        }

    @staticmethod
    def _tool_reply(call_id: str, name: str, content: str) -> "ChatMessage":
        """Build the ``tool`` message carrying a call's result."""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

    @staticmethod
    def _render_result(result: ToolResult) -> str:
        """Render a :class:`ToolResult` into compact text for the model.

        Successful outputs are JSON-encoded when structured; failures surface the
        error message. Everything is truncated to a sane budget.
        """
        if result.success:
            output = result.output
            if isinstance(output, str):
                text = output
            else:
                try:
                    text = json.dumps(output, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    text = str(output)
        else:
            text = f"ERROR: {result.error}"

        if len(text) > _MAX_TOOL_RESULT_CHARS:
            text = (
                text[:_MAX_TOOL_RESULT_CHARS]
                + f"\n…[truncated {len(text) - _MAX_TOOL_RESULT_CHARS} chars]"
            )
        return text

    @staticmethod
    def _context_for(
        base: ToolContext | None, *, confirmed: bool
    ) -> ToolContext | None:
        """Return a per-call context with the confirmation flag set.

        A fresh copy is made (via :func:`dataclasses.replace`) so flipping
        ``confirmed`` for one dangerous call never leaks into sibling calls.
        """
        if base is None:
            return None
        if base.confirmed == confirmed:
            return base
        return replace(base, confirmed=confirmed)


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
def _final_event(result: ExecutionResult) -> AgentEvent:
    """Build the terminal ``FINAL`` event carrying the execution result."""
    return AgentEvent(
        EventType.FINAL, text=result.final_text, data={"result": result}
    )
