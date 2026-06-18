"""Tests for the streaming ReAct executor loop."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.context_builder import AgentMode, LLMResponse, StreamChunk
from app.agent.executor import AgentEvent, EventType, ExecutionResult, Executor
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import ToolRegistry, tool
from pydantic import BaseModel, Field
from tests.conftest import FakeRouter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _registry_with_echo() -> ToolRegistry:
    """Build an isolated registry with a single 'echo' tool.

    Uses the global registry (to avoid the falsy-empty-registry footgun) then
    returns a fresh ToolRegistry populated by copying over what we need.
    """
    from app.tools.registry import get_registry

    global_reg = get_registry()
    # Register onto the global so the decorator's `registry or _registry` resolves.

    class EchoArgs(BaseModel):
        value: str = Field(description="text")

    @tool(name="echo_exec_test", description="Echo text for executor tests.",
          args=EchoArgs)
    async def _echo(args: EchoArgs, ctx: ToolContext | None) -> ToolResult:
        return ToolResult.ok(f"echoed:{args.value}")

    # Build an isolated registry containing only our test tool.
    reg = ToolRegistry()
    echo_tool = global_reg.get("echo_exec_test")
    reg._tools["echo_exec_test"] = echo_tool
    return reg


def _executor(router, registry=None) -> Executor:
    class _Settings:
        agent_max_steps = 4
        agent_max_tokens = 128
        llm_temperature = 0.7
        require_tool_confirmation = True

    return Executor(
        provider_router=router,
        registry=registry or ToolRegistry(),
        settings=_Settings(),
    )


async def _collect(executor, **kwargs) -> tuple[list[AgentEvent], ExecutionResult | None]:
    """Drain the executor and return all events + the final result."""
    events = []
    result = None
    async for ev in executor.run(**kwargs):
        events.append(ev)
        if ev.type is EventType.FINAL:
            result = ev.data.get("result")
    return events, result


# --------------------------------------------------------------------------- #
# Direct answer (no tools)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_direct_answer_streams_tokens_and_emits_final():
    router = FakeRouter(script=[LLMResponse(content="Hello world")])
    ex = _executor(router)
    events, result = await _collect(
        ex,
        messages=[{"role": "user", "content": "hi"}],
        mode=AgentMode.CHAT,
    )
    token_texts = "".join(e.text for e in events if e.type is EventType.TOKEN)
    assert "Hello world" in token_texts
    assert result is not None
    assert result.final_text == "Hello world"
    assert result.stop_reason == "completed"
    assert result.iterations == 1


# --------------------------------------------------------------------------- #
# Tool round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tool_roundtrip_emits_start_and_result():
    from app.agent.context_builder import ToolCallRequest

    reg = _registry_with_echo()

    class ToolThenAnswerRouter:
        _turn = 0

        async def stream(self, messages, **_k):
            self._turn += 1
            if self._turn == 1:
                resp = LLMResponse(
                    tool_calls=[ToolCallRequest(id="c1", name="echo_exec_test",
                                               arguments={"value": "hi"})],
                    finish_reason="tool_calls",
                )
                yield StreamChunk(type="done", response=resp)
            else:
                yield StreamChunk(type="text", text="Done")
                yield StreamChunk(type="done", response=LLMResponse(content="Done"))

    ex = _executor(ToolThenAnswerRouter(), reg)
    events, result = await _collect(
        ex,
        messages=[{"role": "user", "content": "echo hi"}],
        mode=AgentMode.AGENT,
        tool_schemas=reg.openai_schemas(),
        tool_context=ToolContext(settings=ex._settings),
    )
    types = [e.type for e in events]
    assert EventType.TOOL_START in types
    assert EventType.TOOL_RESULT in types
    assert EventType.TOKEN in types

    tool_result_ev = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert tool_result_ev.success is True
    assert "echoed:hi" in tool_result_ev.text
    assert len(result.tool_invocations) == 1
    assert result.tool_invocations[0]["name"] == "echo_exec_test"

    assert result.stop_reason == "completed"


# --------------------------------------------------------------------------- #
# Unknown tool
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_unknown_tool_reported_not_raised():
    from app.agent.context_builder import ToolCallRequest

    class UnknownToolRouter:
        _turn = 0

        async def stream(self, messages, **_k):
            self._turn += 1
            if self._turn == 1:
                resp = LLMResponse(
                    tool_calls=[ToolCallRequest(id="c1", name="nonexistent",
                                               arguments={})],
                    finish_reason="tool_calls",
                )
                yield StreamChunk(type="done", response=resp)
            else:
                yield StreamChunk(type="text", text="sorry")
                yield StreamChunk(type="done", response=LLMResponse(content="sorry"))

    ex = _executor(UnknownToolRouter(), ToolRegistry())
    events, result = await _collect(
        ex,
        messages=[{"role": "user", "content": "call nonexistent"}],
        mode=AgentMode.AGENT,
    )
    tool_results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert any(not e.success for e in tool_results)
    assert result is not None  # loop completed, did not crash


# --------------------------------------------------------------------------- #
# Dangerous tool denied without confirmer
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dangerous_tool_denied_without_confirmer():
    from app.agent.context_builder import ToolCallRequest

    class DangerArgs(BaseModel):
        cmd: str = Field(description="cmd")

    @tool(name="danger_test_deny", description="Danger deny test.",
          args=DangerArgs, dangerous=True)
    async def _danger(args: DangerArgs, ctx: Any) -> ToolResult:
        raise AssertionError("must not run")

    # Build isolated registry with the registered dangerous tool.
    from app.tools.registry import get_registry
    reg = ToolRegistry()
    reg._tools["danger_test_deny"] = get_registry().get("danger_test_deny")

    class DangerRouter:
        _turn = 0

        async def stream(self, messages, **_k):
            self._turn += 1
            if self._turn == 1:
                resp = LLMResponse(
                    tool_calls=[ToolCallRequest(id="c1", name="danger_test_deny",
                                               arguments={"cmd": "x"})],
                    finish_reason="tool_calls",
                )
                yield StreamChunk(type="done", response=resp)
            else:
                yield StreamChunk(type="text", text="ok")
                yield StreamChunk(type="done", response=LLMResponse(content="ok"))

    ex = _executor(DangerRouter(), reg)
    events, _ = await _collect(
        ex,
        messages=[{"role": "user", "content": "run danger"}],
        mode=AgentMode.AGENT,
        tool_schemas=reg.openai_schemas(),
        tool_context=ToolContext(settings=ex._settings),
        confirmer=None,
    )
    assert any(e.type is EventType.TOOL_DENIED for e in events)


# --------------------------------------------------------------------------- #
# Dangerous tool allowed with confirmer
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dangerous_tool_allowed_with_confirmer():
    from app.agent.context_builder import ToolCallRequest

    class DangerArgs2(BaseModel):
        cmd: str = Field(description="cmd")

    @tool(name="danger_test_allow", description="Danger allow test.",
          args=DangerArgs2, dangerous=True)
    async def _danger2(args: DangerArgs2, ctx: Any) -> ToolResult:
        return ToolResult.ok("ran safely")

    from app.tools.registry import get_registry
    reg = ToolRegistry()
    reg._tools["danger_test_allow"] = get_registry().get("danger_test_allow")

    class DangerRouter2:
        _turn = 0

        async def stream(self, messages, **_k):
            self._turn += 1
            if self._turn == 1:
                resp = LLMResponse(
                    tool_calls=[ToolCallRequest(id="c1", name="danger_test_allow",
                                               arguments={"cmd": "y"})],
                    finish_reason="tool_calls",
                )
                yield StreamChunk(type="done", response=resp)
            else:
                yield StreamChunk(type="text", text="done")
                yield StreamChunk(type="done", response=LLMResponse(content="done"))

    async def allow_confirmer(t, args):
        return True

    ex = _executor(DangerRouter2(), reg)
    events, result = await _collect(
        ex,
        messages=[{"role": "user", "content": "run danger_test_allow"}],
        mode=AgentMode.AGENT,
        tool_schemas=reg.openai_schemas(),
        tool_context=ToolContext(settings=ex._settings),
        confirmer=allow_confirmer,
    )
    tool_results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert any(e.success for e in tool_results)
    assert result.stop_reason == "completed"


# --------------------------------------------------------------------------- #
# Max-iterations cap
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_max_iterations_forces_final_answer():
    from app.agent.context_builder import ToolCallRequest

    # Router that always asks for a tool call, never finishes normally.
    class InfiniteToolRouter:
        async def stream(self, messages, *, tools=None, **_k):
            if tools:
                resp = LLMResponse(
                    tool_calls=[ToolCallRequest(id="c1", name="echo_exec_test",
                                               arguments={"value": "x"})],
                    finish_reason="tool_calls",
                )
                yield StreamChunk(type="done", response=resp)
            else:
                yield StreamChunk(type="text", text="forced answer")
                yield StreamChunk(type="done",
                                  response=LLMResponse(content="forced answer"))

    reg = _registry_with_echo()
    ex = _executor(InfiniteToolRouter(), reg)
    _, result = await _collect(
        ex,
        messages=[{"role": "user", "content": "loop"}],
        mode=AgentMode.AGENT,
        tool_schemas=reg.openai_schemas(),
        tool_context=ToolContext(settings=ex._settings),
        max_iterations=2,
    )
    # Either the loop capped at max_iterations or the forced-final router answered.
    assert result is not None
    assert result.iterations <= 2
