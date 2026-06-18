"""Tests for prompt assembly, persona prompts, and token budgeting."""

from __future__ import annotations

from app.agent.context_builder import (
    AgentMode,
    ContextBuilder,
    TokenCounter,
    system_prompt,
)


def test_persona_present_in_every_mode():
    for mode in AgentMode:
        prompt = system_prompt(mode)
        assert "Adit" in prompt
    # Mode contracts differ.
    assert system_prompt(AgentMode.CHAT) != system_prompt(AgentMode.DEEP)


def test_tool_guidance_only_when_requested():
    assert "Tool use:" not in system_prompt(AgentMode.AGENT, with_tools=False)
    assert "Tool use:" in system_prompt(AgentMode.AGENT, with_tools=True)


def test_extra_sections_appended():
    prompt = system_prompt(AgentMode.CHAT, extra_sections=["REMEMBER: be brief"])
    assert "REMEMBER: be brief" in prompt


def test_token_counter_counts_and_falls_back():
    tc = TokenCounter()
    assert tc.count_text("") == 0
    assert tc.count_text("hello world") > 0
    msg = {"role": "user", "content": "hello"}
    assert tc.count_message(msg) >= tc.count_text("hello")


def test_build_orders_system_first_and_input_last():
    cb = ContextBuilder(max_context_tokens=10_000, response_reserve_tokens=128)
    built = cb.build(
        mode=AgentMode.CHAT,
        user_input="current question",
        history=[{"role": "user", "content": "earlier"}],
    )
    assert built.messages[0]["role"] == "system"
    assert built.messages[-1]["content"] == "current question"
    assert built.prompt_tokens > 0
    assert built.response_budget == 128


def test_build_drops_old_history_under_tight_budget():
    cb = ContextBuilder(max_context_tokens=80, response_reserve_tokens=16)
    history = [
        {"role": "user", "content": "very old " * 30},
        {"role": "assistant", "content": "old reply " * 30},
        {"role": "user", "content": "recent " * 30},
    ]
    built = cb.build(mode=AgentMode.CHAT, user_input="now", history=history)
    assert built.dropped_messages >= 1
    # System prompt and current input always survive.
    assert built.messages[0]["role"] == "system"
    assert built.messages[-1]["content"] == "now"


def test_build_trims_dangling_tool_prefix():
    cb = ContextBuilder(max_context_tokens=10_000)
    # A leading orphan tool reply must be stripped so providers don't reject it.
    history = [
        {"role": "tool", "tool_call_id": "x", "name": "t", "content": "orphan"},
        {"role": "user", "content": "hi"},
    ]
    built = cb.build(mode=AgentMode.AGENT, user_input="go", history=history)
    roles = [m["role"] for m in built.messages]
    assert "tool" not in roles[:2]  # the orphan tool message was trimmed
