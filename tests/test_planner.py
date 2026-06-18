"""Tests for the Planner: heuristic gating and JSON parsing."""

from __future__ import annotations

import pytest

from app.agent.context_builder import AgentMode, LLMResponse
from app.agent.planner import Plan, PlanStep, Planner, _extract_json_object
from tests.conftest import FakeRouter


# --------------------------------------------------------------------------- #
# should_plan heuristics
# --------------------------------------------------------------------------- #
class TestShouldPlan:
    def _planner(self) -> Planner:
        return Planner(FakeRouter(), max_steps=5)

    def test_deep_mode_always_plans(self):
        p = self._planner()
        assert p.should_plan("hi", mode=AgentMode.DEEP, tool_count=0)

    def test_chat_never_plans(self):
        p = self._planner()
        assert not p.should_plan("complex task with many steps", mode=AgentMode.CHAT, tool_count=5)

    def test_agent_no_tools_does_not_plan(self):
        p = self._planner()
        assert not p.should_plan("do x then y", mode=AgentMode.AGENT, tool_count=0)

    def test_agent_multi_step_hint_plans(self):
        p = self._planner()
        assert p.should_plan("first search, then summarize, finally write the file",
                             mode=AgentMode.AGENT, tool_count=2)

    def test_agent_long_request_plans(self):
        p = self._planner()
        long_text = "please " + ("do this complex thing " * 10)
        assert p.should_plan(long_text, mode=AgentMode.AGENT, tool_count=2)

    def test_agent_short_simple_request_no_plan(self):
        p = self._planner()
        assert not p.should_plan("what time is it", mode=AgentMode.AGENT, tool_count=2)


# --------------------------------------------------------------------------- #
# Plan parsing
# --------------------------------------------------------------------------- #
class TestPlanParsing:
    def _planner(self) -> Planner:
        return Planner.__new__(Planner)

    def setup_method(self):
        p = self._planner()
        p._max_steps = 5

    @pytest.fixture
    def pl(self) -> Planner:
        p = Planner.__new__(Planner)
        p._max_steps = 5
        return p

    def test_bare_json(self, pl):
        raw = '{"goal": "do X", "steps": [{"description": "step one", "tool": "web_search"}]}'
        plan = pl._parse_plan(raw)
        assert plan.goal == "do X"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "web_search"

    def test_fenced_json(self, pl):
        raw = 'Here is the plan:\n```json\n{"goal": "g", "steps": [{"description": "a", "tool": null}]}\n```\nDone.'
        plan = pl._parse_plan(raw)
        assert plan.goal == "g"
        assert plan.steps[0].tool is None

    def test_empty_response(self, pl):
        plan = pl._parse_plan("")
        assert not plan
        assert plan.steps == []

    def test_max_steps_enforced(self, pl):
        steps = [{"description": f"step {i}", "tool": None} for i in range(20)]
        raw = f'{{"goal": "big", "steps": {__import__("json").dumps(steps)}}}'
        plan = pl._parse_plan(raw)
        assert len(plan.steps) <= 5

    def test_plan_truthiness(self, pl):
        empty = Plan(goal="x", steps=[])
        full = Plan(goal="x", steps=[PlanStep(description="do it")])
        assert not empty
        assert full

    def test_plan_pending_and_next(self):
        plan = Plan(
            goal="g",
            steps=[
                PlanStep(description="done", done=True),
                PlanStep(description="open"),
            ],
        )
        assert len(plan.pending) == 1
        assert plan.next_open().description == "open"
        assert not plan.is_complete

    def test_plan_complete_when_all_done(self):
        plan = Plan(goal="g", steps=[PlanStep(description="x", done=True)])
        assert plan.is_complete


# --------------------------------------------------------------------------- #
# JSON extraction helper
# --------------------------------------------------------------------------- #
class TestExtractJsonObject:
    def test_clean(self):
        result = _extract_json_object('{"a": 1}')
        assert result == {"a": 1}

    def test_embedded_in_prose(self):
        result = _extract_json_object('noise before {"key": "val"} noise after')
        assert result == {"key": "val"}

    def test_fenced(self):
        result = _extract_json_object('```json\n{"k": 2}\n```')
        assert result == {"k": 2}

    def test_invalid_returns_none(self):
        assert _extract_json_object("not json at all") is None
        assert _extract_json_object("") is None

    def test_non_object_returns_none(self):
        assert _extract_json_object("[1, 2, 3]") is None


# --------------------------------------------------------------------------- #
# make_plan integration (with fake provider)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_make_plan_returns_plan_from_provider():
    response = LLMResponse(
        content='{"goal": "search and summarize", "steps": ['
                '{"description": "search the web", "tool": "web_search"},'
                '{"description": "write summary", "tool": null}]}'
    )
    planner = Planner(FakeRouter(script=[response]), max_steps=5)
    plan = await planner.make_plan(
        "search cats and summarize", mode=AgentMode.AGENT, tool_names=["web_search"]
    )
    assert plan.goal == "search and summarize"
    assert len(plan.steps) == 2


@pytest.mark.asyncio
async def test_make_plan_degrades_on_provider_error():
    class ErrorRouter:
        async def complete(self, *a, **k):
            raise RuntimeError("network")
        async def stream(self, *a, **k):
            raise RuntimeError("network")
            yield  # make it an async generator

    planner = Planner(ErrorRouter(), max_steps=5)
    plan = await planner.make_plan("anything", mode=AgentMode.DEEP)
    assert not plan  # empty plan, not a crash
