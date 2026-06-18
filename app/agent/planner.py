"""Task planning for Adit-Agent.

Before the executor starts spending tool calls, the :class:`Planner` decides
whether a request is simple enough to answer directly or complex enough to
warrant an explicit, ordered plan — and, when it is, produces that plan.

Planning earns its keep on multi-step tasks ("research X, then summarize it into
a file"): a short, upfront decomposition keeps the ReAct loop on track, makes
progress legible to the user, and reduces aimless tool wandering. For ordinary
questions it would only add latency, so :meth:`Planner.should_plan` gates it
with cheap heuristics first and the orchestrator skips planning entirely in
plain chat mode.

The plan itself is intentionally lightweight: a goal plus a list of
:class:`PlanStep` objects. It guides the model (it is rendered into the system
prompt by the context builder) rather than rigidly driving execution — the
executor remains free to adapt as tool results come in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agent.context_builder import AgentMode
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.agent.context_builder import ChatMessage, LLMRouter

log = get_logger(__name__)

__all__ = ["PlanStep", "Plan", "Planner"]


# Words/patterns that hint a request is inherently multi-step. Deliberately
# conservative — false positives only cost one planning call, false negatives
# just fall back to direct execution.
_MULTISTEP_HINTS: tuple[str, ...] = (
    "then ", "after that", "afterwards", "first ", "second ", "third ",
    "finally", "step by step", "step-by-step", "and then", "once you",
    "compare", "summarize", "research", "investigate", "analyze",
    "for each", "every ", "all of the", "multiple", "several",
)
# Rough word-count above which an AGENT-mode request is treated as complex.
_LONG_REQUEST_WORDS = 35


@dataclass(slots=True)
class PlanStep:
    """A single, actionable step in a plan.

    ``tool`` is an optional hint naming the capability the step likely needs; it
    is advisory only — the executor still selects tools per turn.
    """

    description: str
    tool: str | None = None
    done: bool = False
    result: str | None = None


@dataclass(slots=True)
class Plan:
    """An ordered decomposition of a task into steps.

    A plan with no steps is falsy, so callers can write ``if plan:`` to mean
    "there is real planned work to execute against".
    """

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    raw: str = ""

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def pending(self) -> list[PlanStep]:
        """Steps not yet marked done, in order."""
        return [s for s in self.steps if not s.done]

    @property
    def is_complete(self) -> bool:
        """True when every step is done (vacuously true for an empty plan)."""
        return all(s.done for s in self.steps)

    def next_open(self) -> PlanStep | None:
        """Return the first not-yet-done step, or ``None`` if all are done."""
        return next((s for s in self.steps if not s.done), None)

    def render(self) -> str:
        """Render a compact, human-readable checklist of the plan."""
        lines = [f"Goal: {self.goal}"] if self.goal else []
        for i, step in enumerate(self.steps, start=1):
            mark = "x" if step.done else " "
            suffix = f"  (tool: {step.tool})" if step.tool else ""
            lines.append(f"[{mark}] {i}. {step.description}{suffix}")
        return "\n".join(lines)


class Planner:
    """Decides whether to plan and, if so, builds the plan.

    Parameters
    ----------
    provider_router:
        Provider used to generate the decomposition.
    max_steps:
        Hard cap on plan length (defaults to ``settings.agent_max_steps`` via the
        orchestrator). Keeps plans focused and bounds executor work.
    default_model:
        Model id to use for the planning call; ``None`` lets the router decide.
    """

    def __init__(
        self,
        provider_router: "LLMRouter",
        *,
        max_steps: int = 8,
        default_model: str | None = None,
    ) -> None:
        self._provider = provider_router
        self._max_steps = max(1, max_steps)
        self._model = default_model

    # ------------------------------------------------------------------ #
    # Gating
    # ------------------------------------------------------------------ #
    def should_plan(
        self,
        user_input: str,
        *,
        mode: AgentMode,
        tool_count: int = 0,
    ) -> bool:
        """Heuristically decide whether ``user_input`` warrants a plan.

        Deep-reasoning always plans. Chat never does (it answers directly).
        Agent mode plans only for requests that look genuinely multi-step, and
        only when tools exist to act with.
        """
        if mode is AgentMode.DEEP:
            return True
        if mode is AgentMode.CHAT:
            return False
        if tool_count == 0:
            return False

        text = (user_input or "").lower()
        word_count = len(text.split())
        if word_count >= _LONG_REQUEST_WORDS:
            return True
        if any(hint in text for hint in _MULTISTEP_HINTS):
            return True
        # Several distinct imperative clauses (separated by ; or newline) also
        # signal multi-step work.
        if len([c for c in re.split(r"[;\n]", text) if c.strip()]) >= 3:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Plan generation
    # ------------------------------------------------------------------ #
    async def make_plan(
        self,
        user_input: str,
        *,
        mode: AgentMode,
        tool_names: list[str] | None = None,
        history: list["ChatMessage"] | None = None,
    ) -> Plan:
        """Generate an ordered :class:`Plan` for ``user_input``.

        On any failure to produce a usable plan (provider error, unparseable
        output), returns an empty plan so the orchestrator can proceed with
        direct execution rather than blocking the turn.
        """
        depth_hint = (
            "Think rigorously and decompose thoroughly; prefer 3-7 substantive "
            "steps that include verification."
            if mode is AgentMode.DEEP
            else "Keep the plan minimal — only as many steps as the task truly "
            "needs. A simple task may need just one or two steps."
        )
        tools_line = (
            f"Tools available to later steps: {', '.join(tool_names)}."
            if tool_names
            else "No external tools are available; plan reasoning steps only."
        )
        system = (
            "You are the planning module of the Adit agent. Decompose the user's "
            "request into a short, ordered list of concrete steps that, executed "
            "in sequence, accomplish it.\n"
            f"{depth_hint}\n{tools_line}\n\n"
            "Respond with STRICT JSON and nothing else, in this exact shape:\n"
            '{"goal": "<one-sentence restatement of the objective>", '
            '"steps": [{"description": "<imperative step>", '
            '"tool": "<tool name or null>"}]}\n'
            f"Use at most {self._max_steps} steps. Do not wrap the JSON in code "
            "fences or add commentary."
        )

        messages: list["ChatMessage"] = [{"role": "system", "content": system}]
        # A little recent context helps the planner resolve references like "it".
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_input})

        try:
            response = await self._provider.complete(
                messages,
                model=self._model,
                temperature=0.2,
                max_tokens=700,
            )
        except Exception as exc:  # noqa: BLE001 - planning is optional; degrade
            log.warning("Planning call failed; proceeding without a plan: {}", exc)
            return Plan(goal="", steps=[], raw="")

        plan = self._parse_plan(response.content or "")
        if plan.steps:
            log.info("Built plan with {} step(s): {}", len(plan.steps), plan.goal)
        else:
            log.debug("Planner returned no usable steps; executing directly.")
        return plan

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse_plan(self, text: str) -> Plan:
        """Parse the model's JSON plan, tolerating fences and stray prose."""
        payload = _extract_json_object(text)
        if payload is None:
            return Plan(goal="", steps=[], raw=text)

        goal = str(payload.get("goal", "")).strip()
        steps: list[PlanStep] = []
        for item in payload.get("steps", []) or []:
            if isinstance(item, str):
                description, tool = item.strip(), None
            elif isinstance(item, dict):
                description = str(item.get("description", "")).strip()
                tool_val = item.get("tool")
                tool = str(tool_val).strip() if tool_val not in (None, "", "null") else None
            else:
                continue
            if description:
                steps.append(PlanStep(description=description, tool=tool))
            if len(steps) >= self._max_steps:
                break

        return Plan(goal=goal, steps=steps, raw=text)


# --------------------------------------------------------------------------- #
# JSON extraction helper
# --------------------------------------------------------------------------- #
def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from ``text``, tolerating noise.

    Handles three common model behaviors: clean JSON, JSON wrapped in
    ``` ```json fences ```, and JSON preceded/followed by prose. Returns
    ``None`` if no parseable object is found.
    """
    if not text:
        return None

    # Strip Markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    if candidate is None:
        # Fall back to the first balanced-looking {...} span.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]

    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
