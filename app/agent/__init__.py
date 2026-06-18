"""The intelligent core of Adit-Agent.

This package houses the agent's reasoning pipeline and exposes it through the
:class:`~app.agent.orchestrator.Orchestrator`, the single entry point the bot
layer uses to turn a user message into a streamed answer.

Pipeline at a glance::

    Orchestrator
      ├─ MemoryManager   – short-term history + long-term semantic recall
      ├─ ContextBuilder  – persona/system prompts + token-budgeted assembly
      ├─ Planner         – decomposes complex tasks before execution
      └─ Executor        – streaming ReAct loop with dynamic tool calling

The provider contract the pipeline depends on (:class:`LLMRouter`,
:class:`LLMResponse`, :class:`StreamChunk`) is defined in
:mod:`app.agent.context_builder` and implemented by :mod:`app.providers`.
"""

from __future__ import annotations

from app.agent.context_builder import (
    AgentMode,
    BuiltContext,
    ChatMessage,
    ContextBuilder,
    LLMResponse,
    LLMRouter,
    StreamChunk,
    TokenCounter,
    ToolCallRequest,
    system_prompt,
)
from app.agent.executor import (
    AgentEvent,
    Confirmer,
    EventType,
    ExecutionResult,
    Executor,
)
from app.agent.memory_manager import MemoryManager, RecalledMemory
from app.agent.orchestrator import AgentRequest, AgentResponse, Orchestrator
from app.agent.planner import Plan, PlanStep, Planner

__all__ = [
    # Orchestration
    "Orchestrator",
    "AgentRequest",
    "AgentResponse",
    # Modes & prompts
    "AgentMode",
    "system_prompt",
    # Context building
    "ContextBuilder",
    "BuiltContext",
    "TokenCounter",
    # Provider contract
    "LLMRouter",
    "LLMResponse",
    "StreamChunk",
    "ChatMessage",
    "ToolCallRequest",
    # Planning
    "Planner",
    "Plan",
    "PlanStep",
    # Execution
    "Executor",
    "ExecutionResult",
    "AgentEvent",
    "EventType",
    "Confirmer",
    # Memory
    "MemoryManager",
    "RecalledMemory",
]
