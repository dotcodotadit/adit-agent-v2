"""LLM provider layer for Adit-Agent.

Exposes :class:`~app.providers.base.ProviderRouter`, the failover-capable
implementation of the agent's :class:`~app.agent.context_builder.LLMRouter`
contract, plus :class:`~app.providers.base.ProviderError`.
"""

from __future__ import annotations

from app.providers.base import ProviderError, ProviderRouter

__all__ = ["ProviderRouter", "ProviderError"]
