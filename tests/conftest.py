"""Shared pytest fixtures and fakes for Adit-Agent tests.

Provides lightweight, dependency-free stand-ins for the heavy collaborators
(LLM provider, vector store) plus an in-memory SQLite session factory, so the
agent logic can be exercised without a real model or external services.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from app.agent.context_builder import LLMResponse, StreamChunk, ToolCallRequest


# --------------------------------------------------------------------------- #
# Fake provider router
# --------------------------------------------------------------------------- #
class FakeRouter:
    """A scriptable LLMRouter for tests.

    ``script`` is a list of :class:`LLMResponse` objects returned in order by
    successive ``complete``/``stream`` calls; once exhausted, the last response
    repeats. ``embeddings`` is the canned vector returned per input string.
    """

    def __init__(
        self,
        script: list[LLMResponse] | None = None,
        *,
        embeddings: list[float] | None = None,
    ) -> None:
        self.script = list(script or [LLMResponse(content="OK")])
        self.calls = 0
        self._embedding = embeddings or [0.1, 0.2, 0.3]

    def _next(self) -> LLMResponse:
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]

    async def complete(self, messages: Any, **_kwargs: Any) -> LLMResponse:
        return self._next()

    async def stream(self, messages: Any, **_kwargs: Any):
        response = self._next()
        if response.content and not response.tool_calls:
            # Emit the content as a couple of deltas to exercise streaming.
            mid = max(1, len(response.content) // 2)
            yield StreamChunk(type="text", text=response.content[:mid])
            yield StreamChunk(type="text", text=response.content[mid:])
        yield StreamChunk(type="done", response=response)

    async def embed(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [list(self._embedding) for _ in texts]


@pytest.fixture
def fake_router() -> FakeRouter:
    return FakeRouter()


def tool_call(name: str, arguments: dict[str, Any], *, call_id: str = "c1") -> ToolCallRequest:
    """Helper to build a tool-call request."""
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


# --------------------------------------------------------------------------- #
# In-memory database session factory
# --------------------------------------------------------------------------- #
@pytest_asyncio.fixture
async def session_factory():
    """Yield an async_sessionmaker backed by a fresh in-memory SQLite database."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.database.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Settings stub
# --------------------------------------------------------------------------- #
@pytest.fixture
def settings(tmp_path):
    """Return real Settings pointed at a temp directory (no .env needed)."""
    from app.config import Settings

    return Settings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
        vector_store_dir=tmp_path / "vectors",
        sandbox_root=tmp_path / "uploads",
        agent_max_steps=4,
        agent_max_tokens=256,
        short_term_window=10,
    )
