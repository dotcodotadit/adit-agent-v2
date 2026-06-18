"""Tests for MemoryManager, VectorStore, and EmbeddingService."""

from __future__ import annotations

import pytest

from app.agent.memory_manager import MemoryManager
from app.database.models import MemoryType, MessageRole
from app.memory.embeddings import EmbeddingService
from app.memory.vector_store import VectorStore
from tests.conftest import FakeRouter


# --------------------------------------------------------------------------- #
# EmbeddingService
# --------------------------------------------------------------------------- #
class TestEmbeddingService:
    def test_unavailable_without_provider(self):
        svc = EmbeddingService(None, model="text-embedding-3-small")
        assert not svc.available

    @pytest.mark.asyncio
    async def test_embed_one_returns_vector(self):
        router = FakeRouter(embeddings=[0.1, 0.2, 0.3])
        svc = EmbeddingService(router, model="text-embedding-3-small")
        assert svc.available
        vec = await svc.embed_one("hello")
        assert vec == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_one_empty_returns_none(self):
        router = FakeRouter()
        svc = EmbeddingService(router, model="m")
        result = await svc.embed_one("")
        assert result is None

    @pytest.mark.asyncio
    async def test_embed_one_provider_error_returns_none(self):
        class ErrorRouter:
            async def embed(self, texts, **k):
                raise RuntimeError("boom")

        svc = EmbeddingService(ErrorRouter(), model="m")
        result = await svc.embed_one("text")
        assert result is None


# --------------------------------------------------------------------------- #
# VectorStore (disabled)
# --------------------------------------------------------------------------- #
class TestVectorStoreDisabled:
    def test_not_available_without_client(self):
        vs = VectorStore(None, "test")
        assert not vs.available

    @pytest.mark.asyncio
    async def test_add_returns_false_when_disabled(self):
        vs = VectorStore(None, "test")
        ok = await vs.add(id="x", embedding=[0.1], document="d")
        assert not ok

    @pytest.mark.asyncio
    async def test_query_returns_empty_when_disabled(self):
        vs = VectorStore(None, "test")
        result = await vs.query(embedding=[0.1], limit=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_count_returns_zero_when_disabled(self):
        vs = VectorStore(None, "test")
        assert await vs.count() == 0


# --------------------------------------------------------------------------- #
# MemoryManager — user and conversation management
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_or_create_user_creates_and_retrieves(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(12345, username="alice", first_name="Alice")
    assert user.id is not None
    assert user.telegram_id == 12345
    assert user.username == "alice"

    # Second call returns the same user.
    user2 = await mm.get_or_create_user(12345, username="alice_updated")
    assert user2.id == user.id
    assert user2.username == "alice_updated"


@pytest.mark.asyncio
async def test_start_new_conversation(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(1)
    conv1 = await mm.get_or_create_conversation(user.id)
    conv2 = await mm.start_new_conversation(user.id)
    assert conv2.id != conv1.id


@pytest.mark.asyncio
async def test_record_and_load_history(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(2)
    conv = await mm.get_or_create_conversation(user.id)

    await mm.record_message(conv.id, MessageRole.USER, "hello")
    await mm.record_message(conv.id, MessageRole.ASSISTANT, "hi back")

    history = await mm.load_history(conv.id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_record_message_with_tool_calls(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(3)
    conv = await mm.get_or_create_conversation(user.id)

    tool_calls = [
        {"name": "web_search", "arguments": {"query": "cats"},
         "result": "many results", "success": True, "execution_time": 0.5}
    ]
    await mm.record_message(conv.id, MessageRole.USER, "search cats")
    await mm.record_message(conv.id, MessageRole.ASSISTANT, "",
                            tool_calls=tool_calls)

    history = await mm.load_history(conv.id)
    # Assistant message with tool calls expands to 2 entries (assistant + tool reply).
    assert len(history) == 3
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "tool"


# --------------------------------------------------------------------------- #
# Long-term memory (no vector store — relational fallback)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_remember_and_relational_recall(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(4)

    await mm.remember(user.id, "User likes Python", importance=0.9)
    await mm.remember(user.id, "User dislikes Java", importance=0.7,
                      memory_type=MemoryType.PREFERENCE)

    memories = await mm.recall(user.id, "programming language preference")
    assert len(memories) >= 1
    contents = [m.content for m in memories]
    assert any("Python" in c for c in contents)


@pytest.mark.asyncio
async def test_forget_user_removes_memories(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(5)

    await mm.remember(user.id, "fact one")
    await mm.remember(user.id, "fact two")
    count_before = await mm.count_memories(user.id)
    assert count_before == 2

    removed = await mm.forget_user(user.id)
    assert removed == 2
    assert await mm.count_memories(user.id) == 0


@pytest.mark.asyncio
async def test_remember_empty_content_returns_none(session_factory, settings):
    mm = MemoryManager(session_factory=session_factory, settings=settings)
    user = await mm.get_or_create_user(6)
    result = await mm.remember(user.id, "   ")
    assert result is None


@pytest.mark.asyncio
async def test_summarize_if_needed_no_op_when_small(session_factory, settings):
    mm = MemoryManager(
        session_factory=session_factory,
        settings=settings,
        provider_router=FakeRouter(),
    )
    user = await mm.get_or_create_user(7)
    conv = await mm.get_or_create_conversation(user.id)
    # Only 2 messages — well below the 2×window threshold.
    await mm.record_message(conv.id, MessageRole.USER, "hi")
    result = await mm.summarize_if_needed(conv.id, user.id)
    assert result is None
