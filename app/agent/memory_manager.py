"""Short- and long-term memory for Adit-Agent.

The :class:`MemoryManager` is the single gateway between the agent and its
persistence layer. It owns two complementary stores:

* **Short-term memory** — the recent turns of the *current* conversation, read
  straight from the relational ``messages`` table and rendered as OpenAI-style
  chat messages for the prompt. This is the agent's working context.
* **Long-term memory** — durable facts, preferences, episodes and rolled-up
  summaries stored in the ``memories`` table and, when available, indexed in a
  vector store for semantic recall across conversations.

Everything degrades gracefully: if the vector store or an embedding provider is
unavailable, recall falls back to the most important recent memories from the
relational store, and the agent simply remembers a little less precisely rather
than crashing. All blocking vector-store calls are off-loaded to a worker thread
so the asyncio event loop stays responsive.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func as sa_func, select
from sqlalchemy.orm import selectinload

from app.database.models import (
    Conversation,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    ToolCall,
    User,
)
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    from app.agent.context_builder import ChatMessage, LLMRouter
    from app.config import Settings

log = get_logger(__name__)

__all__ = ["RecalledMemory", "MemoryManager"]


# --------------------------------------------------------------------------- #
# Recall result
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RecalledMemory:
    """A long-term memory surfaced for the current turn.

    ``score`` is a unit-interval relevance estimate (higher = more relevant);
    for semantic hits it derives from vector distance, for the relational
    fallback it derives from stored importance.
    """

    id: int | None
    content: str
    memory_type: str
    score: float

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.content


# --------------------------------------------------------------------------- #
# Memory manager
# --------------------------------------------------------------------------- #
class MemoryManager:
    """Coordinates short-term history and long-term memory.

    Parameters
    ----------
    session_factory:
        An ``async_sessionmaker`` bound to the application engine.
    settings:
        Application settings (window size, embedding model, collection name).
    vector_store:
        A ChromaDB client (``PersistentClient``) or ``None`` to disable semantic
        recall.
    provider_router:
        Provider used for embeddings and summary generation; optional.
    """

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        settings: "Settings",
        vector_store: Any = None,
        provider_router: "LLMRouter | None" = None,
    ) -> None:
        from app.memory import EmbeddingService, VectorStore

        self._session_factory = session_factory
        self._settings = settings
        self._provider = provider_router
        # Storage layer (both degrade to no-ops when their backends are absent).
        self._vectors = VectorStore(vector_store, settings.chroma_collection)
        self._embeddings = EmbeddingService(
            provider_router, model=settings.embedding_model
        )

    # ================================================================== #
    # User helpers
    # ================================================================== #
    async def get_or_create_user(
        self,
        telegram_id: int,
        *,
        username: str | None = None,
        first_name: str | None = None,
    ) -> User:
        """Return the :class:`User` for ``telegram_id``, creating it if new."""
        async with self._session_factory() as session:
            user = await session.scalar(
                select(User).where(User.telegram_id == telegram_id)
            )
            if user is None:
                user = User(
                    telegram_id=telegram_id,
                    username=username,
                    first_name=first_name,
                )
                session.add(user)
                await session.flush()
                log.info("Created new user record for telegram_id={}.", telegram_id)
            else:
                # Keep the lightweight profile fields fresh.
                if username and user.username != username:
                    user.username = username
                if first_name and user.first_name != first_name:
                    user.first_name = first_name
            await session.commit()
            await session.refresh(user)
            return user

    async def get_conversation(self, conversation_id: int) -> Conversation | None:
        """Return the conversation with ``conversation_id``, or ``None``."""
        async with self._session_factory() as session:
            return await session.get(Conversation, conversation_id)

    async def start_new_conversation(
        self, user_id: int, *, mode: Any = None, title: str | None = None
    ) -> Conversation:
        """Force-open a fresh conversation for the user (used by ``/new``)."""
        async with self._session_factory() as session:
            conversation = Conversation(user_id=user_id, title=title)
            if mode is not None:
                conversation.mode = mode
            session.add(conversation)
            await session.flush()
            await session.commit()
            await session.refresh(conversation)
            log.info("Started new conversation id={} for user_id={}.",
                     conversation.id, user_id)
            return conversation

    async def set_conversation_mode(self, conversation_id: int, mode: Any) -> None:
        """Persist a new default mode on a conversation."""
        async with self._session_factory() as session:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.mode = mode
                await session.commit()

    async def forget_user(self, user_id: int) -> int:
        """Delete all long-term memories for a user. Returns the count removed.

        Also removes the corresponding vectors from the vector store so semantic
        recall does not resurrect forgotten content.
        """
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(Memory).where(Memory.user_id == user_id)
                )
            ).all()
            embedding_ids = [m.embedding_id for m in rows if m.embedding_id]
            count = len(rows)
            for memory in rows:
                await session.delete(memory)
            await session.commit()

        if embedding_ids:
            await self._vectors.delete(embedding_ids)
        log.info("Forgot {} memories for user_id={}.", count, user_id)
        return count

    async def count_memories(self, user_id: int) -> int:
        """Return how many long-term memories a user has."""
        async with self._session_factory() as session:
            total = await session.scalar(
                select(sa_func.count(Memory.id)).where(Memory.user_id == user_id)
            )
            return int(total or 0)

    async def get_or_create_conversation(
        self, user_id: int, *, mode: Any = None, title: str | None = None
    ) -> Conversation:
        """Return the user's most recent conversation, or open a new one."""
        async with self._session_factory() as session:
            conversation: Conversation | None = await session.scalar(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .limit(1)
            )
            if conversation is None:
                conversation = Conversation(user_id=user_id, title=title)
                if mode is not None:
                    conversation.mode = mode
                session.add(conversation)
                await session.flush()
                log.info("Opened new conversation for user_id={}.", user_id)
            await session.commit()
            await session.refresh(conversation)
            return conversation

    # ================================================================== #
    # Short-term memory (conversation history)
    # ================================================================== #
    async def load_history(
        self, conversation_id: int, *, window: int | None = None
    ) -> list["ChatMessage"]:
        """Return the recent turns of a conversation as chat messages.

        Loads at most ``window`` (default ``settings.short_term_window``) of the
        newest messages, then returns them oldest-first with any tool calls and
        their results reconstructed into the OpenAI message sequence.
        """
        window = window or self._settings.short_term_window
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .options(selectinload(Message.tool_calls))
                    .order_by(Message.created_at.desc())
                    .limit(window)
                )
            ).all()

        chat: list["ChatMessage"] = []
        for msg in reversed(rows):  # back to chronological order
            chat.extend(self._message_to_chat(msg))
        return chat

    @staticmethod
    def _message_to_chat(msg: Message) -> list["ChatMessage"]:
        """Render a stored :class:`Message` into one or more chat messages.

        An assistant turn that invoked tools expands into the assistant message
        (carrying ``tool_calls``) followed by one ``tool`` message per call so
        the model sees a well-formed request/result sequence.
        """
        role = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)

        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            tool_calls: list[dict[str, Any]] = []
            tool_replies: list["ChatMessage"] = []
            for call in msg.tool_calls:
                call_id = f"call_{call.id}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": _json_dumps(call.arguments or {}),
                        },
                    }
                )
                tool_replies.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": call.tool_name,
                        "content": call.result if call.result is not None else "",
                    }
                )
            assistant_msg: "ChatMessage" = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }
            return [assistant_msg, *tool_replies]

        return [{"role": role, "content": msg.content or ""}]

    async def record_message(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        *,
        token_count: int = 0,
        has_attachments: bool = False,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> int:
        """Persist a message (and any tool calls); return its new id.

        ``tool_calls`` items use the shape produced by the executor::

            {"name": str, "arguments": dict, "result": str,
             "success": bool, "execution_time": float | None}
        """
        async with self._session_factory() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content or "",
                token_count=token_count,
                has_attachments=has_attachments,
            )
            session.add(message)
            await session.flush()  # assigns message.id

            for tc in tool_calls or []:
                session.add(
                    ToolCall(
                        message_id=message.id,
                        tool_name=str(tc.get("name", "")),
                        arguments=tc.get("arguments") or {},
                        result=_as_text(tc.get("result")),
                        success=bool(tc.get("success", False)),
                        execution_time=tc.get("execution_time"),
                    )
                )

            # Touch the parent conversation so "most recent" ordering is correct.
            # Assigning a SQL expression forces a real UPDATE (a no-op Python
            # assignment would not flag the row dirty).
            conversation = await session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.updated_at = sa_func.now()
                await session.flush()

            await session.commit()
            return message.id

    async def count_messages(self, conversation_id: int) -> int:
        """Return the total number of messages stored in a conversation."""
        async with self._session_factory() as session:
            total = await session.scalar(
                select(sa_func.count(Message.id)).where(
                    Message.conversation_id == conversation_id
                )
            )
            return int(total or 0)

    # ================================================================== #
    # Long-term memory
    # ================================================================== #
    async def remember(
        self,
        user_id: int,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        conversation_id: int | None = None,
    ) -> int | None:
        """Store a durable memory and, when possible, index it for recall.

        Returns the new memory's id, or ``None`` if ``content`` was empty.
        Embedding/indexing failures are logged and swallowed — the relational
        record is always written so the memory is never silently lost.
        """
        content = (content or "").strip()
        if not content:
            return None

        embedding_id: str | None = None
        embedding: list[float] | None = None
        if self._semantic_enabled:
            embedding = await self._embeddings.embed_one(content)
            if embedding is not None:
                embedding_id = uuid.uuid4().hex

        async with self._session_factory() as session:
            memory = Memory(
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type=memory_type,
                content=content,
                importance_score=_clamp_unit(importance),
                embedding_id=embedding_id,
            )
            session.add(memory)
            await session.flush()
            memory_id = memory.id
            await session.commit()

        if embedding_id and embedding is not None:
            await self._vectors.add(
                id=embedding_id,
                embedding=embedding,
                document=content,
                metadata={
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "memory_type": memory_type.value,
                },
            )
        log.debug("Stored memory id={} (type={}) for user_id={}.",
                  memory_id, memory_type.value, user_id)
        return memory_id

    async def recall(
        self, user_id: int, query: str, *, limit: int = 5
    ) -> list[RecalledMemory]:
        """Return up to ``limit`` memories most relevant to ``query``.

        Tries semantic search over the vector store first; on any failure (or
        when semantic search is disabled) falls back to the user's most
        important recent memories from the relational store.
        """
        query = (query or "").strip()
        if self._semantic_enabled and query:
            try:
                hits = await self._semantic_recall(user_id, query, limit=limit)
                if hits:
                    return hits
            except Exception as exc:  # noqa: BLE001 - degrade to relational fallback
                log.warning("Semantic recall failed, falling back: {}", exc)
        return await self._relational_recall(user_id, limit=limit)

    async def _semantic_recall(
        self, user_id: int, query: str, *, limit: int
    ) -> list[RecalledMemory]:
        """Vector-store backed recall, scoped to ``user_id``."""
        embedding = await self._embeddings.embed_one(query)
        if embedding is None:
            return []

        matches = await self._vectors.query(
            embedding=embedding, limit=limit, where={"user_id": user_id}
        )
        return [
            RecalledMemory(
                id=match.metadata.get("memory_id"),
                content=match.document,
                memory_type=str(match.metadata.get("memory_type", "fact")),
                score=match.score,
            )
            for match in matches
        ]

    async def _relational_recall(
        self, user_id: int, *, limit: int
    ) -> list[RecalledMemory]:
        """Fallback recall: most important, then most recent, memories."""
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(Memory)
                    .where(Memory.user_id == user_id)
                    .order_by(
                        Memory.importance_score.desc(),
                        Memory.created_at.desc(),
                    )
                    .limit(limit)
                )
            ).all()
        return [
            RecalledMemory(
                id=m.id,
                content=m.content,
                memory_type=(
                    m.memory_type.value
                    if isinstance(m.memory_type, MemoryType)
                    else str(m.memory_type)
                ),
                score=_clamp_unit(m.importance_score),
            )
            for m in rows
        ]

    # ================================================================== #
    # Summarization (memory consolidation)
    # ================================================================== #
    async def summarize_if_needed(
        self, conversation_id: int, user_id: int
    ) -> int | None:
        """Roll older history into a single summary memory when it grows large.

        Triggered when stored messages exceed twice the short-term window. The
        oldest messages beyond the window are summarized by the LLM and saved as
        a high-importance :attr:`MemoryType.SUMMARY` memory, so their gist
        survives even though they fall out of the working context. Best-effort:
        any failure is logged and ``None`` returned.
        """
        if self._provider is None:
            return None

        window = self._settings.short_term_window
        total = await self.count_messages(conversation_id)
        if total <= window * 2:
            return None

        async with self._session_factory() as session:
            overflow = (
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.asc())
                    .limit(total - window)
                )
            ).all()

        transcript = "\n".join(
            f"{m.role.value if isinstance(m.role, MessageRole) else m.role}: {m.content}"
            for m in overflow
            if m.content
        )
        if not transcript.strip():
            return None

        try:
            response = await self._provider.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation excerpt into a "
                            "compact, factual note capturing decisions, user "
                            "preferences, and unresolved threads. Write 3-6 terse "
                            "bullet points. Do not invent details."
                        ),
                    },
                    {"role": "user", "content": transcript[:12_000]},
                ],
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001 - summarization is best-effort
            log.warning("Conversation summarization failed: {}", exc)
            return None

        summary = (response.content or "").strip()
        if not summary:
            return None
        return await self.remember(
            user_id,
            summary,
            memory_type=MemoryType.SUMMARY,
            importance=0.8,
            conversation_id=conversation_id,
        )

    # ================================================================== #
    # Semantic-memory availability
    # ================================================================== #
    @property
    def _semantic_enabled(self) -> bool:
        """True when both the vector store and embeddings are available."""
        return self._vectors.available and self._embeddings.available


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clamp_unit(value: float) -> float:
    """Clamp ``value`` into the closed unit interval ``[0.0, 1.0]``."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _as_text(value: Any) -> str | None:
    """Coerce a tool result into stored text (JSON for non-strings)."""
    if value is None or isinstance(value, str):
        return value
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    """Compact, non-ASCII-preserving JSON dump that never raises."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
