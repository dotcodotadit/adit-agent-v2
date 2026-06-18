"""SQLAlchemy 2.0 ORM models for Adit-Agent.

All persistent entities live here, declared with the typed, async-friendly
SQLAlchemy 2.0 style (:class:`~sqlalchemy.orm.DeclarativeBase` +
:func:`~sqlalchemy.orm.mapped_column`). The schema captures the agent's core
domain:

    User ──< Conversation ──< Message ──< ToolCall
      │            │              └────< Attachment
      └──< Memory ─┘  (Memory may optionally belong to a Conversation)

Conventions
-----------
* Surrogate integer primary keys (``id``) everywhere; natural keys (e.g.
  ``telegram_id``) get unique indexes instead.
* Timestamps are timezone-aware and set by the database via ``func.now()`` so
  they are consistent regardless of the app server's clock.
* Foreign keys use ``ON DELETE`` rules that match the ownership graph, and ORM
  relationships mirror them with matching ``cascade`` settings.
* Enums are stored as strings (``native_enum=False``) for portability across
  SQLite (dev) and Postgres/MySQL (prod) without ALTER TYPE migrations.

The :class:`Base` metadata is what :mod:`app.database.migrations` reflects to
create/drop tables, and :mod:`app.database.session` provides the engine/session
plumbing.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON

__all__ = [
    "Base",
    "ConversationMode",
    "MessageRole",
    "MemoryType",
    "AttachmentType",
    "User",
    "Conversation",
    "Message",
    "ToolCall",
    "Memory",
    "Attachment",
    "IMPORTANT_MEMORY_THRESHOLD",
]

# A memory at or above this importance is considered worth persisting/recalling
# long-term. Used by the :attr:`Memory.is_persistent` hybrid property.
IMPORTANT_MEMORY_THRESHOLD: float = 0.5


class Base(DeclarativeBase):
    """Declarative base shared by every model.

    A ``type_annotation_map`` lets us annotate columns with plain Python types
    (``dict``) and have SQLAlchemy pick the right column type (``JSON``).
    """

    type_annotation_map = {
        dict[str, Any]: JSON,
    }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        ident = getattr(self, "id", None)
        return f"<{type(self).__name__} id={ident}>"


# --------------------------------------------------------------------------- #
# Mixins
# --------------------------------------------------------------------------- #
class CreatedAtMixin:
    """Adds an immutable, DB-populated ``created_at`` timestamp."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class TimestampMixin(CreatedAtMixin):
    """Adds ``created_at`` plus an auto-updating ``updated_at`` timestamp."""

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ConversationMode(str, enum.Enum):
    """How the agent treats a conversation."""

    CHAT = "chat"          # plain conversational replies
    AGENT = "agent"        # full ReAct tool-using loop
    PLANNER = "planner"    # planner decomposes then executes


class MessageRole(str, enum.Enum):
    """Author role of a message, mirroring the OpenAI chat schema."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MemoryType(str, enum.Enum):
    """Category of a stored memory."""

    FACT = "fact"                # objective fact about the user/world
    PREFERENCE = "preference"    # a user preference/setting
    EPISODIC = "episodic"        # something that happened in a conversation
    SUMMARY = "summary"          # rolled-up summary of older messages


class AttachmentType(str, enum.Enum):
    """Kind of media attached to a message."""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


def _enum_col(enum_cls: type[enum.Enum]) -> SAEnum:
    """Build a portable, string-backed Enum column for ``enum_cls``."""
    return SAEnum(
        enum_cls,
        native_enum=False,                       # store as VARCHAR + CHECK
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
        length=32,
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class User(TimestampMixin, Base):
    """A Telegram user known to the bot."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Telegram user IDs exceed 32-bit, so use BigInteger.
    telegram_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False
    )
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Free-form JSON bag of per-user settings (locale, tone, default mode...).
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    is_whitelisted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Relationships -------------------------------------------------------- #
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    memories: Mapped[list[Memory]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @hybrid_property
    def display_name(self) -> str:
        """Best human-friendly label for the user (Python side)."""
        return self.username or self.first_name or f"user:{self.telegram_id}"

    @display_name.inplace.expression
    @classmethod
    def _display_name_expr(cls):
        # SQL side: prefer username, then first_name (telegram_id fallback is
        # Python-only to avoid a cross-dialect cast).
        return func.coalesce(cls.username, cls.first_name)


class Conversation(TimestampMixin, Base):
    """A chat thread between a user and the agent."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mode: Mapped[ConversationMode] = mapped_column(
        _enum_col(ConversationMode),
        default=ConversationMode.CHAT,
        nullable=False,
    )

    # Relationships -------------------------------------------------------- #
    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )
    memories: Mapped[list[Memory]] = relationship(
        back_populates="conversation",
        passive_deletes=True,
    )


class Message(CreatedAtMixin, Base):
    """A single message within a conversation."""

    __tablename__ = "messages"
    # Composite index supports the common "messages for a conversation in
    # chronological order" query used to rebuild context windows.
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[MessageRole] = mapped_column(
        _enum_col(MessageRole), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    has_attachments: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Relationships -------------------------------------------------------- #
    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    attachments: Mapped[list[Attachment]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @hybrid_property
    def is_user(self) -> bool:
        """True for user-authored messages."""
        return self.role == MessageRole.USER

    @is_user.inplace.expression
    @classmethod
    def _is_user_expr(cls):
        return cls.role == MessageRole.USER

    @hybrid_property
    def is_assistant(self) -> bool:
        """True for assistant-authored messages."""
        return self.role == MessageRole.ASSISTANT

    @is_assistant.inplace.expression
    @classmethod
    def _is_assistant_expr(cls):
        return cls.role == MessageRole.ASSISTANT


class ToolCall(CreatedAtMixin, Base):
    """A tool invocation made while producing an assistant message."""

    __tablename__ = "tool_calls"
    __table_args__ = (
        Index("ix_tool_calls_name_created", "tool_name", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Wall-clock execution time in seconds; NULL until the call completes.
    execution_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships -------------------------------------------------------- #
    message: Mapped[Message] = relationship(back_populates="tool_calls")


class Memory(CreatedAtMixin, Base):
    """A long-term memory item, optionally backed by a vector embedding."""

    __tablename__ = "memories"
    __table_args__ = (
        # Recall queries fetch a user's memories newest-first.
        Index("ix_memories_user_created", "user_id", "created_at"),
        Index("ix_memories_user_type", "user_id", "memory_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # If the source conversation is deleted, keep the memory but null the link.
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    memory_type: Mapped[MemoryType] = mapped_column(
        _enum_col(MemoryType),
        default=MemoryType.FACT,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # ID of the corresponding vector in ChromaDB (NULL if not yet embedded).
    embedding_id: Mapped[str | None] = mapped_column(
        String(128), unique=True, index=True, nullable=True
    )
    importance_score: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )

    # Relationships -------------------------------------------------------- #
    user: Mapped[User] = relationship(back_populates="memories")
    conversation: Mapped[Conversation | None] = relationship(
        back_populates="memories"
    )

    @hybrid_property
    def is_persistent(self) -> bool:
        """True when the memory is important enough to keep long-term."""
        return self.importance_score >= IMPORTANT_MEMORY_THRESHOLD

    @is_persistent.inplace.expression
    @classmethod
    def _is_persistent_expr(cls):
        return cls.importance_score >= IMPORTANT_MEMORY_THRESHOLD

    @hybrid_property
    def is_embedded(self) -> bool:
        """True once a vector embedding has been associated."""
        return self.embedding_id is not None

    @is_embedded.inplace.expression
    @classmethod
    def _is_embedded_expr(cls):
        return cls.embedding_id.is_not(None)


class Attachment(CreatedAtMixin, Base):
    """A file (image, document, audio, video) attached to a message."""

    __tablename__ = "attachments"
    __table_args__ = (
        Index("ix_attachments_message_kind", "message_id", "kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[AttachmentType] = mapped_column(
        _enum_col(AttachmentType),
        default=AttachmentType.OTHER,
        nullable=False,
    )
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    # Local path under the configured upload/sandbox root.
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Telegram's file_id allows lazy re-download without re-uploading.
    telegram_file_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    # Kind-specific metadata: {"width":.., "height":..} / {"duration":..} etc.
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Relationships -------------------------------------------------------- #
    message: Mapped[Message] = relationship(back_populates="attachments")

    @hybrid_property
    def is_image(self) -> bool:
        """True for image attachments."""
        return self.kind == AttachmentType.IMAGE

    @is_image.inplace.expression
    @classmethod
    def _is_image_expr(cls):
        return cls.kind == AttachmentType.IMAGE
