"""Database layer for Adit-Agent.

Exports the ORM models, enum types, session management helpers, and the
migration entry point so the rest of the application can do::

    from app.database import User, Conversation, Message, Memory, MemoryType

The SQLAlchemy async engine and session factory are owned by
:class:`~app.dependencies.AppContainer` at startup; this package only defines
the schema and provides low-level session management tools.
"""

from __future__ import annotations

from app.database.migrations import create_all, drop_all, run_migrations
from app.database.models import (
    Attachment,
    AttachmentType,
    Base,
    Conversation,
    ConversationMode,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    ToolCall,
    User,
)
from app.database.session import (
    DatabaseNotInitializedError,
    DatabaseSessionManager,
    get_session,
    sessionmanager,
)

__all__ = [
    # Models
    "Base",
    "User",
    "Conversation",
    "Message",
    "ToolCall",
    "Memory",
    "Attachment",
    # Enums
    "ConversationMode",
    "MessageRole",
    "MemoryType",
    "AttachmentType",
    # Session management
    "DatabaseSessionManager",
    "DatabaseNotInitializedError",
    "sessionmanager",
    "get_session",
    # Migrations
    "create_all",
    "drop_all",
    "run_migrations",
]
