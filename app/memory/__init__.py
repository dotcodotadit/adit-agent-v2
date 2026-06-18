"""Long-term memory storage layer for Adit-Agent.

This package holds the low-level stores used by the high-level
:class:`~app.agent.memory_manager.MemoryManager`:

* :class:`~app.memory.vector_store.VectorStore` — async ChromaDB wrapper for
  semantic recall.
* :class:`~app.memory.embeddings.EmbeddingService` — provider-backed embedding
  generation.

Both are defensive and optional: when their backends are absent the memory
manager transparently falls back to relational recall.
"""

from __future__ import annotations

from app.memory.embeddings import EmbeddingService
from app.memory.vector_store import VectorMatch, VectorStore

__all__ = ["VectorStore", "VectorMatch", "EmbeddingService"]
