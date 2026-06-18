"""Vector store abstraction for Adit-Agent's long-term memory.

:class:`VectorStore` is a thin, async-friendly wrapper around a ChromaDB
collection. It exists so the rest of the app (notably
:class:`~app.agent.memory_manager.MemoryManager`) can ``add`` and ``query``
embedded memories without knowing anything about Chroma's synchronous API or
its result shapes.

Two design points:

* **Non-blocking.** ChromaDB's client is synchronous; every call is dispatched
  to a worker thread via :func:`asyncio.to_thread` so the event loop is never
  blocked.
* **Optional.** The whole long-term memory subsystem is best-effort. If no
  Chroma client is supplied (or it errors), the store reports itself unavailable
  and callers fall back to relational recall instead of crashing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.utils.logger import get_logger

log = get_logger(__name__)

__all__ = ["VectorMatch", "VectorStore"]


@dataclass(slots=True)
class VectorMatch:
    """A single similarity-search hit.

    ``score`` is a unit-interval relevance estimate derived from the backend's
    distance metric (higher = more similar).
    """

    id: str
    document: str
    metadata: dict[str, Any]
    score: float


class VectorStore:
    """Async wrapper around one ChromaDB collection.

    Parameters
    ----------
    client:
        A ``chromadb`` client (e.g. ``PersistentClient``) or ``None`` to run in
        a disabled, no-op mode.
    collection_name:
        Name of the collection to read/write.
    """

    def __init__(self, client: Any, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._collection: Any = None

    @property
    def available(self) -> bool:
        """True when a backing client was provided."""
        return self._client is not None

    # ------------------------------------------------------------------ #
    # Collection access
    # ------------------------------------------------------------------ #
    async def _collection_handle(self) -> Any:
        """Lazily resolve and cache the underlying collection (or ``None``)."""
        if self._collection is not None:
            return self._collection
        if self._client is None:
            return None
        try:
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection, self._collection_name
            )
        except Exception as exc:  # noqa: BLE001 - degrade to disabled
            log.warning("Could not open vector collection {!r}: {}",
                        self._collection_name, exc)
            self._collection = None
        return self._collection

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def add(
        self,
        *,
        id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Index a single embedded document. Returns ``True`` on success."""
        collection = await self._collection_handle()
        if collection is None:
            return False
        try:
            await asyncio.to_thread(
                collection.add,
                ids=[id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata or {}],
            )
            return True
        except Exception as exc:  # noqa: BLE001 - indexing must never break a turn
            log.warning("Vector add failed for id={}: {}", id, exc)
            return False

    async def delete(self, ids: list[str]) -> None:
        """Remove documents by id (best-effort)."""
        collection = await self._collection_handle()
        if collection is None or not ids:
            return
        try:
            await asyncio.to_thread(collection.delete, ids=ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector delete failed: {}", exc)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    async def query(
        self,
        *,
        embedding: list[float],
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Return the ``limit`` nearest documents to ``embedding``.

        Distances are mapped to a ``[0, 1]`` relevance score assuming a cosine
        metric (Chroma's default), where distance ``0`` → score ``1``.
        """
        collection = await self._collection_handle()
        if collection is None:
            return []
        try:
            result = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=limit,
                where=where or None,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector query failed: {}", exc)
            return []

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: list[VectorMatch] = []
        for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
            if not doc:
                continue
            matches.append(
                VectorMatch(
                    id=id_,
                    document=doc,
                    metadata=meta or {},
                    score=_distance_to_score(dist),
                )
            )
        return matches

    async def count(self) -> int:
        """Return the number of documents in the collection (0 if unavailable)."""
        collection = await self._collection_handle()
        if collection is None:
            return 0
        try:
            return int(await asyncio.to_thread(collection.count))
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector count failed: {}", exc)
            return 0


def _distance_to_score(distance: Any) -> float:
    """Map a cosine distance in ``[0, 2]`` to a relevance score in ``[0, 1]``."""
    try:
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
    except (TypeError, ValueError):
        return 0.0
