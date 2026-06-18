"""Embedding generation for Adit-Agent's long-term memory.

:class:`EmbeddingService` wraps whatever provider exposes an ``embed`` method
(the :class:`~app.providers.base.ProviderRouter`) behind a small, defensive
interface. It centralizes the embedding model choice and turns provider failures
into ``None`` results so the memory layer can degrade to relational recall rather
than propagating errors into a user's turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.agent.context_builder import LLMRouter

log = get_logger(__name__)

__all__ = ["EmbeddingService"]


class EmbeddingService:
    """Produces embedding vectors via a provider, or ``None`` when unavailable.

    Parameters
    ----------
    provider:
        An object with ``async embed(texts, *, model=...) -> list[list[float]]``
        (typically the provider router), or ``None`` to disable embeddings.
    model:
        The embedding model id to request.
    """

    def __init__(self, provider: "LLMRouter | None", *, model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def available(self) -> bool:
        """True when a provider capable of embedding was supplied."""
        return self._provider is not None

    async def embed_one(self, text: str) -> list[float] | None:
        """Embed a single string, or return ``None`` on failure/empty input."""
        if not text or self._provider is None:
            return None
        vectors = await self.embed_many([text])
        return vectors[0] if vectors else None

    async def embed_many(self, texts: list[str]) -> list[list[float]] | None:
        """Embed several strings, or return ``None`` on failure."""
        if not texts or self._provider is None:
            return None
        try:
            return await self._provider.embed(texts, model=self._model)
        except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
            log.warning("Embedding request failed: {}", exc)
            return None
