"""Multimodal dispatch pipeline for Adit-Agent.

:class:`MediaPipeline` is the single front door the bot layer uses to turn any
uploaded file into a :class:`~app.multimodal.base.ProcessedMedia`. It detects the
media kind (or accepts a hint), routes to the right extractor, and ensures
blocking work (PDF/DOCX parsing) runs off the event loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from app.database.models import AttachmentType
from app.multimodal.audio import process_audio
from app.multimodal.base import ProcessedMedia, detect_kind
from app.multimodal.documents import extract_document
from app.multimodal.images import process_image
from app.multimodal.video import process_video
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.config import Settings
    from app.providers.base import ProviderRouter

log = get_logger(__name__)

__all__ = ["MediaPipeline"]


class MediaPipeline:
    """Routes media files to the appropriate processing pipeline.

    Parameters
    ----------
    provider_router:
        Provider used for vision and transcription; optional (text/document
        extraction works without it).
    settings:
        Application settings (used for cache directories in video processing).
    """

    def __init__(
        self,
        *,
        provider_router: "ProviderRouter | None" = None,
        settings: "Settings | None" = None,
    ) -> None:
        self._provider = provider_router
        self._settings = settings

    async def process(
        self,
        path: str | Path,
        *,
        kind: AttachmentType | None = None,
        prompt: str | None = None,
    ) -> ProcessedMedia:
        """Process ``path`` and return normalized text/metadata.

        Parameters
        ----------
        path:
            Local file path (already downloaded into the sandbox/upload dir).
        kind:
            Optional attachment kind; inferred from the extension when omitted.
        prompt:
            Optional instruction passed to vision processing (images).
        """
        path = Path(path)
        kind = kind or detect_kind(path)
        log.debug("Processing {} as {}.", path.name, kind.value)

        try:
            if kind is AttachmentType.IMAGE:
                extra = {"prompt": prompt} if prompt else {}
                return await process_image(path, provider=self._provider, **extra)
            if kind is AttachmentType.AUDIO:
                return await process_audio(path, provider=self._provider)
            if kind is AttachmentType.VIDEO:
                return await process_video(
                    path, provider=self._provider, settings=self._settings
                )
            if kind is AttachmentType.DOCUMENT:
                # Synchronous parsing → run in a worker thread.
                return await asyncio.to_thread(extract_document, path)
            # Unknown type: try to read it as a document as a last resort.
            return await asyncio.to_thread(extract_document, path)
        except Exception as exc:  # noqa: BLE001 - never propagate to the handler
            log.exception("Media pipeline failed for {}.", path.name)
            return ProcessedMedia.failed(kind, f"unexpected error: {exc}")
