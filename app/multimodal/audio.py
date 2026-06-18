"""Audio transcription for Adit-Agent.

Delegates speech-to-text to a provider exposing ``transcribe(audio_path=...,
language=...)`` (the provider router). Returns the transcript as the media text
so the agent can reason over spoken input the same way it does typed input.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.database.models import AttachmentType
from app.multimodal.base import ProcessedMedia
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.providers.base import ProviderRouter

log = get_logger(__name__)

__all__ = ["process_audio"]


async def process_audio(
    path: str | Path,
    *,
    provider: "ProviderRouter | None" = None,
    language: str | None = None,
) -> ProcessedMedia:
    """Transcribe an audio file to text."""
    p = Path(path)
    if not p.is_file():
        return ProcessedMedia.failed(AttachmentType.AUDIO, f"file not found: {p.name}")

    if provider is None or not hasattr(provider, "transcribe"):
        return ProcessedMedia.failed(
            AttachmentType.AUDIO, "no speech-to-text provider configured"
        )

    try:
        transcript = await provider.transcribe(audio_path=str(p), language=language)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        log.warning("Transcription failed for {}: {}", p.name, exc)
        return ProcessedMedia.failed(AttachmentType.AUDIO, f"transcription failed: {exc}")

    transcript = (transcript or "").strip()
    return ProcessedMedia(
        kind=AttachmentType.AUDIO,
        success=bool(transcript),
        text=transcript,
        metadata={"language": language, "chars": len(transcript)},
        error=None if transcript else "empty transcript",
    )
