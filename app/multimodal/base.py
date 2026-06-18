"""Shared types and helpers for Adit-Agent's multimodal pipelines.

Each pipeline (documents, images, audio, video) turns a file on disk into a
:class:`ProcessedMedia` — a uniform, text-centric result the agent can fold into
a prompt. Keeping the result shape uniform means the bot layer handles every
attachment type the same way: process the file, then attach
``ProcessedMedia.as_context()`` to the user's message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.database.models import AttachmentType

__all__ = ["MediaError", "ProcessedMedia", "detect_kind"]

# Extension → attachment kind. Used when the caller doesn't already know the type
# (e.g. a generic document upload).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".aac"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_DOC_EXTS = {
    ".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".tsv", ".json",
    ".log", ".rtf", ".html", ".htm", ".xml", ".yaml", ".yml", ".ini",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".sql",
}


class MediaError(RuntimeError):
    """Raised for unrecoverable media-processing failures."""


@dataclass(slots=True)
class ProcessedMedia:
    """The normalized outcome of processing one media file.

    Attributes
    ----------
    kind:
        Detected/declared attachment kind.
    success:
        Whether usable content was produced.
    text:
        The primary extracted content (document text, transcript, or image/video
        description). May be empty on failure.
    summary:
        Optional short summary or caption.
    metadata:
        Side info (dimensions, page/char counts, language, ...).
    error:
        Human-readable failure reason when ``success`` is False.
    """

    kind: AttachmentType
    success: bool
    text: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def failed(cls, kind: AttachmentType, error: str) -> "ProcessedMedia":
        """Build a failed result with a reason."""
        return cls(kind=kind, success=False, error=error)

    def as_context(self, *, file_name: str | None = None) -> str:
        """Render this result as a prompt-ready context block.

        Produces a labeled section the orchestrator can append to the user's
        message so the model "sees" the attachment as text.
        """
        label = self.kind.value.upper()
        name = f" ({file_name})" if file_name else ""
        if not self.success:
            return f"[{label} ATTACHMENT{name}: could not be processed — {self.error}]"
        body = self.text.strip() or self.summary.strip() or "(no extractable content)"
        return f"[{label} ATTACHMENT{name}]\n{body}"


def detect_kind(path: str | Path) -> AttachmentType:
    """Infer the :class:`AttachmentType` of ``path`` from its extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return AttachmentType.IMAGE
    if suffix in _AUDIO_EXTS:
        return AttachmentType.AUDIO
    if suffix in _VIDEO_EXTS:
        return AttachmentType.VIDEO
    if suffix in _DOC_EXTS:
        return AttachmentType.DOCUMENT
    return AttachmentType.OTHER
