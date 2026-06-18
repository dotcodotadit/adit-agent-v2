"""``audio_reader`` tool — transcribe speech from an audio file.

Transcription is delegated to a speech-to-text backend exposed on
``ctx.provider_router``. Two duck-typed contracts are accepted, tried in order:

    async def transcribe(self, *, audio_path: str, language: str | None) -> str
    async def transcribe(self, audio_path: str) -> str

This keeps the tool independent of the eventual STT choice (faster-whisper, a
hosted Whisper endpoint, etc. — see the TODO in requirements.txt). If no STT
backend is configured, a :class:`ToolNotConfiguredError` is raised.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolNotConfiguredError,
    ToolResult,
    resolve_in_sandbox,
)
from app.tools.registry import tool

# Common container/codec extensions we accept; ffmpeg-backed STT can take more.
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".webm"}


class AudioReaderArgs(BaseModel):
    """Arguments for :func:`audio_reader`."""

    path: str = Field(
        description="Path to the audio file, relative to the sandbox root.",
        min_length=1,
    )
    language: str | None = Field(
        None,
        description="Optional ISO-639-1 language hint (e.g. 'en', 'id'). "
        "Leave unset for auto-detection.",
    )


@tool(
    name="audio_reader",
    description=(
        "Transcribe spoken audio to text. Accepts common audio formats and "
        "returns the transcript plus the detected/declared language."
    ),
    args=AudioReaderArgs,
    category="media",
    dangerous=False,
)
async def audio_reader(args: AudioReaderArgs, ctx: ToolContext | None) -> ToolResult:
    """Transcribe an audio file.

    Returns
    -------
    ToolResult
        ``output`` is a dict with ``transcript`` and ``language``.
    """
    if ctx is None:
        raise ToolExecutionError("audio_reader requires a ToolContext with settings.")

    target = resolve_in_sandbox(args.path, ctx.settings.sandbox_root)
    if not target.is_file():
        raise ToolExecutionError(f"Audio file not found: {args.path}")
    if target.suffix.lower() not in _AUDIO_SUFFIXES:
        raise ToolExecutionError(
            f"Unsupported audio extension {target.suffix!r}. "
            f"Expected one of: {', '.join(sorted(_AUDIO_SUFFIXES))}."
        )

    router = ctx.provider_router
    if router is None or not hasattr(router, "transcribe"):
        raise ToolNotConfiguredError(
            "audio_reader needs a speech-to-text backend on "
            "ctx.provider_router.transcribe()."
        )

    # Support both keyword and positional transcribe signatures.
    try:
        try:
            transcript = await router.transcribe(
                audio_path=str(target), language=args.language
            )
        except TypeError:
            transcript = await router.transcribe(str(target))
    except Exception as exc:  # noqa: BLE001 - backend failures are runtime
        raise ToolExecutionError(f"Transcription failed: {exc}") from exc

    return ToolResult.ok(
        {"transcript": transcript, "language": args.language},
        chars=len(transcript or ""),
    )
