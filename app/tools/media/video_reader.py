"""``video_reader`` tool — summarize a video via keyframe sampling.

Pipeline:

1. Probe the video and sample evenly-spaced **keyframes** with ``ffmpeg``
   (must be installed at the OS level — see requirements.txt).
2. Describe each keyframe with the vision provider on ``ctx.provider_router``
   (same ``vision(prompt, images)`` contract as :mod:`image_reader`).
3. Ask the text provider to fuse the per-frame descriptions into a single
   coherent summary.

ffmpeg is invoked asynchronously via :mod:`asyncio.subprocess`. If ffmpeg or a
provider is missing, the tool raises a clear :class:`ToolNotConfiguredError`.
"""

from __future__ import annotations

import asyncio
import base64
import shutil

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolNotConfiguredError,
    ToolResult,
    resolve_in_sandbox,
)
from app.tools.registry import tool


class VideoReaderArgs(BaseModel):
    """Arguments for :func:`video_reader`."""

    path: str = Field(
        description="Path to the video file, relative to the sandbox root.",
        min_length=1,
    )
    num_keyframes: int = Field(
        6, ge=1, le=24, description="How many keyframes to sample and describe."
    )
    prompt: str = Field(
        "Summarize what happens in this video.",
        description="Guides the final summary produced from the keyframes.",
    )


@tool(
    name="video_reader",
    description=(
        "Understand a video by sampling keyframes, describing each with a "
        "vision model, and summarizing the sequence into a coherent overview."
    ),
    args=VideoReaderArgs,
    category="media",
    dangerous=False,
)
async def video_reader(args: VideoReaderArgs, ctx: ToolContext | None) -> ToolResult:
    """Extract keyframes and produce a summary.

    Returns
    -------
    ToolResult
        ``output`` is a dict with ``frame_descriptions`` (list) and ``summary``.
    """
    if ctx is None:
        raise ToolExecutionError("video_reader requires a ToolContext with settings.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ToolNotConfiguredError(
            "ffmpeg is not installed/on PATH; it is required to sample keyframes."
        )

    router = ctx.provider_router
    if router is None or not hasattr(router, "vision"):
        raise ToolNotConfiguredError(
            "video_reader needs a vision-capable provider on ctx.provider_router."
        )

    source = resolve_in_sandbox(args.path, ctx.settings.sandbox_root)
    if not source.is_file():
        raise ToolExecutionError(f"Video not found: {args.path}")

    # Sample keyframes into a per-run subdirectory of the cache dir.
    out_dir = ctx.settings.cache_dir / "keyframes" / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%03d.jpg"

    # `thumbnail` + fps filter gives representative, evenly-spread frames.
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        "thumbnail,fps=1,scale=512:-1",
        "-frames:v",
        str(args.num_keyframes),
        str(pattern),
    ]
    await _run_ffmpeg(cmd)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise ToolExecutionError(
            "ffmpeg produced no keyframes; the file may be unreadable or empty."
        )

    # Describe each keyframe.
    descriptions: list[str] = []
    for idx, frame in enumerate(frames, start=1):
        b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        try:
            desc = await router.vision(
                prompt=f"Describe frame {idx} of {len(frames)} concisely.",
                images=[data_url],
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(f"Vision failed on frame {idx}: {exc}") from exc
        descriptions.append(desc)

    # Fuse into a single summary using the text provider when available.
    summary = "\n".join(f"Frame {i}: {d}" for i, d in enumerate(descriptions, 1))
    if hasattr(router, "complete"):
        joined = "\n".join(f"- {d}" for d in descriptions)
        try:
            response = await router.complete(
                f"{args.prompt}\n\nKeyframe descriptions:\n{joined}"
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(f"Summary generation failed: {exc}") from exc
        # The provider router returns a structured LLMResponse; use its text.
        summary = getattr(response, "content", None) or summary

    return ToolResult.ok(
        {"frame_descriptions": descriptions, "summary": summary},
        keyframes=len(frames),
    )


async def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command, raising :class:`ToolExecutionError` on failure."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode("utf-8", "ignore")[-500:]
        raise ToolExecutionError(f"ffmpeg failed (exit {proc.returncode}): {tail}")
