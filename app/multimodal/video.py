"""Video understanding for Adit-Agent.

Samples evenly-spaced keyframes with ``ffmpeg``, describes each with the vision
provider, and fuses the descriptions into a short summary. Requires ffmpeg on the
OS PATH and a vision-capable provider; when either is missing it returns a clear,
non-fatal failure result.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from app.database.models import AttachmentType
from app.multimodal.base import ProcessedMedia
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.config import Settings
    from app.providers.base import ProviderRouter

log = get_logger(__name__)

__all__ = ["process_video"]

_DEFAULT_KEYFRAMES = 5


async def process_video(
    path: str | Path,
    *,
    provider: "ProviderRouter | None" = None,
    settings: "Settings | None" = None,
    num_keyframes: int = _DEFAULT_KEYFRAMES,
) -> ProcessedMedia:
    """Summarize a video by describing sampled keyframes."""
    p = Path(path)
    if not p.is_file():
        return ProcessedMedia.failed(AttachmentType.VIDEO, f"file not found: {p.name}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return ProcessedMedia.failed(AttachmentType.VIDEO, "ffmpeg is not installed")
    if provider is None or not hasattr(provider, "vision"):
        return ProcessedMedia.failed(AttachmentType.VIDEO, "no vision provider configured")

    # Sample keyframes into a cache subdirectory.
    cache_root = settings.cache_dir if settings is not None else p.parent
    out_dir = Path(cache_root) / "keyframes" / p.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%03d.jpg"

    cmd = [
        ffmpeg, "-y", "-i", str(p),
        "-vf", "thumbnail,fps=1,scale=512:-1",
        "-frames:v", str(num_keyframes),
        str(pattern),
    ]
    try:
        await _run_ffmpeg(cmd)
    except Exception as exc:  # noqa: BLE001
        return ProcessedMedia.failed(AttachmentType.VIDEO, f"ffmpeg failed: {exc}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        return ProcessedMedia.failed(AttachmentType.VIDEO, "no keyframes produced")

    descriptions: list[str] = []
    for idx, frame in enumerate(frames, start=1):
        try:
            b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
            desc = await provider.vision(
                prompt=f"Describe frame {idx} of {len(frames)} concisely.",
                images=[f"data:image/jpeg;base64,{b64}"],
            )
            descriptions.append(desc)
        except Exception as exc:  # noqa: BLE001 - skip a bad frame, keep going
            log.warning("Vision failed on frame {}: {}", idx, exc)

    if not descriptions:
        return ProcessedMedia.failed(AttachmentType.VIDEO, "could not describe any frames")

    per_frame = "\n".join(f"Frame {i}: {d}" for i, d in enumerate(descriptions, 1))
    summary = per_frame
    if hasattr(provider, "complete"):
        try:
            joined = "\n".join(f"- {d}" for d in descriptions)
            resp = await provider.complete(
                f"Fuse these keyframe descriptions into a short, coherent "
                f"summary of the video:\n{joined}"
            )
            summary = resp.content or per_frame
        except Exception as exc:  # noqa: BLE001
            log.warning("Video summary fusion failed: {}", exc)

    return ProcessedMedia(
        kind=AttachmentType.VIDEO,
        success=True,
        text=summary,
        summary=summary,
        metadata={"keyframes": len(frames), "frame_descriptions": descriptions},
    )


async def _run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg, raising on a non-zero exit."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode("utf-8", "ignore")[-300:]
        raise RuntimeError(f"exit {proc.returncode}: {tail}")
