"""Image understanding for Adit-Agent.

Extracts cheap local metadata with Pillow and, when a vision-capable provider is
available, produces a description + OCR by sending the image to the model. Mirrors
the contract used by the ``image_reader`` tool but is driven directly by the bot
layer when a user simply uploads a picture.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

from app.database.models import AttachmentType
from app.multimodal.base import ProcessedMedia
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.providers.base import ProviderRouter

log = get_logger(__name__)

__all__ = ["process_image", "MAX_IMAGE_BYTES"]

MAX_IMAGE_BYTES = 8_000_000

_DEFAULT_PROMPT = (
    "Describe this image in detail and transcribe any visible text (OCR). "
    "Be concise but complete."
)


async def process_image(
    path: str | Path,
    *,
    provider: "ProviderRouter | None" = None,
    prompt: str = _DEFAULT_PROMPT,
) -> ProcessedMedia:
    """Read image metadata and (if possible) a vision description."""
    p = Path(path)
    if not p.is_file():
        return ProcessedMedia.failed(AttachmentType.IMAGE, f"file not found: {p.name}")

    size = p.stat().st_size
    if size > MAX_IMAGE_BYTES:
        return ProcessedMedia.failed(
            AttachmentType.IMAGE, f"image too large ({size} bytes)"
        )

    meta: dict = {"size_bytes": size}
    fmt = p.suffix.lstrip(".").lower() or "png"
    try:
        from PIL import Image

        with Image.open(p) as img:
            meta.update(
                format=img.format, mode=img.mode, width=img.width, height=img.height
            )
            fmt = (img.format or fmt).lower()
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        log.debug("Pillow could not read image metadata: {}", exc)

    description = ""
    if provider is not None and hasattr(provider, "vision"):
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            data_url = f"data:image/{fmt};base64,{b64}"
            description = await provider.vision(prompt=prompt, images=[data_url])
        except Exception as exc:  # noqa: BLE001 - degrade to metadata-only
            log.warning("Vision description failed for {}: {}", p.name, exc)

    if not description:
        return ProcessedMedia(
            kind=AttachmentType.IMAGE,
            success=True,
            text="",
            summary="(image received; no vision model available to describe it)",
            metadata=meta,
        )

    return ProcessedMedia(
        kind=AttachmentType.IMAGE,
        success=True,
        text=description,
        metadata=meta,
    )
