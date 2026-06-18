"""``image_reader`` tool — understand an image (OCR + description + reasoning).

The tool always extracts cheap, local metadata (format, dimensions, EXIF) using
Pillow. For the *semantic* part — OCR, a visual description, and answering a
question about the image — it delegates to a vision-capable LLM exposed on
``ctx.provider_router``.

Expected provider contract (duck-typed)::

    async def vision(self, *, prompt: str, images: list[str]) -> str
        # images are data URLs / base64-encoded image strings

If no vision provider is configured the tool still returns metadata and reports
that semantic analysis was skipped, rather than failing outright.
"""

from __future__ import annotations

import base64

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
    resolve_in_sandbox,
)
from app.tools.registry import tool

# Cap the image size we are willing to base64 into a prompt (~bytes on disk).
_MAX_IMAGE_BYTES = 8_000_000


class ImageReaderArgs(BaseModel):
    """Arguments for :func:`image_reader`."""

    path: str = Field(
        description="Path to the image file, relative to the sandbox root.",
        min_length=1,
    )
    prompt: str = Field(
        "Describe this image in detail. Transcribe any visible text (OCR).",
        description="Instruction guiding what the vision model should report.",
    )
    extract_text_only: bool = Field(
        False,
        description="If true, ask the model to return only transcribed text (OCR).",
    )


@tool(
    name="image_reader",
    description=(
        "Analyze an image: extract metadata, perform OCR, produce a visual "
        "description, and reason about its contents in response to a prompt."
    ),
    args=ImageReaderArgs,
    category="media",
    dangerous=False,
)
async def image_reader(args: ImageReaderArgs, ctx: ToolContext | None) -> ToolResult:
    """Read and (optionally) describe an image.

    Returns
    -------
    ToolResult
        ``output`` is a dict with ``metadata`` (format/size/dimensions/EXIF) and
        ``analysis`` (the vision model's text, or ``None`` if unavailable).
    """
    if ctx is None:
        raise ToolExecutionError("image_reader requires a ToolContext with settings.")

    target = resolve_in_sandbox(args.path, ctx.settings.sandbox_root)
    if not target.is_file():
        raise ToolExecutionError(f"Image not found: {args.path}")

    size = target.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise ToolExecutionError(
            f"Image too large ({size} bytes; limit {_MAX_IMAGE_BYTES})."
        )

    # ---- Local metadata via Pillow (lazy import) ----------------------------
    try:
        from PIL import Image, ExifTags
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ToolExecutionError(
            "Pillow is required for image_reader but is not installed."
        ) from exc

    try:
        with Image.open(target) as img:
            meta = {
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_bytes": size,
            }
            exif_raw = getattr(img, "_getexif", lambda: None)()
            if exif_raw:
                meta["exif"] = {
                    ExifTags.TAGS.get(k, str(k)): str(v)
                    for k, v in exif_raw.items()
                    if isinstance(v, (str, int, float))
                }
    except Exception as exc:  # noqa: BLE001 - Pillow raises many error types
        raise ToolExecutionError(f"Could not open image {args.path}: {exc}") from exc

    # ---- Semantic analysis via vision provider (optional) -------------------
    analysis: str | None = None
    router = ctx.provider_router
    if router is not None and hasattr(router, "vision"):
        prompt = (
            "Transcribe all text in this image verbatim. Output only the text."
            if args.extract_text_only
            else args.prompt
        )
        b64 = base64.b64encode(target.read_bytes()).decode("ascii")
        data_url = f"data:image/{(meta['format'] or 'png').lower()};base64,{b64}"
        try:
            analysis = await router.vision(prompt=prompt, images=[data_url])
        except Exception as exc:  # noqa: BLE001 - provider failures are runtime
            raise ToolExecutionError(f"Vision analysis failed: {exc}") from exc

    return ToolResult.ok(
        {"metadata": meta, "analysis": analysis},
        analyzed=analysis is not None,
        note=None if analysis is not None else "No vision provider configured; "
        "returned metadata only.",
    )
