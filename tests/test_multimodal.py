"""Tests for multimodal pipelines and the base ProcessedMedia type."""

from __future__ import annotations

import pytest

from app.database.models import AttachmentType
from app.multimodal.base import MediaError, ProcessedMedia, detect_kind
from app.multimodal.documents import extract_document


# --------------------------------------------------------------------------- #
# ProcessedMedia helpers
# --------------------------------------------------------------------------- #
class TestProcessedMedia:
    def test_failed_factory(self):
        pm = ProcessedMedia.failed(AttachmentType.IMAGE, "not found")
        assert not pm.success
        assert pm.error == "not found"

    def test_as_context_success(self):
        pm = ProcessedMedia(
            kind=AttachmentType.DOCUMENT,
            success=True,
            text="hello world",
        )
        ctx = pm.as_context(file_name="notes.txt")
        assert "DOCUMENT" in ctx
        assert "notes.txt" in ctx
        assert "hello world" in ctx

    def test_as_context_failure(self):
        pm = ProcessedMedia.failed(AttachmentType.AUDIO, "no provider")
        ctx = pm.as_context()
        assert "AUDIO" in ctx
        assert "no provider" in ctx


# --------------------------------------------------------------------------- #
# detect_kind
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,expected", [
    ("report.pdf", AttachmentType.DOCUMENT),
    ("photo.jpg", AttachmentType.IMAGE),
    ("clip.mp4", AttachmentType.VIDEO),
    ("voice.ogg", AttachmentType.AUDIO),
    ("script.py", AttachmentType.DOCUMENT),
    ("data.csv", AttachmentType.DOCUMENT),
    ("unknown.xyz", AttachmentType.OTHER),
])
def test_detect_kind(filename, expected):
    assert detect_kind(filename) is expected


# --------------------------------------------------------------------------- #
# Document extraction (pure Python — no external deps needed beyond pypdf)
# --------------------------------------------------------------------------- #
class TestDocumentExtraction:
    def test_text_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        result = extract_document(f)
        assert result.success
        assert "Hello, world!" in result.text
        assert result.kind is AttachmentType.DOCUMENT

    def test_missing_file(self, tmp_path):
        result = extract_document(tmp_path / "ghost.txt")
        assert not result.success
        assert result.error is not None

    def test_python_source_extracted(self, tmp_path):
        f = tmp_path / "example.py"
        f.write_text("def hello():\n    return 42\n", encoding="utf-8")
        result = extract_document(f)
        assert result.success
        assert "def hello" in result.text

    def test_large_file_truncated(self, tmp_path):
        from app.multimodal.documents import MAX_DOCUMENT_CHARS

        f = tmp_path / "big.txt"
        f.write_bytes(b"A" * (MAX_DOCUMENT_CHARS * 2))
        result = extract_document(f)
        assert result.success
        assert result.metadata.get("truncated") is True
        assert len(result.text) <= MAX_DOCUMENT_CHARS + 50  # small overhead for marker

    def test_binary_file_decoded_leniently(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(bytes(range(256)))
        result = extract_document(f)
        # Should not raise; replaces bad bytes with replacement character.
        assert result.success


# --------------------------------------------------------------------------- #
# MediaPipeline dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pipeline_routes_text_document(tmp_path):
    from app.multimodal.pipeline import MediaPipeline

    f = tmp_path / "notes.txt"
    f.write_text("pipeline test content", encoding="utf-8")

    pipeline = MediaPipeline()
    result = await pipeline.process(f)
    assert result.success
    assert "pipeline test content" in result.text


@pytest.mark.asyncio
async def test_pipeline_returns_failure_for_missing_file(tmp_path):
    from app.multimodal.pipeline import MediaPipeline

    pipeline = MediaPipeline()
    result = await pipeline.process(tmp_path / "missing.txt")
    assert not result.success


@pytest.mark.asyncio
async def test_pipeline_image_no_provider(tmp_path):
    """Image processing without a provider returns metadata-only (not an error)."""
    from PIL import Image

    from app.multimodal.pipeline import MediaPipeline

    img_path = tmp_path / "test.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(str(img_path))

    pipeline = MediaPipeline(provider_router=None)
    result = await pipeline.process(img_path, kind=AttachmentType.IMAGE)
    assert result.success  # metadata-only result is still considered success
    assert result.kind is AttachmentType.IMAGE
