"""Document text extraction for Adit-Agent.

Extracts plain text from PDFs (``pypdf``), Word documents (``python-docx``) and a
broad set of plain-text/code formats. Heavy parsing is synchronous; the pipeline
calls :func:`extract_document` inside a worker thread so the event loop stays
free.
"""

from __future__ import annotations

from pathlib import Path

from app.database.models import AttachmentType
from app.multimodal.base import ProcessedMedia
from app.utils.logger import get_logger

log = get_logger(__name__)

__all__ = ["extract_document", "MAX_DOCUMENT_CHARS"]

# Cap on extracted text so a giant document can't blow the context window.
MAX_DOCUMENT_CHARS = 50_000


def extract_document(path: str | Path) -> ProcessedMedia:
    """Extract text from a document file (synchronous).

    Dispatches on file extension. Always returns a :class:`ProcessedMedia`;
    failures are captured rather than raised so the caller can degrade.
    """
    p = Path(path)
    if not p.is_file():
        return ProcessedMedia.failed(AttachmentType.DOCUMENT, f"file not found: {p.name}")

    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            text, meta = _extract_pdf(p)
        elif suffix == ".docx":
            text, meta = _extract_docx(p)
        else:
            text, meta = _extract_text(p)
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        log.warning("Document extraction failed for {}: {}", p.name, exc)
        return ProcessedMedia.failed(AttachmentType.DOCUMENT, str(exc))

    truncated = len(text) > MAX_DOCUMENT_CHARS
    if truncated:
        text = text[:MAX_DOCUMENT_CHARS] + "\n…[truncated]"
    meta["truncated"] = truncated
    meta["char_count"] = len(text)

    return ProcessedMedia(
        kind=AttachmentType.DOCUMENT,
        success=bool(text.strip()),
        text=text,
        metadata=meta,
        error=None if text.strip() else "no extractable text",
    )


def _extract_pdf(path: Path) -> tuple[str, dict]:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdf is required to read PDF files.") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(pages).strip()
    return text, {"format": "pdf", "pages": len(reader.pages)}


def _extract_docx(path: Path) -> tuple[str, dict]:
    """Extract text from a Word document using python-docx."""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("python-docx is required to read .docx files.") from exc

    document = docx.Document(str(path))
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    text = "\n".join(paragraphs).strip()
    return text, {"format": "docx", "paragraphs": len(paragraphs)}


def _extract_text(path: Path) -> tuple[str, dict]:
    """Read a plain-text/code file, decoding leniently."""
    raw = path.read_bytes()[: MAX_DOCUMENT_CHARS * 4]
    text = raw.decode("utf-8", errors="replace").strip()
    return text, {"format": path.suffix.lstrip(".") or "text"}
