"""Multimodal input pipelines for Adit-Agent.

Turns uploaded files — documents, images, audio, video — into prompt-ready text
via a single entry point, :class:`~app.multimodal.pipeline.MediaPipeline`.
"""

from __future__ import annotations

from app.multimodal.base import MediaError, ProcessedMedia, detect_kind
from app.multimodal.pipeline import MediaPipeline

__all__ = ["MediaPipeline", "ProcessedMedia", "MediaError", "detect_kind"]
