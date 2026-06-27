"""``url_analyzer`` tool — analyze any URL and extract content.

Handles various content types: HTML pages, images, videos, PDFs,
JSON, and more. Provides appropriate analysis based on content type.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from app.tools.base import ToolContext, ToolResult
from app.tools.registry import tool
from app.utils.logger import get_logger

log = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class UrlAnalyzerArgs(BaseModel):
    """Arguments for URL analyzer."""

    url: HttpUrl = Field(description="URL to analyze.")
    extract_text: bool = Field(
        True, description="Extract text content from HTML pages."
    )
    max_chars: int = Field(
        20000, ge=1000, le=100000, description="Maximum characters to return."
    )


@tool(
    name="url_analyzer",
    description=(
        "Analyze any URL and extract its content. Handles HTML pages, images, "
        "videos, PDFs, JSON, and more. Returns structured information based on "
        "content type. Use this to read web pages, view images, or download files."
    ),
    args=UrlAnalyzerArgs,
    category="web",
    dangerous=False,
)
async def url_analyzer(args: UrlAnalyzerArgs, ctx: ToolContext | None) -> ToolResult:
    """Analyze a URL and extract content."""
    try:
        import httpx
    except ImportError as exc:
        return ToolResult.fail("httpx is required for url_analyzer.")

    url = str(args.url)

    # Detect special URLs and handle them
    special_result = await _handle_special_url(url, ctx)
    if special_result:
        return special_result

    # Generic URL fetch
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
    except Exception as exc:
        return ToolResult.fail(f"Failed to fetch URL: {exc}")

    content_type = resp.headers.get("content-type", "").lower()
    content_length = int(resp.headers.get("content-length", 0))

    # Route based on content type
    if "image" in content_type:
        return _analyze_image(url, content_type, content_length)
    elif "video" in content_type:
        return _analyze_video(url, content_type, content_length)
    elif "audio" in content_type:
        return _analyze_audio(url, content_type, content_length)
    elif "pdf" in content_type:
        return _analyze_pdf(url, content_length)
    elif "json" in content_type:
        return _analyze_json(resp.text, url, args.max_chars)
    elif "html" in content_type:
        return _analyze_html(resp.text, url, args.extract_text, args.max_chars)
    else:
        return _analyze_generic(resp.text, url, content_type, args.max_chars)


async def _handle_special_url(url: str, ctx: ToolContext | None) -> ToolResult | None:
    """Handle special URLs like GitHub, YouTube, Twitter, etc."""

    # GitHub repository
    if "github.com" in url:
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if match:
            owner, repo = match.group(1), match.group(2).rstrip(".git")
            return await _analyze_github_repo(owner, repo, ctx)

    # YouTube video
    if "youtube.com" in url or "youtu.be" in url:
        video_id = _extract_youtube_id(url)
        if video_id:
            return ToolResult.ok({
                "type": "video",
                "platform": "YouTube",
                "video_id": video_id,
                "url": url,
                "embed_url": f"https://www.youtube.com/embed/{video_id}",
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                "note": "Use the YouTube API or a transcript service to get video content.",
            })

    # Twitter/X post
    if "twitter.com" in url or "x.com" in url:
        match = re.search(r"(?:twitter|x)\.com/\w+/status/(\d+)", url)
        if match:
            return ToolResult.ok({
                "type": "social_media",
                "platform": "Twitter/X",
                "post_id": match.group(1),
                "url": url,
                "note": "Use the Twitter API to get full post content.",
            })

    return None


async def _analyze_github_repo(owner: str, repo: str, ctx: ToolContext | None) -> ToolResult:
    """Analyze a GitHub repository."""
    try:
        import httpx
    except ImportError:
        return ToolResult.fail("httpx not available.")

    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "User-Agent": "Adit-Agent/1.0",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(api_url, headers=headers, timeout=30)

            if resp.status_code == 404:
                return ToolResult.fail(f"Repository not found: {owner}/{repo}")

            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return ToolResult.fail(f"Failed to fetch repo: {exc}")

    return ToolResult.ok({
        "type": "github_repo",
        "name": data.get("full_name"),
        "description": data.get("description"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "language": data.get("language"),
        "topics": data.get("topics", []),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "default_branch": data.get("default_branch"),
        "url": data.get("html_url"),
        "clone_url": data.get("clone_url"),
        "homepage": data.get("homepage"),
        "license": data.get("license", {}).get("name") if data.get("license") else None,
        "open_issues": data.get("open_issues_count"),
    })


def _extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _analyze_image(url: str, content_type: str, size: int) -> ToolResult:
    """Analyze an image URL."""
    return ToolResult.ok({
        "type": "image",
        "url": url,
        "content_type": content_type,
        "size_bytes": size,
        "format": content_type.split("/")[-1] if "/" in content_type else "unknown",
        "note": "To view this image, open the URL in a browser. For AI analysis, provide the image directly to the bot.",
    })


def _analyze_video(url: str, content_type: str, size: int) -> ToolResult:
    """Analyze a video URL."""
    return ToolResult.ok({
        "type": "video",
        "url": url,
        "content_type": content_type,
        "size_bytes": size,
        "format": content_type.split("/")[-1] if "/" in content_type else "unknown",
        "note": "Video content cannot be analyzed directly. Use a video processing service or provide a transcript.",
    })


def _analyze_audio(url: str, content_type: str, size: int) -> ToolResult:
    """Analyze an audio URL."""
    return ToolResult.ok({
        "type": "audio",
        "url": url,
        "content_type": content_type,
        "size_bytes": size,
        "format": content_type.split("/")[-1] if "/" in content_type else "unknown",
        "note": "Audio content requires transcription. Use an STT service to convert to text.",
    })


def _analyze_pdf(url: str, size: int) -> ToolResult:
    """Analyze a PDF URL."""
    return ToolResult.ok({
        "type": "pdf",
        "url": url,
        "size_bytes": size,
        "note": "PDF content requires extraction. Download and use a PDF parser to extract text.",
    })


def _analyze_json(text: str, url: str, max_chars: int) -> ToolResult:
    """Analyze JSON content."""
    import json
    try:
        data = json.loads(text)
        return ToolResult.ok({
            "type": "json",
            "url": url,
            "data": data if len(text) <= max_chars else "JSON too large to display",
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "length": len(data) if isinstance(data, list) else None,
        })
    except json.JSONDecodeError:
        return ToolResult.ok({
            "type": "text",
            "url": url,
            "content": text[:max_chars],
        })


def _analyze_html(html: str, url: str, extract_text: bool, max_chars: int) -> ToolResult:
    """Analyze HTML content."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Extract metadata
    title = soup.title.get_text(strip=True) if soup.title else None

    description = None
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if meta and meta.get("content"):
        description = meta["content"].strip()

    # Extract OG image
    og_image = None
    og_meta = soup.find("meta", attrs={"property": "og:image"})
    if og_meta and og_meta.get("content"):
        og_image = og_meta["content"]

    result: dict[str, Any] = {
        "type": "html",
        "url": url,
        "title": title,
        "description": description,
        "og_image": og_image,
    }

    if extract_text:
        # Remove boilerplate
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        content = "\n".join(lines)

        result["content"] = content[:max_chars]
        result["truncated"] = len(content) > max_chars

    return ToolResult.ok(result)


def _analyze_generic(text: str, url: str, content_type: str, max_chars: int) -> ToolResult:
    """Analyze generic text content."""
    return ToolResult.ok({
        "type": "text",
        "url": url,
        "content_type": content_type,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    })
