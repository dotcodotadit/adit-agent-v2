"""``web_scrape`` tool — fetch a URL and extract its readable content.

A lightweight, dependency-free (httpx + BeautifulSoup) reader for static pages.
It strips boilerplate (``script``/``style``/``nav``/``footer``/...), pulls the
main textual content, the title, a meta description, and outbound links — the
kind of clean text an LLM can actually reason over.

For JavaScript-rendered pages that return little/no text here, fall back to the
``browser`` tool (Playwright).
"""

from __future__ import annotations

from urllib.parse import urljoin

from pydantic import BaseModel, Field, HttpUrl

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
)
from app.tools.registry import tool

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# Tags whose text is boilerplate, not content.
_STRIP_TAGS = ("script", "style", "noscript", "template", "nav", "footer", "header", "aside")
_DEFAULT_MAX_CHARS = 15_000


class WebScrapeArgs(BaseModel):
    """Arguments for :func:`web_scrape`."""

    url: HttpUrl = Field(description="The page URL to fetch and read.")
    max_chars: int = Field(
        _DEFAULT_MAX_CHARS,
        ge=500,
        le=100_000,
        description="Maximum characters of extracted text to return.",
    )
    include_links: bool = Field(
        True, description="Whether to include outbound links in the result."
    )
    max_links: int = Field(
        30, ge=0, le=200, description="Maximum number of links to return."
    )


@tool(
    name="web_scrape",
    description=(
        "Fetch a web page and extract its readable text, title, description, "
        "and links. For static/server-rendered pages; use 'browser' for "
        "JavaScript-heavy sites."
    ),
    args=WebScrapeArgs,
    category="web",
    dangerous=False,
)
async def web_scrape(args: WebScrapeArgs, ctx: ToolContext | None) -> ToolResult:
    """Scrape and clean a single page.

    Returns
    -------
    ToolResult
        ``output`` is ``{"url", "title", "description", "text", "links"}``.
        ``metadata`` carries ``status_code``, ``content_type``, and
        ``truncated``.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ToolExecutionError("httpx is required for web_scrape.") from exc

    client = ctx.http_client if ctx is not None else None
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)

    try:
        resp = await client.get(
            str(args.url), headers={"User-Agent": _USER_AGENT}
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"Failed to fetch {args.url}: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "xml" not in content_type:
        # Non-HTML (PDF, image, JSON...) — return as-is, truncated.
        body = resp.text[: args.max_chars]
        return ToolResult.ok(
            {
                "url": str(args.url),
                "title": None,
                "description": None,
                "text": body,
                "links": [],
            },
            status_code=resp.status_code,
            content_type=content_type,
            truncated=len(resp.text) > len(body),
        )

    parsed = _parse_html(resp.text, base_url=str(resp.url), args=args)
    return ToolResult(
        success=True,
        output=parsed["output"],
        metadata={
            "status_code": resp.status_code,
            "content_type": content_type,
            "truncated": parsed["truncated"],
        },
    )


def _parse_html(html: str, *, base_url: str, args: WebScrapeArgs) -> dict:
    """Extract title/description/text/links from raw HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None

    description = None
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if meta and meta.get("content"):
        description = meta["content"].strip()

    # Collapse whitespace from the remaining visible text.
    raw_text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    text = "\n".join(lines)
    truncated = len(text) > args.max_chars
    text = text[: args.max_chars]

    links: list[dict] = []
    if args.include_links and args.max_links:
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if href in seen or href.startswith(("javascript:", "mailto:")):
                continue
            seen.add(href)
            links.append({"text": a.get_text(strip=True)[:120], "url": href})
            if len(links) >= args.max_links:
                break

    return {
        "output": {
            "url": base_url,
            "title": title,
            "description": description,
            "text": text,
            "links": links,
        },
        "truncated": truncated,
    }
