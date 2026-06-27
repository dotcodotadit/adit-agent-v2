"""``web_search`` tool — web search via SerpAPI (Google/Bing) with DuckDuckGo fallback.

When a SerpAPI key is configured (``SERPAPI_API_KEY``), searches use Google via
SerpAPI for superior results. Without a key, the tool falls back to DuckDuckGo's
free HTML endpoint.

Returns a ranked list of ``{title, url, snippet}`` results.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
)
from app.tools.registry import tool
from app.utils.logger import get_logger

log = get_logger(__name__)

_DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class WebSearchArgs(BaseModel):
    """Arguments for :func:`web_search`."""

    query: str = Field(description="Search query.", min_length=1, max_length=500)
    max_results: int = Field(
        5, ge=1, le=25, description="Maximum number of results to return."
    )
    region: str = Field(
        "wt-wt",
        description="Region code. For SerpAPI: 'us', 'id', 'jp', etc. For DuckDuckGo: 'wt-wt', 'us-en'.",
    )
    language: str = Field(
        "en",
        description="Language code for results (e.g. 'en', 'id', 'zh').",
    )


@tool(
    name="web_search",
    description=(
        "Search the web and return a ranked list of results (title, URL, and "
        "snippet). Uses Google via SerpAPI when available, falls back to DuckDuckGo. "
        "Use this to find pages, then 'web_scrape' to read one."
    ),
    args=WebSearchArgs,
    category="web",
    dangerous=False,
)
async def web_search(args: WebSearchArgs, ctx: ToolContext | None) -> ToolResult:
    """Run a web search.

    Returns
    -------
    ToolResult
        ``output`` is a list of ``{"title", "url", "snippet"}`` dicts.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ToolExecutionError("httpx is required for web_search.") from exc

    # Check for SerpAPI key
    serpapi_key = os.environ.get("SERPAPI_API_KEY", "").strip()

    client = ctx.http_client if ctx is not None else None
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    try:
        if serpapi_key:
            results = await _search_serpapi(client, args, serpapi_key)
            source = "google"
        else:
            results = await _search_duckduckgo(client, args)
            source = "duckduckgo"
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"Search request failed: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()

    return ToolResult.ok(
        results,
        result_count=len(results),
        query=args.query,
        source=source,
    )


async def _search_serpapi(
    client, args: WebSearchArgs, api_key: str
) -> list[dict]:
    """Query Google via SerpAPI for high-quality results."""
    from serpapi import GoogleSearch

    params = {
        "q": args.query,
        "api_key": api_key,
        "num": args.max_results,
        "gl": args.region if args.region != "wt-wt" else "us",
        "hl": args.language,
    }

    # Run in executor since SerpAPI client is synchronous
    import asyncio

    def _run_search():
        search = GoogleSearch(params)
        return search.get_dict()

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _run_search)

    results: list[dict] = []

    # Parse organic results
    for item in data.get("organic_results", [])[: args.max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })

    # If no organic results, try answer box
    if not results and "answer_box" in data:
        box = data["answer_box"]
        results.append({
            "title": box.get("title", "Answer Box"),
            "url": box.get("link", ""),
            "snippet": box.get("answer", box.get("snippet", "")),
        })

    # Add knowledge graph if available
    if "knowledge_graph" in data and len(results) < args.max_results:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title", ""),
            "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description", ""),
        })

    return results


async def _search_duckduckgo(client, args: WebSearchArgs) -> list[dict]:
    """Query the DuckDuckGo HTML endpoint and parse the result list (fallback)."""
    from bs4 import BeautifulSoup

    resp = await client.post(
        _DDG_HTML_ENDPOINT,
        data={"q": args.query, "kl": args.region},
        headers={"User-Agent": _USER_AGENT},
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict] = []
    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "")
        title = anchor.get_text(strip=True)
        if not href or not title:
            continue

        # DDG wraps targets in a redirect (.../l/?uddg=<encoded-url>): unwrap it.
        url = _unwrap_ddg_redirect(href)

        snippet_el = anchor.find_parent("div", class_="result__body")
        snippet = ""
        if snippet_el is not None:
            snip = snippet_el.select_one(".result__snippet")
            if snip is not None:
                snippet = snip.get_text(" ", strip=True)

        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= args.max_results:
            break

    return results


def _unwrap_ddg_redirect(href: str) -> str:
    """Extract the real destination from a DuckDuckGo redirect URL."""
    if "uddg=" not in href:
        return href
    parsed = urlparse(href if href.startswith("http") else f"https:{href}")
    target = parse_qs(parsed.query).get("uddg", [])
    return target[0] if target else href
