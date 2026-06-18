"""``web_search`` tool — keyless web search via the DuckDuckGo HTML endpoint.

Returns a ranked list of ``{title, url, snippet}`` results. It uses DuckDuckGo's
no-API-key HTML endpoint so it works out of the box; swapping in a paid search
API later only means changing :func:`_search_duckduckgo`.

The shared ``ctx.http_client`` (an ``httpx.AsyncClient``) is reused when
present; otherwise a short-lived client is created and closed per call.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
)
from app.tools.registry import tool

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
        "wt-wt", description="DuckDuckGo region code (e.g. 'us-en', 'wt-wt')."
    )


@tool(
    name="web_search",
    description=(
        "Search the web and return a ranked list of results (title, URL, and "
        "snippet). Use this to find pages, then 'web_scrape' to read one."
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

    client = ctx.http_client if ctx is not None else None
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    try:
        results = await _search_duckduckgo(client, args)
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"Search request failed: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()

    return ToolResult.ok(results, result_count=len(results), query=args.query)


async def _search_duckduckgo(client, args: WebSearchArgs) -> list[dict]:
    """Query the DuckDuckGo HTML endpoint and parse the result list."""
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

        # DDG wraps targets in a redirect (…/l/?uddg=<encoded-url>): unwrap it.
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
