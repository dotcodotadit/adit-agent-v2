"""``browser`` tool — fetch and render a page with a headless browser (Playwright).

For JavaScript-heavy pages that the plain HTTP scraper can't handle, this drives
a headless Chromium via Playwright to navigate, optionally interact, and return
the page title, visible text, and (optionally) a screenshot saved to the cache
dir.

Playwright is an optional dependency (see the TODO in requirements.txt). If it
isn't installed — or its browser binaries aren't downloaded — the tool raises a
clear :class:`ToolNotConfiguredError` explaining how to enable it.

**Dangerous** — it performs outbound network requests and runs page scripts, so
it is gated behind confirmation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolNotConfiguredError,
    ToolResult,
)
from app.tools.registry import tool

_TEXT_LIMIT = 20_000


class BrowserArgs(BaseModel):
    """Arguments for :func:`browser`."""

    url: HttpUrl = Field(description="The URL to open.")
    wait_until: str = Field(
        "load",
        pattern="^(load|domcontentloaded|networkidle)$",
        description="Playwright load state to wait for before reading the page.",
    )
    timeout: int = Field(
        30, ge=1, le=120, description="Navigation timeout in seconds."
    )
    screenshot: bool = Field(
        False, description="If true, save a full-page screenshot to the cache dir."
    )


@tool(
    name="browser",
    description=(
        "Open a URL in a headless browser (Playwright), render JavaScript, and "
        "return the page title and visible text. Can capture a screenshot."
    ),
    args=BrowserArgs,
    category="system",
    dangerous=True,
)
async def browser(args: BrowserArgs, ctx: ToolContext | None) -> ToolResult:
    """Render a page and extract its text.

    Returns
    -------
    ToolResult
        ``output`` is a dict ``{"url", "title", "text", "screenshot"}`` where
        ``screenshot`` is a sandbox path or ``None``.
    """
    if ctx is None:
        raise ToolExecutionError("browser requires a ToolContext with settings.")

    if ctx.settings.require_tool_confirmation and not ctx.confirmed:
        raise ToolExecutionError(
            "browser is a dangerous tool and requires user confirmation "
            "(ctx.confirmed is False)."
        )

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ToolNotConfiguredError(
            "Playwright is not installed. Add `playwright` to requirements and "
            "run `playwright install chromium` to enable the browser tool."
        ) from exc

    timeout_ms = args.timeout * 1000
    screenshot_path: str | None = None

    try:
        async with async_playwright() as pw:
            try:
                browser_obj = await pw.chromium.launch(headless=True)
            except Exception as exc:  # noqa: BLE001 - missing browser binaries etc.
                raise ToolNotConfiguredError(
                    "Could not launch Chromium. Run `playwright install chromium`. "
                    f"Underlying error: {exc}"
                ) from exc

            try:
                page = await browser_obj.new_page()
                await page.goto(
                    str(args.url),
                    wait_until=args.wait_until,
                    timeout=timeout_ms,
                )
                title = await page.title()
                text = (await page.inner_text("body"))[:_TEXT_LIMIT]

                if args.screenshot:
                    shots = ctx.settings.cache_dir / "screenshots"
                    shots.mkdir(parents=True, exist_ok=True)
                    dest = shots / f"{abs(hash(str(args.url)))}.png"
                    await page.screenshot(path=str(dest), full_page=True)
                    screenshot_path = str(dest)
            finally:
                await browser_obj.close()
    except ToolNotConfiguredError:
        raise
    except Exception as exc:  # noqa: BLE001 - navigation/timeout/runtime failures
        raise ToolExecutionError(f"Browser navigation failed: {exc}") from exc

    return ToolResult.ok(
        {
            "url": str(args.url),
            "title": title,
            "text": text,
            "screenshot": screenshot_path,
        },
        chars=len(text),
    )
