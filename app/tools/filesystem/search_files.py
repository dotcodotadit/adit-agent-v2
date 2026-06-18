"""``search_files`` tool — find files by name and/or content within the sandbox.

Supports two complementary modes that can be combined:

* **name search** — glob the sandbox tree (e.g. ``**/*.py``).
* **content search** — match a regular expression inside the matched files,
  returning the matching lines with line numbers.

Both are confined to the sandbox root and bounded by result/​file-size limits so
a broad query can't blow up memory or the prompt.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
    resolve_in_sandbox,
)
from app.tools.registry import tool

# Skip files larger than this when scanning content (likely binaries/blobs).
_MAX_SCAN_BYTES = 1_000_000


class SearchFilesArgs(BaseModel):
    """Arguments for :func:`search_files`."""

    glob: str = Field(
        "**/*",
        description="Glob pattern (relative to sandbox root) selecting files.",
    )
    contains: str | None = Field(
        None,
        description=(
            "Optional regular expression; only files containing a match are "
            "returned, along with the matching lines."
        ),
    )
    max_results: int = Field(
        50, ge=1, le=500, description="Maximum number of files to return."
    )
    max_matches_per_file: int = Field(
        20, ge=1, le=200, description="Max matching lines reported per file."
    )
    case_sensitive: bool = Field(
        False, description="Whether the content regex is case-sensitive."
    )


@tool(
    name="search_files",
    description=(
        "Search the sandbox for files by glob pattern and optionally by a "
        "regular expression on their contents. Returns matching file paths and "
        "(when searching content) the matching lines with line numbers."
    ),
    args=SearchFilesArgs,
    category="filesystem",
    dangerous=False,
)
async def search_files(args: SearchFilesArgs, ctx: ToolContext | None) -> ToolResult:
    """Search files by name and/or content.

    Returns
    -------
    ToolResult
        ``output`` is a list of ``{"path", "matches"}`` dicts (``matches`` is a
        list of ``{"line", "text"}`` when content searching, else empty).
        ``metadata`` carries ``files_scanned`` and ``truncated``.
    """
    if ctx is None:
        raise ToolExecutionError("search_files requires a ToolContext with settings.")

    root = ctx.settings.sandbox_root.resolve()
    if not root.exists():
        raise ToolExecutionError(f"Sandbox root does not exist: {root}")

    pattern: re.Pattern[str] | None = None
    if args.contains:
        flags = 0 if args.case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(args.contains, flags)
        except re.error as exc:
            raise ToolExecutionError(f"Invalid 'contains' regex: {exc}") from exc

    results: list[dict] = []
    files_scanned = 0
    truncated = False

    for candidate in sorted(root.glob(args.glob)):
        if not candidate.is_file():
            continue
        # Defense in depth: ensure glob didn't follow a symlink out of bounds.
        try:
            resolve_in_sandbox(candidate, root)
        except ToolExecutionError:
            continue

        files_scanned += 1

        if pattern is None:
            results.append({"path": _rel(candidate, root), "matches": []})
        else:
            matches = _scan_content(candidate, pattern, args.max_matches_per_file)
            if matches:
                results.append({"path": _rel(candidate, root), "matches": matches})

        if len(results) >= args.max_results:
            truncated = True
            break

    return ToolResult.ok(
        results,
        files_scanned=files_scanned,
        result_count=len(results),
        truncated=truncated,
    )


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` using forward slashes."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _scan_content(
    path: Path, pattern: re.Pattern[str], limit: int
) -> list[dict]:
    """Return up to ``limit`` matching lines from ``path`` (best-effort)."""
    try:
        if path.stat().st_size > _MAX_SCAN_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    found: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            found.append({"line": lineno, "text": line.strip()[:500]})
            if len(found) >= limit:
                break
    return found
