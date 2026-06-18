"""Tool system for Adit-Agent.

Exposes the core abstractions (``Tool``, ``ToolResult``, ``ToolContext``) and
the ``@tool`` decorator / ``ToolRegistry`` so callers can both define new tools
and query the registry::

    from app.tools import tool, get_registry, ToolResult, ToolContext

    @tool(name="ping", description="Return pong.", category="web")
    async def ping(args, ctx):
        return ToolResult.ok("pong")

    registry = get_registry()
    registry.discover()   # auto-imports all modules under app/tools/
    schemas = registry.openai_schemas()

Concrete tools live in sub-packages:

* ``app/tools/filesystem/`` — read_file, write_file, search_files
* ``app/tools/web/``        — web_search, web_scrape
* ``app/tools/system/``     — shell, process, browser
* ``app/tools/media/``      — image_reader, audio_reader, video_reader
"""

from __future__ import annotations

from app.tools.base import (
    EmptyArgs,
    Tool,
    ToolContext,
    ToolError,
    ToolExecutionError,
    ToolNotConfiguredError,
    ToolResult,
    ToolValidationError,
    resolve_in_sandbox,
)
from app.tools.registry import ToolRegistry, get_registry, tool

__all__ = [
    # Core abstractions
    "Tool",
    "ToolResult",
    "ToolContext",
    "EmptyArgs",
    "resolve_in_sandbox",
    # Errors
    "ToolError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolNotConfiguredError",
    # Registry
    "ToolRegistry",
    "get_registry",
    "tool",
]
