"""Tool registry and the ``@tool`` decorator for Adit-Agent.

The registry is the single source of truth for which capabilities the agent
has. Tools register themselves at import time via the :func:`tool` decorator,
and :meth:`ToolRegistry.discover` imports every module under ``app/tools/`` so
adding a new tool is just dropping a file in the right folder — no central list
to edit.

Typical wiring (e.g. in :class:`app.dependencies.AppContainer`)::

    from app.tools.registry import get_registry
    registry = get_registry()
    registry.discover()                         # auto-import all tool modules
    schemas = registry.openai_schemas()         # advertise to the LLM
    result = await registry.get("read_file").invoke({"path": "notes.txt"}, ctx)

Defining a tool::

    from pydantic import BaseModel, Field
    from app.tools.registry import tool
    from app.tools.base import ToolResult, ToolContext

    class EchoArgs(BaseModel):
        text: str = Field(description="Text to echo back.")

    @tool(name="echo", description="Echo text.", args=EchoArgs, category="web")
    async def echo(args: EchoArgs, ctx: ToolContext | None) -> ToolResult:
        return ToolResult.ok(args.text)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from typing import Callable

from pydantic import BaseModel

from app.tools.base import EmptyArgs, Tool, ToolError, ToolFunc
from app.utils.logger import get_logger

__all__ = ["ToolRegistry", "tool", "get_registry"]

log = get_logger(__name__)

# Modules under app/tools that are infrastructure, not tools.
_NON_TOOL_MODULES = {"base", "registry", "__init__"}
_TOOLS_PACKAGE = "app.tools"


class ToolRegistry:
    """An in-memory collection of :class:`~app.tools.base.Tool` objects."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def register(self, tool_obj: Tool, *, override: bool = False) -> Tool:
        """Add ``tool_obj`` to the registry.

        Raises
        ------
        ToolError
            If a tool with the same name is already registered and ``override``
            is False.
        """
        existing = self._tools.get(tool_obj.name)
        if existing is not None and not override:
            raise ToolError(
                f"A tool named {tool_obj.name!r} is already registered "
                f"(category={existing.category!r}). Use override=True to replace it."
            )
        if existing is not None:
            log.warning("Overriding already-registered tool {!r}.", tool_obj.name)
        self._tools[tool_obj.name] = tool_obj
        log.debug(
            "Registered tool {!r} (category={}, dangerous={}).",
            tool_obj.name,
            tool_obj.category,
            tool_obj.dangerous,
        )
        return tool_obj

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if absent)."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Drop all registered tools (mainly for tests)."""
        self._tools.clear()

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def get(self, name: str) -> Tool:
        """Return the named tool or raise :class:`ToolError` if unknown."""
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "<none>"
            raise ToolError(
                f"Unknown tool {name!r}. Available: {available}."
            ) from None

    def try_get(self, name: str) -> Tool | None:
        """Return the named tool or ``None`` if not registered."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        """All registered tools, ordered by name."""
        return [self._tools[n] for n in self.names()]

    def by_category(self, category: str) -> list[Tool]:
        """Tools belonging to ``category``."""
        return [t for t in self.all() if t.category == category]

    def dangerous_tools(self) -> list[str]:
        """Names of tools flagged dangerous (need confirmation/sandbox)."""
        return [t.name for t in self.all() if t.dangerous]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.all())

    # ------------------------------------------------------------------ #
    # Schemas
    # ------------------------------------------------------------------ #
    def openai_schemas(self, *, include_dangerous: bool = True) -> list[dict]:
        """Return OpenAI ``tools`` definitions for the registered tools.

        Parameters
        ----------
        include_dangerous:
            When False, omit tools flagged ``dangerous`` (e.g. to present a
            restricted toolset to an unprivileged user).
        """
        return [
            t.openai_schema()
            for t in self.all()
            if include_dangerous or not t.dangerous
        ]

    # ------------------------------------------------------------------ #
    # Auto-discovery
    # ------------------------------------------------------------------ #
    def discover(self, package: str = _TOOLS_PACKAGE) -> int:
        """Import every tool module under ``package`` so they self-register.

        Walks the package tree, importing each non-infrastructure submodule.
        Import errors (e.g. a tool whose optional dependency is missing) are
        logged and skipped rather than aborting discovery — tools keep heavy
        imports lazy, so this should be rare.

        Returns
        -------
        int
            The total number of tools registered after discovery.
        """
        try:
            pkg = importlib.import_module(package)
        except ImportError as exc:
            log.error("Cannot import tools package {!r}: {}", package, exc)
            return len(self)

        if not hasattr(pkg, "__path__"):
            log.error("{!r} is not a package; nothing to discover.", package)
            return len(self)

        for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
            leaf = modinfo.name.rsplit(".", 1)[-1]
            if leaf in _NON_TOOL_MODULES or modinfo.ispkg:
                continue
            try:
                importlib.import_module(modinfo.name)
            except Exception as exc:  # noqa: BLE001 - one bad tool must not kill discovery
                log.warning("Skipping tool module {!r}: {}", modinfo.name, exc)

        log.info("Tool discovery complete: {} tool(s) registered.", len(self))
        return len(self)


# --------------------------------------------------------------------------- #
# Global registry + decorator
# --------------------------------------------------------------------------- #
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Return the process-wide :class:`ToolRegistry` singleton."""
    return _registry


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    args: type[BaseModel] | None = None,
    category: str = "general",
    dangerous: bool = False,
    registry: ToolRegistry | None = None,
) -> Callable[[ToolFunc], Tool]:
    """Decorator that wraps an async function into a :class:`Tool` and registers it.

    Parameters
    ----------
    name:
        Tool name exposed to the LLM. Defaults to the function name.
    description:
        Human/LLM-facing description. Defaults to the function's docstring.
    args:
        A pydantic model describing the tool's parameters (gives validation and
        the JSON schema). Defaults to :class:`~app.tools.base.EmptyArgs`.
    category:
        Logical grouping (``filesystem``/``media``/``system``/``web``/...).
    dangerous:
        Mark tools that mutate the system or run code; the orchestrator gates
        these behind the confirmation flow.
    registry:
        Target registry. Defaults to the global one from :func:`get_registry`.

    The decorated name is bound to the resulting :class:`Tool`, which is itself
    awaitable (``await mytool(arg=...)``).
    """

    def decorator(func: ToolFunc) -> Tool:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@tool requires an async function; {getattr(func, '__name__', func)!r} is not."
            )
        resolved_name = name or func.__name__
        resolved_desc = (description or inspect.getdoc(func) or "").strip()
        if not resolved_desc:
            raise ValueError(
                f"Tool {resolved_name!r} needs a description (pass description= "
                "or add a docstring)."
            )
        tool_obj = Tool(
            name=resolved_name,
            description=resolved_desc,
            func=func,
            args_model=args or EmptyArgs,
            category=category,
            dangerous=dangerous,
        )
        (registry or _registry).register(tool_obj, override=True)
        return tool_obj

    return decorator
