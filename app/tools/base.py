"""Core abstractions for the Adit-Agent tool system.

A *tool* is an async, schema-described capability the agent can invoke (read a
file, search the web, run a shell command, ...). Tools are deliberately small
and uniform so the orchestrator can:

* advertise them to the LLM as OpenAI-compatible *function* definitions
  (:meth:`Tool.openai_schema`),
* validate the model's arguments against a typed schema before running
  anything (pydantic ``args_model``),
* execute them safely, with every failure mode collapsed into a structured
  :class:`ToolResult` rather than a raised exception.

Shared, long-lived resources (settings, the provider router, an HTTP client,
the vector store) are passed in at call time via :class:`ToolContext` instead
of being imported as globals, which keeps tools pure and testable.

This module defines the primitives; concrete tools live under
``app/tools/<category>/`` and register themselves with the
:class:`~app.tools.registry.ToolRegistry` via the ``@tool`` decorator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.config import Settings

log = get_logger(__name__)

__all__ = [
    "ToolError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolNotConfiguredError",
    "ToolResult",
    "ToolContext",
    "Tool",
    "EmptyArgs",
    "resolve_in_sandbox",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ToolError(Exception):
    """Base class for all tool-related errors.

    Tools should raise :class:`ToolError` (or a subclass) for *expected*
    failures; :meth:`Tool.invoke` converts these into a failed
    :class:`ToolResult` with a clean message instead of propagating.
    """


class ToolValidationError(ToolError):
    """Raised when arguments fail schema validation."""


class ToolExecutionError(ToolError):
    """Raised when a tool fails while doing its work."""


class ToolNotConfiguredError(ToolError):
    """Raised when a required dependency, binary, or credential is missing.

    Used by tools that depend on subsystems which may not be wired yet (a
    vision provider, Playwright, ffmpeg, ...). Distinct from
    :class:`ToolExecutionError` so callers can tell "can't run here" apart from
    "ran and failed".
    """


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ToolResult:
    """Structured outcome of a tool invocation.

    Attributes
    ----------
    success:
        Whether the tool completed its work.
    output:
        The successful payload (any JSON-serializable value). ``None`` on
        failure.
    error:
        Human-readable error message. ``None`` on success.
    metadata:
        Side-channel info (timings, counts, error_type, ...) that is useful for
        logging/UX but not part of the primary output.
    """

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any = None, **metadata: Any) -> "ToolResult":
        """Build a successful result."""
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str | Exception, **metadata: Any) -> "ToolResult":
        """Build a failed result from a message or exception."""
        return cls(success=False, error=str(error), metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for transport back to the model / persistence."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Execution context
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    """Shared resources handed to a tool at invocation time.

    Most fields are optional and typed loosely (``Any``) because the concrete
    subsystems live in sibling packages that may not yet be implemented. Tools
    that need one should fetch it via :meth:`require` to get a clear
    :class:`ToolNotConfiguredError` when it is missing.
    """

    settings: "Settings"
    provider_router: Any = None          # LLM router (text + vision)
    vector_store: Any = None             # ChromaDB client
    http_client: Any = None              # shared httpx.AsyncClient
    user_id: int | None = None           # acting user, for auditing/limits
    confirmed: bool = False              # user approved a dangerous action
    extra: dict[str, Any] = field(default_factory=dict)

    def require(self, attr: str, *, tool: str | None = None) -> Any:
        """Return ``self.<attr>`` or raise if it is missing/None."""
        value = getattr(self, attr, None)
        if value is None:
            where = f" required by tool {tool!r}" if tool else ""
            raise ToolNotConfiguredError(
                f"ToolContext.{attr} is not configured{where}."
            )
        return value


# --------------------------------------------------------------------------- #
# Tool wrapper
# --------------------------------------------------------------------------- #
# Every tool implementation has the uniform signature ``(args, ctx) -> result``.
ToolFunc = Callable[[BaseModel, "ToolContext | None"], Awaitable[Any]]


class EmptyArgs(BaseModel):
    """Argument model for tools that take no parameters."""


def _strip_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop pydantic-generated ``title`` keys for a cleaner schema."""
    if isinstance(schema, dict):
        return {
            k: _strip_titles(v)
            for k, v in schema.items()
            if k != "title"
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]  # type: ignore[return-value]
    return schema


@dataclass
class Tool:
    """A registered, callable capability.

    Wraps a plain async function together with its metadata and a pydantic
    argument model. Use :meth:`openai_schema` to advertise it to the LLM and
    :meth:`invoke` to run it with validation + error handling.
    """

    name: str
    description: str
    func: ToolFunc
    args_model: type[BaseModel] = EmptyArgs
    category: str = "general"
    dangerous: bool = False

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def parameters_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this tool's parameters."""
        return _strip_titles(self.args_model.model_json_schema())

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI ``tools`` function definition for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def invoke(
        self,
        arguments: dict[str, Any] | None = None,
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Validate ``arguments`` and run the tool, never raising.

        All outcomes are returned as a :class:`ToolResult`:

        * invalid arguments → ``success=False`` with a validation message,
        * a raised :class:`ToolError` → ``success=False`` with its message,
        * any other exception → ``success=False`` with a generic message
          (full traceback logged),
        * a returned :class:`ToolResult` is passed through; any other return
          value is wrapped via :meth:`ToolResult.ok`.
        """
        # 1) Validate arguments against the typed schema.
        try:
            args = self.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            log.warning("Tool {} called with invalid arguments: {}", self.name, exc)
            return ToolResult.fail(
                f"Invalid arguments for tool {self.name!r}: {exc}",
                error_type="ToolValidationError",
            )

        # 2) Run, funneling every failure into a structured result.
        try:
            result = await self.func(args, context)
        except ToolError as exc:
            log.warning("Tool {} failed: {}", self.name, exc)
            return ToolResult.fail(str(exc), error_type=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            log.exception("Unhandled error in tool {}", self.name)
            return ToolResult.fail(
                f"Unexpected error in tool {self.name!r}: {exc}",
                error_type="UnexpectedError",
            )

        if isinstance(result, ToolResult):
            return result
        return ToolResult.ok(result)

    async def __call__(
        self, *, context: ToolContext | None = None, **arguments: Any
    ) -> ToolResult:
        """Convenience caller: ``await tool(path="x", context=ctx)``."""
        return await self.invoke(arguments, context=context)


# --------------------------------------------------------------------------- #
# Sandbox helper (shared by filesystem / shell tools)
# --------------------------------------------------------------------------- #
def resolve_in_sandbox(path: str | Path, root: Path) -> Path:
    """Resolve ``path`` and ensure it stays within ``root``.

    Relative paths are taken relative to ``root``; absolute paths are still
    checked against it. Prevents directory-traversal escapes (``../../etc``).

    Raises
    ------
    ToolExecutionError
        If the resolved path falls outside ``root``.
    """
    root = root.resolve()
    p = Path(path)
    candidate = (p if p.is_absolute() else root / p).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolExecutionError(
            f"Path {str(path)!r} resolves outside the sandbox root ({root})."
        )
    return candidate
