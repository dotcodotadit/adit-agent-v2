"""``read_file`` tool — read a UTF-8 text file from the sandbox.

Reads are confined to the configured sandbox root (``settings.sandbox_root``)
to prevent the agent from exfiltrating arbitrary host files. Binary files are
rejected with a clear message; large files are truncated to a caller-specified
byte budget.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
    resolve_in_sandbox,
)
from app.tools.registry import tool

# Default/maximum amount of a file we are willing to load into a prompt.
_DEFAULT_MAX_BYTES = 100_000
_HARD_MAX_BYTES = 2_000_000


class ReadFileArgs(BaseModel):
    """Arguments for :func:`read_file`."""

    path: str = Field(
        description="Path to the file, relative to the sandbox root.",
        min_length=1,
    )
    max_bytes: int = Field(
        _DEFAULT_MAX_BYTES,
        ge=1,
        le=_HARD_MAX_BYTES,
        description="Maximum number of bytes to read before truncating.",
    )
    encoding: str = Field(
        "utf-8",
        description="Text encoding used to decode the file.",
    )


@tool(
    name="read_file",
    description=(
        "Read the contents of a UTF-8 text file located inside the sandbox. "
        "Returns the decoded text (truncated to max_bytes) plus file metadata."
    ),
    args=ReadFileArgs,
    category="filesystem",
    dangerous=False,
)
async def read_file(args: ReadFileArgs, ctx: ToolContext | None) -> ToolResult:
    """Read a text file and return its contents.

    Parameters
    ----------
    args:
        Validated :class:`ReadFileArgs`.
    ctx:
        Execution context; ``ctx.settings.sandbox_root`` bounds access.

    Returns
    -------
    ToolResult
        ``output`` is the decoded text. ``metadata`` carries ``path``, ``size``
        (bytes on disk), ``bytes_read``, and ``truncated``.
    """
    if ctx is None:
        raise ToolExecutionError("read_file requires a ToolContext with settings.")

    target = resolve_in_sandbox(args.path, ctx.settings.sandbox_root)

    if not target.exists():
        raise ToolExecutionError(f"File not found: {args.path}")
    if not target.is_file():
        raise ToolExecutionError(f"Not a regular file: {args.path}")

    size = target.stat().st_size
    try:
        raw = target.read_bytes()[: args.max_bytes]
    except OSError as exc:
        raise ToolExecutionError(f"Could not read {args.path}: {exc}") from exc

    try:
        text = raw.decode(args.encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise ToolExecutionError(
            f"File {args.path} is not valid {args.encoding} text ({exc}). "
            "It may be binary; use a media tool instead."
        ) from exc

    return ToolResult.ok(
        text,
        path=str(target),
        size=size,
        bytes_read=len(raw),
        truncated=size > len(raw),
    )
