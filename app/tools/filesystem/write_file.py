"""``write_file`` tool — write or append text to a file in the sandbox.

This is a **dangerous** tool: it mutates the filesystem and is therefore gated
behind the orchestrator's confirmation flow. Writes are confined to the sandbox
root, parent directories are created on demand, and an existing-file overwrite
must be explicit.
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

_HARD_MAX_BYTES = 5_000_000


class WriteFileArgs(BaseModel):
    """Arguments for :func:`write_file`."""

    path: str = Field(
        description="Destination path, relative to the sandbox root.",
        min_length=1,
    )
    content: str = Field(description="Text content to write.")
    mode: str = Field(
        "overwrite",
        pattern="^(overwrite|append|create)$",
        description=(
            "overwrite = replace if exists; append = add to end; "
            "create = fail if the file already exists."
        ),
    )
    encoding: str = Field("utf-8", description="Text encoding for the output.")


@tool(
    name="write_file",
    description=(
        "Write text to a file inside the sandbox. Supports overwrite, append, "
        "and create-only modes. Creates parent directories as needed."
    ),
    args=WriteFileArgs,
    category="filesystem",
    dangerous=True,
)
async def write_file(args: WriteFileArgs, ctx: ToolContext | None) -> ToolResult:
    """Write ``content`` to ``path``.

    Returns
    -------
    ToolResult
        ``output`` is a short confirmation string; ``metadata`` carries
        ``path``, ``bytes_written``, and ``mode``.
    """
    if ctx is None:
        raise ToolExecutionError("write_file requires a ToolContext with settings.")

    encoded = args.content.encode(args.encoding, errors="strict")
    if len(encoded) > _HARD_MAX_BYTES:
        raise ToolExecutionError(
            f"Refusing to write {len(encoded)} bytes (limit {_HARD_MAX_BYTES})."
        )

    target = resolve_in_sandbox(args.path, ctx.settings.sandbox_root)

    if args.mode == "create" and target.exists():
        raise ToolExecutionError(
            f"File already exists (mode=create): {args.path}"
        )
    if target.exists() and target.is_dir():
        raise ToolExecutionError(f"Path is a directory, not a file: {args.path}")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "ab" if args.mode == "append" else "wb"
        with target.open(file_mode) as fh:
            fh.write(encoded)
    except OSError as exc:
        raise ToolExecutionError(f"Could not write {args.path}: {exc}") from exc

    return ToolResult.ok(
        f"Wrote {len(encoded)} bytes to {args.path} (mode={args.mode}).",
        path=str(target),
        bytes_written=len(encoded),
        mode=args.mode,
    )
