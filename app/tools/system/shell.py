"""``shell`` tool — run a shell command inside the sandbox.

**Dangerous.** This executes arbitrary commands and is gated three ways:

1. **Confirmation** — when ``settings.require_tool_confirmation`` is on, the
   command only runs if the caller set ``ctx.confirmed = True`` (the
   orchestrator flips this after the user approves).
2. **Working directory** — commands run with ``cwd`` pinned to the sandbox
   root, never the host project tree.
3. **Denylist** — a few catastrophic patterns (fork bombs, ``rm -rf /``, disk
   writes) are refused outright.

NOTE: a denylist is a backstop, not real isolation. The actual security
boundary is the container the agent runs in plus the sandbox root; do not run
this tool with host-level privileges.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolExecutionError, ToolResult, resolve_in_sandbox
from app.tools.registry import tool

_OUTPUT_LIMIT = 20_000  # chars of stdout/stderr returned to the model

# Obviously-destructive patterns we refuse regardless of confirmation.
_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brm\s+-rf?\s+(/|~|\*)",   # recursive root/home/glob deletes
        r":\(\)\s*\{",                # bash fork bomb  :(){ :|:& };:
        r"\bmkfs\b",                  # format a filesystem
        r"\bdd\b.*\bof=/dev/",        # raw disk writes
        r">\s*/dev/sd",               # clobber a block device
        r"\b(shutdown|reboot|halt|poweroff)\b",
    )
)


class ShellArgs(BaseModel):
    """Arguments for :func:`shell`."""

    command: str = Field(description="The shell command to execute.", min_length=1)
    timeout: int = Field(
        30, ge=1, le=300, description="Max seconds to allow before killing."
    )
    workdir: str | None = Field(
        None,
        description="Optional sub-path of the sandbox to run in (default: root).",
    )


@tool(
    name="shell",
    description=(
        "Execute a shell command inside the sandbox and return its stdout, "
        "stderr, and exit code. Subject to confirmation and a safety denylist."
    ),
    args=ShellArgs,
    category="system",
    dangerous=True,
)
async def shell(args: ShellArgs, ctx: ToolContext | None) -> ToolResult:
    """Run a shell command with a timeout.

    Returns
    -------
    ToolResult
        ``output`` is a dict ``{"stdout", "stderr", "exit_code"}``. A non-zero
        exit code yields ``success=False`` but still returns the captured
        output for the model to inspect.
    """
    if ctx is None:
        raise ToolExecutionError("shell requires a ToolContext with settings.")

    if ctx.settings.require_tool_confirmation and not ctx.confirmed:
        raise ToolExecutionError(
            "shell is a dangerous tool and requires user confirmation "
            "(ctx.confirmed is False)."
        )

    for pat in _DENY_PATTERNS:
        if pat.search(args.command):
            raise ToolExecutionError(
                f"Refused: command matches a blocked dangerous pattern "
                f"({pat.pattern!r})."
            )

    # Pin the working directory inside the sandbox.
    cwd = (
        resolve_in_sandbox(args.workdir, ctx.settings.sandbox_root)
        if args.workdir
        else ctx.settings.sandbox_root.resolve()
    )
    cwd.mkdir(parents=True, exist_ok=True)

    try:
        proc = await asyncio.create_subprocess_shell(
            args.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
    except OSError as exc:
        raise ToolExecutionError(f"Failed to launch command: {exc}") from exc

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=args.timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolExecutionError(
            f"Command timed out after {args.timeout}s and was killed."
        ) from None

    stdout = stdout_b.decode("utf-8", "replace")[:_OUTPUT_LIMIT]
    stderr = stderr_b.decode("utf-8", "replace")[:_OUTPUT_LIMIT]
    payload = {"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode}

    if proc.returncode != 0:
        return ToolResult(
            success=False,
            output=payload,
            error=f"Command exited with code {proc.returncode}.",
            metadata={"exit_code": proc.returncode},
        )
    return ToolResult.ok(payload, exit_code=0)
