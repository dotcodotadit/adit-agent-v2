"""``process`` tool — manage long-running background processes.

Unlike :mod:`shell` (which runs a command to completion), this tool starts and
supervises **background** processes — a dev server, a watcher, a training job —
and lets the agent list, inspect, and terminate them across turns.

Only processes started *through this tool* are managed; it deliberately does
not enumerate or kill arbitrary host processes (that would need elevated,
unsandboxed access). Processes inherit the sandbox root as their working
directory and are tracked in a module-level registry for the lifetime of the
app process.

**Dangerous** — starting/terminating processes is gated behind confirmation.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.tools.base import (
    ToolContext,
    ToolExecutionError,
    ToolResult,
    resolve_in_sandbox,
)
from app.tools.registry import tool


@dataclass
class _ManagedProcess:
    """Bookkeeping for one supervised background process."""

    pid: int
    command: str
    proc: asyncio.subprocess.Process
    cwd: str
    _stdout: bytearray = field(default_factory=bytearray)


# Tracks processes started via this tool, keyed by PID. Module-level so it
# survives across tool invocations within the same app process.
_PROCESSES: dict[int, _ManagedProcess] = {}


class ProcessArgs(BaseModel):
    """Arguments for :func:`process`."""

    action: str = Field(
        description="One of: start, list, status, terminate.",
        pattern="^(start|list|status|terminate)$",
    )
    command: str | None = Field(
        None, description="Command line to run (required for action=start)."
    )
    pid: int | None = Field(
        None, description="Target PID (required for status/terminate)."
    )
    timeout: float = Field(
        5.0, ge=0.5, le=60.0,
        description="Seconds to wait for graceful termination before killing.",
    )


@tool(
    name="process",
    description=(
        "Manage background processes started by the agent: start a long-running "
        "command, list managed processes, check status, or terminate one."
    ),
    args=ProcessArgs,
    category="system",
    dangerous=True,
)
async def process(args: ProcessArgs, ctx: ToolContext | None) -> ToolResult:
    """Dispatch a process-management action.

    Returns
    -------
    ToolResult
        Shape depends on the action (see each handler below).
    """
    if ctx is None:
        raise ToolExecutionError("process requires a ToolContext with settings.")

    mutating = args.action in {"start", "terminate"}
    if mutating and ctx.settings.require_tool_confirmation and not ctx.confirmed:
        raise ToolExecutionError(
            f"process action {args.action!r} requires user confirmation "
            "(ctx.confirmed is False)."
        )

    if args.action == "start":
        return await _start(args, ctx)
    if args.action == "list":
        return _list()
    if args.action == "status":
        return _status(args)
    if args.action == "terminate":
        return await _terminate(args)
    # Unreachable: pattern-validated.
    raise ToolExecutionError(f"Unknown action: {args.action!r}")


async def _start(args: ProcessArgs, ctx: ToolContext) -> ToolResult:
    if not args.command:
        raise ToolExecutionError("action=start requires 'command'.")

    cwd = ctx.settings.sandbox_root.resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    # On POSIX use an argv split; on Windows pass through the shell line.
    try:
        if sys.platform == "win32":
            proc = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd),
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(args.command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd),
            )
    except (OSError, ValueError) as exc:
        raise ToolExecutionError(f"Failed to start process: {exc}") from exc

    managed = _ManagedProcess(
        pid=proc.pid, command=args.command, proc=proc, cwd=str(cwd)
    )
    _PROCESSES[proc.pid] = managed
    return ToolResult.ok(
        {"pid": proc.pid, "command": args.command, "cwd": str(cwd)},
        running=True,
    )


def _list() -> ToolResult:
    items = [
        {"pid": p.pid, "command": p.command, "running": p.proc.returncode is None}
        for p in _PROCESSES.values()
    ]
    return ToolResult.ok(items, count=len(items))


def _status(args: ProcessArgs) -> ToolResult:
    if args.pid is None:
        raise ToolExecutionError("action=status requires 'pid'.")
    managed = _PROCESSES.get(args.pid)
    if managed is None:
        raise ToolExecutionError(f"No managed process with PID {args.pid}.")
    rc = managed.proc.returncode
    return ToolResult.ok(
        {
            "pid": managed.pid,
            "command": managed.command,
            "running": rc is None,
            "exit_code": rc,
        }
    )


async def _terminate(args: ProcessArgs) -> ToolResult:
    if args.pid is None:
        raise ToolExecutionError("action=terminate requires 'pid'.")
    managed = _PROCESSES.get(args.pid)
    if managed is None:
        raise ToolExecutionError(f"No managed process with PID {args.pid}.")

    proc = managed.proc
    if proc.returncode is not None:
        _PROCESSES.pop(args.pid, None)
        return ToolResult.ok(
            {"pid": args.pid, "already_exited": True, "exit_code": proc.returncode}
        )

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=args.timeout)
        killed = False
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        killed = True

    _PROCESSES.pop(args.pid, None)
    return ToolResult.ok(
        {"pid": args.pid, "exit_code": proc.returncode, "force_killed": killed}
    )
