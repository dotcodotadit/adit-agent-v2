"""Developer-mode state for Adit-Agent.

Developer mode lets power users peek inside the agent: its planning, its
intermediate reasoning ("thoughts"), and the raw event stream. Flags are held
per user, in memory (they are a debugging convenience, not durable settings).

Three independent toggles, exposed via the ``/dev``, ``/thoughts`` and ``/raw``
commands:

* ``dev``      — master switch; shows plans, tool activity detail, timings.
* ``thoughts`` — surface the model's interim reasoning between tool calls.
* ``raw``      — dump the raw event stream (most verbose; implies ``dev``).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DevFlags", "DevModeStore"]


@dataclass(slots=True)
class DevFlags:
    """Per-user developer-mode toggles."""

    dev: bool = False
    thoughts: bool = False
    raw: bool = False

    def any_on(self) -> bool:
        return self.dev or self.thoughts or self.raw

    def summary(self) -> str:
        """Human-readable one-liner of the current flags."""
        def _s(b: bool) -> str:
            return "on" if b else "off"
        return f"dev={_s(self.dev)}, thoughts={_s(self.thoughts)}, raw={_s(self.raw)}"


class DevModeStore:
    """In-memory registry of per-user :class:`DevFlags`."""

    def __init__(self) -> None:
        self._users: dict[int, DevFlags] = {}

    def flags(self, user_id: int) -> DevFlags:
        """Return (creating if needed) the flags for ``user_id``."""
        return self._users.setdefault(user_id, DevFlags())

    def toggle(self, user_id: int, name: str) -> bool:
        """Flip one flag and return its new value.

        Enabling ``thoughts`` or ``raw`` implicitly enables ``dev``; disabling
        ``dev`` clears the dependent flags so the UI stays coherent.
        """
        flags = self.flags(user_id)
        new_value = not getattr(flags, name)
        setattr(flags, name, new_value)

        if name in ("thoughts", "raw") and new_value:
            flags.dev = True
        if name == "dev" and not new_value:
            flags.thoughts = False
            flags.raw = False
        return new_value
