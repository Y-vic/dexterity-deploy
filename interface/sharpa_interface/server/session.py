"""Session + revision state machine.

The v3 server issues a session_id on /reset and expects every /infer to
reference it. Revisions increment whenever the client decides to invalidate
the current action plan (e.g. safety event, replan on user input).

Both embodiments must produce revision numbers with the same semantics so
that the server's scheduler behaves identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Session:
    session_id: str
    revision: int = 0
    execute_start: int = 0

    def bump_revision(self, new_execute_start: Optional[int] = None) -> int:
        self.revision += 1
        if new_execute_start is not None:
            self.execute_start = new_execute_start
        return self.revision


@dataclass
class SessionRegistry:
    """One active session at a time per embodiment client."""

    active: Optional[Session] = field(default=None)

    def attach(self, session_id: str) -> Session:
        self.active = Session(session_id=session_id, revision=0, execute_start=0)
        return self.active

    def detach(self) -> None:
        self.active = None

    def require(self) -> Session:
        if self.active is None:
            raise RuntimeError("no active session; call /reset first")
        return self.active
