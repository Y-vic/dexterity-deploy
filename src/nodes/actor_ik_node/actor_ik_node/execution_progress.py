"""Actor command publication and ACK diagnostics for the current action slice."""

from __future__ import annotations

from typing import Any

from sharpa_interface.server.executed_steps import make_progress


def action_identity(action: Any) -> tuple[str, int, str, int]:
    return (
        str(action.session_id), int(action.request_id),
        str(action.action_id), int(action.revision),
    )


class ActorExecutionProgress:
    """Observe scheduling without gating it or treating ACKs as physical motion.

    The snapshot describes the latest activated action, including its terminal
    state. Late results for it can update diagnostics until another action is
    activated. Results for older identities never alter the new action.
    """

    def __init__(self) -> None:
        self._action = None
        self._provider = "policy"
        self._scope = "full"
        self._dry_run = False
        self._required_devices: frozenset[str] = frozenset()
        self._active = False
        self._success: bool | None = None
        self._error = ""
        self._dispatched_steps = 0
        self._acknowledged_steps = 0
        self._published_at: dict[int, int] = {}
        self._results: dict[int, dict[str, bool]] = {}

    def start(
        self, action: Any, *, provider: str, execution_scope: str,
        dry_run: bool, required_devices: frozenset[str],
    ) -> None:
        self._action = action
        self._provider = str(provider)
        self._scope = str(execution_scope)
        self._dry_run = bool(dry_run)
        self._required_devices = frozenset(required_devices)
        self._active = True
        self._success = None
        self._error = ""
        self._dispatched_steps = 0
        self._acknowledged_steps = 0
        self._published_at.clear()
        self._results.clear()

    def _matches(self, action: Any) -> bool:
        return self._action is not None and action_identity(action) == action_identity(self._action)

    def dispatched(self, action: Any, step: int, command_timestamp_ns: int) -> bool:
        step = int(step)
        if (
            not self._active or not self._matches(action)
            or step != self._dispatched_steps or step >= int(action.execute_length)
        ):
            return False
        self._published_at[step] = int(command_timestamp_ns)
        self._dispatched_steps += 1
        return True

    def acknowledged(
        self, action: Any, step: int, *, device: str, source: int,
        dry_run: bool, success: bool, timestamp_ns: int,
    ) -> bool:
        step = int(step)
        if (
            not self._matches(action) or int(source) != 3
            or str(device) not in self._required_devices
            or bool(dry_run) != self._dry_run
            or step not in self._published_at
            or int(timestamp_ns) < self._published_at[step]
        ):
            return False
        results = self._results.setdefault(step, {})
        if str(device) in results:
            return False
        results[str(device)] = bool(success)
        previous = self._acknowledged_steps
        while self._acknowledged_steps < self._dispatched_steps:
            step_results = self._results.get(self._acknowledged_steps, {})
            if not all(step_results.get(device, False) for device in self._required_devices):
                break
            self._acknowledged_steps += 1
        return self._acknowledged_steps != previous

    def finish(self, action: Any, *, success: bool, error: str = "") -> bool:
        if not self._matches(action):
            return False
        self._active = False
        self._success = bool(success)
        self._error = str(error)
        return True

    def snapshot(self, *, paused: bool = False) -> dict[str, Any]:
        action = self._action
        return {
            "active": self._active,
            "paused": bool(paused and self._active),
            "provider": self._provider,
            "session_id": str(action.session_id) if action is not None else "",
            "request_id": int(action.request_id) if action is not None else None,
            "action_id": str(action.action_id) if action is not None else "",
            "revision": int(action.revision) if action is not None else None,
            "action_length": int(action.action_length) if action is not None else 0,
            "execute_start": int(action.execute_start) if action is not None else 0,
            "execute_length": int(action.execute_length) if action is not None else 0,
            "dispatched_steps": self._dispatched_steps,
            "acknowledged_steps": self._acknowledged_steps,
            "executed_steps": (
                make_progress(int(action.execute_start), self._dispatched_steps,
                              int(action.revision)).executed_steps
                if action is not None else 0
            ),
            "execution_scope": self._scope,
            "dry_run": self._dry_run,
            "success": self._success,
            "error": self._error,
        }
