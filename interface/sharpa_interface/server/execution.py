"""Execution feedback and the synchronous policy-loop gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .action import ParsedPolicyAction


EXECUTION_DONE_SCHEMA = "sharpa_execution_done.v1"


class ExecutionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionDone:
    request_id: int
    action_id: str
    revision: int
    execute_start: int
    execute_length: int
    executed_steps: int
    success: bool
    done: bool
    error: str | None


class SyncPhase(str, Enum):
    READY = "ready"
    INFERENCE = "inference"
    EXECUTING = "executing"
    FAILED = "failed"
    CLOSED = "closed"


def initial_execution_feedback() -> dict[str, Any]:
    return {"last_action_id": None, "executed_steps": 0, "success": True}


def validate_execution_feedback(value: Any) -> dict[str, Any]:
    feedback = _required(value, "execution_feedback", ("last_action_id", "executed_steps", "success"))
    action_id = feedback["last_action_id"]
    if action_id is not None and (not isinstance(action_id, str) or not action_id.strip()):
        raise ExecutionValidationError("execution_feedback.last_action_id must be nonempty or None")
    executed_steps = _integer(feedback["executed_steps"], "execution_feedback.executed_steps")
    success = _boolean(feedback["success"], "execution_feedback.success")
    return {**dict(feedback), "last_action_id": action_id, "executed_steps": executed_steps, "success": success}


def validate_execution_done(value: Any) -> ExecutionDone:
    payload = _required(
        value,
        "execution_done",
        (
            "schema",
            "request_id",
            "action_id",
            "revision",
            "execute_start",
            "execute_length",
            "executed_steps",
            "success",
            "done",
            "error",
        ),
    )
    if payload["schema"] != EXECUTION_DONE_SCHEMA:
        raise ExecutionValidationError(f"execution_done.schema must be {EXECUTION_DONE_SCHEMA}")
    action_id = payload["action_id"]
    if not isinstance(action_id, str) or not action_id.strip():
        raise ExecutionValidationError("execution_done.action_id must be nonempty")
    error = payload["error"]
    if error is not None and not isinstance(error, str):
        raise ExecutionValidationError("execution_done.error must be a string or None")
    return ExecutionDone(
        request_id=_integer(payload["request_id"], "execution_done.request_id"),
        action_id=action_id,
        revision=_integer(payload["revision"], "execution_done.revision"),
        execute_start=_integer(payload["execute_start"], "execution_done.execute_start"),
        execute_length=_integer(payload["execute_length"], "execution_done.execute_length", minimum=1),
        executed_steps=_integer(payload["executed_steps"], "execution_done.executed_steps"),
        success=_boolean(payload["success"], "execution_done.success"),
        done=_boolean(payload["done"], "execution_done.done"),
        error=error,
    )


def build_execution_done(
    action: ParsedPolicyAction | Mapping[str, Any],
    *,
    executed_in_slice: int,
    success: bool,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the event published once by ``action_execute``."""

    if isinstance(action, ParsedPolicyAction):
        request_id = action.request_id
        action_id = action.action_id
        revision = action.revision
        execute_start = action.execution.execute_start
        execute_length = action.execution.execute_length
    else:
        command = _required(
            action,
            "executable_action",
            ("request_id", "action_id", "revision", "execution"),
        )
        execution = _required(
            command["execution"],
            "executable_action.execution",
            ("execute_start", "execute_length"),
        )
        request_id = _integer(command["request_id"], "executable_action.request_id")
        action_id = command["action_id"]
        if not isinstance(action_id, str) or not action_id.strip():
            raise ExecutionValidationError("executable_action.action_id must be nonempty")
        revision = _integer(command["revision"], "executable_action.revision")
        execute_start = _integer(
            execution["execute_start"], "executable_action.execution.execute_start"
        )
        execute_length = _integer(
            execution["execute_length"],
            "executable_action.execution.execute_length",
            minimum=1,
        )
    if executed_in_slice < 0 or executed_in_slice > execute_length:
        raise ValueError("executed_in_slice is outside the active execution slice")
    return {
        "schema": EXECUTION_DONE_SCHEMA,
        "request_id": request_id,
        "action_id": action_id,
        "revision": revision,
        "execute_start": execute_start,
        "execute_length": execute_length,
        "executed_steps": execute_start + executed_in_slice,
        "success": bool(success),
        "done": True,
        "error": error,
    }


class SyncExecutionGate:
    """Allow one inference and one action execution at a time."""

    def __init__(self, execution_mode: str = "synchronous") -> None:
        mode = execution_mode.strip().lower()
        if mode == "async":
            raise NotImplementedError("async execution mode is not supported")
        if mode not in {"sync", "synchronous"}:
            raise ValueError("execution_mode must be synchronous; async is unsupported")
        self.phase = SyncPhase.READY
        self.next_request_id = 0
        self.active_request_id: int | None = None
        self.active_action: ParsedPolicyAction | None = None
        self.feedback = initial_execution_feedback()

    def begin_inference(self) -> tuple[int, dict[str, Any]]:
        if self.phase is not SyncPhase.READY:
            raise RuntimeError(f"cannot begin inference while phase={self.phase.value}")
        request_id = self.next_request_id
        self.next_request_id += 1
        self.active_request_id = request_id
        self.phase = SyncPhase.INFERENCE
        return request_id, dict(self.feedback)

    def accept_action(self, action: ParsedPolicyAction) -> dict[str, Any]:
        if self.phase is not SyncPhase.INFERENCE:
            raise RuntimeError(f"cannot accept action while phase={self.phase.value}")
        if action.request_id != self.active_request_id:
            raise ExecutionValidationError("action request_id does not match active inference")
        self.active_action = action
        self.phase = SyncPhase.EXECUTING
        return action.execution_command()

    def cancel_inference(self) -> None:
        """Release an inference slot after transport or validation failure."""

        if self.phase is not SyncPhase.INFERENCE:
            raise RuntimeError(f"cannot cancel inference while phase={self.phase.value}")
        self.active_request_id = None
        self.phase = SyncPhase.READY

    def complete(self, value: Any) -> dict[str, Any]:
        if self.phase is not SyncPhase.EXECUTING or self.active_action is None:
            raise RuntimeError(f"cannot complete execution while phase={self.phase.value}")
        done = validate_execution_done(value)
        action = self.active_action
        expected_steps = action.execution.execute_stop
        identity = (
            done.request_id == action.request_id
            and done.action_id == action.action_id
            and done.revision == action.revision
            and done.execute_start == action.execution.execute_start
            and done.execute_length == action.execution.execute_length
        )
        if not identity:
            raise ExecutionValidationError("execution_done does not match the active action")
        if not done.done:
            raise ExecutionValidationError("execution_done.done must be true")
        if done.executed_steps > expected_steps:
            raise ExecutionValidationError("execution_done.executed_steps exceeds the active slice")
        if done.success and done.executed_steps != expected_steps:
            raise ExecutionValidationError("successful execution must complete the full active slice")

        self.feedback = {
            "last_action_id": done.action_id,
            "executed_steps": done.executed_steps,
            "success": done.success,
        }
        self.active_request_id = None
        self.active_action = None
        self.phase = SyncPhase.READY if done.success else SyncPhase.FAILED
        return dict(self.feedback)

    def reset(self) -> None:
        self.phase = SyncPhase.READY
        self.next_request_id = 0
        self.active_request_id = None
        self.active_action = None
        self.feedback = initial_execution_feedback()

    def close(self) -> None:
        self.phase = SyncPhase.CLOSED
        self.active_request_id = None
        self.active_action = None


def _required(value: Any, path: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionValidationError(f"{path} must be an object")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ExecutionValidationError(f"{path} missing fields: {', '.join(missing)}")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExecutionValidationError(f"{path} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ExecutionValidationError(f"{path} must be a boolean")
    return value
