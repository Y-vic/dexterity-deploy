"""Independent SharpA command/state tracking for Deploy and Replay.

This executable deliberately runs outside ``actor_ik_node`` and
``sharpa_node``.  Tracking I/O must never share an executor with command ACKs
or the hardware SDK path: a slow disk may lose diagnostics, but it must not
delay motion or turn a successful hardware command into an ACK timeout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


TRACKING_SCHEMA = "sharpa.hand_tracking.v1"
HAND_JOINTS = 44
JOINTS_PER_SIDE = 22


def _vector(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (HAND_JOINTS,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {HAND_JOINTS} finite values")
    return result


@dataclass(frozen=True)
class CommandIdentity:
    action_id: str
    revision: int
    step_index: int
    dry_run: bool
    requested_q_cmd: np.ndarray
    received_at_ns: int


class CommandMatcher:
    """Associate SDK-dispatched targets with recent Actor command identities."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("command matcher capacity must be positive")
        self._commands: deque[CommandIdentity] = deque(maxlen=int(capacity))

    def clear(self) -> None:
        self._commands.clear()

    def add(
        self,
        *,
        action_id: str,
        revision: int,
        step_index: int,
        dry_run: bool,
        requested_q_cmd: Any,
        received_at_ns: int | None = None,
    ) -> None:
        self._commands.append(
            CommandIdentity(
                action_id=str(action_id),
                revision=int(revision),
                step_index=int(step_index),
                dry_run=bool(dry_run),
                requested_q_cmd=_vector(requested_q_cmd, "requested_q_cmd"),
                received_at_ns=(
                    time.time_ns() if received_at_ns is None else int(received_at_ns)
                ),
            )
        )

    def nearest(self, dispatched_q_cmd: Any) -> tuple[CommandIdentity | None, float | None]:
        if not self._commands:
            return None, None
        target = _vector(dispatched_q_cmd, "dispatched_q_cmd")
        # Reverse order makes the newest command win when repeated targets are
        # exactly equal, which is common in slowly moving distal joints.
        candidates = list(reversed(self._commands))
        errors = [
            float(np.max(np.abs(command.requested_q_cmd - target)))
            for command in candidates
        ]
        index = int(np.argmin(errors))
        return candidates[index], errors[index]


def tracking_record(
    *,
    hand_timestamp_ns: int,
    measured_q: Any,
    left_valid: bool,
    right_valid: bool,
    dispatched_q_cmd: Any,
    q_cmd_valid: bool,
    q_cmd_timestamp_ns: int,
    q_cmd_source: int,
    control_mode: str,
    position_control_ready: bool,
    control_error: str,
    command: CommandIdentity | None = None,
    command_match_max_abs_rad: float | None = None,
    recorded_at_ns: int | None = None,
    recorded_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe, same-snapshot hand tracking record."""
    measured = _vector(measured_q, "measured_q")
    dispatched = _vector(dispatched_q_cmd, "dispatched_q_cmd")
    state_valid = bool(left_valid and right_valid)
    command_valid = bool(q_cmd_valid)
    tracking_valid = bool(state_valid and command_valid)
    error = measured - dispatched if tracking_valid else None
    q_cmd_age_ms = None
    if command_valid and int(hand_timestamp_ns) > 0 and int(q_cmd_timestamp_ns) > 0:
        q_cmd_age_ms = (int(hand_timestamp_ns) - int(q_cmd_timestamp_ns)) / 1_000_000.0
    record: dict[str, Any] = {
        "schema": TRACKING_SCHEMA,
        "timestamp_ns": time.time_ns() if recorded_at_ns is None else int(recorded_at_ns),
        "monotonic_ns": (
            time.monotonic_ns()
            if recorded_monotonic_ns is None
            else int(recorded_monotonic_ns)
        ),
        "hand_timestamp_ns": int(hand_timestamp_ns),
        "left_valid": bool(left_valid),
        "right_valid": bool(right_valid),
        "control_mode": str(control_mode),
        "position_control_ready": bool(position_control_ready),
        "control_error": str(control_error),
        "q": measured.tolist(),
        "q_cmd": dispatched.tolist(),
        "q_cmd_valid": command_valid,
        "q_cmd_timestamp_ns": int(q_cmd_timestamp_ns),
        "q_cmd_source": int(q_cmd_source),
        "q_cmd_age_ms": q_cmd_age_ms,
        "tracking_valid": tracking_valid,
        "tracking_error": None if error is None else error.tolist(),
        "left_max_abs_error_rad": (
            None if error is None else float(np.max(np.abs(error[:JOINTS_PER_SIDE])))
        ),
        "right_max_abs_error_rad": (
            None if error is None else float(np.max(np.abs(error[JOINTS_PER_SIDE:])))
        ),
        "action_id": "" if command is None else command.action_id,
        "revision": None if command is None else command.revision,
        "step_index": None if command is None else command.step_index,
        "dry_run": None if command is None else command.dry_run,
        "requested_q_cmd": (
            None if command is None else command.requested_q_cmd.tolist()
        ),
        "command_received_at_ns": (
            None if command is None else command.received_at_ns
        ),
        "command_match_max_abs_rad": (
            None
            if command_match_max_abs_rad is None
            else float(command_match_max_abs_rad)
        ),
    }
    return record


class HandTrackingWriter:
    """Append tracking samples to the active Actor execution directory."""

    def __init__(self) -> None:
        self.trace_dir: Path | None = None
        self.path: Path | None = None
        self._stream = None

    def select_trace(self, trace_dir: str | Path) -> bool:
        path = Path(trace_dir).expanduser().resolve()
        if self.trace_dir == path and self._stream is not None:
            return False
        self.close()
        if not path.is_dir():
            raise ValueError(f"Actor trace directory does not exist: {path}")
        self.trace_dir = path
        self.path = path / "hand_tracking.jsonl"
        self._stream = self.path.open("a", encoding="utf-8")
        return True

    def append(self, record: dict[str, Any]) -> None:
        if self._stream is None:
            return
        json.dump(record, self._stream, ensure_ascii=False, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._stream = None
        self.trace_dir = None
        self.path = None


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from ur_msgs.msg import ActorTrajectory, ControlStatus, HandCommand, HandStateFrame

    class HandTrackingNode(Node):
        def __init__(self) -> None:
            super().__init__("hand_tracking_node")
            self._writer = HandTrackingWriter()
            self._matcher = CommandMatcher()
            self._execution_active = False
            self._pending_trace_dir = ""
            self.create_subscription(
                ActorTrajectory, "/actor_ik/trajectory", self._trajectory, 10
            )
            self.create_subscription(
                HandCommand, "/sharpa/command", self._command, 100
            )
            self.create_subscription(
                HandStateFrame, "/sharpa/state", self._state, 10
            )
            self.create_subscription(
                ControlStatus, "/control/status", self._control, 10
            )

        def _trajectory(self, message) -> None:
            trace_dir = str(message.trace_dir).strip()
            if not trace_dir:
                return
            self._pending_trace_dir = trace_dir
            if self._execution_active:
                self._select_trace(trace_dir)

        def _select_trace(self, trace_dir: str) -> None:
            try:
                if self._writer.select_trace(trace_dir):
                    self.get_logger().info(
                        f"recording SharpA tracking to {trace_dir}/hand_tracking.jsonl"
                    )
            except (OSError, ValueError) as exc:
                self.get_logger().error(f"cannot open hand tracking log: {exc}")

        def _control(self, message) -> None:
            active = bool(
                (
                    int(message.ur_state) == int(ControlStatus.UR_RUNNING)
                    and int(message.sharpa_state) == int(ControlStatus.SHARPA_RUNNING)
                )
                or bool(message.execution_dry_run)
            )
            if active and not self._execution_active:
                self._matcher.clear()
                if self._pending_trace_dir:
                    self._select_trace(self._pending_trace_dir)
            elif not active and self._execution_active:
                self._writer.close()
                self._matcher.clear()
                self._pending_trace_dir = ""
            self._execution_active = active

        def _command(self, message) -> None:
            if int(message.source) != int(HandCommand.SOURCE_ACTOR):
                return
            try:
                self._matcher.add(
                    action_id=message.action_id,
                    revision=message.revision,
                    step_index=message.step_index,
                    dry_run=message.dry_run,
                    requested_q_cmd=message.q_cmd,
                )
            except ValueError as exc:
                self.get_logger().warning(f"invalid Actor hand command: {exc}")

        def _state(self, message) -> None:
            if not self._execution_active or self._writer.trace_dir is None:
                return
            command = None
            match_error = None
            if bool(message.q_cmd_valid):
                try:
                    command, match_error = self._matcher.nearest(message.q_cmd)
                except ValueError:
                    command = None
                    match_error = None
            try:
                record = tracking_record(
                    hand_timestamp_ns=message.timestamp_ns,
                    measured_q=message.q,
                    left_valid=message.left_valid,
                    right_valid=message.right_valid,
                    dispatched_q_cmd=message.q_cmd,
                    q_cmd_valid=message.q_cmd_valid,
                    q_cmd_timestamp_ns=message.q_cmd_timestamp_ns,
                    q_cmd_source=message.q_cmd_source,
                    control_mode=message.control_mode,
                    position_control_ready=message.position_control_ready,
                    control_error=message.control_error,
                    command=command,
                    command_match_max_abs_rad=match_error,
                )
                self._writer.append(record)
            except (OSError, ValueError) as exc:
                self.get_logger().error(f"failed to record hand tracking state: {exc}")
                self._writer.close()

        def destroy_node(self):
            self._writer.close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = HandTrackingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
