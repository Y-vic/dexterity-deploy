"""Recorded joint trajectory source for the Deploy execution pipeline."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .sample import ReplaySample
from .sequencer import ReplayConfig, ReplaySequencer


DEFAULT_SAMPLE_DIR = ""


def execution_identity(message: Any) -> tuple[Any, ...]:
    """Return every field required to associate an ExecutionDone with an action."""
    return (
        message.session_id,
        int(message.request_id),
        message.action_id,
        int(message.revision),
        int(message.execute_start),
        int(message.execute_length),
    )


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import ControlStatus, ExecutableAction, ExecutionDone, HandStateFrame, UrStateFrame
    from ur_common.geometry import CANONICAL_EEF_FRAME

    class ReplayNode(Node):
        def __init__(self) -> None:
            super().__init__("replay_node")
            self.declare_parameter("sample_dir", DEFAULT_SAMPLE_DIR)
            self.declare_parameter("execution_steps", 40)
            self.declare_parameter("frequency_hz", 30.0)
            self.declare_parameter("joint_source", "measured")
            self.declare_parameter("loop", False)
            self.declare_parameter("dry_run", True)
            self.declare_parameter("execution_scope", "full")
            self._lock = threading.RLock()
            self._sample: ReplaySample | None = None
            self._sequencer: ReplaySequencer | None = None
            self._state = "idle"
            self._error = ""
            self._provider = "policy"
            self._machine_running = False
            self._waiting_action_id = ""
            self._waiting_identity: tuple[Any, ...] | None = None
            self._waiting_kind = ""
            self._waiting_segment_start = 0
            self._stop_requested = False
            self._unload_requested = False
            self._last_done: dict[str, Any] = {}
            self._command_result: dict[str, Any] = {}
            self._requested_sample_dir = ""
            self._preview_active = False
            self._ur_joint: np.ndarray | None = None
            self._hand_joint: np.ndarray | None = None
            self._action_pub = self.create_publisher(
                ExecutableAction, "/replay/executable_action", 10
            )
            self._status_pub = self.create_publisher(String, "/replay/status", 10)
            self._control_command_pub = self.create_publisher(
                String, "/control/ui_command", 10
            )
            self.create_subscription(String, "/replay/command", self._command, 10)
            self.create_subscription(String, "/control/execution_provider", self._provider_update, 10)
            self.create_subscription(ControlStatus, "/control/status", self._control, 10)
            self.create_subscription(ExecutionDone, "/policy/execution_done", self._done, 10)
            self.create_subscription(UrStateFrame, "/ur/state", self._ur, 10)
            self.create_subscription(HandStateFrame, "/sharpa/state", self._hand, 10)
            self.create_timer(0.5, self._publish_status)
            if not str(self.get_parameter("sample_dir").value).strip():
                self._publish_status()
                return
            try:
                self._load(
                    str(self.get_parameter("sample_dir").value),
                    ReplayConfig(
                        int(self.get_parameter("execution_steps").value),
                        float(self.get_parameter("frequency_hz").value),
                        bool(self.get_parameter("loop").value),
                        bool(self.get_parameter("dry_run").value),
                        str(self.get_parameter("execution_scope").value),
                    ),
                    str(self.get_parameter("joint_source").value),
                )
            except Exception as exc:
                self._state = "fault"
                self._error = str(exc)
                self.get_logger().error(f"initial replay load failed: {exc}")

        def _load(self, sample_dir: str, config: ReplayConfig, joint_source: str) -> None:
            with self._lock:
                if self._state in {"playing", "aligning"} or self._waiting_action_id:
                    raise RuntimeError("cannot load a sample while replay is active")
                if self._state == "complete":
                    raise RuntimeError(
                        "replay buffer is complete; clear progress before loading again"
                    )
                loaded_sample = self._sample
            requested_path = Path(sample_dir).expanduser().resolve()
            requested_source = str(joint_source).strip().lower()
            sample = (
                loaded_sample
                if loaded_sample is not None
                and loaded_sample.path == requested_path
                and loaded_sample.joint_source == requested_source
                else ReplaySample.load(requested_path, requested_source)
            )
            with self._lock:
                self._sample = sample
                self._sequencer = ReplaySequencer(sample, config)
                self._state = "ready"
                self._error = ""
                self._clear_waiting()
                self._stop_requested = False
                self._unload_requested = False
                self._preview_active = False

        @staticmethod
        def _to_action(payload: dict[str, Any]) -> ExecutableAction:
            message = ExecutableAction()
            message.session_id = payload["session_id"]
            message.request_id = int(payload["request_id"])
            message.action_id = payload["action_id"]
            message.revision = int(payload["revision"])
            message.timestamp_ns = time.time_ns()
            message.frequency_hz = float(payload["frequency_hz"])
            message.action_length = int(payload["action_length"])
            message.execute_start = int(payload["execute_start"])
            message.execute_length = int(payload["execute_length"])
            message.rtc = 0
            message.eef_frame = CANONICAL_EEF_FRAME
            message.left_arm_joint = payload["left_arm_joint"]
            message.right_arm_joint = payload["right_arm_joint"]
            message.hand_joint = payload["hand_joint"]
            return message

        def _provider_update(self, message: String) -> None:
            with self._lock:
                self._provider = message.data.strip().lower() or "policy"

        def _clear_waiting(self) -> None:
            self._waiting_action_id = ""
            self._waiting_identity = None
            self._waiting_kind = ""
            self._waiting_segment_start = 0

        def _set_waiting(
            self, message: ExecutableAction, kind: str, segment_start: int
        ) -> None:
            self._waiting_action_id = message.action_id
            self._waiting_identity = execution_identity(message)
            self._waiting_kind = kind
            self._waiting_segment_start = int(segment_start)

        def _matches_waiting(self, message: ExecutionDone) -> bool:
            return self._waiting_identity == execution_identity(message)

        def _request_machine_stop(self) -> None:
            command = String()
            command.data = json.dumps(
                {"action": "stop_deploy"}, separators=(",", ":")
            )
            self._control_command_pub.publish(command)

        def _finish_stop(self) -> None:
            if self._unload_requested:
                self._sample = None
                self._sequencer = None
            elif self._sequencer is not None:
                # STOP discards progress but keeps the validated sample in RAM,
                # so the next run can start at row zero without loading it again.
                self._sequencer.reset()
            self._clear_waiting()
            self._stop_requested = False
            self._preview_active = False
            self._state = "ready" if self._sequencer is not None else "idle"
            self._unload_requested = False

        def _control(self, message: ControlStatus) -> None:
            with self._lock:
                running = bool(
                    (
                        message.sharpa_state == ControlStatus.SHARPA_RUNNING
                        and (
                            int(message.execution_scope)
                            == int(ControlStatus.EXECUTION_SCOPE_HANDS_ONLY)
                            or message.ur_state == ControlStatus.UR_RUNNING
                        )
                    )
                    or message.execution_dry_run
                )
                if (
                    self._machine_running
                    and not running
                    and self._state in {"playing", "aligning"}
                ):
                    self._state = "paused"
                elif not running and self._state == "stopping":
                    self._finish_stop()
                self._machine_running = running

        def _ur(self, message: UrStateFrame) -> None:
            values = np.asarray(
                list(message.left_joint) + list(message.right_joint), dtype=np.float32
            )
            with self._lock:
                self._ur_joint = values if message.valid and np.isfinite(values).all() else None

        def _hand(self, message: HandStateFrame) -> None:
            values = np.asarray(message.q, dtype=np.float32)
            with self._lock:
                self._hand_joint = values if message.left_valid and message.right_valid and np.isfinite(values).all() else None

        def _alignment_payload(self) -> dict[str, Any] | None:
            sample = self._sample
            sequencer = self._sequencer
            if (
                sample is None
                or sequencer is None
                or self._hand_joint is None
                or (
                    sequencer.config.execution_scope == "full"
                    and self._ur_joint is None
                )
            ):
                required = (
                    "SharpA"
                    if sequencer is not None
                    and sequencer.config.execution_scope == "hands_only"
                    else "UR and SharpA"
                )
                raise RuntimeError(f"valid current {required} states are required")
            target_row = min(sequencer.next_row, sample.row_count - 1)
            arm_target = sample.arm_joint[target_row]
            hand_target = sample.hand_joint[target_row]
            hand_delta = float(np.max(np.abs(hand_target - self._hand_joint)))
            if sequencer.config.execution_scope == "hands_only":
                steps = math.ceil(hand_delta / 0.08) + 1
            else:
                arm_delta = float(np.max(np.abs(arm_target - self._ur_joint)))
                steps = max(
                    math.ceil(arm_delta / 0.04), math.ceil(hand_delta / 0.08)
                ) + 1
            if steps <= 1:
                return None
            steps = min(300, int(steps))
            arm = (
                np.repeat(arm_target[None, :], steps, axis=0).astype(np.float32)
                if self._ur_joint is None
                else np.linspace(self._ur_joint, arm_target, steps, dtype=np.float32)
            )
            hand = np.linspace(self._hand_joint, hand_target, steps, dtype=np.float32)
            return {
                **sequencer.next_identity(f"alignment_{sequencer.cycle}_{target_row}"),
                "frequency_hz": float(sequencer.config.frequency_hz),
                "action_length": steps,
                "execute_start": 0,
                "execute_length": steps,
                "left_arm_joint": arm[:, :6].reshape(-1).tolist(),
                "right_arm_joint": arm[:, 6:].reshape(-1).tolist(),
                "hand_joint": hand.reshape(-1).tolist(),
            }

        def _publish_alignment_or_next(self) -> None:
            alignment = self._alignment_payload()
            if alignment is None:
                self._publish_next()
                return
            message = self._to_action(alignment)
            self._set_waiting(
                message,
                "alignment",
                self._sequencer.next_row,
            )
            self._state = "aligning"
            self._action_pub.publish(message)

        def _publish_next(self) -> None:
            sequencer = self._sequencer
            if sequencer is None:
                raise RuntimeError("no replay sample is loaded")
            payload = sequencer.next_segment()
            if payload is None:
                self._state = "complete"
                self._preview_active = False
                self._clear_waiting()
                self._request_machine_stop()
                return
            message = self._to_action(payload)
            self._set_waiting(message, "replay", int(payload["execute_start"]))
            self._state = "playing"
            self._action_pub.publish(message)

        def _start(self) -> None:
            with self._lock:
                if self._sequencer is None:
                    raise RuntimeError("no replay sample is loaded")
                if self._waiting_action_id:
                    raise RuntimeError("a replay action is already executing")
                if self._provider != "replay":
                    raise RuntimeError("execution provider must be replay")
                if not self._machine_running:
                    raise RuntimeError("Actor live or preview gate must be running")
                if self._state == "complete":
                    raise RuntimeError(
                        "replay buffer is complete; clear progress before starting again"
                    )
                if self._state == "fault":
                    raise RuntimeError("replay is faulted; clear progress or reload")
                self._preview_active = bool(self._sequencer.config.dry_run)
                self._publish_alignment_or_next()

        def _done(self, message: ExecutionDone) -> None:
            with self._lock:
                if not bool(message.execute_done):
                    return
                if not self._waiting_action_id or not self._matches_waiting(message):
                    return
                self._last_done = {
                    "action_id": message.action_id,
                    "success": bool(message.success),
                    "executed_steps": int(message.executed_steps),
                    "error": message.error,
                }
                continuation_requested = self._state in {"aligning", "playing"}
                waiting_kind = self._waiting_kind
                waiting_start = self._waiting_segment_start
                self._clear_waiting()
                if not continuation_requested:
                    if self._sequencer is not None:
                        if self._stop_requested:
                            if self._machine_running:
                                self._state = "stopping"
                            else:
                                self._finish_stop()
                        elif waiting_kind == "replay":
                            self._sequencer.next_row = max(
                                waiting_start,
                                min(int(message.executed_steps), self._sample.row_count),
                            )
                    if self._state != "stopping":
                        self._stop_requested = False
                    return
                if not message.success:
                    if message.error == "machine execution gate closed":
                        self._state = "paused"
                        if waiting_kind == "replay":
                            self._sequencer.next_row = max(
                                waiting_start,
                                min(int(message.executed_steps), self._sample.row_count),
                            )
                        self._stop_requested = False
                        return
                    self._state = "fault"
                    self._error = message.error or "replay execution failed"
                    return
                try:
                    self._publish_next()
                except Exception as exc:
                    self._state = "fault"
                    self._error = str(exc)

        def _command(self, message: String) -> None:
            action = ""
            preserve_complete = False
            body = {}
            try:
                body = json.loads(message.data or "{}")
                action = str(body.get("action", "")).strip().lower()
                with self._lock:
                    preserve_complete = self._state == "complete"
                if action == "load":
                    self._requested_sample_dir = str(body.get("sample_dir") or self.get_parameter("sample_dir").value)
                    current = self._sequencer.config if self._sequencer else ReplayConfig()
                    config = ReplayConfig(
                        body.get("execution_steps", current.execution_steps),
                        body.get("frequency_hz", current.frequency_hz),
                        body.get("loop", current.loop),
                        body.get("dry_run", current.dry_run),
                        str(body.get("execution_scope", current.execution_scope)),
                    )
                    self._load(
                        str(body.get("sample_dir") or self.get_parameter("sample_dir").value),
                        config,
                        str(body.get("joint_source") or "measured"),
                    )
                elif action in {"start", "resume"}:
                    self._start()
                elif action in {"pause", "stop", "unload"}:
                    with self._lock:
                        waiting = bool(self._waiting_action_id)
                        terminal = action in {"stop", "unload"}
                        stopping = bool(
                            terminal and (waiting or self._machine_running)
                        )
                        self._state = (
                            "paused"
                            if action == "pause"
                            else "stopping" if stopping else self._state
                        )
                        self._error = ""
                        self._stop_requested = terminal and waiting
                        self._unload_requested = action == "unload"
                        self._preview_active = False
                        if terminal and not self._machine_running:
                            self._finish_stop()
                        close_machine_gate = self._machine_running
                    if close_machine_gate:
                        self._request_machine_stop()
                else:
                    raise ValueError(f"unknown replay action: {action}")
            except Exception as exc:
                with self._lock:
                    if action == "load":
                        self._preview_active = False
                    if not preserve_complete:
                        self._state = "fault"
                    self._error = str(exc)
                self._command_result = dict(id=body.get('id', '') if isinstance(body, dict) else '',
                    action=action, accepted=False, reason=str(exc), sample_dir=self._requested_sample_dir)
                self.get_logger().error(f"replay {action} failed [{self._requested_sample_dir}]: {exc}")
            else:
                self._command_result = dict(id=body.get('id', ''), action=action, accepted=True,
                    reason='', sample_dir=self._requested_sample_dir)
            self._publish_status()

        def _publish_status(self) -> None:
            with self._lock:
                sample = self._sample
                sequencer = self._sequencer
                config = sequencer.config if sequencer else ReplayConfig()
                payload = {
                    "state": self._state,
                    "error": self._error,
                    "provider": self._provider,
                    "machine_running": self._machine_running,
                    "sample_dir": str(sample.path) if sample else "",
                    "sample": sample.name if sample else "",
                    "row_count": sample.row_count if sample else 0,
                    "duration_s": round(sample.duration_s, 3) if sample else 0.0,
                    "recorded_rate_hz": sample.recorded_rate_hz if sample else 0.0,
                    "repaired_hand_rows": len(sample.repaired_hand_rows) if sample else 0,
                    "joint_source": sample.joint_source if sample else "",
                    "execution_id": sequencer.execution_id if sequencer else "",
                    "next_row": sequencer.next_row if sequencer else 0,
                    "preview_row": sequencer.next_row if sequencer else 0,
                    "preview_active": self._preview_active,
                    "cycle": sequencer.cycle if sequencer else 0,
                    "execution_steps": config.execution_steps,
                    "frequency_hz": config.frequency_hz,
                    "loop": config.loop,
                    "dry_run": config.dry_run,
                    "execution_scope": config.execution_scope,
                    "waiting_action_id": self._waiting_action_id,
                    "waiting_kind": self._waiting_kind,
                    "last_done": dict(self._last_done),
                    "command_result": dict(self._command_result),
                    "requested_sample_dir": self._requested_sample_dir,
                }
            output = String()
            output.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self._status_pub.publish(output)

    rclpy.init(args=args)
    node = ReplayNode()
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
