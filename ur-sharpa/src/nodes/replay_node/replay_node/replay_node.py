"""Recorded joint trajectory source for the Deploy execution pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SAMPLE_DIR = (
    "/media/yns/T9/0824_yns_unscrew_cap/success/"
    "2026-08-24_16-01-44.683076"
)


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_checksums(path: Path) -> None:
    checksum_path = path / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {checksum_path}") from exc
    for line in lines:
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line: {line!r}") from exc
        target = path / relative
        if not target.is_file():
            raise ValueError(f"checksum target is missing: {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"checksum mismatch: {relative}")


def _verify_joint_quality(path: Path, joint_source: str) -> None:
    """Require the validation report entries used by joint-only replay."""
    validation = _load_json(path / "validation.json")
    if not bool(validation.get("available", False)):
        raise ValueError("sample validation report is unavailable")
    items = validation.get("items")
    if not isinstance(items, list):
        raise ValueError("sample validation items must be a list")
    by_key = {
        str(item.get("key")): item
        for item in items
        if isinstance(item, dict) and item.get("key")
    }
    required = ("arm_state", "hand_state" if joint_source == "measured" else "hand_action")
    for key in required:
        item = by_key.get(key)
        if item is None:
            raise ValueError(f"sample validation is missing required item: {key}")
        if not bool(item.get("ok", False)):
            details = ", ".join(str(value) for value in item.get("details") or [])
            suffix = f": {details}" if details else ""
            raise ValueError(f"sample validation failed for {key}{suffix}")


def _hold_invalid_rows(values: np.ndarray, valid: np.ndarray, label: str) -> tuple[np.ndarray, list[int]]:
    if valid.shape != (values.shape[0],):
        raise ValueError(f"{label} validity must have shape ({values.shape[0]},)")
    invalid_rows = np.flatnonzero(~valid).astype(int).tolist()
    if not invalid_rows:
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite values in valid rows")
        return np.ascontiguousarray(values, dtype=np.float32), []
    if not bool(valid[0]):
        raise ValueError(f"{label} first row is invalid and cannot be held")
    repaired = np.asarray(values, dtype=np.float32).copy()
    for row in range(repaired.shape[0]):
        if valid[row]:
            if not np.isfinite(repaired[row]).all():
                raise ValueError(f"{label} row {row} is marked valid but is non-finite")
        else:
            repaired[row] = repaired[row - 1]
    return np.ascontiguousarray(repaired), invalid_rows


def _resample_source_rows(
    values: np.ndarray,
    valid: np.ndarray,
    source_timestamp_ns: np.ndarray,
    target_timestamp_ns: np.ndarray,
    label: str,
) -> tuple[np.ndarray, list[int]]:
    _, invalid_rows = _hold_invalid_rows(values, valid, label)
    if source_timestamp_ns.shape != (values.shape[0],):
        raise ValueError(
            f"{label} source timestamps must have shape ({values.shape[0]},)"
        )
    if target_timestamp_ns.shape != (values.shape[0],):
        raise ValueError(
            f"{label} target timestamps must have shape ({values.shape[0]},)"
        )
    if np.any(np.diff(target_timestamp_ns) <= 0):
        raise ValueError(f"{label} target timestamps must be strictly increasing")

    valid_indices = np.flatnonzero(valid)
    source_times = source_timestamp_ns[valid_indices]
    if np.any(source_times <= 0):
        raise ValueError(f"{label} source timestamps must be positive")
    if np.any(np.diff(source_times) < 0):
        raise ValueError(f"{label} source timestamps must be nondecreasing")

    unique_indices: list[int] = []
    for index in valid_indices:
        index = int(index)
        is_new_timestamp = (
            not unique_indices
            or source_timestamp_ns[index] != source_timestamp_ns[unique_indices[-1]]
        )
        if is_new_timestamp:
            unique_indices.append(index)
        elif not np.array_equal(values[index], values[unique_indices[-1]]):
            raise ValueError(
                f"{label} has conflicting rows for one source timestamp"
            )

    source_indices = np.asarray(unique_indices, dtype=np.int64)
    origin_ns = min(
        int(target_timestamp_ns[0]), int(source_timestamp_ns[source_indices[0]])
    )
    target_seconds = (target_timestamp_ns - origin_ns).astype(np.float64) / 1.0e9
    source_seconds = (
        source_timestamp_ns[source_indices] - origin_ns
    ).astype(np.float64) / 1.0e9
    resampled = np.empty(values.shape, dtype=np.float32)
    for joint in range(values.shape[1]):
        resampled[:, joint] = np.interp(
            target_seconds,
            source_seconds,
            values[source_indices, joint],
        )
    return np.ascontiguousarray(resampled), invalid_rows


@dataclass(frozen=True)
class ReplaySample:
    path: Path
    name: str
    elapsed_ns: np.ndarray
    arm_joint: np.ndarray
    hand_joint: np.ndarray
    recorded_rate_hz: float
    repaired_hand_rows: tuple[int, ...]
    joint_source: str

    @property
    def row_count(self) -> int:
        return int(self.elapsed_ns.size)

    @property
    def duration_s(self) -> float:
        return float(self.elapsed_ns[-1]) / 1.0e9 if self.row_count else 0.0

    @classmethod
    def load(cls, sample_dir: str | Path, joint_source: str = "measured") -> "ReplaySample":
        path = Path(sample_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"sample directory not found: {path}")
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"sample is not finalized (missing COMPLETE): {path}")
        manifest = _load_json(path / "manifest.json")
        schema = _load_json(path / "schema.json")
        if manifest.get("schema") != "ur_sharpa_sample.v1":
            raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
        if manifest.get("state") != "complete":
            raise ValueError(f"sample manifest state is not complete: {manifest.get('state')!r}")
        if not bool((manifest.get("quality") or {}).get("ok", False)):
            raise ValueError("sample manifest quality.ok is false")
        if schema.get("schema") != "ur_sharpa_sample.v1":
            raise ValueError(f"unsupported sample schema: {schema.get('schema')!r}")
        _verify_checksums(path)

        source = str(joint_source).strip().lower()
        if source not in {"measured", "commanded"}:
            raise ValueError("joint_source must be measured or commanded")
        _verify_joint_quality(path, source)
        try:
            with np.load(path / "timeline" / "clock.npz", allow_pickle=False) as clock:
                time_unix_ns = np.asarray(clock["time_unix_ns"], dtype=np.int64)
                elapsed_ns = np.asarray(clock["elapsed_ns"], dtype=np.int64)
                rate_hz = float(np.asarray(clock["sample_rate_hz"]).item())
            with np.load(path / "ur" / "physical_12.npz", allow_pickle=False) as arm:
                arm_joint = np.asarray(arm["q"], dtype=np.float32)
                arm_valid = np.asarray(arm["valid"], dtype=bool)
            with np.load(path / "sharpa" / "joints_44.npz", allow_pickle=False) as hand:
                hand_key = "q" if source == "measured" else "q_cmd"
                valid_key = "q_valid" if source == "measured" else "q_cmd_valid"
                timestamp_key = (
                    "q_timestamp_ns" if source == "measured" else "q_cmd_timestamp_ns"
                )
                hand_joint = np.asarray(hand[hand_key], dtype=np.float32)
                hand_valid = np.asarray(hand[valid_key], dtype=bool)
                hand_timestamp_ns = np.asarray(hand[timestamp_key], dtype=np.int64)
                hand_source = (
                    np.asarray(hand["q_cmd_source"], dtype=np.uint8)
                    if source == "commanded"
                    else None
                )
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            raise ValueError(f"failed to load replay arrays: {exc}") from exc

        if elapsed_ns.ndim != 1 or elapsed_ns.size < 2:
            raise ValueError(f"timeline elapsed_ns must have shape (T>=2,), got {elapsed_ns.shape}")
        rows = int(elapsed_ns.size)
        if time_unix_ns.shape != (rows,):
            raise ValueError(f"timeline time_unix_ns must have shape ({rows},)")
        if arm_joint.shape != (rows, 12):
            raise ValueError(f"arm q must have shape ({rows}, 12), got {arm_joint.shape}")
        if hand_joint.shape != (rows, 44):
            raise ValueError(f"hand joints must have shape ({rows}, 44), got {hand_joint.shape}")
        if source == "commanded":
            if hand_source is None or hand_source.shape != (rows,):
                raise ValueError(f"hand q_cmd_source must have shape ({rows},)")
            if np.any(hand_source[hand_valid] != 2):
                raise ValueError("commanded hand trajectory contains a non-teleop source")
        if arm_valid.shape != (rows,) or not np.all(arm_valid):
            invalid = np.flatnonzero(~arm_valid).astype(int).tolist()
            raise ValueError(f"arm trajectory contains invalid rows: {invalid[:20]}")
        if not np.isfinite(arm_joint).all():
            raise ValueError("arm trajectory contains non-finite values")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("recorded sample rate must be finite and positive")
        normalized_elapsed = elapsed_ns - elapsed_ns[0]
        if normalized_elapsed[0] != 0 or np.any(np.diff(normalized_elapsed) <= 0):
            raise ValueError("timeline elapsed_ns must be strictly increasing")
        repaired_hand, repaired_rows = _resample_source_rows(
            hand_joint,
            hand_valid,
            hand_timestamp_ns,
            time_unix_ns,
            f"hand {source} trajectory",
        )
        arm_step = float(np.max(np.abs(np.diff(arm_joint, axis=0))))
        hand_step = float(np.max(np.abs(np.diff(repaired_hand, axis=0))))
        if arm_step > 0.25:
            raise ValueError(f"arm trajectory step {arm_step:.4f} rad exceeds 0.25 rad")
        if hand_step > 0.4:
            raise ValueError(f"hand trajectory step {hand_step:.4f} rad exceeds 0.4 rad")
        return cls(
            path=path,
            name=path.name,
            elapsed_ns=np.ascontiguousarray(normalized_elapsed),
            arm_joint=np.ascontiguousarray(arm_joint),
            hand_joint=repaired_hand,
            recorded_rate_hz=rate_hz,
            repaired_hand_rows=tuple(repaired_rows),
            joint_source=source,
        )


@dataclass(frozen=True)
class ReplayConfig:
    execution_steps: int = 40
    frequency_hz: float = 30.0
    loop: bool = False
    dry_run: bool = True
    execution_scope: str = "full"

    def __post_init__(self) -> None:
        if (
            isinstance(self.execution_steps, bool)
            or not isinstance(self.execution_steps, Integral)
            or int(self.execution_steps) <= 0
        ):
            raise ValueError("execution_steps must be positive")
        if (
            isinstance(self.frequency_hz, bool)
            or not isinstance(self.frequency_hz, Real)
            or not math.isfinite(float(self.frequency_hz))
            or not 5.0 <= float(self.frequency_hz) <= 125.0
        ):
            raise ValueError("frequency_hz must be in [5, 125]")
        if type(self.loop) is not bool or type(self.dry_run) is not bool:
            raise ValueError("loop and dry_run must be boolean")
        if self.loop:
            raise ValueError("loop replay is disabled; clear progress to run again")
        if self.execution_scope not in {"full", "hands_only"}:
            raise ValueError("execution_scope must be full or hands_only")


class ReplaySequencer:
    """Pure segment and identity state used by the ROS wrapper."""

    def __init__(self, sample: ReplaySample, config: ReplayConfig) -> None:
        self.sample = sample
        self.config = config
        self.execution_id = ""
        self.next_row = 0
        self.cycle = 0
        self.segment_index = 0
        self.request_id = 0
        self.reset()

    def reset(self) -> None:
        self.execution_id = uuid.uuid4().hex
        self.next_row = 0
        self.cycle = 0
        self.segment_index = 0
        self.request_id = 0

    def next_identity(self, label: str) -> dict[str, Any]:
        request_id = self.request_id
        self.request_id += 1
        safe_label = str(label).strip().replace(" ", "_") or "action"
        return {
            "session_id": f"replay_{self.sample.name}_{self.execution_id}",
            "request_id": request_id,
            "action_id": f"replay_{self.execution_id}_{safe_label}_{request_id}",
            "revision": self.cycle,
        }

    def next_segment(self) -> dict[str, Any] | None:
        if self.next_row >= self.sample.row_count:
            return None
        start = self.next_row
        end = min(start + int(self.config.execution_steps), self.sample.row_count)
        self.segment_index += 1
        self.next_row = end
        return {
            **self.next_identity(f"segment_{self.cycle}_{start}_{end}"),
            "frequency_hz": float(self.config.frequency_hz),
            "action_length": self.sample.row_count,
            "execute_start": start,
            "execute_length": end - start,
            "left_arm_joint": self.sample.arm_joint[start:end, :6].reshape(-1).tolist(),
            "right_arm_joint": self.sample.arm_joint[start:end, 6:].reshape(-1).tolist(),
            "hand_joint": self.sample.hand_joint[start:end].reshape(-1).tolist(),
        }

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
            try:
                body = json.loads(message.data or "{}")
                action = str(body.get("action", "")).strip().lower()
                with self._lock:
                    preserve_complete = self._state == "complete"
                if action == "load":
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
                self.get_logger().error(f"replay command failed: {exc}")
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
