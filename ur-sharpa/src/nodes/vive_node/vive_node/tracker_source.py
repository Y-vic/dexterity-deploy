"""Vive tracker transport used by the FQX teleoperation stack."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ur_common.geometry import matrix_to_rotvec, rotvec_to_matrix, validate_transform


DEFAULT_ZMQ_ADDRESS = "tcp://172.16.13.167:5555"
TRACKER_TOPICS = ("left_elbow", "right_elbow", "chest")
OPENXR_TO_ROBOT = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
LEFT_VIVE_TO_WRIST = np.asarray(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
RIGHT_VIVE_TO_WRIST = np.asarray(
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def quat_xyzw_to_matrix(quaternion: Any) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("orientation must be four finite values [x, y, z, w]")
    norm = float(np.linalg.norm(values))
    if norm < 1e-9:
        raise ValueError("orientation quaternion is degenerate")
    x_value, y_value, z_value, w_value = values / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y_value * y_value + z_value * z_value),
                2.0 * (x_value * y_value - z_value * w_value),
                2.0 * (x_value * z_value + y_value * w_value),
            ],
            [
                2.0 * (x_value * y_value + z_value * w_value),
                1.0 - 2.0 * (x_value * x_value + z_value * z_value),
                2.0 * (y_value * z_value - x_value * w_value),
            ],
            [
                2.0 * (x_value * z_value - y_value * w_value),
                2.0 * (y_value * z_value + x_value * w_value),
                1.0 - 2.0 * (x_value * x_value + y_value * y_value),
            ],
        ],
        dtype=np.float64,
    )


def parse_tracker_message(message: str) -> tuple[str, np.ndarray]:
    try:
        topic, payload_text = str(message).split(" ", 1)
    except ValueError as exc:
        raise ValueError("Vive message must be '<topic> <json>'") from exc
    if topic not in TRACKER_TOPICS:
        raise ValueError(f"unexpected Vive topic {topic!r}")
    try:
        payload = json.loads(payload_text)
        position = payload["position"]
        orientation = payload["orientation"]
        xyz = np.asarray(
            [position["x"], position["y"], position["z"]], dtype=np.float64
        )
        quaternion = np.asarray(
            [
                orientation["x"],
                orientation["y"],
                orientation["z"],
                orientation["w"],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {topic} Vive JSON: {exc}") from exc
    if not np.isfinite(xyz).all():
        raise ValueError(f"invalid {topic} position")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_xyzw_to_matrix(quaternion)
    transform[:3, 3] = xyz
    return topic, transform


@dataclass(frozen=True)
class TrackerFrame:
    left: np.ndarray | None
    right: np.ndarray | None
    chest: np.ndarray
    topic_received: dict[str, float]
    topic_sequence: dict[str, int]
    recv_wall_ns: int
    sequence: int

    def wrist(self, side: str) -> np.ndarray:
        pose = self.left if side == "left" else self.right
        if pose is None:
            raise ValueError(f"{side} wrist is not available")
        return pose

    def topic_ages(self, now: float | None = None) -> dict[str, float]:
        current = time.monotonic() if now is None else float(now)
        return {
            topic: max(0.0, current - self.topic_received[topic])
            for topic in TRACKER_TOPICS
            if topic in self.topic_received
        }

    def stale_topics(
        self, timeout_s: float, now: float | None = None
    ) -> tuple[str, ...]:
        ages = self.topic_ages(now)
        return tuple(
            topic
            for topic in TRACKER_TOPICS
            if topic not in ages or ages[topic] > float(timeout_s)
        )

    def wrist_sequence(self, side: str) -> int:
        return int(self.topic_sequence[f"{side}_elbow"])


def retarget_vive_frame(
    frame: TrackerFrame,
    head_origin: Any | None = None,
    base_offset: Any = (0.05, 0.0, 0.40),
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if frame.left is None or frame.right is None:
        raise ValueError("both Vive wrists are required for a full-frame retarget")
    raw = {"left": frame.left, "right": frame.right, "chest": frame.chest}
    robot_from_tracker = {}
    for name, pose in raw.items():
        transformed = np.eye(4, dtype=np.float64)
        transformed[:3, :3] = OPENXR_TO_ROBOT @ pose[:3, :3]
        transformed[:3, 3] = OPENXR_TO_ROBOT @ pose[:3, 3]
        robot_from_tracker[name] = transformed
    origin = (
        robot_from_tracker["chest"][:3, 3].copy()
        if head_origin is None
        else np.asarray(head_origin, dtype=np.float64).reshape(3).copy()
    )
    offset = np.asarray(base_offset, dtype=np.float64).reshape(3)
    targets = {}
    for side, wrist_rotation in (
        ("left", LEFT_VIVE_TO_WRIST),
        ("right", RIGHT_VIVE_TO_WRIST),
    ):
        target = robot_from_tracker[side].copy()
        target[:3, :3] = target[:3, :3] @ wrist_rotation.T
        target[:3, 3] = target[:3, 3] - origin + offset
        targets[side] = target
    return targets, origin


def retarget_vive_wrist(
    frame: TrackerFrame,
    side: str,
    head_origin: Any | None = None,
    base_offset: Any = (0.05, 0.0, 0.40),
) -> tuple[np.ndarray, np.ndarray]:
    """Retarget one wrist so each arm can be aligned independently."""
    if side not in ("left", "right"):
        raise ValueError(f"unknown Vive side {side!r}")
    chest = frame.chest
    pose = frame.wrist(side)
    chest_robot = np.eye(4, dtype=np.float64)
    chest_robot[:3, :3] = OPENXR_TO_ROBOT @ chest[:3, :3]
    chest_robot[:3, 3] = OPENXR_TO_ROBOT @ chest[:3, 3]
    origin = (
        chest_robot[:3, 3].copy()
        if head_origin is None
        else np.asarray(head_origin, dtype=np.float64).reshape(3).copy()
    )
    transformed = np.eye(4, dtype=np.float64)
    transformed[:3, :3] = OPENXR_TO_ROBOT @ pose[:3, :3]
    transformed[:3, 3] = OPENXR_TO_ROBOT @ pose[:3, 3]
    wrist_rotation = LEFT_VIVE_TO_WRIST if side == "left" else RIGHT_VIVE_TO_WRIST
    transformed[:3, :3] = transformed[:3, :3] @ wrist_rotation.T
    transformed[:3, 3] = transformed[:3, 3] - origin + np.asarray(
        base_offset, dtype=np.float64
    ).reshape(3)
    return transformed, origin


def apply_relative_pose_delta(
    tracker_pose: Any,
    tracker_origin: Any,
    robot_origin: Any,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Apply tracker translation and rotation deltas to a robot origin pose."""
    tracker = np.asarray(tracker_pose, dtype=np.float64)
    tracker_zero = np.asarray(tracker_origin, dtype=np.float64)
    robot_zero = np.asarray(robot_origin, dtype=np.float64)
    if any(value.shape != (4, 4) for value in (tracker, tracker_zero, robot_zero)):
        raise ValueError("relative poses must be 4x4 transforms")
    scale = float(translation_scale)
    if not np.isfinite(scale):
        raise ValueError("translation_scale must be finite")
    if np.array_equal(tracker, tracker_zero):
        return robot_zero.copy()

    target = robot_zero.copy()
    target[:3, 3] = (
        robot_zero[:3, 3]
        + scale * (tracker[:3, 3] - tracker_zero[:3, 3])
    )
    target[:3, :3] = (
        tracker[:3, :3]
        @ tracker_zero[:3, :3].T
        @ robot_zero[:3, :3]
    )
    return target


def _smoothing_alpha(cutoff_hz: float, dt_s: float) -> float:
    time_constant = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    return 1.0 / (1.0 + time_constant / float(dt_s))


def _soft_deadband(error: np.ndarray, width: float) -> np.ndarray:
    magnitude = float(np.linalg.norm(error))
    if magnitude <= width:
        return np.zeros_like(error)
    return error * ((magnitude - width) / magnitude)


class AdaptivePoseFilter:
    """Deadband plus One Euro filtering for a Cartesian TCP target.

    Translation is filtered in metres. Rotation is filtered on SO(3) through
    rotation vectors, avoiding Euler-angle wrapping and invalid averaged
    rotation matrices. The cutoff increases with measured motion speed, so a
    stationary tracker is smoothed more strongly than an intentional motion.
    """

    def __init__(
        self,
        frequency_hz: float = 30.0,
        position_deadband_m: float = 0.0015,
        rotation_deadband_rad: float = np.deg2rad(0.5),
        position_min_cutoff_hz: float = 1.5,
        position_beta: float = 5.0,
        rotation_min_cutoff_hz: float = 2.0,
        rotation_beta: float = 1.5,
        derivative_cutoff_hz: float = 1.0,
    ) -> None:
        values = {
            "frequency_hz": frequency_hz,
            "position_min_cutoff_hz": position_min_cutoff_hz,
            "rotation_min_cutoff_hz": rotation_min_cutoff_hz,
            "derivative_cutoff_hz": derivative_cutoff_hz,
        }
        for name, value in values.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "position_deadband_m": position_deadband_m,
            "rotation_deadband_rad": rotation_deadband_rad,
            "position_beta": position_beta,
            "rotation_beta": rotation_beta,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        self.frequency_hz = float(frequency_hz)
        self.position_deadband_m = float(position_deadband_m)
        self.rotation_deadband_rad = float(rotation_deadband_rad)
        self.position_min_cutoff_hz = float(position_min_cutoff_hz)
        self.position_beta = float(position_beta)
        self.rotation_min_cutoff_hz = float(rotation_min_cutoff_hz)
        self.rotation_beta = float(rotation_beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.reset()

    def reset(
        self, pose: Any | None = None, timestamp: float | None = None
    ) -> None:
        self._output: np.ndarray | None = None
        self._previous_raw: np.ndarray | None = None
        self._position_velocity = np.zeros(3, dtype=np.float64)
        self._rotation_velocity = np.zeros(3, dtype=np.float64)
        self._timestamp: float | None = None
        if pose is not None:
            initial = validate_transform(pose, "initial filtered TCP pose").copy()
            stamp = time.monotonic() if timestamp is None else float(timestamp)
            if not np.isfinite(stamp):
                raise ValueError("filter timestamp must be finite")
            self._output = initial
            self._previous_raw = initial.copy()
            self._timestamp = stamp

    def update(self, pose: Any, timestamp: float | None = None) -> np.ndarray:
        target = validate_transform(pose, "unfiltered TCP pose").copy()
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(stamp):
            raise ValueError("filter timestamp must be finite")
        if self._output is None or self._previous_raw is None:
            self.reset(target, stamp)
            return target.copy()

        dt_s = stamp - float(self._timestamp)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = 1.0 / self.frequency_hz
        derivative_alpha = _smoothing_alpha(self.derivative_cutoff_hz, dt_s)

        raw_position_velocity = (
            target[:3, 3] - self._previous_raw[:3, 3]
        ) / dt_s
        self._position_velocity += derivative_alpha * (
            raw_position_velocity - self._position_velocity
        )
        position_cutoff = self.position_min_cutoff_hz + self.position_beta * float(
            np.linalg.norm(self._position_velocity)
        )
        position_alpha = _smoothing_alpha(position_cutoff, dt_s)
        position_error = target[:3, 3] - self._output[:3, 3]
        self._output[:3, 3] += position_alpha * _soft_deadband(
            position_error, self.position_deadband_m
        )

        raw_rotation_delta = self._previous_raw[:3, :3].T @ target[:3, :3]
        raw_rotation_velocity = matrix_to_rotvec(raw_rotation_delta) / dt_s
        self._rotation_velocity += derivative_alpha * (
            raw_rotation_velocity - self._rotation_velocity
        )
        rotation_cutoff = self.rotation_min_cutoff_hz + self.rotation_beta * float(
            np.linalg.norm(self._rotation_velocity)
        )
        rotation_alpha = _smoothing_alpha(rotation_cutoff, dt_s)
        rotation_error = matrix_to_rotvec(
            self._output[:3, :3].T @ target[:3, :3]
        )
        filtered_rotation_delta = rotation_alpha * _soft_deadband(
            rotation_error, self.rotation_deadband_rad
        )
        self._output[:3, :3] = self._output[:3, :3] @ rotvec_to_matrix(
            filtered_rotation_delta
        )

        self._previous_raw = target
        self._timestamp = stamp
        return self._output.copy()


def limit_pose_delta(
    previous: Any,
    target: Any,
    max_translation_delta_m: float,
    max_rotation_delta_rad: float,
) -> np.ndarray:
    """Limit one TCP pose step by translation norm and rotation angle."""
    previous_pose = validate_transform(previous, "previous TCP pose")
    target_pose = validate_transform(target, "target TCP pose")
    translation_limit = float(max_translation_delta_m)
    rotation_limit = float(max_rotation_delta_rad)
    if not np.isfinite(translation_limit) or translation_limit <= 0.0:
        raise ValueError("max_translation_delta_m must be finite and positive")
    if not np.isfinite(rotation_limit) or rotation_limit <= 0.0:
        raise ValueError("max_rotation_delta_rad must be finite and positive")

    limited = target_pose.copy()
    translation_delta = target_pose[:3, 3] - previous_pose[:3, 3]
    translation_norm = float(np.linalg.norm(translation_delta))
    if translation_norm > translation_limit:
        limited[:3, 3] = (
            previous_pose[:3, 3]
            + translation_delta * (translation_limit / translation_norm)
        )

    relative_rotation = previous_pose[:3, :3].T @ target_pose[:3, :3]
    rotation_vector = matrix_to_rotvec(relative_rotation)
    rotation_angle = float(np.linalg.norm(rotation_vector))
    if rotation_angle > rotation_limit:
        limited[:3, :3] = previous_pose[:3, :3] @ rotvec_to_matrix(
            rotation_vector * (rotation_limit / rotation_angle)
        )
    return limited


class TrackerLatch:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._poses: dict[str, np.ndarray] = {}
        self._topic_received: dict[str, float] = {}
        self._topic_sequence: dict[str, int] = {}
        self.messages_received = 0
        self.parse_errors = 0
        self.last_error = ""
        self._sequence = 0
        self._last_wall_ns = 0

    def put(
        self,
        topic: str,
        transform: np.ndarray,
        received: float | None = None,
        wall_ns: int | None = None,
    ) -> None:
        with self._lock:
            self._poses[topic] = np.asarray(transform, dtype=np.float64).copy()
            self._topic_received[topic] = (
                time.monotonic() if received is None else float(received)
            )
            self.messages_received += 1
            self._sequence += 1
            self._topic_sequence[topic] = self._topic_sequence.get(topic, 0) + 1
            self._last_wall_ns = time.time_ns() if wall_ns is None else int(wall_ns)

    def peek(self) -> TrackerFrame | None:
        with self._lock:
            if "chest" not in self._poses or not any(
                topic in self._poses for topic in ("left_elbow", "right_elbow")
            ):
                return None
            return TrackerFrame(
                left=(
                    None
                    if "left_elbow" not in self._poses
                    else self._poses["left_elbow"].copy()
                ),
                right=(
                    None
                    if "right_elbow" not in self._poses
                    else self._poses["right_elbow"].copy()
                ),
                chest=self._poses["chest"].copy(),
                topic_received=dict(self._topic_received),
                topic_sequence=dict(self._topic_sequence),
                recv_wall_ns=self._last_wall_ns,
                sequence=self._sequence,
            )

    def missing_topics(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(topic for topic in TRACKER_TOPICS if topic not in self._poses)

    def note_error(self, message: str) -> None:
        with self._lock:
            self.parse_errors += 1
            self.last_error = str(message)[:300]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "messages_received": self.messages_received,
                "parse_errors": self.parse_errors,
                "last_error": self.last_error,
                "missing_topics": [
                    topic for topic in TRACKER_TOPICS if topic not in self._poses
                ],
                "topic_age_s": {
                    topic: round(max(0.0, now - received), 3)
                    for topic, received in self._topic_received.items()
                },
            }


class ZmqTrackerReceiver:
    def __init__(
        self,
        address: str = DEFAULT_ZMQ_ADDRESS,
        latch: TrackerLatch | None = None,
        poll_timeout_ms: int = 100,
    ) -> None:
        self.address = str(address)
        self.latch = latch if latch is not None else TrackerLatch()
        self.poll_timeout_ms = int(poll_timeout_ms)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._socket_ready = False
        self.start_error = ""

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        try:
            import zmq  # noqa: F401
        except Exception as exc:
            self.start_error = f"Vive ZMQ unavailable: {exc}"
            return False
        self.start_error = ""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="vive-zmq-recv", daemon=True
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._socket_ready = False

    @property
    def socket_ready(self) -> bool:
        return self._socket_ready

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 30)
        socket.setsockopt(zmq.LINGER, 0)
        for topic in TRACKER_TOPICS:
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        try:
            socket.connect(self.address)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while not self._stop.is_set():
                if not dict(poller.poll(self.poll_timeout_ms)):
                    continue
                try:
                    message = socket.recv_string(flags=zmq.NOBLOCK)
                    topic, transform = parse_tracker_message(message)
                    self.latch.put(topic, transform)
                    self._socket_ready = True
                except zmq.Again:
                    continue
                except Exception as exc:
                    self.latch.note_error(f"Vive receive/parse failed: {exc}")
        except Exception as exc:
            self.start_error = f"Vive receiver failed: {exc}"
        finally:
            self._socket_ready = False
            socket.close(linger=0)
