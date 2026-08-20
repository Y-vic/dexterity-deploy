"""Validation and coordinate conversion for the PND Quest WebVR protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


POSE_NAMES = ("Head", "LeftHand", "RightHand")
JOY_AXIS_COUNT = 8
JOY_BUTTON_COUNT = 6


class WebVRProtocolError(ValueError):
    """Raised when a WebVR frame does not match the expected wire format."""


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class TrackingInfo:
    connected: bool = False
    position: str = "Unavailable"
    rotation: str = "Unavailable"

    @property
    def position_is_known(self) -> bool:
        return self.connected and self.position.casefold() == "known"

    @property
    def position_is_usable(self) -> bool:
        return self.connected and self.position.casefold() in {"known", "inferred"}

    @property
    def rotation_is_known(self) -> bool:
        return self.connected and self.rotation.casefold() == "known"

    @property
    def rotation_is_usable(self) -> bool:
        return self.connected and self.rotation.casefold() in {"known", "inferred"}


@dataclass(frozen=True)
class Pose:
    position: Vector3
    quaternion: Quaternion
    tracking: TrackingInfo = TrackingInfo()


@dataclass
class HandExecutionGate:
    """Reject discontinuities and re-anchor stable tracking without a pose jump."""

    name: str
    mode: str = "Normal"
    last_normal_pose: Pose | None = None
    last_raw_pose: Pose | None = None
    recovery_candidate: Pose | None = None
    recovery_frames: int = 0
    position_offset: Vector3 = Vector3(0.0, 0.0, 0.0)
    quaternion_offset: Quaternion = Quaternion(0.0, 0.0, 0.0, 1.0)
    jump_distance: float | None = None
    error: str = ""

    @property
    def state(self) -> str:
        return self.mode

    @property
    def ready(self) -> bool:
        return self.mode == "Normal" and self.last_normal_pose is not None

    def _execution_pose(self, current: Pose, position: Vector3) -> Pose:
        quaternion = (
            _multiply_quaternions(current.quaternion, self.quaternion_offset)
            if current.tracking.rotation_is_usable
            else (
                self.last_normal_pose.quaternion
                if self.last_normal_pose is not None
                else current.quaternion
            )
        )
        return Pose(
            position=position,
            quaternion=quaternion,
            tracking=current.tracking,
        )

    def _accept(self, current: Pose) -> Pose:
        execution = self._execution_pose(
            current,
            Vector3(
                current.position.x + self.position_offset.x,
                current.position.y + self.position_offset.y,
                current.position.z + self.position_offset.z,
            ),
        )
        self.mode = "Normal"
        self.last_raw_pose = current
        self.last_normal_pose = execution
        self.recovery_candidate = None
        self.recovery_frames = 0
        self.error = ""
        return execution

    def _hold(self, current: Pose) -> Pose | None:
        if self.last_normal_pose is None:
            return None
        return self._execution_pose(current, self.last_normal_pose.position)

    def observe(
        self,
        current: Pose,
        *,
        calibration_command: bool,
        jump_threshold: float,
        recovery_frame_count: int = 3,
        recovery_motion_threshold: float = 0.03,
    ) -> Pose | None:
        if jump_threshold <= 0.0:
            raise ValueError("jump_threshold must be positive")
        if recovery_frame_count <= 0:
            raise ValueError("recovery_frame_count must be positive")
        if recovery_motion_threshold <= 0.0:
            raise ValueError("recovery_motion_threshold must be positive")
        usable = current.tracking.position_is_usable

        if usable and calibration_command:
            self.position_offset = Vector3(0.0, 0.0, 0.0)
            self.quaternion_offset = Quaternion(0.0, 0.0, 0.0, 1.0)
            self.jump_distance = None
            return self._accept(current)

        if self.mode == "Normal":
            if not usable:
                self.mode = "Lost"
                self.jump_distance = None
                self.error = (
                    f"{self.name} position tracking is {current.tracking.position}"
                )
                return self._hold(current)
            if self.last_raw_pose is None:
                return self._accept(current)

            jump = position_distance(self.last_raw_pose, current)
            self.jump_distance = jump
            if jump <= jump_threshold:
                return self._accept(current)

            self.mode = "Suspect"
            self.recovery_candidate = current
            self.recovery_frames = 1
            self.error = (
                f"{self.name} position jumped {jump:.3f}m "
                f"(threshold {jump_threshold:.3f}m); checking recovery"
            )
            return self._hold(current)

        if not usable:
            self.mode = "Lost"
            self.recovery_candidate = None
            self.recovery_frames = 0
            self.jump_distance = None
            self.error = f"{self.name} position tracking is {current.tracking.position}"
            return self._hold(current)

        if self.last_raw_pose is None or self.last_normal_pose is None:
            return self._accept(current)

        distance_from_last = position_distance(self.last_raw_pose, current)
        self.jump_distance = distance_from_last
        if distance_from_last <= jump_threshold:
            return self._accept(current)

        if self.recovery_candidate is None:
            self.recovery_frames = 1
        elif (
            position_distance(self.recovery_candidate, current)
            <= recovery_motion_threshold
        ):
            self.recovery_frames += 1
        else:
            self.recovery_frames = 1
        self.recovery_candidate = current
        self.mode = "Recovering"

        if self.recovery_frames >= recovery_frame_count:
            held = self.last_normal_pose.position
            self.position_offset = Vector3(
                held.x - current.position.x,
                held.y - current.position.y,
                held.z - current.position.z,
            )
            if current.tracking.rotation_is_usable:
                self.quaternion_offset = _multiply_quaternions(
                    _inverse_quaternion(current.quaternion),
                    self.last_normal_pose.quaternion,
                )
            self.jump_distance = distance_from_last
            return self._accept(current)

        self.error = (
            f"{self.name} tracking recovery {self.recovery_frames}/"
            f"{recovery_frame_count}; {distance_from_last:.3f}m from last raw pose"
        )
        return self._hold(current)


def hand_execution_is_ready(
    gates: Mapping[str, HandExecutionGate],
    execution_poses: Mapping[str, Pose],
) -> bool:
    """Return whether both hand execution poses are current and safe to consume."""

    return all(
        name in execution_poses and gates[name].ready
        for name in ("LeftHand", "RightHand")
    )


def pose_status(pose: Pose) -> dict[str, dict[str, float]]:
    """Return a JSON-ready pose without dropping coordinate precision."""

    return {
        "position": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "quaternion": {
            "x": pose.quaternion.x,
            "y": pose.quaternion.y,
            "z": pose.quaternion.z,
            "w": pose.quaternion.w,
        },
    }


@dataclass(frozen=True)
class WebVRSample:
    timestamp_ms: float
    poses: Mapping[str, Pose]
    joy_axes: tuple[float, ...]
    joy_buttons: tuple[tuple[int, int], ...]
    calibration_pressed: bool = False
    source_sequence: int | None = None
    source_monotonic_ms: float | None = None


@dataclass(frozen=True)
class WebVRCalibration:
    scale: float
    position_rotation: Quaternion
    position_offsets: Mapping[str, Vector3]
    quaternion_offsets: Mapping[str, Quaternion]


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise WebVRProtocolError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WebVRProtocolError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise WebVRProtocolError(f"{field} must be a finite number")
    return number


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WebVRProtocolError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str, length: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WebVRProtocolError(f"{field} must be an array")
    if len(value) != length:
        raise WebVRProtocolError(f"{field} must contain exactly {length} values")
    return value


def _optional_nonnegative_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    number = _finite_number(value, field)
    if number < 0.0 or not number.is_integer():
        raise WebVRProtocolError(f"{field} must be a non-negative integer")
    return int(number)


def _parse_vector3(value: Any, field: str) -> Vector3:
    data = _mapping(value, field)
    return Vector3(
        x=_finite_number(data.get("x"), f"{field}.x"),
        y=_finite_number(data.get("y"), f"{field}.y"),
        z=_finite_number(data.get("z"), f"{field}.z"),
    )


def _parse_quaternion(value: Any, field: str) -> Quaternion:
    data = _mapping(value, field)
    x = _finite_number(data.get("x"), f"{field}.x")
    y = _finite_number(data.get("y"), f"{field}.y")
    z = _finite_number(data.get("z"), f"{field}.z")
    w = _finite_number(data.get("w"), f"{field}.w")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-6:
        raise WebVRProtocolError(f"{field} has a zero-length quaternion")
    return Quaternion(x=x / norm, y=y / norm, z=z / norm, w=w / norm)


def _parse_pose(value: Any, field: str) -> Pose:
    data = _mapping(value, field)
    return Pose(
        position=_parse_vector3(data.get("position"), f"{field}.position"),
        quaternion=_parse_quaternion(data.get("quaternion"), f"{field}.quaternion"),
        tracking=_parse_tracking(data.get("tracking"), f"{field}.tracking"),
    )


def _parse_tracking(value: Any, field: str) -> TrackingInfo:
    if value is None:
        return TrackingInfo()
    data = _mapping(value, field)
    connected = data.get("connected", False)
    if not isinstance(connected, bool):
        raise WebVRProtocolError(f"{field}.connected must be a boolean")
    position = data.get("position", "Unavailable")
    rotation = data.get("rotation", "Unavailable")
    if not isinstance(position, str) or not position.strip():
        raise WebVRProtocolError(f"{field}.position must be a non-empty string")
    if not isinstance(rotation, str) or not rotation.strip():
        raise WebVRProtocolError(f"{field}.rotation must be a non-empty string")
    return TrackingInfo(
        connected=connected,
        position=position,
        rotation=rotation,
    )


def hand_position_states(sample: WebVRSample) -> dict[str, str]:
    """Return the reported positional tracking state for both controllers."""

    return {
        name: sample.poses[name].tracking.position
        for name in ("LeftHand", "RightHand")
    }


def hand_positions_are_usable(sample: WebVRSample) -> bool:
    """Return whether both controllers provide current or inferred positions."""

    return all(
        sample.poses[name].tracking.position_is_usable
        for name in ("LeftHand", "RightHand")
    )


def position_distance(previous: Pose, current: Pose) -> float:
    """Return the Euclidean distance between two pose positions."""

    delta_x = current.position.x - previous.position.x
    delta_y = current.position.y - previous.position.y
    delta_z = current.position.z - previous.position.z
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


def _parse_button(value: Any, field: str) -> tuple[int, int]:
    pair = _sequence(value, field, 2)
    parsed: list[int] = []
    for index, item in enumerate(pair):
        number = (
            float(int(item))
            if isinstance(item, bool)
            else _finite_number(
                item,
                f"{field}[{index}]",
            )
        )
        if number < 0.0 or number > 1.0:
            raise WebVRProtocolError(f"{field}[{index}] must be between 0 and 1")
        parsed.append(int(number))
    return parsed[0], parsed[1]


def parse_webvr_message(message: str | bytes | Mapping[str, Any]) -> WebVRSample:
    """Parse one browser frame using the upstream PND WebVR wire contract."""

    if isinstance(message, Mapping):
        data = message
    else:
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise WebVRProtocolError("WebVR payload must be valid JSON") from exc
        data = _mapping(data, "payload")

    poses = {name: _parse_pose(data.get(name), name) for name in POSE_NAMES}
    joy = _mapping(data.get("Joy"), "Joy")
    axes_raw = _sequence(joy.get("axes"), "Joy.axes", JOY_AXIS_COUNT)
    buttons_raw = _sequence(joy.get("buttons"), "Joy.buttons", JOY_BUTTON_COUNT)

    calibration = data.get("Calibration")
    if calibration is None:
        calibration_pressed = bool(buttons_raw[4][0])
    else:
        calibration_data = _mapping(calibration, "Calibration")
        calibration_pressed = calibration_data.get("pressed")
        if not isinstance(calibration_pressed, bool):
            raise WebVRProtocolError("Calibration.pressed must be a boolean")

    return WebVRSample(
        timestamp_ms=_finite_number(data.get("timestamp"), "timestamp"),
        poses=poses,
        joy_axes=tuple(
            _finite_number(value, f"Joy.axes[{index}]")
            for index, value in enumerate(axes_raw)
        ),
        joy_buttons=tuple(
            _parse_button(value, f"Joy.buttons[{index}]")
            for index, value in enumerate(buttons_raw)
        ),
        calibration_pressed=calibration_pressed,
        source_sequence=_optional_nonnegative_integer(
            data.get("sequence"),
            "sequence",
        ),
        source_monotonic_ms=(
            None
            if data.get("monotonicTimestampMs") is None
            else _finite_number(
                data.get("monotonicTimestampMs"),
                "monotonicTimestampMs",
            )
        ),
    )


def flatten_joy_buttons(buttons: Sequence[tuple[int, int]]) -> list[int]:
    """Return PND's ROS Joy layout: six presses followed by six touches."""

    if len(buttons) != JOY_BUTTON_COUNT:
        raise WebVRProtocolError(
            f"Joy.buttons must contain exactly {JOY_BUTTON_COUNT} pairs"
        )
    return [button[0] for button in buttons] + [button[1] for button in buttons]


def vr_pose_to_ros(
    pose: Pose,
    *,
    scale: float = 1.0,
    position_rotation: Quaternion = Quaternion(0.0, 0.0, 0.0, 1.0),
    position_offset: Vector3 = Vector3(0.0, 0.0, 0.0),
    quaternion_offset: Quaternion = Quaternion(0.0, 0.0, 0.0, 1.0),
) -> Pose:
    """Convert a VR pose to ROS and apply a calibrated rigid transform."""

    scale_value = _finite_number(scale, "scale")
    offset = Vector3(
        _finite_number(position_offset.x, "position_offset.x"),
        _finite_number(position_offset.y, "position_offset.y"),
        _finite_number(position_offset.z, "position_offset.z"),
    )
    converted_quaternion = Quaternion(
        x=-pose.quaternion.z,
        y=-pose.quaternion.x,
        z=pose.quaternion.y,
        w=pose.quaternion.w,
    )
    converted_position = _rotate_vector(
        position_rotation,
        Vector3(
            x=-pose.position.z * scale_value,
            y=-pose.position.x * scale_value,
            z=pose.position.y * scale_value,
        ),
    )
    return Pose(
        position=Vector3(
            x=converted_position.x + offset.x,
            y=converted_position.y + offset.y,
            z=converted_position.z + offset.z,
        ),
        quaternion=_multiply_quaternions(
            _multiply_quaternions(position_rotation, converted_quaternion),
            quaternion_offset,
        ),
        tracking=pose.tracking,
    )


def _multiply_quaternions(left: Quaternion, right: Quaternion) -> Quaternion:
    x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y
    y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x
    z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w
    w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-6:
        raise WebVRProtocolError("calibrated quaternion has zero length")
    return Quaternion(x=x / norm, y=y / norm, z=z / norm, w=w / norm)


def _inverse_quaternion(quaternion: Quaternion) -> Quaternion:
    return Quaternion(
        x=-quaternion.x,
        y=-quaternion.y,
        z=-quaternion.z,
        w=quaternion.w,
    )


def _rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    norm = math.sqrt(
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + quaternion.w * quaternion.w
    )
    if norm < 1.0e-6:
        raise WebVRProtocolError("position rotation has zero length")
    qx = quaternion.x / norm
    qy = quaternion.y / norm
    qz = quaternion.z / norm
    qw = quaternion.w / norm
    tx = 2.0 * (qy * vector.z - qz * vector.y)
    ty = 2.0 * (qz * vector.x - qx * vector.z)
    tz = 2.0 * (qx * vector.y - qy * vector.x)
    return Vector3(
        x=vector.x + qw * tx + qy * tz - qz * ty,
        y=vector.y + qw * ty + qz * tx - qx * tz,
        z=vector.z + qw * tz + qx * ty - qy * tx,
    )


def calibration_from_sample(
    sample: WebVRSample,
    *,
    robot_arm_length: float = 0.53,
    head_target: Vector3 = Vector3(0.0186, 0.0204, 1.5715),
    left_hand_target: Vector3 = Vector3(0.54, 0.2, 1.4),
    right_hand_target: Vector3 = Vector3(0.54, -0.2, 1.4),
    head_quaternion: Quaternion = Quaternion(0.0, 0.0, 0.0, 1.0),
    left_hand_quaternion: Quaternion = Quaternion(0.5, -0.5, -0.5, 0.5),
    right_hand_quaternion: Quaternion = Quaternion(0.5, 0.5, -0.5, -0.5),
) -> WebVRCalibration:
    """Map the arms-forward Quest calibration pose into Adam Pro targets."""

    _validate_calibration_tracking(sample)
    arm_length = _finite_number(robot_arm_length, "robot_arm_length")
    if arm_length <= 0.0:
        raise WebVRProtocolError("robot_arm_length must be positive")
    converted = {name: vr_pose_to_ros(sample.poses[name]) for name in POSE_NAMES}
    head_forward = _rotate_vector(
        converted["Head"].quaternion,
        Vector3(1.0, 0.0, 0.0),
    )
    head_up = _rotate_vector(
        converted["Head"].quaternion,
        Vector3(0.0, 0.0, 1.0),
    )
    horizontal_forward = math.hypot(head_forward.x, head_forward.y)
    level_tolerance = math.cos(math.radians(20.0))
    if horizontal_forward < level_tolerance or head_up.z < level_tolerance:
        raise WebVRProtocolError(
            "head must face horizontally forward within 20 degrees during calibration"
        )
    yaw = math.atan2(head_forward.y, head_forward.x)
    half_yaw = -0.5 * yaw
    frame_rotation = Quaternion(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    head_position = converted["Head"].position
    forward_distances: list[float] = []
    hand_heights: list[float] = []
    for name in ("LeftHand", "RightHand"):
        hand_position = converted[name].position
        relative = _rotate_vector(
            frame_rotation,
            Vector3(
                hand_position.x - head_position.x,
                hand_position.y - head_position.y,
                hand_position.z - head_position.z,
            ),
        )
        forward_distances.append(relative.x)
        hand_heights.append(relative.z)
    if min(forward_distances) < 0.25:
        raise WebVRProtocolError(
            "both hands must be extended horizontally forward before calibration"
        )
    average_arm_distance = 0.5 * sum(forward_distances)
    if abs(forward_distances[0] - forward_distances[1]) > max(
        0.15,
        0.3 * average_arm_distance,
    ):
        raise WebVRProtocolError(
            "left and right hands must be extended by similar distances"
        )
    if abs(hand_heights[0] - hand_heights[1]) > 0.12:
        raise WebVRProtocolError("left and right hands must be held at the same height")
    scale = arm_length / average_arm_distance
    targets = {
        "Head": head_target,
        "LeftHand": left_hand_target,
        "RightHand": right_hand_target,
    }
    target_quaternions = {
        "Head": head_quaternion,
        "LeftHand": left_hand_quaternion,
        "RightHand": right_hand_quaternion,
    }
    position_offsets: dict[str, Vector3] = {}
    quaternion_offsets: dict[str, Quaternion] = {}
    for name in POSE_NAMES:
        aligned_position = _rotate_vector(
            frame_rotation,
            Vector3(
                converted[name].position.x * scale,
                converted[name].position.y * scale,
                converted[name].position.z * scale,
            ),
        )
        position_offsets[name] = Vector3(
            targets[name].x - aligned_position.x,
            targets[name].y - aligned_position.y,
            targets[name].z - aligned_position.z,
        )
        aligned_quaternion = _multiply_quaternions(
            frame_rotation,
            converted[name].quaternion,
        )
        quaternion_offsets[name] = _multiply_quaternions(
            _inverse_quaternion(aligned_quaternion),
            target_quaternions[name],
        )
    return WebVRCalibration(
        scale=scale,
        position_rotation=frame_rotation,
        position_offsets=position_offsets,
        quaternion_offsets=quaternion_offsets,
    )


def validate_zero_pose_sample(sample: WebVRSample) -> None:
    """Require enough horizontal hand separation for a stable calibration frame."""

    _validate_calibration_tracking(sample)

    converted = {name: vr_pose_to_ros(sample.poses[name]) for name in POSE_NAMES}
    left = converted["LeftHand"].position
    right = converted["RightHand"].position
    horizontal_separation = math.hypot(left.x - right.x, left.y - right.y)
    if horizontal_separation < 0.05:
        raise WebVRProtocolError(
            "left and right hands must be separated before zero-pose calibration"
        )


def _validate_calibration_tracking(sample: WebVRSample) -> None:
    for name in ("LeftHand", "RightHand"):
        tracking = sample.poses[name].tracking
        if not tracking.position_is_usable:
            raise WebVRProtocolError(
                f"{name} position tracking must be Known or Inferred during calibration "
                f"(got {tracking.position})"
            )
