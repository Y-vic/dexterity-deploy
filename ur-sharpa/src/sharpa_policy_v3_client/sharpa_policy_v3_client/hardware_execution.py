"""Controller-independent validation and scheduling for policy-v3 actions."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol

import numpy as np

from .hardware_geometry import (
    SIDES,
    UrSharpAWireGeometry,
    relative_wire_pose_to_absolute,
    rtde_pose_to_transform,
)


class ActionExecutionError(RuntimeError):
    """A policy action cannot be executed safely."""


@dataclass(frozen=True)
class ExecutionLimits:
    workspace_min: tuple[float, float, float] = (0.0, -0.5, 0.03)
    workspace_max: tuple[float, float, float] = (0.6, 0.5, 0.65)
    max_translation_step_m: float = 0.035
    max_rotation_step_rad: float = 0.45
    max_joint_step_rad: float = 0.35
    max_hand_step_rad: float = 0.5
    max_abs_hand_rad: float = 3.5
    max_frequency_hz: float = 60.0

    def __post_init__(self) -> None:
        minimum = np.asarray(self.workspace_min, dtype=np.float64)
        maximum = np.asarray(self.workspace_max, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            raise ValueError("workspace bounds must be increasing XYZ triples")
        positive = (
            self.max_translation_step_m,
            self.max_rotation_step_rad,
            self.max_joint_step_rad,
            self.max_hand_step_rad,
            self.max_abs_hand_rad,
            self.max_frequency_hz,
        )
        if not np.isfinite(positive).all() or any(value <= 0.0 for value in positive):
            raise ValueError("execution limits must be finite and positive")


@dataclass(frozen=True)
class HardwareSnapshot:
    left_rtde_pose: np.ndarray
    right_rtde_pose: np.ndarray
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    normal_mode: bool = True


@dataclass(frozen=True)
class ExecutableAction:
    action_id: str
    frequency_hz: float
    execute_start: int
    execute_length: int
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    hand_joint: np.ndarray


@dataclass(frozen=True)
class ExecutionResult:
    executed_steps: int
    success: bool
    failure_code: str = ""
    failure_message: str = ""


class HardwareCommandBackend(Protocol):
    def snapshot(self) -> HardwareSnapshot: ...

    def inverse_kinematics(
        self, side: str, target_rtde_pose: np.ndarray, near_joint: np.ndarray
    ) -> np.ndarray: ...

    def command_arms(
        self, left_joint: np.ndarray, right_joint: np.ndarray, period_s: float
    ) -> tuple[bool, bool]: ...

    def command_hands(
        self, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> tuple[bool, bool]: ...

    def safe_stop(self) -> None: ...


def _float_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ActionExecutionError(f"{label} must be finite float32{shape}")
    return np.array(array, dtype=np.float32, order="C", copy=True)


def prepare_executable_action(
    *,
    action_id: str,
    frequency_hz: float,
    action_length: int,
    execute_start: int,
    execute_length: int,
    left_wrist_action_type: str,
    right_wrist_action_type: str,
    left_wrist: Any,
    right_wrist: Any,
    hand_joint: Any,
    current_left_wire: np.ndarray,
    current_right_wire: np.ndarray,
    max_frequency_hz: float = 60.0,
) -> ExecutableAction:
    if not isinstance(action_id, str) or not action_id:
        raise ActionExecutionError("action_id must be nonempty")
    if action_length <= 0 or execute_length <= 0:
        raise ActionExecutionError("action and execution lengths must be positive")
    if execute_start < 0 or execute_start + execute_length > action_length:
        raise ActionExecutionError("execution slice exceeds action length")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ActionExecutionError("frequency_hz must be finite and positive")
    if frequency_hz > max_frequency_hz:
        raise ActionExecutionError(
            f"frequency_hz {frequency_hz:g} exceeds hardware limit {max_frequency_hz:g}"
        )
    wrists = []
    for side, action_type, value, current in (
        ("left", left_wrist_action_type, left_wrist, current_left_wire),
        ("right", right_wrist_action_type, right_wrist, current_right_wire),
    ):
        if action_type not in {"eef", "relative_eef"}:
            raise ActionExecutionError(
                f"{side} wrist action type {action_type!r} is not supported by the UR bridge"
            )
        wrist = _float_matrix(value, (action_length, 9), f"{side}_wrist")
        if action_type == "relative_eef":
            try:
                wrist = relative_wire_pose_to_absolute(wrist, current)
            except ValueError as exc:
                raise ActionExecutionError(f"invalid {side} relative EEF: {exc}") from exc
        wrists.append(wrist[execute_start : execute_start + execute_length])
    hands = _float_matrix(hand_joint, (action_length, 44), "hand_joint")
    hands = hands[execute_start : execute_start + execute_length]
    return ExecutableAction(
        action_id=action_id,
        frequency_hz=float(frequency_hz),
        execute_start=execute_start,
        execute_length=execute_length,
        left_wrist=wrists[0],
        right_wrist=wrists[1],
        hand_joint=hands,
    )


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def _unwrap_to_near(joint: np.ndarray, near: np.ndarray) -> np.ndarray:
    return joint + np.round((near - joint) / (2.0 * np.pi)) * (2.0 * np.pi)


class PolicyActionExecutor:
    """Execute one validated slice while enforcing robot-side safety limits."""

    def __init__(
        self,
        backend: HardwareCommandBackend,
        *,
        geometry: UrSharpAWireGeometry | None = None,
        limits: ExecutionLimits | None = None,
        enabled: bool = False,
    ) -> None:
        self.backend = backend
        self.geometry = geometry or UrSharpAWireGeometry()
        self.limits = limits or ExecutionLimits()
        self.enabled = bool(enabled)

    def current_wire_poses(self, snapshot: HardwareSnapshot) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.geometry.rtde_pose_to_wire_pose(snapshot.left_rtde_pose, "left"),
            self.geometry.rtde_pose_to_wire_pose(snapshot.right_rtde_pose, "right"),
        )

    def execute(
        self,
        action: ExecutableAction,
        cancel: threading.Event | None = None,
    ) -> ExecutionResult:
        cancel = cancel or threading.Event()
        if not self.enabled:
            return ExecutionResult(0, False, "execution_disabled", "hardware execution is disabled")
        period_s = 1.0 / action.frequency_hz
        executed_steps = 0
        try:
            snapshot = self._snapshot()
            previous_capture = {
                "left": self.geometry.ur_base_tcp_to_capture_tcp(
                    rtde_pose_to_transform(snapshot.left_rtde_pose), "left"
                ),
                "right": self.geometry.ur_base_tcp_to_capture_tcp(
                    rtde_pose_to_transform(snapshot.right_rtde_pose), "right"
                ),
            }
            previous_joint = {
                "left": np.asarray(snapshot.left_joint, dtype=np.float64),
                "right": np.asarray(snapshot.right_joint, dtype=np.float64),
            }
            previous_hand = np.concatenate((snapshot.left_hand, snapshot.right_hand)).astype(
                np.float64
            )
            for index in range(action.execute_length):
                if cancel.is_set():
                    raise ActionExecutionError("execution cancelled")
                started = time.monotonic()
                targets = {
                    "left": self.geometry.wire_pose_to_capture_tcp(
                        action.left_wrist[index], "left"
                    ),
                    "right": self.geometry.wire_pose_to_capture_tcp(
                        action.right_wrist[index], "right"
                    ),
                }
                target_joints: dict[str, np.ndarray] = {}
                for side in SIDES:
                    self._validate_capture_step(previous_capture[side], targets[side], side)
                    rtde_target = self.geometry.wire_pose_to_rtde_pose(
                        action.left_wrist[index] if side == "left" else action.right_wrist[index],
                        side,
                    )
                    solved = np.asarray(
                        self.backend.inverse_kinematics(side, rtde_target, previous_joint[side]),
                        dtype=np.float64,
                    )
                    if solved.shape != (6,) or not np.isfinite(solved).all():
                        raise ActionExecutionError(f"{side} IK did not return finite joint[6]")
                    solved = _unwrap_to_near(solved, previous_joint[side])
                    if np.max(np.abs(solved - previous_joint[side])) > self.limits.max_joint_step_rad:
                        raise ActionExecutionError(f"{side} IK exceeds per-step joint limit")
                    target_joints[side] = solved
                hand = action.hand_joint[index].astype(np.float64)
                if np.max(np.abs(hand)) > self.limits.max_abs_hand_rad:
                    raise ActionExecutionError("hand target exceeds absolute joint limit")
                if np.max(np.abs(hand - previous_hand)) > self.limits.max_hand_step_rad:
                    raise ActionExecutionError("hand target exceeds per-step joint limit")
                arm_ok = self.backend.command_arms(
                    target_joints["left"], target_joints["right"], period_s
                )
                hand_ok = self.backend.command_hands(hand[:22], hand[22:])
                if arm_ok != (True, True):
                    raise ActionExecutionError(f"UR servo command failed: {arm_ok}")
                if hand_ok != (True, True):
                    raise ActionExecutionError(f"SharpA command failed: {hand_ok}")
                executed_steps += 1
                previous_capture = targets
                previous_joint = target_joints
                previous_hand = hand
                remaining = period_s - (time.monotonic() - started)
                if remaining > 0.0 and cancel.wait(remaining):
                    raise ActionExecutionError("execution cancelled")
            return ExecutionResult(executed_steps, True)
        except Exception as exc:
            self.backend.safe_stop()
            code = "execution_cancelled" if cancel.is_set() else "execution_failed"
            return ExecutionResult(executed_steps, False, code, str(exc))

    def _snapshot(self) -> HardwareSnapshot:
        snapshot = self.backend.snapshot()
        if not snapshot.normal_mode:
            raise ActionExecutionError("UR pair is not in normal running mode")
        for name in (
            "left_rtde_pose",
            "right_rtde_pose",
            "left_joint",
            "right_joint",
            "left_hand",
            "right_hand",
        ):
            value = np.asarray(getattr(snapshot, name))
            expected = (22,) if "hand" in name else (6,)
            if value.shape != expected or not np.isfinite(value).all():
                raise ActionExecutionError(f"snapshot {name} must be finite{expected}")
        return snapshot

    def _validate_capture_step(
        self, previous: np.ndarray, target: np.ndarray, side: str
    ) -> None:
        position = target[:3, 3]
        minimum = np.asarray(self.limits.workspace_min)
        maximum = np.asarray(self.limits.workspace_max)
        if np.any(position < minimum) or np.any(position > maximum):
            raise ActionExecutionError(f"{side} target is outside the shared workspace")
        translation = float(np.linalg.norm(position - previous[:3, 3]))
        if translation > self.limits.max_translation_step_m:
            raise ActionExecutionError(
                f"{side} translation step {translation:.4f} m exceeds limit"
            )
        rotation = _rotation_distance(previous[:3, :3], target[:3, :3])
        if rotation > self.limits.max_rotation_step_rad:
            raise ActionExecutionError(
                f"{side} rotation step {rotation:.4f} rad exceeds limit"
            )
