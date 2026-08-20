"""Self-contained adapters for the local UR, SharpA, and ZED hardware."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import importlib
from numbers import Integral
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np

from .hardware_geometry import (
    SIDES,
    UrSharpAWireGeometry,
    column_pose9_to_transform,
    rtde_pose_to_transform,
)


FINGER_NAMES = ("pinky", "ring", "middle", "index", "thumb")
TACTILE_CHANNELS = {
    "left": (5, 6, 7, 8, 9),
    "right": (0, 1, 2, 3, 4),
}


class HardwareDependencyError(RuntimeError):
    pass


class HardwareConnectionError(RuntimeError):
    pass


class HardwareCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class UrPairSnapshot:
    timestamp_ns: int
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_rtde_pose: np.ndarray
    right_rtde_pose: np.ndarray
    left_wire_eef: np.ndarray
    right_wire_eef: np.ndarray
    normal_mode: bool


@dataclass(frozen=True)
class SharpASnapshot:
    timestamp_ns: int
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_tau: np.ndarray
    right_tau: np.ndarray
    left_joint_valid: bool
    right_joint_valid: bool
    left_tau_valid: np.ndarray
    right_tau_valid: np.ndarray
    left_wrench: np.ndarray
    right_wrench: np.ndarray
    left_wrench_valid: np.ndarray
    right_wrench_valid: np.ndarray
    left_deformation: np.ndarray
    right_deformation: np.ndarray
    left_deformation_valid: np.ndarray
    right_deformation_valid: np.ndarray


@dataclass(frozen=True)
class CameraSnapshot:
    timestamp_ns: int
    jpeg: bytes


def _finite_vector(value: Any, dimension: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise HardwareCommandError(f"{label} must be finite[{dimension}]")
    return result


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def _unwrap_to_near(joint: np.ndarray, near: np.ndarray) -> np.ndarray:
    return joint + np.round((near - joint) / (2.0 * np.pi)) * (2.0 * np.pi)


class UrRtdePair:
    def __init__(
        self,
        *,
        left_ip: str = "192.168.56.20",
        right_ip: str = "192.168.56.10",
        frequency_hz: float = 125.0,
        enable_control: bool = False,
        geometry: UrSharpAWireGeometry | None = None,
        workspace_min: tuple[float, float, float] = (0.0, -0.5, 0.03),
        workspace_max: tuple[float, float, float] = (0.6, 0.5, 0.65),
        max_translation_step_m: float = 0.035,
        max_rotation_step_rad: float = 0.45,
        max_joint_step_rad: float = 0.35,
        control_connection_delay_s: float = 0.5,
        control_ready_timeout_s: float = 10.0,
        control_ready_poll_s: float = 0.1,
    ) -> None:
        self.ips = {"left": left_ip, "right": right_ip}
        self.frequency_hz = float(frequency_hz)
        self.enable_control = bool(enable_control)
        self.geometry = geometry or UrSharpAWireGeometry()
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.max_translation_step_m = float(max_translation_step_m)
        self.max_rotation_step_rad = float(max_rotation_step_rad)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.control_connection_delay_s = float(control_connection_delay_s)
        self.control_ready_timeout_s = float(control_ready_timeout_s)
        self.control_ready_poll_s = float(control_ready_poll_s)
        if self.control_ready_timeout_s <= 0.0:
            raise ValueError("UR control ready timeout must be positive")
        if self.control_ready_poll_s <= 0.0:
            raise ValueError("UR control ready poll interval must be positive")
        self.receivers: dict[str, Any] = {}
        self.controllers: dict[str, Any] = {}
        self.last_target_joint: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._command_lock = threading.Lock()

    def connect(self) -> None:
        try:
            receive_module = importlib.import_module("rtde_receive")
            control_module = importlib.import_module("rtde_control")
        except ImportError as exc:
            raise HardwareDependencyError("install ur-rtde for Python 3") from exc
        try:
            flags = (
                control_module.RTDEControlInterface.FLAG_VERBOSE
                | control_module.RTDEControlInterface.FLAG_UPLOAD_SCRIPT
                | control_module.RTDEControlInterface.FLAG_NO_WAIT
            )
            for side in SIDES:
                self.receivers[side] = receive_module.RTDEReceiveInterface(
                    self.ips[side],
                    self.frequency_hz,
                    [],
                    True,
                    False,
                    -1,
                )
                if self.enable_control:
                    time.sleep(self.control_connection_delay_s)
                    controller = control_module.RTDEControlInterface(
                        self.ips[side],
                        self.frequency_hz,
                        flags,
                        50013,
                        -1,
                    )
                    self.controllers[side] = controller
                    self._wait_for_control_program(controller, side)
        except Exception as exc:
            self.close()
            raise HardwareConnectionError(f"failed to connect UR pair: {exc}") from exc

    def _wait_for_control_program(self, controller: Any, side: str) -> None:
        deadline = time.monotonic() + self.control_ready_timeout_s
        reupload_attempted = False
        last_error: Exception | None = None
        while True:
            try:
                if not controller.isConnected():
                    last_error = RuntimeError("RTDE control socket is disconnected")
                elif controller.isProgramRunning():
                    return
                elif not reupload_attempted:
                    reupload_attempted = True
                    if not controller.reuploadScript():
                        last_error = RuntimeError("RTDE control script reupload failed")
            except Exception as exc:
                last_error = exc
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                detail = f": {last_error}" if last_error is not None else ""
                raise HardwareConnectionError(
                    f"{side} UR RTDE control program did not become ready"
                    f" within {self.control_ready_timeout_s:.1f}s{detail}"
                )
            time.sleep(min(self.control_ready_poll_s, remaining_s))

    def read(self) -> UrPairSnapshot:
        if set(self.receivers) != set(SIDES):
            raise HardwareConnectionError("UR receive interfaces are not connected")
        joints: dict[str, np.ndarray] = {}
        poses: dict[str, np.ndarray] = {}
        wire: dict[str, np.ndarray] = {}
        normal = True
        for side in SIDES:
            receiver = self.receivers[side]
            joints[side] = _finite_vector(receiver.getActualQ(), 6, f"{side} joint")
            poses[side] = _finite_vector(
                receiver.getActualTCPPose(), 6, f"{side} RTDE pose"
            )
            wire[side] = self.geometry.rtde_pose_to_wire_pose(poses[side], side)
            normal = normal and self._normal_mode(receiver)
        return UrPairSnapshot(
            timestamp_ns=time.time_ns(),
            left_joint=joints["left"].astype(np.float32),
            right_joint=joints["right"].astype(np.float32),
            left_rtde_pose=poses["left"],
            right_rtde_pose=poses["right"],
            left_wire_eef=wire["left"],
            right_wire_eef=wire["right"],
            normal_mode=normal,
        )

    @staticmethod
    def _normal_mode(receiver: Any) -> bool:
        try:
            if hasattr(receiver, "isConnected") and not receiver.isConnected():
                return False
            if hasattr(receiver, "isProtectiveStopped") and receiver.isProtectiveStopped():
                return False
            if hasattr(receiver, "isEmergencyStopped") and receiver.isEmergencyStopped():
                return False
            if hasattr(receiver, "getSafetyMode") and int(receiver.getSafetyMode()) != 1:
                return False
            if hasattr(receiver, "getRobotMode") and int(receiver.getRobotMode()) != 7:
                return False
        except Exception:
            return False
        return True

    def command_eef_step(
        self,
        left_wire_eef: Any,
        right_wire_eef: Any,
        period_s: float,
    ) -> None:
        if not self.enable_control or set(self.controllers) != set(SIDES):
            raise HardwareCommandError("UR hardware execution is disabled")
        if not np.isfinite(period_s) or period_s <= 0.0 or period_s > 0.2:
            raise HardwareCommandError("UR command period must be in (0, 0.2] seconds")
        wire = {
            "left": np.asarray(left_wire_eef, dtype=np.float32),
            "right": np.asarray(right_wire_eef, dtype=np.float32),
        }
        if wire["left"].shape != (9,) or wire["right"].shape != (9,):
            raise HardwareCommandError("UR EEF commands must be pose9")
        with self._command_lock:
            snapshot = self.read()
            if not snapshot.normal_mode:
                raise HardwareCommandError("UR pair is not in normal running mode")
            target_joint: dict[str, np.ndarray] = {}
            for side in SIDES:
                self._validate_cartesian_step(snapshot, wire[side], side)
                target_rtde = self.geometry.wire_pose_to_rtde_pose(wire[side], side)
                current_joint = (
                    snapshot.left_joint if side == "left" else snapshot.right_joint
                ).astype(np.float64)
                near = self.last_target_joint[side]
                if near is None:
                    near = current_joint
                solved = self.controllers[side].getInverseKinematics(
                    target_rtde.tolist(), near.tolist()
                )
                solved = _finite_vector(solved, 6, f"{side} IK")
                solved = _unwrap_to_near(solved, near)
                if np.any(np.abs(solved) > 2.0 * np.pi):
                    raise HardwareCommandError(f"{side} IK exceeds joint position limits")
                if np.max(np.abs(solved - near)) > self.max_joint_step_rad:
                    raise HardwareCommandError(f"{side} IK exceeds joint step limit")
                target_joint[side] = solved
            try:
                for side in SIDES:
                    self.controllers[side].servoJ(
                        target_joint[side].tolist(),
                        0.5,
                        0.5,
                        float(period_s),
                        0.2,
                        100.0,
                    )
                    self.last_target_joint[side] = target_joint[side].copy()
            except Exception as exc:
                self.safe_stop()
                raise HardwareCommandError(f"UR servoJ failed: {exc}") from exc

    def initialize_joints(
        self,
        left_joint: Any,
        right_joint: Any,
        *,
        steps: int = 120,
        step_delay_s: float = 0.02,
        tolerance_rad: float = 0.08,
    ) -> None:
        if not self.enable_control or set(self.controllers) != set(SIDES):
            raise HardwareCommandError("UR hardware execution is disabled")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) <= 0:
            raise HardwareCommandError("UR initialization steps must be positive")
        if not np.isfinite(step_delay_s) or step_delay_s < 0.0:
            raise HardwareCommandError("UR initialization delay must be nonnegative")
        if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise HardwareCommandError("UR initialization tolerance must be positive")
        targets = {
            "left": _finite_vector(left_joint, 6, "left initial joint"),
            "right": _finite_vector(right_joint, 6, "right initial joint"),
        }
        for side in SIDES:
            if np.any(np.abs(targets[side]) > 2.0 * np.pi):
                raise HardwareCommandError(
                    f"{side} initial joint exceeds position limits"
                )

        with self._command_lock:
            snapshot = self.read()
            if not snapshot.normal_mode:
                raise HardwareCommandError("UR pair is not in normal running mode")
            starts = {
                "left": snapshot.left_joint.astype(np.float64),
                "right": snapshot.right_joint.astype(np.float64),
            }
            try:
                for step_index in range(1, int(steps) + 1):
                    alpha = step_index / int(steps)
                    for side in SIDES:
                        target = starts[side] + alpha * (
                            targets[side] - starts[side]
                        )
                        self.controllers[side].servoJ(
                            target.tolist(),
                            0.5,
                            0.5,
                            max(float(step_delay_s), 1.0 / self.frequency_hz),
                            0.2,
                            100.0,
                        )
                    if step_delay_s > 0.0:
                        time.sleep(float(step_delay_s))
                for controller in self.controllers.values():
                    controller.servoStop()
                reached = self.read()
                reached_joints = {
                    "left": reached.left_joint.astype(np.float64),
                    "right": reached.right_joint.astype(np.float64),
                }
                for side in SIDES:
                    actual = _unwrap_to_near(reached_joints[side], targets[side])
                    if np.max(np.abs(actual - targets[side])) > tolerance_rad:
                        raise HardwareCommandError(
                            f"{side} UR did not reach its initial joint target"
                        )
                    self.last_target_joint[side] = targets[side].copy()
            except Exception as exc:
                self.safe_stop()
                if isinstance(exc, HardwareCommandError):
                    raise
                raise HardwareCommandError(
                    f"UR joint initialization failed: {exc}"
                ) from exc

    def _validate_cartesian_step(
        self,
        snapshot: UrPairSnapshot,
        target_wire: np.ndarray,
        side: str,
    ) -> None:
        current_rtde = (
            snapshot.left_rtde_pose if side == "left" else snapshot.right_rtde_pose
        )
        current_capture = self.geometry.ur_base_tcp_to_capture_tcp(
            rtde_pose_to_transform(current_rtde), side
        )
        target_capture = self.geometry.wire_pose_to_capture_tcp(target_wire, side)
        position = target_capture[:3, 3]
        if np.any(position < self.workspace_min) or np.any(position > self.workspace_max):
            raise HardwareCommandError(f"{side} EEF target is outside workspace")
        translation = float(np.linalg.norm(position - current_capture[:3, 3]))
        if translation > self.max_translation_step_m:
            raise HardwareCommandError(f"{side} EEF translation step is too large")
        rotation = _rotation_distance(
            current_capture[:3, :3], target_capture[:3, :3]
        )
        if rotation > self.max_rotation_step_rad:
            raise HardwareCommandError(f"{side} EEF rotation step is too large")

    def safe_stop(self) -> None:
        for controller in tuple(self.controllers.values()):
            try:
                controller.servoStop()
            except Exception:
                pass

    def close(self) -> None:
        self.safe_stop()
        for controller in tuple(self.controllers.values()):
            try:
                controller.stopScript()
            except Exception:
                pass
            try:
                controller.disconnect()
            except Exception:
                pass
        for receiver in tuple(self.receivers.values()):
            try:
                receiver.disconnect()
            except Exception:
                pass
        self.controllers.clear()
        self.receivers.clear()


def load_sharpa_sdk(sdk_root: str | Path) -> ModuleType:
    root = Path(sdk_root).expanduser().resolve()
    python_dir = root / "python"
    if not python_dir.is_dir():
        raise HardwareDependencyError(f"SharpA SDK python directory not found: {python_dir}")
    path = str(python_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        return importlib.import_module("sharpa")
    except Exception as exc:
        raise HardwareDependencyError(f"failed to load SharpA SDK from {root}: {exc}") from exc


class SharpADevicePair:
    def __init__(
        self,
        *,
        sdk_root: str | Path,
        discovery_timeout_s: float = 5.0,
        speed_coefficient: float = 0.5,
        current_coefficient: float = 0.6,
        enable_control: bool = False,
        max_hand_step_rad: float = 0.5,
        max_abs_hand_rad: float = 3.5,
    ) -> None:
        self.sdk_root = Path(sdk_root)
        self.discovery_timeout_s = float(discovery_timeout_s)
        self.speed_coefficient = float(speed_coefficient)
        self.current_coefficient = float(current_coefficient)
        self.enable_control = bool(enable_control)
        self.max_hand_step_rad = float(max_hand_step_rad)
        self.max_abs_hand_rad = float(max_abs_hand_rad)
        self.sdk: ModuleType | None = None
        self.manager: Any = None
        self.hands: dict[str, Any] = {}
        self._tactile: dict[str, dict[int, dict[str, np.ndarray]]] = {
            "left": {},
            "right": {},
        }
        self._tactile_lock = threading.Lock()
        self._command_lock = threading.Lock()

    def connect(self) -> None:
        self.sdk = load_sharpa_sdk(self.sdk_root)
        self.manager = self.sdk.SharpaWaveManager.get_instance()
        deadline = time.monotonic() + self.discovery_timeout_s
        devices: list[Any] = []
        while time.monotonic() < deadline:
            devices = [
                device
                for device in self.manager.get_all_devices()
                if device.device_type == self.sdk.DeviceType.HAND
            ]
            sides = {self._device_side(device) for device in devices}
            if sides == set(SIDES):
                break
            time.sleep(0.1)
        for device in devices:
            side = self._device_side(device)
            if side in self.hands:
                continue
            hand = self.manager.connect(device.sn)
            self._check_error(hand.set_control_mode(self.sdk.ControlMode.POSITION))
            self._check_error(hand.set_control_source(self.sdk.ControlSource.SDK))
            self._check_error(hand.set_speed_coeff(self.speed_coefficient))
            self._check_error(hand.set_current_coeff(self.current_coefficient))
            hand.set_tactile_callback(self._tactile_callback(side))
            if not hand.start():
                raise HardwareConnectionError(f"SharpA {side} hand failed to start")
            self.hands[side] = hand
        if set(self.hands) != set(SIDES):
            self.close()
            raise HardwareConnectionError("both left and right SharpA hands are required")

    def _device_side(self, device: Any) -> str:
        assert self.sdk is not None
        return "left" if device.hand_side == self.sdk.HandSide.LEFT else "right"

    @staticmethod
    def _check_error(error: Any) -> None:
        if getattr(error, "code", -1) != 0:
            raise HardwareConnectionError(getattr(error, "message", "SharpA SDK error"))

    def _tactile_callback(self, side: str) -> Any:
        def callback(frames: Any) -> None:
            try:
                channel = int(frames["channel"])
                content = frames["content"]
                stored: dict[str, np.ndarray] = {}
                wrench = content.get("F6")
                if wrench is not None:
                    array = np.asarray(wrench, dtype=np.float32).reshape(-1)
                    if array.shape == (6,) and np.isfinite(array).all():
                        stored["wrench"] = array.copy()
                deformation = content.get("DEFORM")
                if deformation is not None:
                    array = np.asarray(deformation, dtype=np.uint8).reshape(-1)
                    if array.size == 240 * 240:
                        stored["deformation"] = array.reshape(240, 240).copy()
                with self._tactile_lock:
                    self._tactile[side][channel] = stored
            except Exception:
                return

        return callback

    def read(self) -> SharpASnapshot:
        joint = {side: np.zeros(22, dtype=np.float32) for side in SIDES}
        tau = {side: np.zeros(22, dtype=np.float32) for side in SIDES}
        joint_valid = {side: False for side in SIDES}
        tau_valid = {side: np.zeros(22, dtype=np.bool_) for side in SIDES}
        for side in SIDES:
            try:
                state = self.hands[side].get_states()
                angles = np.asarray(state.angles, dtype=np.float32)
                torques = np.asarray(state.torques, dtype=np.float32)
                if angles.shape == (22,) and np.isfinite(angles).all():
                    joint[side] = angles
                    joint_valid[side] = True
                if torques.shape == (22,) and np.isfinite(torques).all():
                    tau[side] = torques
                    tau_valid[side][:] = True
            except Exception:
                continue
        wrench = {side: np.zeros((5, 6), dtype=np.float32) for side in SIDES}
        deformation = {
            side: np.zeros((5, 240, 240), dtype=np.uint8) for side in SIDES
        }
        wrench_valid = {side: np.zeros(5, dtype=np.bool_) for side in SIDES}
        deformation_valid = {side: np.zeros(5, dtype=np.bool_) for side in SIDES}
        with self._tactile_lock:
            tactile = {
                side: {channel: values.copy() for channel, values in data.items()}
                for side, data in self._tactile.items()
            }
        for side in SIDES:
            for index, channel in enumerate(TACTILE_CHANNELS[side]):
                values = tactile[side].get(channel, {})
                if "wrench" in values:
                    wrench[side][index] = values["wrench"]
                    wrench_valid[side][index] = True
                if "deformation" in values:
                    deformation[side][index] = values["deformation"]
                    deformation_valid[side][index] = True
        return SharpASnapshot(
            timestamp_ns=time.time_ns(),
            left_joint=joint["left"],
            right_joint=joint["right"],
            left_tau=tau["left"],
            right_tau=tau["right"],
            left_joint_valid=joint_valid["left"],
            right_joint_valid=joint_valid["right"],
            left_tau_valid=tau_valid["left"],
            right_tau_valid=tau_valid["right"],
            left_wrench=wrench["left"],
            right_wrench=wrench["right"],
            left_wrench_valid=wrench_valid["left"],
            right_wrench_valid=wrench_valid["right"],
            left_deformation=deformation["left"],
            right_deformation=deformation["right"],
            left_deformation_valid=deformation_valid["left"],
            right_deformation_valid=deformation_valid["right"],
        )

    def command(self, left_joint: Any, right_joint: Any) -> None:
        if not self.enable_control:
            raise HardwareCommandError("SharpA hardware execution is disabled")
        targets = {
            "left": _finite_vector(left_joint, 22, "left hand target"),
            "right": _finite_vector(right_joint, 22, "right hand target"),
        }
        current = self.read()
        current_joint = {
            "left": current.left_joint.astype(np.float64),
            "right": current.right_joint.astype(np.float64),
        }
        with self._command_lock:
            for side in SIDES:
                if np.max(np.abs(targets[side])) > self.max_abs_hand_rad:
                    raise HardwareCommandError(f"{side} hand target exceeds absolute limit")
                step = np.abs(targets[side] - current_joint[side])
                if np.max(step) > self.max_hand_step_rad:
                    joint_index = int(np.argmax(step))
                    raise HardwareCommandError(
                        f"{side} hand target exceeds step limit: joint {joint_index} "
                        f"delta={step[joint_index]:.6f} rad "
                        f"limit={self.max_hand_step_rad:.6f} rad"
                    )
            for side in SIDES:
                error = self.hands[side].set_joint_position(targets[side].tolist(), False)
                if getattr(error, "code", -1) != 0:
                    self.hold()
                    raise HardwareCommandError(
                        f"SharpA {side} command failed: {getattr(error, 'message', '')}"
                    )

    def initialize_joints(
        self,
        left_joint: Any,
        right_joint: Any,
        *,
        steps: int = 60,
        step_delay_s: float = 0.02,
        tolerance_rad: float = 0.08,
    ) -> None:
        if not self.enable_control or set(self.hands) != set(SIDES):
            raise HardwareCommandError("SharpA hardware execution is disabled")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) <= 0:
            raise HardwareCommandError("SharpA initialization steps must be positive")
        if not np.isfinite(step_delay_s) or step_delay_s < 0.0:
            raise HardwareCommandError("SharpA initialization delay must be nonnegative")
        if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise HardwareCommandError("SharpA initialization tolerance must be positive")
        targets = {
            "left": _finite_vector(left_joint, 22, "left initial hand joint"),
            "right": _finite_vector(right_joint, 22, "right initial hand joint"),
        }
        for side in SIDES:
            if np.max(np.abs(targets[side])) > self.max_abs_hand_rad:
                raise HardwareCommandError(
                    f"{side} initial hand joint exceeds absolute limit"
                )

        current = self.read()
        if not current.left_joint_valid or not current.right_joint_valid:
            raise HardwareCommandError("SharpA initial joint state is unavailable")
        starts = {
            "left": current.left_joint.astype(np.float64),
            "right": current.right_joint.astype(np.float64),
        }
        with self._command_lock:
            try:
                for step_index in range(1, int(steps) + 1):
                    alpha = step_index / int(steps)
                    for side in SIDES:
                        target = starts[side] + alpha * (
                            targets[side] - starts[side]
                        )
                        error = self.hands[side].set_joint_position(
                            target.tolist(),
                            False,
                        )
                        if getattr(error, "code", -1) != 0:
                            raise HardwareCommandError(
                                f"SharpA {side} initialization failed: "
                                f"{getattr(error, 'message', '')}"
                            )
                    if step_delay_s > 0.0:
                        time.sleep(float(step_delay_s))
                reached = self.read()
                if not reached.left_joint_valid or not reached.right_joint_valid:
                    raise HardwareCommandError(
                        "SharpA initialized joint state is unavailable"
                    )
                reached_joints = {
                    "left": reached.left_joint.astype(np.float64),
                    "right": reached.right_joint.astype(np.float64),
                }
                for side in SIDES:
                    if np.max(np.abs(reached_joints[side] - targets[side])) > tolerance_rad:
                        raise HardwareCommandError(
                            f"{side} SharpA did not reach its initial joint target"
                        )
            except Exception as exc:
                self.hold()
                if isinstance(exc, HardwareCommandError):
                    raise
                raise HardwareCommandError(
                    f"SharpA joint initialization failed: {exc}"
                ) from exc

    def hold(self) -> None:
        if not self.enable_control:
            return
        try:
            snapshot = self.read()
            if snapshot.left_joint_valid:
                self.hands["left"].set_joint_position(snapshot.left_joint.tolist(), False)
            if snapshot.right_joint_valid:
                self.hands["right"].set_joint_position(snapshot.right_joint.tolist(), False)
        except Exception:
            pass

    def close(self) -> None:
        self.hold()
        for hand in tuple(self.hands.values()):
            try:
                hand.stop()
            except Exception:
                pass
        self.hands.clear()
        if self.manager is not None:
            try:
                self.manager.disconnect_all()
            except Exception:
                pass
        self.manager = None


class ZedLeftCamera:
    def __init__(
        self,
        *,
        resolution: str = "HD720",
        frequency_hz: int = 30,
        serial_number: int | None = None,
        jpeg_quality: int = 90,
    ) -> None:
        self.resolution = resolution
        self.frequency_hz = int(frequency_hz)
        self.serial_number = serial_number
        self.jpeg_quality = int(jpeg_quality)
        self.sl: Any = None
        self.camera: Any = None
        self.runtime: Any = None
        self.image: Any = None
        self._latest: CameraSnapshot | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        try:
            self.sl = importlib.import_module("pyzed.sl")
            importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise HardwareDependencyError("ZED SDK and Pillow are required") from exc
        self.camera = self.sl.Camera()
        init = self.sl.InitParameters()
        try:
            init.camera_resolution = getattr(self.sl.RESOLUTION, self.resolution)
        except AttributeError as exc:
            raise ValueError(f"unsupported ZED resolution {self.resolution}") from exc
        init.camera_fps = self.frequency_hz
        init.depth_mode = self.sl.DEPTH_MODE.NONE
        init.coordinate_units = self.sl.UNIT.METER
        init.sdk_verbose = 0
        if hasattr(init, "sensors_required"):
            init.sensors_required = False
        if hasattr(init, "camera_disable_self_calib"):
            init.camera_disable_self_calib = True
        if hasattr(init, "enable_image_enhancement"):
            init.enable_image_enhancement = False
        if self.serial_number is not None:
            init.set_from_serial_number(int(self.serial_number))
        error = None
        for _ in range(3):
            error = self.camera.open(init)
            if error == self.sl.ERROR_CODE.SUCCESS:
                break
            self.camera.close()
            time.sleep(1.0)
        if error != self.sl.ERROR_CODE.SUCCESS:
            self.close()
            raise HardwareConnectionError(f"ZED open failed: {error}")
        self.runtime = self.sl.RuntimeParameters()
        self.image = self.sl.Mat()

    def start(self) -> None:
        if self.camera is None:
            raise HardwareConnectionError("ZED camera is not connected")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="zed-left-capture",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        from PIL import Image

        while not self._stop.is_set():
            try:
                if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
                    self._stop.wait(0.001)
                    continue
                self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT)
                bgra = self.image.get_data()
                rgb = np.ascontiguousarray(bgra[:, :, 2::-1])
                stream = BytesIO()
                Image.fromarray(rgb, mode="RGB").save(
                    stream,
                    format="JPEG",
                    quality=self.jpeg_quality,
                )
                timestamp_ns = time.time_ns()
                try:
                    hardware_ns = int(
                        self.camera.get_timestamp(
                            self.sl.TIME_REFERENCE.IMAGE
                        ).get_nanoseconds()
                    )
                    if hardware_ns > 0:
                        timestamp_ns = hardware_ns
                except Exception:
                    pass
                snapshot = CameraSnapshot(timestamp_ns, stream.getvalue())
                with self._lock:
                    self._latest = snapshot
            except Exception:
                self._stop.wait(0.01)

    def read(self) -> CameraSnapshot | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.camera = None
