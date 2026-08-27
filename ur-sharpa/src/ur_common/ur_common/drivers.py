"""Hardware adapters for the dual UR5 + SharpA + camera rig.

Ported from the deployed v3 `hardware_drivers.py` with ONE deliberate change:
**UR commanding is joint space, not EEF space.**

v3's `command_eef_step` took a 9-D pose, ran `getInverseKinematics` on the
controller and then `servoJ`'d the result. That put IK inside the hardware
driver, which means teleop and deploy each got whatever the controller's
numeric solver happened to return, and `ur_common.kinematics` -- the single
FK/IK entry point -- was bypassed. Here the driver only accepts joint targets;
IK belongs to `actor_ik_node` and `vive_node`. Everything else about the v3
driver is preserved: connection + control-program readiness handling, the
`normal_mode` probe, `safe_stop`, native state snapshot, and the tactile
channel map.

Snapshots expose measured joints and the UR-native RTDE TCP pose only. The ROS
node computes canonical EEF from those joints through
`UrCommonKinematics.arm_fk_eef9`; the inverse action path uses
`arm_ik_eef9`. Keeping pose conversion out of this driver prevents a second
geometry path from drifting away from FK/IK.

Missing SDKs raise `HardwareDependencyError`, unreachable/failing hardware
raises `HardwareConnectionError`, and a refused command raises
`HardwareCommandError`. Nodes catch the first two and degrade with a clear log
line instead of dying with an import traceback.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import importlib
import multiprocessing
import os
from numbers import Integral
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np

from .geometry import SIDES


__all__ = [
    "HardwareDependencyError",
    "HardwareConnectionError",
    "HardwareCommandError",
    "UrArmDiagnostic",
    "FINGER_NAMES",
    "TACTILE_CHANNELS",
    "UrPairSnapshot",
    "SharpASnapshot",
    "CameraSnapshot",
    "UrRtdePair",
    "format_ur_pair_diagnostics",
    "SharpADevicePair",
    "ZedEgoCamera",
    "RealSenseWristCamera",
    "load_sharpa_sdk",
    "UR5_NUM_JOINTS",
    "HAND_JOINTS_PER_SIDE",
    "TACTILE_FINGERTIPS",
    "DEFORMATION_HEIGHT",
    "DEFORMATION_WIDTH",
    "ensure_optional_runtime_paths",
]


UR5_NUM_JOINTS = 6
HAND_JOINTS_PER_SIDE = 22
DEFAULT_SERVO_SPEED = 1.0
DEFAULT_SERVO_ACCELERATION = 1.0
DEFAULT_SERVO_LOOKAHEAD_S = 0.1
DEFAULT_SERVO_GAIN = 300.0
TACTILE_FINGERTIPS = 5
DEFORMATION_HEIGHT = 240
DEFORMATION_WIDTH = 240

UR_SAFETY_MODE_NAMES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "VALIDATE_JOINT_ID",
    11: "UNDEFINED",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}
UR_SAFETY_STATUS_BIT_NAMES = (
    "NORMAL_MODE",
    "REDUCED_MODE",
    "PROTECTIVE_STOPPED",
    "RECOVERY_MODE",
    "SAFEGUARD_STOPPED",
    "SYSTEM_EMERGENCY_STOPPED",
    "ROBOT_EMERGENCY_STOPPED",
    "EMERGENCY_STOPPED",
    "VIOLATION",
    "FAULT",
    "STOPPED_DUE_TO_SAFETY",
)


def ensure_optional_runtime_paths() -> tuple[str, ...]:
    """Expose installed hardware wheels to the system-Python ROS process.

    ROS Humble uses system Python, while this workstation's vendor wheels are
    built for the same CPython 3.10 ABI in the SharpA environment and in the
    user-local RTDE install. Only the explicitly listed directories are added;
    the full Conda environment is intentionally not imported because it ships
    an incompatible ``typing.py``.
    """
    configured = os.environ.get("UR_SHARPA_PYTHON_PATHS", "")
    candidates = [p for p in configured.split(os.pathsep) if p]
    added: list[str] = []
    for candidate in candidates:
        path = str(Path(candidate).expanduser())
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return tuple(added)

# Fingertip order on the tactile wire: pinky -> thumb, the REVERSE of the
# hand-joint finger order. Matches ur_msgs/WrenchFrame and DeformationFrame.
FINGER_NAMES = ("pinky", "ring", "middle", "index", "thumb")

# Wave SDK tactile channel ids per side, in FINGER_NAMES order. Frozen by the
# hardware wiring; getting this wrong silently swaps fingers.
TACTILE_CHANNELS = {
    "left": (5, 6, 7, 8, 9),
    "right": (0, 1, 2, 3, 4),
}


class HardwareDependencyError(RuntimeError):
    """A required SDK / Python binding is not importable."""


class HardwareConnectionError(RuntimeError):
    """The device is reachable in principle but did not come up."""


class HardwareCommandError(RuntimeError):
    """The command itself was refused (validation, limits, SDK error)."""


@dataclass(frozen=True)
class UrArmDiagnostic:
    """Read-only controller facts used to diagnose an abnormal UR state."""

    normal_mode: bool
    safety_mode: int | None
    safety_status_bits: int | None
    robot_mode: int | None
    protective_stopped: bool | None
    emergency_stopped: bool | None
    payload_mass_kg: float | None
    payload_cog_m: np.ndarray | None
    tcp_wrench: np.ndarray | None
    probe_errors: tuple[str, ...]


@dataclass(frozen=True)
class UrPairSnapshot:
    """One RTDE sample of both arms.

    `left_rtde_pose` / `right_rtde_pose` are the raw `[xyz, rotvec]` (6,) TCP
    poses; kept so callers that need the UR-native convention (jog, moveL) do
    not have to reconstruct it from canonical EEF.
    """

    timestamp_ns: int
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_rtde_pose: np.ndarray
    right_rtde_pose: np.ndarray
    normal_mode: bool
    left_diagnostic: UrArmDiagnostic
    right_diagnostic: UrArmDiagnostic


@dataclass(frozen=True)
class SharpASnapshot:
    """One sample of both SharpA hands: joints, tau, and the tactile channels."""

    timestamp_ns: int
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_tau: np.ndarray
    right_tau: np.ndarray
    left_joint_valid: bool
    right_joint_valid: bool
    left_tau_valid: bool
    right_tau_valid: bool
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
    """One encoded frame plus the time the sensor sampled it."""

    timestamp_ns: int
    jpeg: bytes


def _finite_vector(value: Any, dimension: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise HardwareCommandError(f"{label} must be finite[{dimension}]")
    return result


def _unwrap_to_near(joint: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Shift each joint by whole turns so it sits closest to `near`.

    UR reports angles in a different branch than the one commanded, so a raw
    difference can read as ~2pi of error on a joint that never moved.
    """
    return joint + np.round((near - joint) / (2.0 * np.pi)) * (2.0 * np.pi)


def _format_optional_vector(value: np.ndarray | None) -> str:
    if value is None:
        return "unavailable"
    return "[" + ",".join(f"{item:.3f}" for item in value) + "]"


def _format_optional_bool(value: bool | None) -> str:
    return "unavailable" if value is None else str(value).lower()


def _format_ur_arm_diagnostic(side: str, diagnostic: UrArmDiagnostic) -> str:
    if diagnostic.safety_mode is None:
        safety_mode = "unavailable"
    else:
        safety_mode = (
            f"{diagnostic.safety_mode}"
            f"({UR_SAFETY_MODE_NAMES.get(diagnostic.safety_mode, 'UNKNOWN')})"
        )
    if diagnostic.safety_status_bits is None:
        safety_bits = "unavailable"
    else:
        active_bits = [
            name
            for index, name in enumerate(UR_SAFETY_STATUS_BIT_NAMES)
            if diagnostic.safety_status_bits & (1 << index)
        ]
        names = "|".join(active_bits) if active_bits else "none"
        safety_bits = f"0x{diagnostic.safety_status_bits:x}({names})"
    payload = (
        "unavailable"
        if diagnostic.payload_mass_kg is None
        else f"{diagnostic.payload_mass_kg:.3f}kg"
    )
    errors = "|".join(diagnostic.probe_errors) if diagnostic.probe_errors else "none"
    return (
        f"{side}: normal={str(diagnostic.normal_mode).lower()} "
        f"safety_mode={safety_mode} safety_bits={safety_bits} "
        f"robot_mode={diagnostic.robot_mode} "
        f"protective_stopped={_format_optional_bool(diagnostic.protective_stopped)} "
        f"emergency_stopped={_format_optional_bool(diagnostic.emergency_stopped)} "
        f"payload={payload} cog_m={_format_optional_vector(diagnostic.payload_cog_m)} "
        f"tcp_wrench={_format_optional_vector(diagnostic.tcp_wrench)} "
        f"probe_errors={errors}"
    )


def format_ur_pair_diagnostics(snapshot: UrPairSnapshot) -> str:
    """Format both arms so a protective-stop report identifies the side."""
    return "; ".join(
        (
            _format_ur_arm_diagnostic("left", snapshot.left_diagnostic),
            _format_ur_arm_diagnostic("right", snapshot.right_diagnostic),
        )
    )


def _encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
    """BGR uint8 HxWx3 -> JPEG bytes without holding Python's GIL."""
    try:
        import cv2
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise HardwareDependencyError(
                "JPEG encoding needs Pillow or opencv-python"
            ) from exc
        stream = BytesIO()
        Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]), mode="RGB").save(
            stream, format="JPEG", quality=int(quality)
        )
        return stream.getvalue()
    ok, buffer = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise HardwareCommandError("JPEG encoding failed")
    return bytes(buffer)


class UrRtdePair:
    """Both UR5 arms behind one RTDE receive + control pair per side.

    Only `ur_node` may construct this: RTDE control uploads a script and takes
    the controller, so two owners means two scripts fighting over one arm.

    Commanding is joint space only (`command_joint_step`). `enable_control` is
    False by default; with it off the object is read-only and any command call
    raises `HardwareCommandError` rather than silently doing nothing.
    """

    def __init__(
        self,
        *,
        left_ip: str = "192.168.56.20",
        right_ip: str = "192.168.56.10",
        frequency_hz: float = 125.0,
        enable_control: bool = False,
        enable_motion_limits: bool = False,
        enable_joint_step_limit: bool | None = None,
        max_joint_step_rad: float = 0.35,
        max_abs_joint_rad: float = 2.0 * np.pi,
        servo_speed: float = DEFAULT_SERVO_SPEED,
        servo_acceleration: float = DEFAULT_SERVO_ACCELERATION,
        servo_lookahead_s: float = DEFAULT_SERVO_LOOKAHEAD_S,
        servo_gain: float = DEFAULT_SERVO_GAIN,
        control_connection_delay_s: float = 0.5,
        control_ready_timeout_s: float = 10.0,
        control_ready_poll_s: float = 0.1,
        enable_payload_configuration: bool = False,
        left_payload_mass_kg: float | None = None,
        left_payload_cog_m: Any = (0.0, 0.0, 0.0),
        right_payload_mass_kg: float | None = None,
        right_payload_cog_m: Any = (0.0, 0.0, 0.0),
        payload_verify_timeout_s: float = 2.0,
        payload_verify_poll_s: float = 0.05,
        payload_mass_tolerance_kg: float = 0.01,
        payload_cog_tolerance_m: float = 0.001,
    ) -> None:
        self.ips = {"left": str(left_ip), "right": str(right_ip)}
        self.frequency_hz = float(frequency_hz)
        self.enable_control = bool(enable_control)
        self.enable_motion_limits = bool(enable_motion_limits)
        self.enable_joint_step_limit = (
            self.enable_motion_limits
            if enable_joint_step_limit is None
            else bool(enable_joint_step_limit)
        )
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.max_abs_joint_rad = float(max_abs_joint_rad)
        self.servo_speed = float(servo_speed)
        self.servo_acceleration = float(servo_acceleration)
        self.servo_lookahead_s = float(servo_lookahead_s)
        self.servo_gain = float(servo_gain)
        self.control_connection_delay_s = float(control_connection_delay_s)
        self.control_ready_timeout_s = float(control_ready_timeout_s)
        self.control_ready_poll_s = float(control_ready_poll_s)
        self.enable_payload_configuration = bool(enable_payload_configuration)
        self.payload_verify_timeout_s = float(payload_verify_timeout_s)
        self.payload_verify_poll_s = float(payload_verify_poll_s)
        self.payload_mass_tolerance_kg = float(payload_mass_tolerance_kg)
        self.payload_cog_tolerance_m = float(payload_cog_tolerance_m)
        self.payload_mass_kg = {
            "left": left_payload_mass_kg,
            "right": right_payload_mass_kg,
        }
        self.payload_cog_m = {
            "left": left_payload_cog_m,
            "right": right_payload_cog_m,
        }
        if self.control_ready_timeout_s <= 0.0:
            raise ValueError("UR control ready timeout must be positive")
        if self.control_ready_poll_s <= 0.0:
            raise ValueError("UR control ready poll interval must be positive")
        if self.max_joint_step_rad <= 0.0:
            raise ValueError("UR max joint step must be positive")
        if not np.isfinite(self.max_abs_joint_rad) or self.max_abs_joint_rad <= 0.0:
            raise ValueError("UR absolute joint limit must be finite and positive")
        if not np.isfinite(self.servo_speed) or self.servo_speed <= 0.0:
            raise ValueError("UR servo speed must be positive")
        if not np.isfinite(self.servo_acceleration) or self.servo_acceleration <= 0.0:
            raise ValueError("UR servo acceleration must be positive")
        self._validate_payload_configuration()
        self.receivers: dict[str, Any] = {}
        self.controllers: dict[str, Any] = {}
        # Last target actually handed to servoJ, per side. Used as the "near"
        # seed for step-size checks so a slow arm cannot accumulate a huge
        # backlog of small-looking steps.
        self.last_target_joint: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._command_lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------
    def connect(self) -> None:
        ensure_optional_runtime_paths()
        try:
            receive_module = importlib.import_module("rtde_receive")
            control_module = importlib.import_module("rtde_control")
        except (ImportError, OSError) as exc:
            raise HardwareDependencyError(
                "ur-rtde is unavailable; run: "
                "python3 -m pip install --user --no-deps ur-rtde==1.6.0"
            ) from exc
        try:
            flags = (
                control_module.RTDEControlInterface.FLAG_VERBOSE
                | control_module.RTDEControlInterface.FLAG_UPLOAD_SCRIPT
            )
            for side in SIDES:
                self.receivers[side] = receive_module.RTDEReceiveInterface(
                    self.ips[side], self.frequency_hz, [], True, False, -1
                )
                if self.enable_control:
                    # The controller refuses a second connection made too soon
                    # after the receive socket on the same arm.
                    time.sleep(self.control_connection_delay_s)
                    controller = control_module.RTDEControlInterface(
                        self.ips[side], self.frequency_hz, flags, 50013, -1
                    )
                    self.controllers[side] = controller
                    self._wait_for_control_program(controller, side)
            self._configure_payloads()
        except HardwareConnectionError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise HardwareConnectionError(f"failed to connect UR pair: {exc}") from exc

    def _validate_payload_configuration(self) -> None:
        if not self.enable_payload_configuration:
            return
        if not self.enable_control:
            raise ValueError("UR payload configuration requires control enabled")
        for value, label in (
            (self.payload_verify_timeout_s, "payload verify timeout"),
            (self.payload_verify_poll_s, "payload verify poll interval"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"UR {label} must be positive")
        for value, label in (
            (self.payload_mass_tolerance_kg, "payload mass tolerance"),
            (self.payload_cog_tolerance_m, "payload CoG tolerance"),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"UR {label} must be nonnegative")
        for side in SIDES:
            mass = self.payload_mass_kg[side]
            if mass is None or not np.isfinite(mass) or float(mass) < 0.0:
                raise ValueError(
                    f"{side} payload mass must be an explicitly configured "
                    "nonnegative value"
                )
            try:
                cog = _finite_vector(self.payload_cog_m[side], 3, f"{side} payload CoG")
            except HardwareCommandError as exc:
                raise ValueError(str(exc)) from exc
            self.payload_mass_kg[side] = float(mass)
            self.payload_cog_m[side] = cog

    def _configure_payloads(self) -> None:
        """Set explicitly requested payloads and verify RTDE readback."""
        if not self.enable_payload_configuration:
            return
        for side in SIDES:
            controller = self.controllers.get(side)
            receiver = self.receivers.get(side)
            if not callable(getattr(controller, "setPayload", None)):
                raise HardwareConnectionError(f"{side} RTDE control lacks setPayload")
            for method in ("getPayload", "getPayloadCog"):
                if not callable(getattr(receiver, method, None)):
                    raise HardwareConnectionError(
                        f"{side} RTDE receive lacks {method}; refusing unverified payload write"
                    )
        for side in SIDES:
            accepted = self.controllers[side].setPayload(
                self.payload_mass_kg[side], self.payload_cog_m[side].tolist()
            )
            if accepted is False:
                raise HardwareConnectionError(f"{side} UR rejected setPayload")
            self._verify_payload_readback(side)

    def _verify_payload_readback(self, side: str) -> None:
        deadline = time.monotonic() + self.payload_verify_timeout_s
        last_mass: float | None = None
        last_cog: np.ndarray | None = None
        last_error: Exception | None = None
        while True:
            try:
                last_mass = float(self.receivers[side].getPayload())
                last_cog = _finite_vector(
                    self.receivers[side].getPayloadCog(), 3, f"{side} payload CoG readback"
                )
                mass_matches = np.isfinite(last_mass) and abs(
                    last_mass - self.payload_mass_kg[side]
                ) <= self.payload_mass_tolerance_kg
                cog_matches = np.allclose(
                    last_cog,
                    self.payload_cog_m[side],
                    rtol=0.0,
                    atol=self.payload_cog_tolerance_m,
                )
                if mass_matches and cog_matches:
                    return
                last_error = None
            except Exception as exc:
                last_error = exc
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                detail = (
                    f"readback_error={last_error}"
                    if last_error is not None
                    else (
                        f"readback_mass={last_mass}kg "
                        f"readback_cog={None if last_cog is None else last_cog.tolist()}"
                    )
                )
                raise HardwareConnectionError(
                    f"{side} UR payload readback did not match requested "
                    f"mass={self.payload_mass_kg[side]}kg "
                    f"cog={self.payload_cog_m[side].tolist()}: {detail}"
                )
            time.sleep(min(self.payload_verify_poll_s, remaining_s))

    def _wait_for_control_program(self, controller: Any, side: str) -> None:
        """Verify that the uploaded RTDE control script remains running."""
        deadline = time.monotonic() + self.control_ready_timeout_s
        last_error: Exception | None = None
        while True:
            try:
                if not controller.isConnected():
                    last_error = RuntimeError("RTDE control socket is disconnected")
                elif controller.isProgramRunning():
                    return
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

    @property
    def connected(self) -> bool:
        return set(self.receivers) == set(SIDES)

    def synchronize_command_state(self, snapshot: UrPairSnapshot | None = None) -> UrPairSnapshot:
        """Adopt the measured joints as the post-reconnect command reference.

        A protective stop can leave the controller at a different joint pose
        than the last servo target. The next command must be based on the
        fresh RTDE sample, never on the old session's target.
        """
        with self._command_lock:
            current = self.read() if snapshot is None else snapshot
            if not current.normal_mode:
                raise HardwareCommandError(
                    "cannot synchronize UR command state outside normal mode"
                )
            self.last_target_joint = {
                "left": np.asarray(current.left_joint, dtype=np.float64).copy(),
                "right": np.asarray(current.right_joint, dtype=np.float64).copy(),
            }
            return current

    # ---- reading ------------------------------------------------------
    def read(self) -> UrPairSnapshot:
        if not self.connected:
            raise HardwareConnectionError("UR receive interfaces are not connected")
        joints: dict[str, np.ndarray] = {}
        poses: dict[str, np.ndarray] = {}
        diagnostics: dict[str, UrArmDiagnostic] = {}
        normal = True
        for side in SIDES:
            receiver = self.receivers[side]
            joints[side] = _finite_vector(
                receiver.getActualQ(), UR5_NUM_JOINTS, f"{side} joint"
            )
            poses[side] = _finite_vector(
                receiver.getActualTCPPose(), 6, f"{side} RTDE pose"
            )
            diagnostics[side] = self._read_arm_diagnostic(receiver)
            normal = normal and diagnostics[side].normal_mode
        return UrPairSnapshot(
            timestamp_ns=time.time_ns(),
            left_joint=joints["left"].astype(np.float32),
            right_joint=joints["right"].astype(np.float32),
            left_rtde_pose=poses["left"],
            right_rtde_pose=poses["right"],
            normal_mode=normal,
            left_diagnostic=diagnostics["left"],
            right_diagnostic=diagnostics["right"],
        )

    @staticmethod
    def _read_arm_diagnostic(receiver: Any) -> UrArmDiagnostic:
        errors: list[str] = []
        failed_methods: set[str] = set()

        def probe(method: str) -> Any:
            if not callable(getattr(receiver, method, None)):
                return None
            try:
                return getattr(receiver, method)()
            except Exception as exc:
                errors.append(f"{method}:{exc}")
                failed_methods.add(method)
                return None

        connected = probe("isConnected")
        protective_stopped = probe("isProtectiveStopped")
        emergency_stopped = probe("isEmergencyStopped")
        safety_mode = probe("getSafetyMode")
        safety_status_bits = probe("getSafetyStatusBits")
        robot_mode = probe("getRobotMode")
        payload_mass_kg = probe("getPayload")
        payload_cog_m = probe("getPayloadCog")
        tcp_wrench = probe("getActualTCPForce")

        def optional_int(value: Any, label: str, method: str) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                errors.append(f"{label}:{exc}")
                failed_methods.add(method)
                return None

        safety_mode = optional_int(safety_mode, "safety_mode", "getSafetyMode")
        safety_status_bits = optional_int(
            safety_status_bits, "safety_status_bits", "getSafetyStatusBits"
        )
        robot_mode = optional_int(robot_mode, "robot_mode", "getRobotMode")
        if payload_mass_kg is not None:
            try:
                payload_mass_kg = float(payload_mass_kg)
                if not np.isfinite(payload_mass_kg):
                    raise ValueError("not finite")
            except (TypeError, ValueError) as exc:
                errors.append(f"payload_mass:{exc}")
                payload_mass_kg = None
        for value, dimension, label in (
            (payload_cog_m, 3, "payload_cog"),
            (tcp_wrench, 6, "tcp_wrench"),
        ):
            if value is None:
                continue
            try:
                converted = _finite_vector(value, dimension, label)
            except HardwareCommandError as exc:
                errors.append(str(exc))
                converted = None
            if label == "payload_cog":
                payload_cog_m = converted
            else:
                tcp_wrench = converted

        normal = True
        safety_probes = {
            "isConnected",
            "isProtectiveStopped",
            "isEmergencyStopped",
            "getSafetyMode",
            "getRobotMode",
        }
        if failed_methods & safety_probes:
            normal = False
        if connected is False or protective_stopped is True or emergency_stopped is True:
            normal = False
        if safety_mode is not None and safety_mode != 1:
            normal = False
        if safety_status_bits is not None:
            abnormal_mask = sum(1 << index for index in range(1, 11))
            if not (safety_status_bits & 1) or safety_status_bits & abnormal_mask:
                normal = False
        if robot_mode is not None and robot_mode != 7:
            normal = False
        return UrArmDiagnostic(
            normal_mode=normal,
            safety_mode=safety_mode,
            safety_status_bits=safety_status_bits,
            robot_mode=robot_mode,
            protective_stopped=(
                None if protective_stopped is None else bool(protective_stopped)
            ),
            emergency_stopped=(
                None if emergency_stopped is None else bool(emergency_stopped)
            ),
            payload_mass_kg=payload_mass_kg,
            payload_cog_m=payload_cog_m,
            tcp_wrench=tcp_wrench,
            probe_errors=tuple(errors),
        )

    @staticmethod
    def _normal_mode(receiver: Any) -> bool:
        """True only when the arm is in normal running mode.

        Every probe is optional (`hasattr`) because ur-rtde versions differ in
        what they expose, and any exception means "assume not normal". Safety
        mode 1 == NORMAL, robot mode 7 == RUNNING.
        """
        return UrRtdePair._read_arm_diagnostic(receiver).normal_mode

    # ---- commanding (JOINT SPACE ONLY) --------------------------------
    def command_joint_step(
        self,
        left_joint: Any,
        right_joint: Any,
        period_s: float,
        *,
        left_enable: bool = True,
        right_enable: bool = True,
    ) -> None:
        """One servoJ step per arm at joint targets already resolved upstream.

        This replaces v3's `command_eef_step`. No IK here: the caller
        (`actor_ik_node` / `vive_node` / `bias_node`) has already gone through
        `ur_common.kinematics`, so observation and execution share one solver.

        Validated before anything is written: control enabled, sane period,
        finite 6-vectors, absolute range, and, when enabled, per-step delta
        against the last commanded target. Any servoJ failure triggers
        `safe_stop` first.
        """
        enables = {"left": bool(left_enable), "right": bool(right_enable)}
        active = [side for side in SIDES if enables[side]]
        if not active:
            return
        if not self.enable_control or any(side not in self.controllers for side in active):
            raise HardwareCommandError("UR hardware execution is disabled")
        if not np.isfinite(period_s) or period_s <= 0.0 or period_s > 0.2:
            raise HardwareCommandError("UR command period must be in (0, 0.2] seconds")
        values = {"left": left_joint, "right": right_joint}
        targets = {
            side: _finite_vector(values[side], UR5_NUM_JOINTS, f"{side} joint target")
            for side in active
        }
        with self._command_lock:
            snapshot = self.read()
            diagnostics = {
                "left": snapshot.left_diagnostic,
                "right": snapshot.right_diagnostic,
            }
            abnormal = [side for side in active if not diagnostics[side].normal_mode]
            if abnormal:
                raise HardwareCommandError(
                    f"UR {', '.join(abnormal)} arm is not in normal running mode: "
                    + format_ur_pair_diagnostics(snapshot)
                )
            measured = {
                "left": snapshot.left_joint.astype(np.float64),
                "right": snapshot.right_joint.astype(np.float64),
            }
            for side in active:
                target = targets[side]
                if self.enable_motion_limits and np.any(np.abs(target) > self.max_abs_joint_rad):
                    raise HardwareCommandError(
                        f"{side} joint target exceeds absolute limit "
                        f"{self.max_abs_joint_rad:.3f} rad"
                    )
                near = self.last_target_joint[side]
                if near is None:
                    near = measured[side]
                unwrapped = _unwrap_to_near(target, near)
                delta = unwrapped - near
                if (
                    self.enable_joint_step_limit
                    and np.any(np.abs(delta) > self.max_joint_step_rad + 1.0e-9)
                ):
                    raise HardwareCommandError(
                        f"{side} joint target step exceeds "
                        f"{self.max_joint_step_rad:.3f} rad"
                    )
                targets[side] = unwrapped
            try:
                errors: list[tuple[str, Exception]] = []

                def servo(side: str) -> None:
                    try:
                        self.controllers[side].servoJ(
                            targets[side].tolist(), self.servo_speed,
                            self.servo_acceleration, float(period_s),
                            self.servo_lookahead_s, self.servo_gain,
                        )
                    except Exception as exc:
                        errors.append((side, exc))

                threads = [threading.Thread(target=servo, args=(side,)) for side in active]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if errors:
                    for failed_side, _ in errors:
                        self.safe_stop(failed_side)
                    for side in active:
                        if side not in {failed_side for failed_side, _ in errors}:
                            self.last_target_joint[side] = targets[side].copy()
                    raise errors[0][1]
                for side in active:
                    self.last_target_joint[side] = targets[side].copy()
            except Exception as exc:
                if not errors:
                    for side in active:
                        self.safe_stop(side)
                raise HardwareCommandError(f"UR servoJ failed: {exc}") from exc

    def move_joint_ramp(
        self,
        left_joint: Any,
        right_joint: Any,
        *,
        steps: int = 120,
        step_delay_s: float = 0.02,
        tolerance_rad: float = 0.08,
    ) -> None:
        """Interpolate both arms from where they are to a fixed joint pose.

        `bias_node` owns homing; this is the primitive it (and `ur_bringup`)
        calls. `ur_node` does NOT call this on startup -- see its README.
        """
        if not self.enable_control or set(self.controllers) != set(SIDES):
            raise HardwareCommandError("UR hardware execution is disabled")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) <= 0:
            raise HardwareCommandError("UR ramp steps must be a positive integer")
        if not np.isfinite(step_delay_s) or step_delay_s < 0.0:
            raise HardwareCommandError("UR ramp delay must be nonnegative")
        if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise HardwareCommandError("UR ramp tolerance must be positive")
        targets = {
            "left": _finite_vector(left_joint, UR5_NUM_JOINTS, "left ramp target"),
            "right": _finite_vector(right_joint, UR5_NUM_JOINTS, "right ramp target"),
        }
        for side in SIDES:
            if self.enable_motion_limits and np.any(np.abs(targets[side]) > self.max_abs_joint_rad):
                raise HardwareCommandError(f"{side} ramp target exceeds position limits")

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
                        target = starts[side] + alpha * (targets[side] - starts[side])
                        self.controllers[side].servoJ(
                            target.tolist(),
                            self.servo_speed,
                            self.servo_acceleration,
                            max(float(step_delay_s), 1.0 / self.frequency_hz),
                            self.servo_lookahead_s,
                            self.servo_gain,
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
                            f"{side} UR did not reach its ramp joint target"
                        )
                    self.last_target_joint[side] = targets[side].copy()
            except Exception as exc:
                self.safe_stop()
                if isinstance(exc, HardwareCommandError):
                    raise
                raise HardwareCommandError(f"UR joint ramp failed: {exc}") from exc

    # ---- stopping -----------------------------------------------------
    def safe_stop(self, side: str | None = None) -> None:
        """Decelerate one arm, or both arms when side is omitted. Never raises.

        This is what `/hardware/stop` and every failed command path call, so it
        must work even when half the connection is already broken.
        """
        controllers = (
            tuple(self.controllers.values())
            if side is None
            else tuple(
                controller for current, controller in self.controllers.items()
                if current == side
            )
        )
        for controller in controllers:
            try:
                controller.servoStop()
            except Exception:
                pass
        if side is None:
            self.last_target_joint = {"left": None, "right": None}
        else:
            self.last_target_joint[side] = None

    def close(self) -> None:
        self.safe_stop()
        for controller in tuple(self.controllers.values()):
            for method in ("stopScript", "disconnect"):
                try:
                    getattr(controller, method)()
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
    """Import the vendored Wave SDK `sharpa` module from an install root."""
    root = Path(sdk_root).expanduser().resolve()
    python_dir = root / "python"
    if not python_dir.is_dir():
        raise HardwareDependencyError(
            f"SharpA SDK python directory not found: {python_dir}"
        )
    path = str(python_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        return importlib.import_module("sharpa")
    except Exception as exc:
        raise HardwareDependencyError(
            f"failed to load SharpA SDK from {root}: {exc}"
        ) from exc


class SharpADevicePair:
    """Both SharpA Wave hands behind one SDK manager.

    The Wave SDK claims the serial link exclusively, so exactly one process may
    hold this -- `sharpa_node`. Teleop and deploy both publish to
    `/sharpa/command` and go through that node.

    Joint vectors here are per-hand 22-D. The 44-D interface concatenation
    (left then right) is assembled by the node, not the driver.
    """

    def __init__(
        self,
        *,
        sdk_root: str | Path,
        discovery_timeout_s: float = 15.0,
        speed_coefficient: float = 0.5,
        current_coefficient: float = 0.6,
        enable_control: bool = False,
        skip_tactile: bool = False,
        max_hand_step_rad: float = 0.5,
        max_abs_hand_rad: float = 3.5,
    ) -> None:
        self.sdk_root = Path(sdk_root)
        self.discovery_timeout_s = float(discovery_timeout_s)
        self.speed_coefficient = float(speed_coefficient)
        self.current_coefficient = float(current_coefficient)
        self.enable_control = bool(enable_control)
        self.skip_tactile = bool(skip_tactile)
        self.max_hand_step_rad = float(max_hand_step_rad)
        self.max_abs_hand_rad = float(max_abs_hand_rad)
        self.sdk: ModuleType | None = None
        self.manager: Any = None
        self.hands: dict[str, Any] = {}
        # channel id -> {"wrench": (6,), "deformation": (240,240)}, filled by
        # the SDK's tactile callback thread.
        self._tactile: dict[str, dict[int, dict[str, np.ndarray]]] = {
            "left": {},
            "right": {},
        }
        self._tactile_lock = threading.Lock()
        self._command_lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------
    def connect(self) -> None:
        ensure_optional_runtime_paths()
        self.sdk = load_sharpa_sdk(self.sdk_root)
        try:
            self.manager = self.sdk.SharpaWaveManager.get_instance()
        except Exception as exc:
            raise HardwareConnectionError(
                f"SharpA Wave manager unavailable: {exc}"
            ) from exc
        deadline = time.monotonic() + self.discovery_timeout_s
        devices: list[Any] = []
        while time.monotonic() < deadline:
            devices = [
                device
                for device in self.manager.get_all_devices()
                if device.device_type == self.sdk.DeviceType.HAND
            ]
            if {self._device_side(device) for device in devices} == set(SIDES):
                break
            time.sleep(0.1)
        for device in devices:
            side = self._device_side(device)
            if side in self.hands:
                continue
            hand = None
            try:
                hand = (
                    self.manager.connect(device.sn, True)
                    if self.skip_tactile
                    else self.manager.connect(device.sn)
                )
                self._check_error(hand.set_control_mode(self.sdk.ControlMode.POSITION))
                self._check_error(hand.set_control_source(self.sdk.ControlSource.SDK))
                self._check_error(hand.set_speed_coeff(self.speed_coefficient))
                self._check_error(hand.set_current_coeff(self.current_coefficient))
                if not self.skip_tactile:
                    hand.set_tactile_callback(self._tactile_callback(side))
                if not hand.start():
                    raise HardwareConnectionError(f"SharpA {side} hand failed to start")
                self.hands[side] = hand
            except Exception:
                if hand is not None:
                    try:
                        hand.stop()
                    except Exception:
                        pass
                continue
        if not self.hands:
            found = sorted(self.hands) or ["none"]
            self.close()
            raise HardwareConnectionError(
                "no SharpA hand is connected; "
                f"discovered: {', '.join(found)}"
            )

    def _device_side(self, device: Any) -> str:
        assert self.sdk is not None
        return "left" if device.hand_side == self.sdk.HandSide.LEFT else "right"

    @staticmethod
    def _check_error(error: Any) -> None:
        if getattr(error, "code", -1) != 0:
            raise HardwareConnectionError(getattr(error, "message", "SharpA SDK error"))

    @property
    def connected(self) -> bool:
        return set(self.hands) == set(SIDES)

    # ---- tactile ------------------------------------------------------
    def _tactile_callback(self, side: str) -> Any:
        """Build the per-side SDK callback. It swallows every error on purpose.

        This runs on an SDK-owned thread. Raising here would take down the
        SDK's reader, and a malformed tactile packet must not cost us the hand.
        A dropped frame simply leaves the previous value in place, and the
        per-fingertip valid flags in `read()` tell consumers what is real.
        """

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
                    if array.size == DEFORMATION_HEIGHT * DEFORMATION_WIDTH:
                        stored["deformation"] = array.reshape(
                            DEFORMATION_HEIGHT, DEFORMATION_WIDTH
                        ).copy()
                with self._tactile_lock:
                    self._tactile[side][channel] = stored
            except Exception:
                return

        return callback

    # ---- reading ------------------------------------------------------
    def read(self) -> SharpASnapshot:
        """Latest joints/tau (polled) plus latest tactile (callback-fed).

        Never raises for a missing hand: an unreachable hand yields zeros with
        its valid flag false, so the node can keep publishing a frame that
        honestly says which half is dead.
        """
        n = HAND_JOINTS_PER_SIDE
        joint = {side: np.zeros(n, dtype=np.float32) for side in SIDES}
        tau = {side: np.zeros(n, dtype=np.float32) for side in SIDES}
        joint_valid = {side: False for side in SIDES}
        tau_valid = {side: False for side in SIDES}
        for side in SIDES:
            hand = self.hands.get(side)
            if hand is None:
                continue
            try:
                state = hand.get_states()
                angles = np.asarray(state.angles, dtype=np.float32).reshape(-1)
                torques = np.asarray(state.torques, dtype=np.float32).reshape(-1)
                if angles.shape == (n,) and np.isfinite(angles).all():
                    joint[side] = angles
                    joint_valid[side] = True
                if torques.shape == (n,) and np.isfinite(torques).all():
                    tau[side] = torques
                    tau_valid[side] = True
            except Exception:
                continue
        f = TACTILE_FINGERTIPS
        wrench = {side: np.zeros((f, 6), dtype=np.float32) for side in SIDES}
        deformation = {
            side: np.zeros(
                (f, DEFORMATION_HEIGHT, DEFORMATION_WIDTH), dtype=np.uint8
            )
            for side in SIDES
        }
        wrench_valid = {side: np.zeros(f, dtype=np.bool_) for side in SIDES}
        deformation_valid = {side: np.zeros(f, dtype=np.bool_) for side in SIDES}
        with self._tactile_lock:
            tactile = {
                side: {channel: dict(values) for channel, values in data.items()}
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

    # ---- commanding ---------------------------------------------------
    def command(
        self,
        left_joint: Any,
        right_joint: Any,
        *,
        left_enable: bool = True,
        right_enable: bool = True,
        event_callback: Any = None,
    ) -> dict[str, np.ndarray]:
        """Set both hands' joint positions.

        `*_enable=False` means "hold this hand where it is": its slice of the
        command is ignored entirely, per `ur_msgs/HandCommand`. Limits are
        checked for every enabled hand BEFORE any hand is written, so a bad
        right-hand target cannot leave the left hand half-moved.
        """
        if not self.enable_control:
            raise HardwareCommandError("SharpA hardware execution is disabled")
        enables = {"left": bool(left_enable), "right": bool(right_enable)}
        if not any(enables.values()):
            return {}
        active = [side for side in SIDES if enables[side]]
        missing = [side for side in active if side not in self.hands]
        if missing:
            raise HardwareCommandError(
                f"SharpA hand not connected: {', '.join(missing)}"
            )
        n = HAND_JOINTS_PER_SIDE
        targets = {
            "left": _finite_vector(left_joint, n, "left hand target"),
            "right": _finite_vector(right_joint, n, "right hand target"),
        }
        def report(event: str, **fields: Any) -> None:
            if not callable(event_callback):
                return
            try:
                event_callback(event, **fields)
            except Exception:
                # Diagnostics must never change the hardware command outcome.
                pass

        report("sharpa_sdk_state_read_started")
        read_started_ns = time.monotonic_ns()
        current = self.read()
        report(
            "sharpa_sdk_state_read_completed",
            duration_ms=(time.monotonic_ns() - read_started_ns) / 1_000_000.0,
        )
        current_joint = {
            "left": current.left_joint.astype(np.float64),
            "right": current.right_joint.astype(np.float64),
        }
        with self._command_lock:
            for side in active:
                if np.max(np.abs(targets[side])) > self.max_abs_hand_rad:
                    raise HardwareCommandError(
                        f"{side} hand target exceeds absolute limit "
                        f"{self.max_abs_hand_rad:.3f} rad"
                    )
                delta = targets[side] - current_joint[side]
                targets[side] = current_joint[side] + np.clip(
                    delta, -self.max_hand_step_rad, self.max_hand_step_rad
                )
            for side in active:
                command_started_ns = time.monotonic_ns()
                report("sharpa_sdk_hand_command_started", side=side)
                error = self.hands[side].set_joint_position(targets[side].tolist(), False)
                report(
                    "sharpa_sdk_hand_command_completed",
                    side=side,
                    duration_ms=(time.monotonic_ns() - command_started_ns)
                    / 1_000_000.0,
                    sdk_code=int(getattr(error, "code", -1)),
                    sdk_message=str(getattr(error, "message", "")),
                    success=getattr(error, "code", -1) == 0,
                )
                if getattr(error, "code", -1) != 0:
                    self.hold(side)
                    raise HardwareCommandError(
                        f"SharpA {side} command failed: {getattr(error, 'message', '')}"
                    )
        # Return exactly what was handed to the SDK. The input can differ when
        # the driver slews a large teleop step, and recording must label the
        # action with the dispatched target rather than the pre-limit request.
        return {side: targets[side].copy() for side in active}

    def set_control_mode(self, mode: Any) -> None:
        """Trigger one control-mode transition on each connected hand.

        Wave control-mode setters are edge-triggered operations, not
        idempotent configuration writes. Lifecycle ownership therefore lives
        in sharpa_node: this method is called once when its state crosses the
        FLOATING/POSITION boundary and never as a health probe.
        """
        if not self.enable_control:
            return
        if self.sdk is None:
            raise HardwareCommandError("SharpA SDK is unavailable")
        for side in SIDES:
            hand = self.hands.get(side)
            if hand is None:
                continue
            error = hand.set_control_mode(mode)
            if getattr(error, "code", -1) != 0:
                raise HardwareCommandError(
                    f"SharpA {side} control mode failed: {getattr(error, 'message', '')}"
                )

    def set_control_source(self, source: Any) -> None:
        """Trigger one command-source transition on each connected hand."""
        if not self.enable_control:
            return
        if self.sdk is None:
            raise HardwareCommandError("SharpA SDK is unavailable")
        for side in SIDES:
            hand = self.hands.get(side)
            if hand is None:
                continue
            error = hand.set_control_source(source)
            if getattr(error, "code", -1) != 0:
                raise HardwareCommandError(
                    f"SharpA {side} control source failed: "
                    f"{getattr(error, 'message', '')}"
                )

    def set_floating(self) -> None:
        """Enable gravity compensation for manual hand adjustment."""
        if self.sdk is not None:
            self.set_control_mode(self.sdk.ControlMode.FLOATING)

    def set_position_control(self) -> None:
        """Enter POSITION once; the SDK source was established at connect."""
        if self.sdk is not None:
            self.set_control_mode(self.sdk.ControlMode.POSITION)

    def move_joint_ramp(
        self,
        left_joint: Any,
        right_joint: Any,
        *,
        steps: int = 60,
        step_delay_s: float = 0.02,
        tolerance_rad: float = 0.08,
    ) -> None:
        """Interpolate both hands to a fixed pose. `bias_node`'s zeroing primitive."""
        if not self.enable_control or not self.connected:
            raise HardwareCommandError("SharpA hardware execution is disabled")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) <= 0:
            raise HardwareCommandError("SharpA ramp steps must be a positive integer")
        if not np.isfinite(step_delay_s) or step_delay_s < 0.0:
            raise HardwareCommandError("SharpA ramp delay must be nonnegative")
        if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise HardwareCommandError("SharpA ramp tolerance must be positive")
        n = HAND_JOINTS_PER_SIDE
        targets = {
            "left": _finite_vector(left_joint, n, "left ramp hand joint"),
            "right": _finite_vector(right_joint, n, "right ramp hand joint"),
        }
        for side in SIDES:
            if np.max(np.abs(targets[side])) > self.max_abs_hand_rad:
                raise HardwareCommandError(f"{side} ramp hand joint exceeds limit")
        current = self.read()
        if not current.left_joint_valid or not current.right_joint_valid:
            raise HardwareCommandError("SharpA joint state is unavailable")
        starts = {
            "left": current.left_joint.astype(np.float64),
            "right": current.right_joint.astype(np.float64),
        }
        with self._command_lock:
            try:
                for step_index in range(1, int(steps) + 1):
                    alpha = step_index / int(steps)
                    for side in SIDES:
                        target = starts[side] + alpha * (targets[side] - starts[side])
                        error = self.hands[side].set_joint_position(
                            target.tolist(), False
                        )
                        if getattr(error, "code", -1) != 0:
                            raise HardwareCommandError(
                                f"SharpA {side} ramp failed: "
                                f"{getattr(error, 'message', '')}"
                            )
                    if step_delay_s > 0.0:
                        time.sleep(float(step_delay_s))
                reached = self.read()
                if not reached.left_joint_valid or not reached.right_joint_valid:
                    raise HardwareCommandError("SharpA joint state is unavailable")
                for side in SIDES:
                    actual = (
                        reached.left_joint if side == "left" else reached.right_joint
                    ).astype(np.float64)
                    if np.max(np.abs(actual - targets[side])) > tolerance_rad:
                        raise HardwareCommandError(
                            f"{side} SharpA did not reach its ramp joint target"
                        )
            except Exception as exc:
                self.hold()
                if isinstance(exc, HardwareCommandError):
                    raise
                raise HardwareCommandError(f"SharpA joint ramp failed: {exc}") from exc

    # ---- stopping -----------------------------------------------------
    def hold(self, side: str | None = None) -> None:
        """Freeze one hand, or both hands when side is omitted. Never raises.

        The hand has no servoStop equivalent, so "stop" means re-commanding the
        current position; releasing the command entirely would let a loaded
        finger sag.
        """
        if not self.enable_control:
            return
        try:
            snapshot = self.read()
            sides = SIDES if side is None else (side,)
            for current_side in sides:
                hand = self.hands.get(current_side)
                valid = (
                    snapshot.left_joint_valid
                    if current_side == "left"
                    else snapshot.right_joint_valid
                )
                if hand is None or not valid:
                    continue
                measured = (
                    snapshot.left_joint if current_side == "left" else snapshot.right_joint
                )
                hand.set_joint_position(measured.tolist(), False)
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


class _ThreadedCamera:
    """Grab and encode independently, hand out the newest encoded frame.

    Camera nodes must never block their ROS executor on a `grab()`, so every
    driver here follows the same shape: `connect()` (may raise), `start()`,
    `read()` returning the latest `CameraSnapshot` or None, `close()`.
    """

    thread_name = "camera-capture"

    def __init__(
        self,
        jpeg_quality: int = 90,
        raw_queue_size: int = 8,
        process_encode: bool = False,
        encode_workers: int = 1,
        encoded_queue_size: int = 128,
    ) -> None:
        self.jpeg_quality = int(jpeg_quality)
        self.raw_queue_size = int(raw_queue_size)
        if self.raw_queue_size < 0:
            raise ValueError("camera raw queue size must be non-negative")
        self.process_encode = bool(process_encode)
        self.encode_workers = int(encode_workers)
        if self.encode_workers <= 0:
            raise ValueError("camera encode worker count must be positive")
        self.encoded_queue_size = int(encoded_queue_size)
        if self.encoded_queue_size <= 0:
            raise ValueError("camera encoded queue size must be positive")
        self._latest: CameraSnapshot | None = None
        self._lock = threading.Lock()
        self._raw_ready = threading.Condition(self._lock)
        self._raw_queue: deque[tuple[int, int, np.ndarray]] = deque()
        self._encoded_pending: dict[int, CameraSnapshot] = {}
        self._encoded_ready: deque[CameraSnapshot] = deque()
        self._dropped_sequences: set[int] = set()
        self._next_capture_sequence = 0
        self._next_output_sequence = 0
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._encode_threads: list[threading.Thread] = []
        self._encode_executor: ProcessPoolExecutor | None = None
        self.last_capture_error = ""
        self.last_encode_error = ""
        self.capture_count = 0
        self.encoded_count = 0
        self.dropped_raw_count = 0
        self.dropped_encoded_count = 0

    def _grab_rgb(self) -> np.ndarray | None:  # pragma: no cover - hardware
        raise NotImplementedError

    def start(self) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        self._stop.clear()
        if self.process_encode:
            self._encode_executor = ProcessPoolExecutor(
                max_workers=self.encode_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            self._encode_executor.submit(
                _encode_jpeg,
                np.zeros((1, 1, 3), dtype=np.uint8),
                self.jpeg_quality,
            ).result(timeout=5.0)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name=self.thread_name, daemon=True
        )
        self._encode_threads = [
            threading.Thread(
                target=self._encode_loop,
                name=f"{self.thread_name}-jpeg-{index}",
                daemon=True,
            )
            for index in range(self.encode_workers)
        ]
        for thread in self._encode_threads:
            thread.start()
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                bgr = self._grab_rgb()
                if bgr is None:
                    self._stop.wait(0.001)
                    continue
                sampled_at_ns = self._sample_timestamp_ns()
                raw = np.asarray(bgr, dtype=np.uint8).copy()
                with self._raw_ready:
                    sequence = self._next_capture_sequence
                    self._next_capture_sequence += 1
                    if (
                        self.raw_queue_size > 0
                        and len(self._raw_queue) >= self.raw_queue_size
                    ):
                        dropped_sequence, _, _ = self._raw_queue.popleft()
                        self._dropped_sequences.add(dropped_sequence)
                        self.dropped_raw_count += 1
                        self._publish_encoded_locked()
                    self._raw_queue.append((sequence, sampled_at_ns, raw))
                    self.capture_count += 1
                    self._raw_ready.notify()
                self.last_capture_error = ""
            except Exception as exc:
                self.last_capture_error = str(exc)
                # A transient grab failure must not kill the thread; back off
                # briefly so a permanently dead camera does not spin a core.
                self._stop.wait(0.01)

    def _encode_loop(self) -> None:
        while not self._stop.is_set():
            with self._raw_ready:
                self._raw_ready.wait_for(
                    lambda: self._stop.is_set()
                    or bool(self._raw_queue),
                    timeout=0.1,
                )
                if self._stop.is_set():
                    return
                if not self._raw_queue:
                    continue
                sequence, timestamp_ns, bgr = self._raw_queue.popleft()
            if bgr is None:
                continue
            try:
                if self._encode_executor is None:
                    jpeg = _encode_jpeg(bgr, self.jpeg_quality)
                else:
                    jpeg = self._encode_executor.submit(
                        _encode_jpeg, bgr, self.jpeg_quality
                    ).result(timeout=2.0)
                snapshot = CameraSnapshot(
                    timestamp_ns=timestamp_ns,
                    jpeg=jpeg,
                )
                with self._lock:
                    # Parallel encoders commonly complete adjacent frames in a
                    # burst or in reverse order. Keep every result until all
                    # earlier capture sequences have completed (or were
                    # explicitly dropped), then publish them in capture order.
                    self._encoded_pending[sequence] = snapshot
                    self.encoded_count += 1
                    self._publish_encoded_locked()
                self.last_encode_error = ""
            except Exception as exc:
                self.last_encode_error = str(exc)
                with self._lock:
                    self._dropped_sequences.add(sequence)
                    self._publish_encoded_locked()

    def _publish_encoded_locked(self) -> None:
        while True:
            sequence = self._next_output_sequence
            if sequence in self._dropped_sequences:
                self._dropped_sequences.remove(sequence)
                self._next_output_sequence += 1
                continue
            snapshot = self._encoded_pending.pop(sequence, None)
            if snapshot is None:
                return
            if len(self._encoded_ready) >= self.encoded_queue_size:
                self._encoded_ready.popleft()
                self.dropped_encoded_count += 1
            self._encoded_ready.append(snapshot)
            self._latest = snapshot
            self._next_output_sequence += 1

    def _sample_timestamp_ns(self) -> int:
        return time.time_ns()

    def read(self) -> CameraSnapshot | None:
        with self._lock:
            return self._latest

    def read_pending(self) -> list[CameraSnapshot]:
        """Drain all encoded frames that have not yet reached camera_node."""

        with self._lock:
            frames = list(self._encoded_ready)
            self._encoded_ready.clear()
            return frames

    def close(self) -> None:
        self._stop.set()
        with self._raw_ready:
            self._raw_ready.notify_all()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        for thread in self._encode_threads:
            thread.join(timeout=2.0)
        if self._encode_executor is not None:
            self._encode_executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._raw_queue.clear()
            self._encoded_pending.clear()
            self._encoded_ready.clear()
            self._dropped_sequences.clear()
            self._latest = None
            self._next_capture_sequence = 0
            self._next_output_sequence = 0
        self._capture_thread = None
        self._encode_threads = []
        self._encode_executor = None


class ZedEgoCamera(_ThreadedCamera):
    """ZED left eye as the ego view.

    Left eye only: the policy's ego_cam is monocular, and skipping the right
    retrieve halves the SDK's per-frame work. Depth is off for the same reason.
    Timestamps prefer the SDK's IMAGE reference over wall clock, because that is
    the actual sampling instant and `camera_node` publishes inter-view skew.
    """

    thread_name = "zed-ego-capture"

    def __init__(
        self,
        *,
        resolution: str = "HD720",
        frequency_hz: int = 30,
        serial_number: int | None = None,
        jpeg_quality: int = 90,
    ) -> None:
        # HD720 JPEG encoding was the Ego bottleneck (~22 delivered Hz while
        # capture remained ~28-30 Hz). Two ordered latest-frame workers keep
        # encoding capacity above capture rate without changing wrist cameras.
        super().__init__(jpeg_quality=jpeg_quality, encode_workers=2)
        self.resolution = str(resolution)
        self.frequency_hz = int(frequency_hz)
        self.serial_number = serial_number
        self.sl: Any = None
        self.camera: Any = None
        self.runtime: Any = None
        self.image: Any = None

    def connect(self) -> None:
        ensure_optional_runtime_paths()
        try:
            self.sl = importlib.import_module("pyzed.sl")
        except (ImportError, OSError) as exc:
            raise HardwareDependencyError(
                "pyzed is unavailable; install the ZED SDK then run "
                "`python /usr/local/zed/get_python_api.py`"
            ) from exc
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
        # These three are version-dependent; each one that exists removes a
        # periodic stall or an unused sensor dependency.
        for attribute, value in (
            ("sensors_required", False),
            ("camera_disable_self_calib", True),
            ("enable_image_enhancement", False),
        ):
            if hasattr(init, attribute):
                setattr(init, attribute, value)
        if self.serial_number is not None:
            init.set_from_serial_number(int(self.serial_number))
        # Open is retried because USB handshake / "motion sensors not detected"
        # failures are usually one-shot. Every failed attempt MUST close the
        # half-initialised SDK object, or its background threads outlive the
        # Python objects and crash at interpreter shutdown.
        error = None
        for _ in range(3):
            error = self.camera.open(init)
            if error == self.sl.ERROR_CODE.SUCCESS:
                break
            try:
                self.camera.close()
            except Exception:
                pass
            time.sleep(1.0)
        if error != self.sl.ERROR_CODE.SUCCESS:
            self.close()
            raise HardwareConnectionError(f"ZED open failed: {error}")
        self.runtime = self.sl.RuntimeParameters()
        self.image = self.sl.Mat()

    def _grab_rgb(self) -> np.ndarray | None:
        if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return None
        self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT)
        bgra = self.image.get_data()
        return np.ascontiguousarray(bgra[:, :, :3])

    def _sample_timestamp_ns(self) -> int:
        try:
            hardware_ns = int(
                self.camera.get_timestamp(
                    self.sl.TIME_REFERENCE.IMAGE
                ).get_nanoseconds()
            )
            if hardware_ns > 0:
                return hardware_ns
        except Exception:
            pass
        return time.time_ns()

    def close(self) -> None:
        super().close()
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.camera = None


class RealSenseWristCamera(_ThreadedCamera):
    """One wrist RealSense, color stream only.

    Ported from `local_camera.LocalCamera2` / `RealSenseCamera`. Serial number
    is required in practice: with two identical wrist cameras on one bus,
    leaving it None makes which arm you get a race.

    Depth is not enabled -- the policy only consumes color, and the depth stream
    costs USB bandwidth that would otherwise show up as dropped color frames.
    """

    thread_name = "realsense-wrist-capture"
    def __init__(
        self,
        *,
        serial_number: str | None = None,
        height: int = 480,
        width: int = 640,
        frequency_hz: int = 30,
        jpeg_quality: int = 90,
        label: str = "wrist",
        frame_timeout_ms: int = 5000,
    ) -> None:
        super().__init__(jpeg_quality=jpeg_quality)
        self.serial_number = serial_number
        self.height = int(height)
        self.width = int(width)
        self.frequency_hz = int(frequency_hz)
        self.label = str(label)
        self.frame_timeout_ms = int(frame_timeout_ms)
        if self.frame_timeout_ms <= 0:
            raise ValueError("RealSense frame timeout must be positive")
        self.rs: Any = None
        self.pipeline: Any = None
        self._frame_timestamp_ns: int | None = None

    def device_available(self) -> bool:
        ensure_optional_runtime_paths()
        try:
            if self.rs is None:
                self.rs = importlib.import_module("pyrealsense2")
            devices = self.rs.context().query_devices()
            if self.serial_number is None:
                return len(devices) > 0
            return any(
                device.get_info(self.rs.camera_info.serial_number)
                == str(self.serial_number)
                for device in devices
            )
        except Exception:
            return False

    def connect(self) -> None:
        ensure_optional_runtime_paths()
        try:
            self.rs = importlib.import_module("pyrealsense2")
        except (ImportError, OSError) as exc:
            raise HardwareDependencyError(
                "pyrealsense2 is unavailable; run: "
                "python3 -m pip install --user pyrealsense2"
            ) from exc
        try:
            self._open_pipeline()
        except Exception as exc:
            self.close()
            raise HardwareConnectionError(
                f"RealSense {self.label} (sn={self.serial_number}) open failed: {exc}"
            ) from exc

    def _open_pipeline(self) -> None:
        pipeline = self.rs.pipeline()
        config = self.rs.config()
        if self.serial_number is not None:
            config.enable_device(str(self.serial_number))
        config.enable_stream(
            self.rs.stream.color,
            self.width,
            self.height,
            self.rs.format.bgr8,
            self.frequency_hz,
        )
        pipeline.start(config)
        self.pipeline = pipeline

    def _stop_pipeline(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _grab_rgb(self) -> np.ndarray | None:
        if self.pipeline is None:
            raise HardwareConnectionError(f"RealSense {self.label} pipeline is closed")
        try:
            frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
        except Exception as exc:
            raise HardwareConnectionError(
                f"RealSense {self.label} capture timed out: {exc}"
            ) from exc
        sampled_at_ns = time.time_ns()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        self._frame_timestamp_ns = sampled_at_ns
        bgr = np.asanyarray(color_frame.get_data())
        return np.ascontiguousarray(bgr)

    def _sample_timestamp_ns(self) -> int:
        if self._frame_timestamp_ns:
            return self._frame_timestamp_ns
        return time.time_ns()

    def close(self) -> None:
        super().close()
        self._stop_pipeline()
