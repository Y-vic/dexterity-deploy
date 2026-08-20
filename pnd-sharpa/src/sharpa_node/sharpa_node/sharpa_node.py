#!/usr/bin/env python3
"""Control Sharpa hands from retargeted joints and hand control mode."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import signal
import sys
import threading
import time
import faulthandler
from array import array
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from teleop_interfaces.msg import (
    SharpaJointState,
    TactileContactPoints,
    TactileContactPointsArray,
    TactileDeformImage,
    TactileDeformImageArray,
    TactileForce6D,
    TactileForce6DArray,
)

from sharpa_node.common import (
    LEFT_JOINT_NAMES,
    MODE_DAMPING,
    MODE_TELEOP,
    MODE_UNSET_TELEOP,
    MODE_ZERO,
    RIGHT_JOINT_NAMES,
    SHARPA_JOINT_NAMES,
    TACTILE_FINGER_NAMES_BY_CHANNEL,
    age_ms,
    as_bool,
    fixed_float_vector,
    json_msg,
    parse_control_status,
    sdk_path,
    transient_local_qos,
)


@dataclass
class TimedTarget:
    stamp: float
    left_rad: list[float] | None
    left_vel: list[float] | None
    left_tau: list[float] | None
    right_rad: list[float] | None
    right_vel: list[float] | None
    right_tau: list[float] | None
    present_count: int
    control_mode: str


# Aggregate tactile entry order is fixed by downstream consumers:
# 0 right_pinky, 1 right_ring, 2 right_middle, 3 right_index, 4 right_thumb,
# 5 left_pinky, 6 left_ring, 7 left_middle, 8 left_index, 9 left_thumb.
TACTILE_AGGREGATE_ORDER: tuple[tuple[str, str], ...] = (
    ("right", "pinky"),
    ("right", "ring"),
    ("right", "middle"),
    ("right", "index"),
    ("right", "thumb"),
    ("left", "pinky"),
    ("left", "ring"),
    ("left", "middle"),
    ("left", "index"),
    ("left", "thumb"),
)


class SharpaNode(Node):
    """Small hardware state machine around the Sharpa Wave SDK."""

    def __init__(self) -> None:
        faulthandler.enable(all_threads=True)
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        except RuntimeError:
            pass
        super().__init__("sharpa")

        self.declare_parameter(
            "sdk_python_path",
            os.environ.get(
                "SHARPA_WAVE_SDK_PYTHON",
                sdk_path("sharpa-wave-sdk", "python"),
            ),
        )
        self.declare_parameter("left_sn", os.environ.get("LEFT_SN", "CF50953ACF51"))
        self.declare_parameter("right_sn", os.environ.get("RIGHT_SN", "C8559538C854"))
        self.declare_parameter(
            "left_ip", os.environ.get("LEFT_HAND_IP", "10.10.10.201")
        )
        self.declare_parameter(
            "right_ip", os.environ.get("RIGHT_HAND_IP", "10.10.10.202")
        )
        self.declare_parameter("allow_missing_hands", True)
        self.declare_parameter("connect_on_start", True)
        self.declare_parameter("status_topic", "/control_status")
        self.declare_parameter("status_json_topic", "/teleop/status_json")
        self.declare_parameter(
            "retargeted_joints_topic", "/sharpa_command_joint_states"
        )
        self.declare_parameter("joint_states_topic", "/sharpa_physical_joint_states")
        self.declare_parameter("command_snapshot_topic", "/sharpa_command_snapshot")
        self.declare_parameter("publish_command_snapshot", True)
        self.declare_parameter("command_snapshot_max_hz", 30.0)
        self.declare_parameter("sharpa_status_topic", "/sharpa_physical_status")
        self.declare_parameter("tactile_topic_prefix", "/sharpa_physical_tactile")
        self.declare_parameter(
            "tactile_status_topic", "/sharpa_physical_tactile_status"
        )
        self.declare_parameter("publish_tactile", True)
        self.declare_parameter("tactile_rate_hz", 30.0)
        self.declare_parameter("tactile_poll_warmup_s", 5.0)
        self.declare_parameter("tactile_fresh_timeout_s", 0.25)
        self.declare_parameter("tactile_sensor_time_max_age_s", 1.0)
        self.declare_parameter("tactile_error_log_period_s", 1.0)
        self.declare_parameter("tactile_auto_retry_alternate_port", True)
        self.declare_parameter("command_rate_hz", 60.0)
        self.declare_parameter("feedback_rate_hz", 30.0)
        self.declare_parameter("status_period", 0.5)
        self.declare_parameter("reconnect_period", 2.0)
        self.declare_parameter("target_timeout", 0.25)
        self.declare_parameter("command_mode", "position")
        self.declare_parameter("mit_torque_limit", 2.0)
        self.declare_parameter("speed_coeff", 0.35)
        self.declare_parameter("current_coeff", 0.50)
        self.declare_parameter("zero_on_shutdown", False)
        self.declare_parameter("zero_when_target_stale", False)
        self.declare_parameter("use_floating_mode", False)
        self.declare_parameter("startup_zero_hold_s", 0.0)
        self.declare_parameter("initial_mode", MODE_ZERO)

        self.sdk_python_path = str(self.get_parameter("sdk_python_path").value)
        self.left_sn = str(self.get_parameter("left_sn").value)
        self.right_sn = str(self.get_parameter("right_sn").value)
        self.left_ip = self._normalize_expected_ip(
            self.get_parameter("left_ip").value,
            "left_ip",
        )
        self.right_ip = self._normalize_expected_ip(
            self.get_parameter("right_ip").value,
            "right_ip",
        )
        self.allow_missing_hands = as_bool(self.get_parameter("allow_missing_hands").value)
        self.connect_on_start = as_bool(self.get_parameter("connect_on_start").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_json_topic = str(self.get_parameter("status_json_topic").value)
        self.retargeted_joints_topic = str(
            self.get_parameter("retargeted_joints_topic").value
        )
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.command_snapshot_topic = str(
            self.get_parameter("command_snapshot_topic").value
        )
        self.publish_command_snapshot = as_bool(
            self.get_parameter("publish_command_snapshot").value
        )
        self.command_snapshot_max_hz = float(
            self.get_parameter("command_snapshot_max_hz").value
        )
        self.sharpa_status_topic = str(self.get_parameter("sharpa_status_topic").value)
        self.tactile_topic_prefix = str(
            self.get_parameter("tactile_topic_prefix").value
        ).rstrip("/")
        self.tactile_status_topic = str(self.get_parameter("tactile_status_topic").value)
        self.publish_tactile = as_bool(self.get_parameter("publish_tactile").value)
        self.tactile_rate_hz = float(self.get_parameter("tactile_rate_hz").value)
        self.tactile_poll_warmup_s = float(
            self.get_parameter("tactile_poll_warmup_s").value
        )
        self.tactile_fresh_timeout_s = float(
            self.get_parameter("tactile_fresh_timeout_s").value
        )
        self.tactile_sensor_time_max_age_s = float(
            self.get_parameter("tactile_sensor_time_max_age_s").value
        )
        self.tactile_error_log_period_s = float(
            self.get_parameter("tactile_error_log_period_s").value
        )
        self.tactile_auto_retry_alternate_port = as_bool(
            self.get_parameter("tactile_auto_retry_alternate_port").value
        )
        self.command_rate_hz = float(self.get_parameter("command_rate_hz").value)
        self.feedback_rate_hz = float(self.get_parameter("feedback_rate_hz").value)
        self.status_period = float(self.get_parameter("status_period").value)
        self.reconnect_period = float(self.get_parameter("reconnect_period").value)
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.command_mode = self._normalize_command_mode(
            self.get_parameter("command_mode").value
        )
        self.mit_torque_limit = float(self.get_parameter("mit_torque_limit").value)
        self.speed_coeff = float(self.get_parameter("speed_coeff").value)
        self.current_coeff = float(self.get_parameter("current_coeff").value)
        self.zero_on_shutdown = as_bool(self.get_parameter("zero_on_shutdown").value)
        self.zero_when_target_stale = as_bool(
            self.get_parameter("zero_when_target_stale").value
        )
        self.use_floating_mode = as_bool(
            self.get_parameter("use_floating_mode").value
        )
        self.startup_zero_hold_s = float(
            self.get_parameter("startup_zero_hold_s").value
        )
        initial_status = parse_control_status(
            str(self.get_parameter("initial_mode").value)
        )
        self.mode = initial_status.mode if initial_status.known else MODE_DAMPING
        self.teleop_state = initial_status.teleop_state
        self.sharpa_active = initial_status.sharpa_active
        if self.command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be positive")
        if self.feedback_rate_hz <= 0.0:
            raise ValueError("feedback_rate_hz must be positive")
        if self.tactile_rate_hz <= 0.0:
            raise ValueError("tactile_rate_hz must be positive")
        if self.tactile_poll_warmup_s < 0.0:
            raise ValueError("tactile_poll_warmup_s must be non-negative")
        if self.tactile_fresh_timeout_s <= 0.0:
            raise ValueError("tactile_fresh_timeout_s must be positive")
        if self.tactile_sensor_time_max_age_s <= 0.0:
            raise ValueError("tactile_sensor_time_max_age_s must be positive")
        if self.tactile_error_log_period_s <= 0.0:
            raise ValueError("tactile_error_log_period_s must be positive")
        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")
        if self.reconnect_period <= 0.0:
            raise ValueError("reconnect_period must be positive")
        if self.mit_torque_limit < 0.0:
            raise ValueError("mit_torque_limit must be non-negative")
        if self.command_snapshot_max_hz < 0.0:
            raise ValueError("command_snapshot_max_hz must be non-negative")
        if self.startup_zero_hold_s < 0.0:
            raise ValueError("startup_zero_hold_s must be non-negative")

        self.create_subscription(
            String, self.status_topic, self._on_status, transient_local_qos()
        )
        self.create_subscription(
            String,
            self.status_json_topic,
            self._on_status_json,
            transient_local_qos(),
        )
        self.create_subscription(
            JointState, self.retargeted_joints_topic, self._on_target, 10
        )
        self.joint_pub = self.create_publisher(
            SharpaJointState, self.joint_states_topic, 10
        )
        self.command_snapshot_pub = self.create_publisher(
            String, self.command_snapshot_topic, 10
        )
        self.status_pub = self.create_publisher(String, self.sharpa_status_topic, 10)

        tactile_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.tactile_deform_pub = self.create_publisher(
            TactileDeformImageArray,
            f"{self.tactile_topic_prefix}/deform_images",
            tactile_qos,
        )
        self.tactile_force_pub = self.create_publisher(
            TactileForce6DArray,
            f"{self.tactile_topic_prefix}/force6d",
            tactile_qos,
        )
        self.tactile_contact_pub = self.create_publisher(
            TactileContactPointsArray,
            f"{self.tactile_topic_prefix}/contact_points",
            tactile_qos,
        )
        self.tactile_status_pub = self.create_publisher(
            String, self.tactile_status_topic, 10
        )

        self.last_status_payload: dict[str, Any] | None = None
        self.recording_active = False
        self.last_mode_event = "startup"
        self.target: TimedTarget | None = None
        self.sdk_lock = threading.Lock()
        self.last_successful_q_cmd = [math.nan] * len(SHARPA_JOINT_NAMES)
        self.last_successful_q_cmd_valid = False
        self.tactile_lock = threading.Lock()
        self.manager: Any | None = None
        self.left: Any | None = None
        self.right: Any | None = None
        self.ControlMode: Any | None = None
        self.ControlSource: Any | None = None
        self.FaultCode: Any | None = None
        self.SharpaWaveManager: Any | None = None
        self.connected_devices: list[str] = []
        self.discovered_ip_by_sn: dict[str, str] = {}
        self.hand_side_by_sn: dict[str, str] = {}
        self.active_sdk_mode = ""
        self.sdk_loaded = False
        self.connected = False
        self.last_error = ""
        self.last_tactile_error = ""
        self.last_tactile_terminal_error = ""
        self.last_tactile_terminal_error_time = 0.0
        self.last_command_reason = "startup_damping"
        self.latest_joint_state_snapshot: dict[str, Any] | None = None
        self.latest_joint_state_mono: float | None = None
        self.latest_joint_state_unix_ns: int | None = None
        self.command_snapshot_count = 0
        self.last_command_snapshot_mono = 0.0
        self.command_count = 0
        self.feedback_count = 0
        self.tactile_frame_count = 0
        self.tactile_publish_count = 0
        self.tactile_window_frames = 0
        self.tactile_window_publishes = 0
        self.tactile_hz = 0.0
        self.tactile_window_time = time.monotonic()
        self.tactile_poll_ready_at = 0.0
        self.tactile_debug_ticks = 0
        self.tactile_ready_by_side: dict[str, bool | None] = {
            "left": None,
            "right": None,
        }
        self.tactile_counts_by_key: dict[str, int] = defaultdict(int)
        self.tactile_duplicate_counts_by_key: dict[str, int] = defaultdict(int)
        self.tactile_invalid_counts_by_key: dict[str, int] = defaultdict(int)
        self.latest_tactile_deforms: dict[tuple[str, str], TactileDeformImage] = {}
        self.latest_tactile_forces: dict[tuple[str, str], TactileForce6D] = {}
        self.latest_tactile_contacts: dict[tuple[str, str], TactileContactPoints] = {}
        self.tactile_last_signatures: dict[tuple[str, str], tuple[int, float]] = {}
        self.tactile_last_update_mono: dict[tuple[str, str], float] = {}
        self.status_window_commands = 0
        self.status_window_feedback = 0
        self.status_window_time = time.monotonic()
        self.command_hz = 0.0
        self.feedback_hz = 0.0
        self.last_command_time: float | None = None
        self.last_side_command_time: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.side_command_count: dict[str, int] = {"left": 0, "right": 0}
        self.last_feedback_time: float | None = None
        self.last_target_time: float | None = None
        self.startup_zero_until = 0.0

        if self.connect_on_start:
            self._connect()
        else:
            self.last_error = "connect_on_start is false; hardware control is disabled"

        self.create_timer(1.0 / self.command_rate_hz, self._command_tick)
        self.create_timer(1.0 / self.feedback_rate_hz, self._feedback_tick)
        self.create_timer(1.0 / self.tactile_rate_hz, self._publish_tactile_aggregates)
        self.create_timer(self.reconnect_period, self._refresh_configured_hands)
        self.create_timer(self.status_period, self._publish_status)
        self.create_timer(self.status_period, self._publish_tactile_status)

        self.get_logger().info(
            f"Sharpa node started in {self.mode}. "
            f"status={self.status_topic}, retarget={self.retargeted_joints_topic}"
        )

    def _connect(self) -> None:
        try:
            self._load_sdk()
            self.manager = self.SharpaWaveManager.get_instance()
            time.sleep(1.5)
            self._refresh_discovered_devices()
            missing = [
                f"{label}={sn}"
                for label, sn in (("left", self.left_sn), ("right", self.right_sn))
                if sn and sn not in self.connected_devices
            ]
            if missing and not self.allow_missing_hands:
                raise RuntimeError(
                    f"Expected {', '.join(missing)}, discovered={self.connected_devices}"
                )
            if self.left_sn and self.left_sn in self.connected_devices:
                self.left = self._connect_hand(self.left_sn, "left")
                self.get_logger().info(f"Sharpa left hand connected: {self.left_sn}")
            if self.right_sn and self.right_sn in self.connected_devices:
                self.right = self._connect_hand(self.right_sn, "right")
                self.get_logger().info(f"Sharpa right hand connected: {self.right_sn}")
            if self.left is None and self.right is None:
                raise RuntimeError(
                    f"No configured Sharpa hands discovered: {self.connected_devices}"
                )
            self.connected = True
            self.last_error = ""
            if self.mode == MODE_DAMPING:
                self._ensure_floating_mode()
            elif self.mode == MODE_ZERO:
                self._ensure_position_mode()
                self._send_zero(f"startup_{self.mode}")
                if self.startup_zero_hold_s > 0.0:
                    self.startup_zero_until = (
                        time.monotonic() + self.startup_zero_hold_s
                    )
                    self.get_logger().info(
                        "Sharpa startup zero hold active for "
                        f"{self.startup_zero_hold_s:.2f}s on both hands"
                    )
            elif self.mode == MODE_UNSET_TELEOP:
                self._transition_unset_teleop("startup_unset_teleop")
            else:
                self._ensure_position_mode()
        except Exception as exc:
            self.connected = False
            self.last_error = str(exc)
            self.get_logger().error(f"Sharpa SDK connection failed: {exc}")

    def _refresh_configured_hands(self) -> None:
        if not self.connect_on_start:
            return
        try:
            if self.manager is None:
                self._connect()
                return

            discovered = self._refresh_discovered_devices()

            for label, sn, attr in (
                ("left", self.left_sn, "left"),
                ("right", self.right_sn, "right"),
            ):
                hand = getattr(self, attr)
                if hand is not None and sn and sn not in discovered:
                    self._drop_hand(label, sn, attr, "missing_from_discovery")
                    continue
                if hand is not None and sn and self._manager_connected(sn) is False:
                    self._drop_hand(label, sn, attr, "manager_not_connected")
                    continue
                if hand is not None and self._hand_ready(hand) is False:
                    self._drop_hand(label, sn, attr, "hand_not_ready")

            connected_new = False
            for label, sn, attr in (
                ("left", self.left_sn, "left"),
                ("right", self.right_sn, "right"),
            ):
                if not sn or sn not in discovered or getattr(self, attr) is not None:
                    continue
                hand = self._connect_hand(sn, label)
                setattr(self, attr, hand)
                connected_new = True
                self.get_logger().info(f"Sharpa {label} hand connected: {sn}")

            if self.left is None and self.right is None:
                self.connected = False
                self.last_error = (
                    f"No configured Sharpa hands discovered: {self.connected_devices}"
                )
                return

            self.connected = True
            if connected_new:
                self._sync_mode_after_reconnect()
                self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
            self.get_logger().warn(f"Sharpa reconnect warning: {exc}")

    def _drop_hand(self, label: str, serial_number: str, attr: str, reason: str) -> None:
        with self.sdk_lock:
            self._invalidate_q_cmd_cache_locked()
            hand = getattr(self, attr)
            setattr(self, attr, None)
            self.hand_side_by_sn.pop(serial_number, None)
            self.active_sdk_mode = ""
            if hand is not None:
                try:
                    hand.stop()
                except Exception as exc:
                    self.get_logger().warn(
                        f"Sharpa {label} hand stop warning after {reason}: {exc}"
                    )
            if self.manager is not None and serial_number:
                try:
                    if self._manager_connected(serial_number) is not False:
                        self.manager.disconnect(serial_number)
                except Exception as exc:
                    self.get_logger().warn(
                        f"Sharpa {label} hand manager disconnect warning "
                        f"after {reason}: {exc}"
                    )
        self.get_logger().warn(
            f"Sharpa {label} hand disconnected: {serial_number} ({reason})"
        )
        self._clear_tactile_side(label)

    def _load_sdk(self) -> None:
        if self.sdk_loaded:
            return
        if self.sdk_python_path not in sys.path:
            sys.path.insert(0, self.sdk_python_path)
        from sharpa import ControlMode, ControlSource, FaultCode, SharpaWaveManager  # type: ignore

        self.ControlMode = ControlMode
        self.ControlSource = ControlSource
        self.FaultCode = FaultCode
        self.SharpaWaveManager = SharpaWaveManager
        self.sdk_loaded = True

    def _active_control_mode(self) -> Any:
        if self.command_mode == "mit":
            return self.ControlMode.MIT
        return self.ControlMode.POSITION

    @staticmethod
    def _normalize_expected_ip(value: Any, parameter_name: str) -> str:
        text = str(value).strip()
        if not text:
            return ""
        try:
            parsed = ipaddress.ip_address(text)
        except ValueError as exc:
            raise ValueError(f"{parameter_name} must be a valid IPv4 address") from exc
        if parsed.version != 4:
            raise ValueError(f"{parameter_name} must be an IPv4 address")
        return str(parsed)

    def _refresh_discovered_devices(self) -> list[str]:
        discovered_ip_by_sn: dict[str, str] = {}
        get_all_devices = getattr(self.manager, "get_all_devices", None)
        if callable(get_all_devices):
            for info in list(get_all_devices()):
                serial_number = str(getattr(info, "sn", "")).strip()
                if not serial_number:
                    continue
                discovered_ip_by_sn[serial_number] = str(
                    getattr(info, "ip", "")
                ).strip()
            discovered = list(discovered_ip_by_sn)
        else:
            discovered = [
                str(serial_number)
                for serial_number in list(self.manager.get_all_device_sn())
            ]
        self.connected_devices = discovered
        self.discovered_ip_by_sn = discovered_ip_by_sn
        return discovered

    def _expected_ip(self, side: str) -> str:
        return self.left_ip if side == "left" else self.right_ip

    def _validate_discovered_ip(self, serial_number: str, side: str) -> None:
        expected_ip = self._expected_ip(side)
        discovered_ip = self.discovered_ip_by_sn.get(serial_number, "")
        if expected_ip and discovered_ip and discovered_ip != expected_ip:
            raise RuntimeError(
                f"Sharpa {side} hand {serial_number} discovered at "
                f"{discovered_ip}, expected {expected_ip}. Migrate the hand "
                "network configuration before starting sharpa_node."
            )

    def _connect_hand(self, serial_number: str, side: str) -> Any:
        with self.sdk_lock:
            self._invalidate_q_cmd_cache_locked()
            self._validate_discovered_ip(serial_number, side)
            if self._manager_connected(serial_number) is True:
                self.manager.disconnect(serial_number)
                time.sleep(0.05)
            hand = self.manager.connect(serial_number, not self.publish_tactile)
            self.hand_side_by_sn[serial_number] = side
            initial_sdk_mode = (
                self.ControlMode.FLOATING
                if self.mode == MODE_DAMPING
                else self._active_control_mode()
            )
            initial_sdk_mode_label = self._sdk_mode_label(initial_sdk_mode)
            self._check(
                f"{serial_number} set_control_mode",
                hand.set_control_mode(initial_sdk_mode),
            )
            self._warn_if_error(
                f"{serial_number} set_speed_coeff",
                hand.set_speed_coeff(self.speed_coeff),
            )
            self._warn_if_error(
                f"{serial_number} set_current_coeff",
                hand.set_current_coeff(self.current_coeff),
            )
            self._check(
                f"{serial_number} set_control_source",
                hand.set_control_source(self.ControlSource.SDK),
            )
            if not hand.start() and not self._retry_tactile_alternate_port(
                hand,
                serial_number,
                side,
            ):
                raise RuntimeError(f"{serial_number} start failed")
            if self.publish_tactile:
                self.tactile_poll_ready_at = max(
                    self.tactile_poll_ready_at,
                    time.monotonic() + self.tactile_poll_warmup_s,
                )
        self.active_sdk_mode = initial_sdk_mode_label
        return hand

    def _sdk_mode_label(self, sdk_mode: Any) -> str:
        if sdk_mode == self.ControlMode.POSITION:
            return "POSITION"
        if sdk_mode == self.ControlMode.MIT:
            return "MIT"
        return "FLOATING"

    def _retry_tactile_alternate_port(
        self,
        hand: Any,
        serial_number: str,
        side: str,
    ) -> bool:
        if not self.publish_tactile or not self.tactile_auto_retry_alternate_port:
            return False
        retry = getattr(hand, "retry_tactile_alternate_port", None)
        if not callable(retry):
            return False

        alternates: list[int] = []
        get_alternates = getattr(hand, "get_tactile_alternate_ports", None)
        if callable(get_alternates):
            try:
                alternates = [int(port) for port in list(get_alternates())]
            except Exception as exc:
                self.get_logger().warn(
                    f"Sharpa {side} alternate tactile port list failed: {exc}"
                )

        suffix = f" candidates={alternates}" if alternates else ""
        self.get_logger().warn(
            f"Sharpa {side} hand start failed; trying alternate tactile port for "
            f"{serial_number}.{suffix}"
        )
        try:
            recovered = bool(retry())
        except Exception as exc:
            self.get_logger().warn(
                f"Sharpa {side} alternate tactile port retry failed: {exc}"
            )
            return False
        if recovered:
            self.get_logger().warn(
                f"Sharpa {side} tactile recovered on an alternate port: "
                f"{serial_number}"
            )
        return recovered

    def _manager_connected(self, serial_number: str) -> bool | None:
        if self.manager is None or not serial_number:
            return None
        if not hasattr(self.manager, "is_connected"):
            return None
        try:
            return bool(self.manager.is_connected(serial_number))
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def _hand_ready(self, hand: Any | None) -> bool | None:
        if hand is None or not hasattr(hand, "is_hand_ready"):
            return None
        try:
            return bool(hand.is_hand_ready())
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def _sync_mode_after_reconnect(self) -> None:
        self.active_sdk_mode = ""
        if not self.connected:
            return
        if self.mode == MODE_DAMPING or not self.sharpa_active:
            self._ensure_floating_mode()
            self.last_command_reason = f"reconnect_{self.teleop_state}_floating"
            return
        target = self._fresh_target()
        if target is not None and self.mode == MODE_TELEOP:
            self._ensure_command_mode(target.control_mode)
            self._send_target(target)
        elif self.mode in {MODE_ZERO, MODE_UNSET_TELEOP}:
            self._ensure_position_mode()
            self._send_zero(f"reconnect_{self.mode}_zero")

    @staticmethod
    def _payload_bool(payload: dict[str, Any] | None, key: str) -> bool | None:
        if not isinstance(payload, dict) or key not in payload:
            return None
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "active"}
        return bool(value)

    def _on_status(self, msg: String) -> None:
        status = parse_control_status(msg.data)
        mode = status.mode
        payload = status.payload
        if (
            mode != self.mode
            or status.teleop_state != self.teleop_state
            or status.sharpa_active != self.sharpa_active
        ):
            self.get_logger().info(
                f"Sharpa status -> {mode} "
                f"(teleop_state={status.teleop_state}, active={status.sharpa_active})"
            )
        self.mode = mode
        self.teleop_state = status.teleop_state
        self.sharpa_active = status.sharpa_active
        self.last_status_payload = payload
        recording_active = self._payload_bool(payload, "t_record")
        if recording_active is not None:
            self.recording_active = recording_active
        self.last_mode_event = "" if payload is None else str(payload.get("event", ""))
        if not status.known:
            self.last_error = f"unknown control status: {msg.data}"
        else:
            self.last_error = ""

    def _on_status_json(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.last_error = f"malformed status_json: {exc}"
            return
        recording_active = self._payload_bool(payload, "t_record")
        if recording_active is not None:
            self.recording_active = recording_active

    def _on_target(self, msg: JointState) -> None:
        receive_mono = time.monotonic()
        receive_unix_ns = time.time_ns()
        receive_stamp = self.get_clock().now().to_msg()
        left, left_vel, left_tau, left_count = self._extract_side_target(
            msg, LEFT_JOINT_NAMES
        )
        right, right_vel, right_tau, right_count = self._extract_side_target(
            msg, RIGHT_JOINT_NAMES
        )
        control_mode = self._target_control_mode_from_msg(msg)
        self.target = TimedTarget(
            stamp=receive_mono,
            left_rad=left,
            left_vel=left_vel,
            left_tau=left_tau,
            right_rad=right,
            right_vel=right_vel,
            right_tau=right_tau,
            present_count=left_count + right_count,
            control_mode=control_mode,
        )
        self.last_target_time = self.target.stamp
        self._publish_command_snapshot_msg(
            msg,
            receive_mono=receive_mono,
            receive_unix_ns=receive_unix_ns,
            receive_stamp=receive_stamp,
            control_mode=control_mode,
            present_count=left_count + right_count,
        )

    def _command_tick(self) -> None:
        if not self.connected:
            return
        try:
            if time.monotonic() < self.startup_zero_until:
                self._ensure_position_mode()
                self._send_zero("startup_zero_hold")
                return
            if self.mode == MODE_DAMPING:
                self._ensure_floating_mode()
                self.last_command_reason = f"inactive_{self.teleop_state}_floating"
                return
            if self.mode == MODE_ZERO:
                self._ensure_position_mode()
                self._send_zero(f"mode_{self.mode}")
                return
            if self.mode == MODE_UNSET_TELEOP:
                self._transition_unset_teleop("command_tick_unset_teleop")
                return
            if not self.sharpa_active:
                self._ensure_floating_mode()
                self.last_command_reason = f"inactive_{self.teleop_state}_floating"
                return
            target = self._fresh_target()
            if target is None:
                if self.zero_when_target_stale:
                    self._send_zero("sharpa_active_target_missing_zero")
                else:
                    self.last_command_reason = "sharpa_active_target_missing_hold"
                return
            self._ensure_command_mode(target.control_mode)
            self._send_target(target)
        except Exception as exc:
            self.last_error = str(exc)
            self.get_logger().warn(f"Sharpa command warning: {exc}")

    def _feedback_tick(self) -> None:
        if not self.connected:
            return
        try:
            snapshot = self._read_joint_state_snapshot()
            if snapshot is None:
                return
            msg = SharpaJointState()
            msg.joint_state.header.stamp = self.get_clock().now().to_msg()
            msg.joint_state.header.frame_id = "sharpa_base"
            msg.joint_state.name = list(snapshot["name"])
            msg.joint_state.position = list(snapshot["position"])
            msg.joint_state.velocity = list(snapshot["velocity"])
            msg.joint_state.effort = list(snapshot["effort"])
            msg.q_cmd = list(snapshot["q_cmd"])
            msg.q_cmd_valid = bool(snapshot["q_cmd_valid"])
            snapshot["header_stamp_ns"] = self._stamp_to_ns(msg.joint_state.header.stamp)
            self.latest_joint_state_snapshot = snapshot
            self.latest_joint_state_mono = float(snapshot["read_mono"])
            self.latest_joint_state_unix_ns = int(snapshot["read_unix_ns"])
            self.joint_pub.publish(msg)
            self.feedback_count += 1
            self.status_window_feedback += 1
            self.last_feedback_time = float(snapshot["read_mono"])
        except Exception as exc:
            self.last_error = str(exc)

    def _read_joint_state_snapshot(self) -> dict[str, Any] | None:
        left_pos = [0.0] * len(LEFT_JOINT_NAMES)
        left_vel = [0.0] * len(LEFT_JOINT_NAMES)
        left_effort = [0.0] * len(LEFT_JOINT_NAMES)
        right_pos = [0.0] * len(RIGHT_JOINT_NAMES)
        right_vel = [0.0] * len(RIGHT_JOINT_NAMES)
        right_effort = [0.0] * len(RIGHT_JOINT_NAMES)
        with self.sdk_lock:
            if self.left is not None:
                state = self.left.get_states()
                left_pos, left_vel, left_effort = self._state_vectors(state)
            if self.right is not None:
                state = self.right.get_states()
                right_pos, right_vel, right_effort = self._state_vectors(state)
            q_cmd = list(self.last_successful_q_cmd)
            q_cmd_valid = bool(self.last_successful_q_cmd_valid)
        if self.left is None and self.right is None:
            return None
        return {
            "name": list(SHARPA_JOINT_NAMES),
            "position": left_pos + right_pos,
            "velocity": left_vel + right_vel,
            "effort": left_effort + right_effort,
            "q_cmd": q_cmd,
            "q_cmd_valid": q_cmd_valid,
            "read_mono": time.monotonic(),
            "read_unix_ns": time.time_ns(),
            "header_stamp_ns": -1,
            "left_present": self.left is not None,
            "right_present": self.right is not None,
        }

    @staticmethod
    def _stamp_to_ns(stamp: Any) -> int:
        sec = int(getattr(stamp, "sec", 0))
        nanosec = int(getattr(stamp, "nanosec", 0))
        if sec == 0 and nanosec == 0:
            return -1
        return sec * 1_000_000_000 + nanosec

    @staticmethod
    def _finite_list(values: Any) -> list[float]:
        output: list[float] = []
        try:
            iterator = list(values)
        except TypeError:
            return output
        for value in iterator:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            output.append(number if math.isfinite(number) else math.nan)
        return output

    def _tactile_force_snapshot(self, receive_mono: float) -> list[dict[str, Any]]:
        now_wall = time.time()
        entries: list[dict[str, Any]] = []
        with self.tactile_lock:
            forces = dict(self.latest_tactile_forces)
            last_updates = dict(self.tactile_last_update_mono)
        for channel, (side, finger) in enumerate(TACTILE_AGGREGATE_ORDER):
            key = (side, finger)
            force = forces.get(key)
            update_mono = last_updates.get(key)
            fresh = False
            age_ms: float | None = None
            if force is not None and update_mono is not None:
                age_ms = (receive_mono - update_mono) * 1000.0
                fresh = (
                    0.0 <= age_ms <= self.tactile_fresh_timeout_s * 1000.0
                    and self._tactile_metadata_is_current(
                        int(getattr(force, "frame_id", 0)),
                        self._float_or_zero(getattr(force, "sensor_time", 0.0)),
                        now_wall,
                    )
                )
            entries.append(
                {
                    "side": side,
                    "finger": finger,
                    "channel": channel,
                    "fresh": fresh,
                    "age_ms": age_ms,
                    "frame_id": -1 if force is None else int(getattr(force, "frame_id", -1)),
                    "sensor_time": math.nan
                    if force is None
                    else self._float_or_zero(getattr(force, "sensor_time", math.nan)),
                    "force": [math.nan, math.nan, math.nan]
                    if force is None
                    else self._finite_list(getattr(force, "force", [])),
                    "torque": [math.nan, math.nan, math.nan]
                    if force is None
                    else self._finite_list(getattr(force, "torque", [])),
                }
            )
        return entries

    def _publish_command_snapshot_msg(
        self,
        msg: JointState,
        *,
        receive_mono: float,
        receive_unix_ns: int,
        receive_stamp: Any,
        control_mode: str,
        present_count: int,
    ) -> None:
        if not self.publish_command_snapshot:
            return
        if self.command_snapshot_max_hz > 0.0:
            min_period = 1.0 / self.command_snapshot_max_hz
            if receive_mono - self.last_command_snapshot_mono < min_period:
                return
        self.last_command_snapshot_mono = receive_mono
        joint_snapshot = self._read_joint_state_snapshot()
        command_joint_snapshot = None if joint_snapshot is None else dict(joint_snapshot)
        if command_joint_snapshot is not None:
            command_joint_snapshot.pop("q_cmd", None)
            command_joint_snapshot.pop("q_cmd_valid", None)
        joint_read_delay_ms = (
            None
            if joint_snapshot is None
            else (float(joint_snapshot["read_mono"]) - receive_mono) * 1000.0
        )
        payload = {
            "schema": "sharpa_command_snapshot_v1",
            "receive_unix_ns": receive_unix_ns,
            "receive_stamp_ns": self._stamp_to_ns(receive_stamp),
            "command_header_stamp_ns": self._stamp_to_ns(msg.header.stamp),
            "command_topic": self.retargeted_joints_topic,
            "control_mode": control_mode,
            "present_count": present_count,
            "mode": self.mode,
            "teleop_state": self.teleop_state,
            "sharpa_active": self.sharpa_active,
            "connected": self.connected,
            "q_cmd": {
                "name": [str(name) for name in msg.name],
                "position": self._finite_list(msg.position),
                "velocity": self._finite_list(msg.velocity),
                "effort": self._finite_list(msg.effort),
            },
            "q_exe": command_joint_snapshot,
            "q_exe_read_delay_ms": joint_read_delay_ms,
            "q_exe_recv_unix_ns": None
            if joint_snapshot is None
            else int(joint_snapshot["read_unix_ns"]),
            "tactile_force6d": self._tactile_force_snapshot(receive_mono),
        }
        out = String()
        out.data = json.dumps(payload, separators=(",", ":"), allow_nan=True)
        self.command_snapshot_pub.publish(out)
        self.command_snapshot_count += 1

    def _fresh_target(self) -> TimedTarget | None:
        if self.target is None:
            return None
        if time.monotonic() - self.target.stamp > self.target_timeout:
            return None
        return self.target

    def _target_control_mode_from_msg(self, msg: JointState) -> str:
        if self.command_mode != "auto":
            return self.command_mode
        frame_id = (msg.header.frame_id or "").strip().lower()
        if frame_id.endswith(":mit") or frame_id == "mit":
            return "mit"
        return "position"

    @staticmethod
    def _normalize_command_mode(value: Any) -> str:
        mode = str(value or "").strip().lower().replace("-", "_")
        if mode in {"mit", "torque", "position_velocity_torque"}:
            return "mit"
        if mode in {"auto", "from_msg", "message"}:
            return "auto"
        return "position"

    def _clamp_torque(self, value: float) -> float:
        if self.mit_torque_limit <= 0.0:
            return value
        return max(-self.mit_torque_limit, min(self.mit_torque_limit, value))

    def _extract_side_target(
        self, msg: JointState, side_names: list[str]
    ) -> tuple[list[float] | None, list[float] | None, list[float] | None, int]:
        index = {name: i for i, name in enumerate(msg.name)}
        positions = [0.0] * len(side_names)
        velocities = [0.0] * len(side_names)
        torques = [0.0] * len(side_names)
        present = 0
        for idx, name in enumerate(side_names):
            source_idx = index.get(name)
            if source_idx is None or source_idx >= len(msg.position):
                continue
            value = float(msg.position[source_idx])
            if not math.isfinite(value):
                continue
            positions[idx] = value
            if source_idx < len(msg.velocity):
                velocity = float(msg.velocity[source_idx])
                if math.isfinite(velocity):
                    velocities[idx] = velocity
            if source_idx < len(msg.effort):
                torque = float(msg.effort[source_idx])
                if math.isfinite(torque):
                    torques[idx] = self._clamp_torque(torque)
            present += 1
        if present == 0:
            return None, None, None, 0
        return positions, velocities, torques, present

    @staticmethod
    def _state_vectors(state: Any) -> tuple[list[float], list[float], list[float]]:
        return (
            fixed_float_vector(getattr(state, "angles", []), 22),
            fixed_float_vector(getattr(state, "velocities", []), 22),
            fixed_float_vector(getattr(state, "torques", []), 22),
        )

    def _send_target(self, target: TimedTarget) -> None:
        sent = False
        with self.sdk_lock:
            left_will_send = self.left is not None and target.left_rad is not None
            right_will_send = self.right is not None and target.right_rad is not None
            if left_will_send or right_will_send:
                self._invalidate_q_cmd_cache_locked()
            if left_will_send:
                self._set_target_rad(
                    self.left,
                    target.left_rad,
                    target.left_vel,
                    target.left_tau,
                    "left",
                    target.control_mode,
                )
                self._mark_side_command("left")
                sent = True
            if right_will_send:
                self._set_target_rad(
                    self.right,
                    target.right_rad,
                    target.right_vel,
                    target.right_tau,
                    "right",
                    target.control_mode,
                )
                self._mark_side_command("right")
                sent = True
            if sent:
                self._set_q_cmd_cache_locked(
                    target.left_rad if self.left is not None else None,
                    target.right_rad if self.right is not None else None,
                )
        if sent:
            self.command_count += 1
            self.status_window_commands += 1
            self.last_command_time = time.monotonic()
            self.last_command_reason = (
                f"{target.control_mode}_teleop_target"
                if self.teleop_state == MODE_TELEOP
                else f"{target.control_mode}_{self.teleop_state}_target"
            )

    def _transition_unset_teleop(self, reason: str) -> None:
        self._ensure_position_mode()
        self._send_zero(reason)
        self.mode = MODE_ZERO
        self.last_command_reason = f"{reason}_to_zero"

    def _send_zero(self, reason: str) -> None:
        if not self.connected:
            self.last_command_reason = reason
            return
        with self.sdk_lock:
            self._invalidate_q_cmd_cache_locked()
            if self.left is not None:
                self._set_joint_position_rad(
                    self.left, [0.0] * len(LEFT_JOINT_NAMES), "left"
                )
                self._mark_side_command("left")
            if self.right is not None:
                self._set_joint_position_rad(
                    self.right, [0.0] * len(RIGHT_JOINT_NAMES), "right"
                )
                self._mark_side_command("right")
            self._set_q_cmd_cache_locked(
                [0.0] * len(LEFT_JOINT_NAMES) if self.left is not None else None,
                [0.0] * len(RIGHT_JOINT_NAMES) if self.right is not None else None,
            )
        self.command_count += 1
        self.status_window_commands += 1
        self.last_command_time = time.monotonic()
        self.last_command_reason = reason

    def _mark_side_command(self, side: str) -> None:
        now = time.monotonic()
        self.side_command_count[side] = self.side_command_count.get(side, 0) + 1
        self.last_side_command_time[side] = now

    def _invalidate_q_cmd_cache_locked(self) -> None:
        self.last_successful_q_cmd = [math.nan] * len(SHARPA_JOINT_NAMES)
        self.last_successful_q_cmd_valid = False

    def _set_q_cmd_cache_locked(
        self,
        left_values: list[float] | None,
        right_values: list[float] | None,
    ) -> None:
        q_cmd = [math.nan] * len(SHARPA_JOINT_NAMES)
        if left_values is not None:
            for index, value in enumerate(left_values[: len(LEFT_JOINT_NAMES)]):
                q_cmd[index] = float(value)
        if right_values is not None:
            right_start = len(LEFT_JOINT_NAMES)
            for index, value in enumerate(right_values[: len(RIGHT_JOINT_NAMES)]):
                q_cmd[right_start + index] = float(value)
        self.last_successful_q_cmd = q_cmd
        self.last_successful_q_cmd_valid = len(q_cmd) == len(SHARPA_JOINT_NAMES) and all(
            math.isfinite(value) for value in q_cmd
        )

    def _set_joint_position_rad(self, hand: Any, values: list[float], label: str) -> None:
        if hasattr(hand, "set_joint_position_rad"):
            self._check(
                f"{label} set_joint_position_rad",
                hand.set_joint_position_rad(values, True),
            )
            return
        self._check(
            f"{label} set_joint_position",
            hand.set_joint_position(values, True),
        )

    def _set_target_rad(
        self,
        hand: Any,
        positions: list[float],
        velocities: list[float] | None,
        torques: list[float] | None,
        label: str,
        control_mode: str,
    ) -> None:
        if control_mode != "mit":
            self._set_joint_position_rad(hand, positions, label)
            return
        if not hasattr(hand, "set_mit_control"):
            raise RuntimeError(f"{label} hand does not expose set_mit_control")
        velocity_values = velocities or [0.0] * len(positions)
        torque_values = torques or [0.0] * len(positions)
        self._check(
            f"{label} set_mit_control",
            hand.set_mit_control(positions, velocity_values, torque_values),
        )

    def _ensure_command_mode(self, control_mode: str) -> None:
        if control_mode == "mit":
            self._ensure_mit_mode()
        else:
            self._ensure_position_mode()

    def _ensure_position_mode(self) -> None:
        if not self.connected:
            return
        with self.sdk_lock:
            if self.active_sdk_mode == "POSITION":
                return
            self._invalidate_q_cmd_cache_locked()
            for label, hand in (("left", self.left), ("right", self.right)):
                if hand is not None:
                    self._check(
                        f"{label} set_control_mode POSITION",
                        hand.set_control_mode(self.ControlMode.POSITION),
                    )
            self.active_sdk_mode = "POSITION"

    def _ensure_mit_mode(self) -> None:
        if not self.connected:
            return
        with self.sdk_lock:
            if self.active_sdk_mode == "MIT":
                return
            self._invalidate_q_cmd_cache_locked()
            for label, hand in (("left", self.left), ("right", self.right)):
                if hand is not None:
                    self._check(
                        f"{label} set_control_mode MIT",
                        hand.set_control_mode(self.ControlMode.MIT),
                    )
            self.active_sdk_mode = "MIT"

    def _ensure_floating_mode(self) -> None:
        if not self.connected:
            return
        if not self.use_floating_mode:
            return
        with self.sdk_lock:
            if self.active_sdk_mode == "FLOATING":
                return
            self._invalidate_q_cmd_cache_locked()
            for label, hand in (("left", self.left), ("right", self.right)):
                if hand is not None:
                    self._check(
                        f"{label} set_control_mode FLOATING",
                        hand.set_control_mode(self.ControlMode.FLOATING),
                    )
            self.active_sdk_mode = "FLOATING"

    def _on_tactile_frame(
        self,
        frame: Any,
        side_override: str | None = None,
        channel_override: int | None = None,
    ) -> None:
        if not self.publish_tactile:
            return
        try:
            side = side_override
            if side not in {"left", "right"}:
                return
            channel = (
                int(channel_override)
                if channel_override is not None
                else self._int_or_default(self._mapping_get(frame, "channel"), -1)
            )
            finger = self._finger_from_channel(channel)
            if finger is None:
                return

            content = self._mapping_get(frame, "content", {}) or {}
            shape = self._mapping_get(frame, "shape", {}) or {}
            frame_id = self._uint32(self._mapping_get(frame, "frame_id", 0))
            sensor_time = self._float_or_zero(self._mapping_get(frame, "ts", 0.0))
            key_tuple = (side, finger)
            key = f"{side}/{finger}"
            wall_now = time.time()
            if not self._tactile_metadata_is_current(frame_id, sensor_time, wall_now):
                with self.tactile_lock:
                    self.tactile_invalid_counts_by_key[key] += 1
                self.last_tactile_error = (
                    f"{key} stale tactile metadata: "
                    f"frame_id={frame_id}, sensor_time={sensor_time:.6f}"
                )
                return

            signature = (frame_id, sensor_time)
            with self.tactile_lock:
                previous_signature = self.tactile_last_signatures.get(key_tuple)
            if previous_signature == signature:
                with self.tactile_lock:
                    self.tactile_duplicate_counts_by_key[key] += 1
                return

            deform = self._mapping_get(content, "DEFORM")
            deform_shape = self._mapping_get(shape, "DEFORM")
            if deform is None:
                deform = self._mapping_get(frame, "deform_data")
                deform_shape = self._mapping_get(frame, "deform_shape")

            force6d = self._mapping_get(content, "F6")
            if force6d is None:
                force6d = self._mapping_get(frame, "f6_data")

            contact_points = self._mapping_get(content, "CONTACT_POINT")
            if contact_points is None:
                contact_points = self._mapping_get(frame, "contact_point_data")

            deform_msg = (
                self._deform_image_msg(
                    deform, deform_shape, side, finger, channel, frame_id, sensor_time
                )
                if deform is not None
                else None
            )
            force_msg = (
                self._force6d_msg(force6d, side, finger, channel, frame_id, sensor_time)
                if force6d is not None
                else None
            )
            if contact_points is not None:
                contact_msg = self._contact_points_msg(
                    contact_points, side, finger, channel, frame_id, sensor_time
                )
            elif deform_msg is not None or force_msg is not None:
                # No CONTACT_POINT payload means no active contact points for this
                # fresh tactile frame, not a missing/stale sensor channel.
                contact_msg = self._empty_contact_points(side, finger, channel)
                contact_msg.frame_id = frame_id
                contact_msg.sensor_time = sensor_time
            else:
                contact_msg = None
            if deform_msg is None and force_msg is None and contact_msg is None:
                with self.tactile_lock:
                    self.tactile_invalid_counts_by_key[key] += 1
                self.last_tactile_error = f"{key} tactile frame has no usable payload"
                return

            with self.tactile_lock:
                if deform_msg is not None:
                    self.latest_tactile_deforms[key_tuple] = deform_msg
                if force_msg is not None:
                    self.latest_tactile_forces[key_tuple] = force_msg
                if contact_msg is not None:
                    self.latest_tactile_contacts[key_tuple] = contact_msg
                self.tactile_last_signatures[key_tuple] = signature
                self.tactile_last_update_mono[key_tuple] = time.monotonic()

            self.tactile_counts_by_key[key] += 1
            self.tactile_frame_count += 1
            self.tactile_window_frames += 1
            self.last_tactile_error = ""
        except Exception as exc:
            self.last_tactile_error = str(exc)

    def _poll_tactile_frames(self) -> None:
        if not self.publish_tactile or not self.connected:
            return
        if time.monotonic() < self.tactile_poll_ready_at:
            self.last_tactile_error = "warming_up"
            return
        frames: list[tuple[Any, str, int]] = []
        self._tactile_trace("poll:enter")
        # SharpaWave is a C++ object shared by command, feedback, status, and
        # tactile timers. The SDK tactile getters are not safe to call
        # concurrently with joint getters/setters, so all SDK access stays
        # under sdk_lock and ROS message conversion happens after the lock.
        with self.sdk_lock:
            for side, hand, channels in (
                ("right", self.right, range(0, 5)),
                ("left", self.left, range(5, 10)),
            ):
                if hand is None:
                    continue
                for channel in channels:
                    try:
                        self._tactile_trace(f"poll:{side}:fetch:{channel}")
                        frame = hand.fetch_tactile_frame(channel, 0.0)
                    except Exception as exc:
                        self.last_tactile_error = (
                            f"{side} ch{channel} fetch failed: {exc}"
                        )
                        continue
                    if frame is not None:
                        frames.append((frame, side, channel))
        self._tactile_trace(f"poll:fetched:{len(frames)}")
        for frame, side, channel in frames:
            self._tactile_trace(f"poll:convert:{side}")
            self._on_tactile_frame(
                frame,
                side_override=side,
                channel_override=channel,
            )
        self._tactile_trace("poll:done")

    def _publish_tactile_aggregates(self) -> None:
        if not self.publish_tactile or not self.connected:
            return
        self._tactile_trace("publish:enter")
        self._poll_tactile_frames()
        self._tactile_trace("publish:after_poll")

        stamp = self.get_clock().now().to_msg()
        now_mono = time.monotonic()
        wall_now = time.time()
        header_frame_id = self.tactile_topic_prefix.strip("/") or "sharpa_tactile"
        with self.tactile_lock:
            deforms = dict(self.latest_tactile_deforms)
            forces = dict(self.latest_tactile_forces)
            contacts = dict(self.latest_tactile_contacts)
            last_updates = dict(self.tactile_last_update_mono)
        self._tactile_trace(
            f"publish:snapshot:d{len(deforms)}:f{len(forces)}:c{len(contacts)}"
        )

        deform_array = TactileDeformImageArray()
        deform_array.header.stamp = stamp
        deform_array.header.frame_id = header_frame_id
        force_array = TactileForce6DArray()
        force_array.header.stamp = stamp
        force_array.header.frame_id = header_frame_id
        contact_array = TactileContactPointsArray()
        contact_array.header.stamp = stamp
        contact_array.header.frame_id = header_frame_id

        deform_images: list[TactileDeformImage] = []
        force_entries: list[TactileForce6D] = []
        contact_entries: list[TactileContactPoints] = []
        stale_keys: list[str] = []
        for channel, (side, finger) in enumerate(TACTILE_AGGREGATE_ORDER):
            key = (side, finger)
            deform = deforms.get(key)
            force = forces.get(key)
            contact = contacts.get(key)
            deform_fresh = self._tactile_msg_is_fresh(
                deform, key, last_updates, now_mono, wall_now
            )
            force_fresh = self._tactile_msg_is_fresh(
                force, key, last_updates, now_mono, wall_now
            )
            contact_fresh = self._tactile_msg_is_fresh(
                contact, key, last_updates, now_mono, wall_now
            )
            deform_images.append(
                deform if deform_fresh else self._empty_deform_image(side, finger, channel)
            )
            force_entries.append(
                force if force_fresh else self._empty_force6d(side, finger, channel)
            )
            contact_entries.append(
                contact if contact_fresh else self._empty_contact_points(side, finger, channel)
            )
            if not (deform_fresh or force_fresh or contact_fresh):
                stale_keys.append(f"{side}/{finger}")
        self._tactile_trace("publish:built")

        deform_array.images = deform_images
        force_array.forces = force_entries
        contact_array.contacts = contact_entries
        self._tactile_trace("publish:deform")
        self.tactile_deform_pub.publish(deform_array)
        self._tactile_trace("publish:force")
        self.tactile_force_pub.publish(force_array)
        self._tactile_trace("publish:contact")
        self.tactile_contact_pub.publish(contact_array)
        self._tactile_trace("publish:done")
        self.tactile_publish_count += 1
        self.tactile_window_publishes += 1
        self.tactile_debug_ticks += 1
        if stale_keys:
            self.last_tactile_error = "stale tactile channels: " + ",".join(stale_keys)
        elif self.last_tactile_error.startswith("stale tactile channels:"):
            self.last_tactile_error = ""

    def _tactile_trace(self, stage: str) -> None:
        if self.tactile_debug_ticks < 4:
            self.get_logger().info(f"tactile trace[{self.tactile_debug_ticks}] {stage}")

    def _clear_tactile_side(self, side: str) -> None:
        if side not in {"left", "right"}:
            return
        with self.tactile_lock:
            for key in [key for key in self.latest_tactile_deforms if key[0] == side]:
                self.latest_tactile_deforms.pop(key, None)
            for key in [key for key in self.latest_tactile_forces if key[0] == side]:
                self.latest_tactile_forces.pop(key, None)
            for key in [key for key in self.latest_tactile_contacts if key[0] == side]:
                self.latest_tactile_contacts.pop(key, None)
            for key in [key for key in self.tactile_last_signatures if key[0] == side]:
                self.tactile_last_signatures.pop(key, None)
            for key in [key for key in self.tactile_last_update_mono if key[0] == side]:
                self.tactile_last_update_mono.pop(key, None)

    def _tactile_msg_is_fresh(
        self,
        msg: Any | None,
        key: tuple[str, str],
        last_updates: dict[tuple[str, str], float],
        now_mono: float,
        wall_now: float,
    ) -> bool:
        if msg is None:
            return False
        last_update = last_updates.get(key)
        if last_update is None:
            return False
        if now_mono - last_update > self.tactile_fresh_timeout_s:
            return False
        return self._tactile_metadata_is_current(
            int(getattr(msg, "frame_id", 0)),
            self._float_or_zero(getattr(msg, "sensor_time", 0.0)),
            wall_now,
        )

    def _tactile_metadata_is_current(
        self,
        frame_id: int,
        sensor_time: float,
        wall_now: float,
    ) -> bool:
        if frame_id <= 0:
            return False
        if not math.isfinite(sensor_time) or sensor_time <= 0.0:
            return False
        return abs(wall_now - sensor_time) <= self.tactile_sensor_time_max_age_s

    def _deform_image_msg(
        self,
        data: Any,
        shape: Any,
        side: str,
        finger: str,
        channel: int,
        frame_id: int,
        sensor_time: float,
    ) -> TactileDeformImage:
        height, width, payload = self._image_payload(data, shape)
        msg = self._empty_deform_image(side, finger, channel)
        msg.frame_id = frame_id
        msg.sensor_time = sensor_time
        msg.height = height
        msg.width = width
        msg.data = array("B", payload)
        return msg

    def _force6d_msg(
        self,
        data: Any,
        side: str,
        finger: str,
        channel: int,
        frame_id: int,
        sensor_time: float,
    ) -> TactileForce6D | None:
        import numpy as np

        values = np.asarray(data, dtype=np.float32).flatten()
        if values.size < 6:
            return None
        msg = self._empty_force6d(side, finger, channel)
        msg.frame_id = frame_id
        msg.sensor_time = sensor_time
        msg.force = [self._float_or_zero(value) for value in values[:3]]
        msg.torque = [self._float_or_zero(value) for value in values[3:6]]
        return msg

    def _contact_points_msg(
        self,
        data: Any,
        side: str,
        finger: str,
        channel: int,
        frame_id: int,
        sensor_time: float,
    ) -> TactileContactPoints:
        import numpy as np

        values = np.asarray(data, dtype=np.float32).flatten()
        msg = self._empty_contact_points(side, finger, channel)
        msg.frame_id = frame_id
        msg.sensor_time = sensor_time
        msg.points = [self._float_or_zero(value) for value in values]
        return msg

    def _image_payload(self, data: Any, shape: Any) -> tuple[int, int, bytes]:
        import numpy as np

        arr = np.asarray(data).squeeze().astype(np.uint8, copy=False)
        if arr.size == 0:
            return 0, 0, b""

        shape_values = self._shape_values(shape)
        if len(shape_values) >= 2:
            height = int(shape_values[-2])
            width = int(shape_values[-1])
            if height > 0 and width > 0 and height * width == int(arr.size):
                return height, width, arr.reshape(height, width).tobytes()

        if arr.ndim >= 2:
            height = int(arr.shape[-2])
            width = int(arr.shape[-1])
            if height > 0 and width > 0 and height * width == int(arr.size):
                return height, width, arr.reshape(height, width).tobytes()

        height = 1
        width = int(arr.size)
        return height, width, arr.reshape(height, width).tobytes()

    @staticmethod
    def _empty_deform_image(
        side: str, finger: str, channel: int
    ) -> TactileDeformImage:
        msg = TactileDeformImage()
        SharpaNode._set_tactile_metadata(msg, side, finger, channel)
        msg.height = 0
        msg.width = 0
        msg.data = array("B")
        return msg

    @staticmethod
    def _empty_force6d(side: str, finger: str, channel: int) -> TactileForce6D:
        msg = TactileForce6D()
        SharpaNode._set_tactile_metadata(msg, side, finger, channel)
        msg.force = [0.0, 0.0, 0.0]
        msg.torque = [0.0, 0.0, 0.0]
        return msg

    @staticmethod
    def _empty_contact_points(
        side: str, finger: str, channel: int
    ) -> TactileContactPoints:
        msg = TactileContactPoints()
        SharpaNode._set_tactile_metadata(msg, side, finger, channel)
        msg.points = []
        return msg

    @staticmethod
    def _set_tactile_metadata(msg: Any, side: str, finger: str, channel: int) -> None:
        msg.side = side
        msg.finger = finger
        msg.channel = max(0, min(255, int(channel)))
        msg.frame_id = 0
        msg.sensor_time = 0.0

    @staticmethod
    def _shape_values(shape: Any) -> list[int]:
        if shape is None:
            return []
        if isinstance(shape, (list, tuple)):
            try:
                return [int(value) for value in shape]
            except (TypeError, ValueError):
                return []
        tolist = getattr(shape, "tolist", None)
        if callable(tolist):
            try:
                return [int(value) for value in tolist()]
            except (TypeError, ValueError):
                return []

        # Sharpa SDK exposes tactile shapes as a pybind object with size()/dim().
        # Iterating that object with list(shape) can segfault, so infer the 2-D
        # image dimensions from the known flat payload sizes instead.
        size_fn = getattr(shape, "size", None)
        if callable(size_fn):
            try:
                size = int(size_fn())
            except (TypeError, ValueError):
                size = 0
            if size == 76800:
                return [240, 320]
            if size == 57600:
                return [240, 240]
            if size > 0:
                return [size]
        return []

    @staticmethod
    def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
        getter = getattr(value, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                try:
                    return getter(key)
                except TypeError:
                    return default
        return getattr(value, key, default)

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _uint32(value: Any) -> int:
        try:
            return int(value) % (2**32)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _float_or_zero(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    @staticmethod
    def _finger_from_channel(channel: int) -> str | None:
        if channel < 0:
            return None
        return TACTILE_FINGER_NAMES_BY_CHANNEL[
            channel % len(TACTILE_FINGER_NAMES_BY_CHANNEL)
        ]

    def _fault_name(self, code: Any) -> str:
        name = getattr(code, "name", None)
        if name:
            return str(name)
        try:
            value = int(code)
        except (TypeError, ValueError):
            return str(code)
        fault_code_enum = self.FaultCode
        if fault_code_enum is None:
            return ""
        for attr in dir(fault_code_enum):
            if attr.startswith("_") or attr in {"name", "value"}:
                continue
            try:
                candidate = getattr(fault_code_enum, attr)
                if int(candidate) == value:
                    return attr
            except (TypeError, ValueError):
                continue
        return ""

    def _hand_fault_payload(self, hand: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        try:
            info = hand.get_device_info()
            status = getattr(info, "status", None)
            fault_code = getattr(status, "error_code", None)
            if fault_code is not None:
                fault_value = int(fault_code)
                payload["heartbeat_fault_code"] = fault_value
                payload["heartbeat_fault_name"] = self._fault_name(fault_value)
            error_joint = getattr(status, "error_joint", None)
            if error_joint is not None:
                payload["heartbeat_error_joint"] = int(error_joint)
        except Exception as exc:
            payload["device_info_error"] = str(exc)

        get_fault_code = getattr(hand, "get_fault_code", None)
        if callable(get_fault_code):
            try:
                active_faults = []
                for code, indices in get_fault_code().items():
                    try:
                        fault_value = int(code)
                    except (TypeError, ValueError):
                        fault_value = 0
                    active_faults.append(
                        {
                            "code": fault_value,
                            "name": self._fault_name(code),
                            "indices": [int(index) for index in list(indices)],
                        }
                    )
                payload["active_faults"] = active_faults
            except Exception as exc:
                payload["active_faults_error"] = str(exc)
        return payload

    @staticmethod
    def _is_tactile_fault_name(name: str) -> bool:
        upper = name.upper()
        return "TOUCH" in upper or "TACTILE" in upper

    def _recording_tactile_problem_reasons(
        self,
        fresh_by_sensor: dict[str, bool],
        faults_by_side: dict[str, dict[str, Any]],
        summaries: dict[str, Any],
        warming_up: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if warming_up:
            reasons.append("tactile warming up")
        stale = [label for label, fresh in fresh_by_sensor.items() if not fresh]
        if stale:
            reasons.append("stale_or_missing_channels=" + ",".join(stale))
        for side, fault_payload in faults_by_side.items():
            heartbeat_name = str(fault_payload.get("heartbeat_fault_name", ""))
            if self._is_tactile_fault_name(heartbeat_name):
                reasons.append(f"{side}:{heartbeat_name}")
            active_faults = fault_payload.get("active_faults", [])
            if isinstance(active_faults, list):
                for fault in active_faults:
                    if not isinstance(fault, dict):
                        continue
                    name = str(fault.get("name", ""))
                    if self._is_tactile_fault_name(name):
                        indices = fault.get("indices", [])
                        reasons.append(f"{side}:{name}:indices={indices}")
        for side, summary in summaries.items():
            if not isinstance(summary, dict):
                continue
            if summary.get("error"):
                reasons.append(f"{side}:summary_error={summary['error']}")
            if summary.get("ready") is False and not warming_up:
                reasons.append(f"{side}:tactile_not_ready")
        return list(dict.fromkeys(reasons))

    def _log_recording_tactile_problems(
        self,
        fresh_by_sensor: dict[str, bool],
        faults_by_side: dict[str, dict[str, Any]],
        summaries: dict[str, Any],
        warming_up: bool,
    ) -> None:
        if not self.recording_active:
            self.last_tactile_terminal_error = ""
            return
        reasons = self._recording_tactile_problem_reasons(
            fresh_by_sensor,
            faults_by_side,
            summaries,
            warming_up,
        )
        if not reasons:
            if self.last_tactile_terminal_error:
                self.get_logger().info("recording tactile recovered")
            self.last_tactile_terminal_error = ""
            return
        message = "; ".join(reasons)
        now = time.monotonic()
        if (
            message != self.last_tactile_terminal_error
            or now - self.last_tactile_terminal_error_time
            >= self.tactile_error_log_period_s
        ):
            self.get_logger().error(f"recording tactile error: {message}")
            self.last_tactile_terminal_error = message
            self.last_tactile_terminal_error_time = now

    def _publish_status(self) -> None:
        now = time.monotonic()
        startup_zero_remaining_s = max(0.0, self.startup_zero_until - now)
        elapsed = now - self.status_window_time
        if elapsed > 0.0:
            self.command_hz = self.status_window_commands / elapsed
            self.feedback_hz = self.status_window_feedback / elapsed
        self.status_window_commands = 0
        self.status_window_feedback = 0
        self.status_window_time = now

        target = self.target
        target_age_ms = None if target is None else round((now - target.stamp) * 1000.0, 1)
        payload = {
            "node": "sharpa",
            "mode": self.mode,
            "teleop_state": self.teleop_state,
            "sharpa_active": self.sharpa_active,
            "last_mode_event": self.last_mode_event,
            "sdk": {
                "loaded": self.sdk_loaded,
                "python_path": self.sdk_python_path,
                "connected": self.connected,
                "active_mode": self.active_sdk_mode,
                "devices": self.connected_devices,
                "left_connected": self.left is not None,
                "right_connected": self.right is not None,
                "hands": {
                    "left": {
                        "sn": self.left_sn,
                        "expected_ip": self.left_ip,
                        "discovered_ip": self.discovered_ip_by_sn.get(self.left_sn),
                        "node_connected": self.left is not None,
                        "manager_connected": self._manager_connected(self.left_sn),
                        "hand_ready": self._hand_ready(self.left),
                    },
                    "right": {
                        "sn": self.right_sn,
                        "expected_ip": self.right_ip,
                        "discovered_ip": self.discovered_ip_by_sn.get(self.right_sn),
                        "node_connected": self.right is not None,
                        "manager_connected": self._manager_connected(self.right_sn),
                        "hand_ready": self._hand_ready(self.right),
                    },
                },
            },
            "command": {
                "status_topic": self.status_topic,
                "retargeted_joints_topic": self.retargeted_joints_topic,
                "joint_states_topic": self.joint_states_topic,
                "command_snapshot_topic": self.command_snapshot_topic,
                "command_snapshot_count": self.command_snapshot_count,
                "command_snapshot_max_hz": self.command_snapshot_max_hz,
                "configured_mode": self.command_mode,
                "active_target_mode": None if target is None else target.control_mode,
                "mit_torque_limit": self.mit_torque_limit,
                "fresh": self._fresh_target() is not None,
                "age_ms": target_age_ms,
                "present_count": 0 if target is None else target.present_count,
                "left_present": bool(target and target.left_rad is not None),
                "right_present": bool(target and target.right_rad is not None),
                "timeout_s": self.target_timeout,
                "side_counts": dict(self.side_command_count),
                "left_last_command_age_ms": age_ms(
                    self.last_side_command_time.get("left"), now
                ),
                "right_last_command_age_ms": age_ms(
                    self.last_side_command_time.get("right"), now
                ),
                "startup_zero_hold_s": self.startup_zero_hold_s,
                "startup_zero_active": startup_zero_remaining_s > 0.0,
                "startup_zero_remaining_s": round(startup_zero_remaining_s, 3),
            },
            "rates": {
                "command_hz": round(self.command_hz, 2),
                "feedback_hz": round(self.feedback_hz, 2),
                "command_count": self.command_count,
                "feedback_count": self.feedback_count,
            },
            "last_command_reason": self.last_command_reason,
            "last_command_age_ms": age_ms(self.last_command_time, now),
            "last_feedback_age_ms": age_ms(self.last_feedback_time, now),
            "last_error": self.last_error,
        }
        self.status_pub.publish(json_msg(payload))

    def _publish_tactile_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.tactile_window_time
        source_hz = 0.0
        if elapsed > 0.0:
            source_hz = self.tactile_window_frames / elapsed
            self.tactile_hz = self.tactile_window_publishes / elapsed
        self.tactile_window_frames = 0
        self.tactile_window_publishes = 0
        self.tactile_window_time = now

        summaries = {}
        faults_by_side = {}
        warming_up = now < self.tactile_poll_ready_at
        if self.connected and self.publish_tactile:
            with self.sdk_lock:
                for side, hand in (("left", self.left), ("right", self.right)):
                    if hand is None:
                        continue
                    faults_by_side[side] = self._hand_fault_payload(hand)

        wall_now = time.time()
        with self.tactile_lock:
            signatures = dict(self.tactile_last_signatures)
            last_updates = dict(self.tactile_last_update_mono)
            duplicate_counts = dict(self.tactile_duplicate_counts_by_key)
            invalid_counts = dict(self.tactile_invalid_counts_by_key)

        fresh_by_sensor: dict[str, bool] = {}
        last_update_age_ms: dict[str, float | None] = {}
        for side, finger in TACTILE_AGGREGATE_ORDER:
            key = (side, finger)
            label = f"{side}/{finger}"
            last_update = last_updates.get(key)
            signature = signatures.get(key)
            last_update_age_ms[label] = (
                None if last_update is None else round((now - last_update) * 1000.0, 1)
            )
            fresh_by_sensor[label] = (
                last_update is not None
                and signature is not None
                and now - last_update <= self.tactile_fresh_timeout_s
                and self._tactile_metadata_is_current(
                    signature[0],
                    signature[1],
                    wall_now,
                )
            )

        for side, hand in (("left", self.left), ("right", self.right)):
            if hand is None:
                self.tactile_ready_by_side[side] = None
                continue
            side_labels = [
                f"{entry_side}/{finger}"
                for entry_side, finger in TACTILE_AGGREGATE_ORDER
                if entry_side == side
            ]
            fresh_count = sum(bool(fresh_by_sensor[label]) for label in side_labels)
            ready = not warming_up and fresh_count == len(side_labels)
            self.tactile_ready_by_side[side] = ready
            summaries[side] = {
                "ready": ready,
                "source": "frame_freshness",
                "fresh_channels": fresh_count,
                "channel_count": len(side_labels),
            }
            if warming_up:
                summaries[side]["reason"] = "warming_up"

        self._log_recording_tactile_problems(
            fresh_by_sensor,
            faults_by_side,
            summaries,
            warming_up,
        )

        self.tactile_status_pub.publish(
            json_msg(
                {
                    "node": "sharpa",
                    "topic_prefix": self.tactile_topic_prefix,
                    "topics": {
                        "deform_images": f"{self.tactile_topic_prefix}/deform_images",
                        "force6d": f"{self.tactile_topic_prefix}/force6d",
                        "contact_points": f"{self.tactile_topic_prefix}/contact_points",
                    },
                    "entry_order": [
                        f"{side}_{finger}" for side, finger in TACTILE_AGGREGATE_ORDER
                    ],
                    "enabled": self.publish_tactile,
                    "frame_count": self.tactile_frame_count,
                    "publish_count": self.tactile_publish_count,
                    "fps": round(self.tactile_hz, 2),
                    "source_fps": round(source_hz, 2),
                    "ready": dict(self.tactile_ready_by_side),
                    "warming_up": warming_up,
                    "warmup_remaining_s": max(
                        0.0, round(self.tactile_poll_ready_at - now, 3)
                    ),
                    "fresh_timeout_s": self.tactile_fresh_timeout_s,
                    "sensor_time_max_age_s": self.tactile_sensor_time_max_age_s,
                    "auto_retry_alternate_port": self.tactile_auto_retry_alternate_port,
                    "fresh_by_sensor": fresh_by_sensor,
                    "last_update_age_ms_by_sensor": last_update_age_ms,
                    "counts_by_sensor": dict(self.tactile_counts_by_key),
                    "duplicate_counts_by_sensor": duplicate_counts,
                    "invalid_counts_by_sensor": invalid_counts,
                    "faults": faults_by_side,
                    "summaries": summaries,
                    "last_error": self.last_tactile_error,
                }
            )
        )

    def _stop_hardware(self) -> None:
        if not self.connected:
            return
        try:
            if self.zero_on_shutdown:
                self._ensure_position_mode()
                self._send_zero("shutdown_zero")
            else:
                self._ensure_floating_mode()
            with self.sdk_lock:
                for hand in (self.left, self.right):
                    if hand is not None:
                        hand.stop()
                if self.manager is not None:
                    self.manager.disconnect_all()
        except Exception as exc:
            self.get_logger().warn(f"Sharpa shutdown warning: {exc}")
        finally:
            self.connected = False

    @staticmethod
    def _check(label: str, err: Any) -> None:
        if getattr(err, "code", 0) != 0:
            message = getattr(err, "message", str(err))
            raise RuntimeError(f"{label} failed: {message}")

    def _warn_if_error(self, label: str, err: Any) -> bool:
        if getattr(err, "code", 0) == 0:
            return True
        message = getattr(err, "message", str(err))
        self.get_logger().warn(f"{label} warning: {message}")
        return False

    def destroy_node(self) -> bool:
        self._stop_hardware()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SharpaNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
