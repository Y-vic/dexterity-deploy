"""bias_node -- the ONLY producer allowed to move hardware during reset.

Subscribes `/control/bias_request` (`home_ur` / `home_sharpa` / `home_both`),
streams an interpolated trajectory to `/ur/command` with
`source=ArmCommand.SOURCE_BIAS` and to `/sharpa/command`, and reports progress on
`/control/bias_status` as a JSON string.

WHY THIS NODE HAS THE HOMING-STATE MONOPOLY (ur_node/arbitration.py): in
`homing` the arbiter accepts SOURCE_BIAS and nothing else. That makes power-on
safe -- a stale actor plan or a live vive stream cannot start the robot moving
just because a node came up. Publishing anything here with a source other than
SOURCE_BIAS silently breaks that property, so the source is set in exactly one
place (`_publish_arm_step`) and asserted in the tests.

ARRIVAL IS VERIFIED, NEVER ASSUMED. Running out of steps is not arrival. When
the measured pose is still outside tolerance after the last step the run is
reported `failed` with the measured error, not `done`. Reporting success on a
timeout would tell `status_node` the arms are at home while they are elsewhere,
and the next `homed -> running` transition would hand a teleop stream an arm in
an unknown pose. See trajectory.check_arm_arrival.

Safety switches (all three default False, per the v1 plan) are passed straight
into `ur_common.safety.SafetyConfig` and are constructed EXPLICITLY -- never via
`SafetyConfig.from_env()`, whose defaults are the opposite polarity (ON), so a
stray environment variable cannot silently arm a gate here.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

from ur_msgs.msg import ArmCommand, HandCommand, HardwareCommandResult, UrStateFrame
from ur_msgs.msg import ControlStatus, HandStateFrame

from ur_common.kinematics import UrCommonKinematics
from ur_common.safety import SafetyConfig

from .home_pose import HomePoseError, both_home_joints
from .trajectory import (
    SHARPA_HOME_STEPS,
    SHARPA_HOME_STEP_DELAY_S,
    SHARPA_HOME_TOLERANCE_RAD,
    SHARPA_JOINTS_TOTAL,
    UR_HOME_STEPS,
    UR_HOME_STEP_DELAY_S,
    UR_HOME_TOLERANCE_RAD,
    check_arm_arrival,
    check_hand_arrival,
    interpolate,
)

REQUEST_HOME_UR = "home_ur"
REQUEST_HOME_SHARPA = "home_sharpa"
REQUEST_HOME_BOTH = "home_both"
VALID_REQUESTS = (REQUEST_HOME_UR, REQUEST_HOME_SHARPA, REQUEST_HOME_BOTH)

PHASE_IDLE = "idle"
PHASE_PREPARING = "preparing"
PHASE_UR = "homing_ur"
PHASE_SHARPA = "homing_sharpa"
PHASE_VERIFY = "verifying"
PHASE_DONE = "done"
PHASE_FAILED = "failed"


@dataclass
class BiasRun:
    """One in-flight homing run."""

    request: str
    action_id: str
    started_at: float
    started_wall_ns: int = field(default_factory=time.time_ns)
    arm_plan: np.ndarray | None = None      # (steps, 12): left 6 then right 6
    arm_target: dict = field(default_factory=dict)
    hand_plan: np.ndarray | None = None     # (steps, 44)
    hand_target: np.ndarray | None = None
    step: int = 0
    phase: str = PHASE_IDLE
    detail: str = ""
    verify_deadline: float = 0.0
    steps_sent: int = 0
    pending_step: int | None = None
    pending_devices: set[str] = field(default_factory=set)
    pending_since: float = 0.0
    next_step_at: float = 0.0

    @property
    def wants_ur(self) -> bool:
        return self.request in (REQUEST_HOME_UR, REQUEST_HOME_BOTH)

    @property
    def wants_sharpa(self) -> bool:
        return self.request in (REQUEST_HOME_SHARPA, REQUEST_HOME_BOTH)

    @property
    def total_steps(self) -> int:
        lengths = [len(p) for p in (self.arm_plan, self.hand_plan) if p is not None]
        return max(lengths) if lengths else 0


class BiasNode(Node):
    """Homing / reset trajectories for both UR5 arms and both SharpA hands."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("bias_node", **kwargs)

        # ---- topics: defaults MUST match docs/ur_topics.md ----
        self.declare_parameter("bias_request_topic", "/control/bias_request")
        self.declare_parameter("bias_status_topic", "/control/bias_status")
        self.declare_parameter("arm_command_topic", "/ur/command")
        self.declare_parameter("hand_command_topic", "/sharpa/command")
        self.declare_parameter("ur_state_topic", "/ur/state")
        self.declare_parameter("sharpa_state_topic", "/sharpa/state")
        self.declare_parameter("arm_result_topic", "/ur/command_result")
        self.declare_parameter("hand_result_topic", "/sharpa/command_result")
        self.declare_parameter("control_status_topic", "/control/status")

        # ---- trajectory shape: v3 initialize_joints defaults ----
        self.declare_parameter("ur_steps", UR_HOME_STEPS)
        self.declare_parameter("ur_step_delay_s", UR_HOME_STEP_DELAY_S)
        self.declare_parameter("ur_tolerance_rad", UR_HOME_TOLERANCE_RAD)
        self.declare_parameter("sharpa_steps", SHARPA_HOME_STEPS)
        self.declare_parameter("sharpa_step_delay_s", SHARPA_HOME_STEP_DELAY_S)
        self.declare_parameter("sharpa_tolerance_rad", SHARPA_HOME_TOLERANCE_RAD)
        # Grace period after the last step before the arrival check gives up.
        # Expires -> FAILURE, never success. See module docstring.
        self.declare_parameter("arrival_timeout_s", 3.0)
        self.declare_parameter("prepare_timeout_s", 5.0)
        self.declare_parameter("command_ack_timeout_s", 1.0)
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("status_rate_hz", 5.0)
        self.declare_parameter("bias_file", "~/.config/ur_sharpa/teleop_bias.json")

        # ---- safety switches: v1 ships with all three OFF ----
        # Nothing is deleted; each flag fully restores one piece of logic in
        # ur_common.safety. Homing itself passes allow_large=True to the IK, so
        # switch 2 does not gate the home excursion either way.
        self.declare_parameter("enable_jump_gate", False)
        # LEAVING enable_flip_hold OFF CRASHED THE REAL ROBOT.
        # ik_solver.py:55-69 records the attempt to drop the wrist-flip hold and
        # rely on downstream limit_joint_vel clipping alone. Clipping bounds the
        # per-frame STEP SIZE but never changes DIRECTION: the target stays the
        # flipped configuration and the arm marches at it 0.35 rad/frame instead
        # of easing in. Measured position error went 9mm -> 37.8mm -> 186.2mm and
        # the protective stop fired within two frames. A hold is safe precisely
        # because it does not move that way. That is what OFF costs.
        self.declare_parameter("enable_flip_hold", False)
        self.declare_parameter("enable_joint_clip", False)

        gp = self.get_parameter
        self.bias_request_topic = gp("bias_request_topic").value
        self.bias_status_topic = gp("bias_status_topic").value
        self.arm_command_topic = gp("arm_command_topic").value
        self.hand_command_topic = gp("hand_command_topic").value
        self.ur_steps = max(1, int(gp("ur_steps").value))
        self.ur_step_delay_s = max(1e-3, float(gp("ur_step_delay_s").value))
        self.ur_tolerance_rad = float(gp("ur_tolerance_rad").value)
        self.sharpa_steps = max(1, int(gp("sharpa_steps").value))
        self.sharpa_step_delay_s = max(1e-3, float(gp("sharpa_step_delay_s").value))
        self.sharpa_tolerance_rad = float(gp("sharpa_tolerance_rad").value)
        self.arrival_timeout_s = float(gp("arrival_timeout_s").value)
        self.prepare_timeout_s = float(gp("prepare_timeout_s").value)
        self.command_ack_timeout_s = float(gp("command_ack_timeout_s").value)
        self.state_timeout_s = float(gp("state_timeout_s").value)
        self.bias_file = Path(str(gp("bias_file").value)).expanduser()

        # EXPLICIT construction. SafetyConfig() defaults all three OFF;
        # SafetyConfig.from_env() defaults them ON. Passing the launch
        # parameters by keyword means no environment variable can flip them.
        self.safety_config = SafetyConfig(
            enable_jump_gate=bool(gp("enable_jump_gate").value),
            enable_flip_hold=bool(gp("enable_flip_hold").value),
            enable_joint_clip=bool(gp("enable_joint_clip").value),
        )
        self.kinematics = UrCommonKinematics(config=self.safety_config)

        # ---- run state ----
        self._lock = threading.RLock()
        self._run: BiasRun | None = None
        self._last_result: dict = {
            "request": "", "phase": PHASE_IDLE, "detail": "",
            "max_error_rad": None, "finished_at": None,
        }
        self._counts = {"requests": 0, "rejected": 0, "completed": 0, "failed": 0,
                        "arm_steps": 0, "hand_steps": 0, "nacks": 0}

        self._ur_state: UrStateFrame | None = None
        self._ur_state_time: float | None = None
        self._hand_state: HandStateFrame | None = None
        self._hand_state_time: float | None = None
        self._ur_active_state: int | None = None
        self._sharpa_active_state: int | None = None
        self._last_nack: str = ""

        # ---- ROS interfaces. QoS: depth=10 reliable, per docs/ur_topics.md ----
        qos = QoSProfile(depth=10)
        self.arm_pub = self.create_publisher(ArmCommand, self.arm_command_topic, qos)
        self.hand_pub = self.create_publisher(HandCommand, self.hand_command_topic, qos)
        self.status_pub = self.create_publisher(String, self.bias_status_topic, qos)

        self.create_subscription(String, self.bias_request_topic, self._on_request, qos)
        self.create_subscription(UrStateFrame, gp("ur_state_topic").value,
                                 self._on_ur_state, qos)
        self.create_subscription(HandStateFrame, gp("sharpa_state_topic").value,
                                 self._on_hand_state, qos)
        self.create_subscription(HardwareCommandResult, gp("arm_result_topic").value,
                                 self._on_command_result, qos)
        self.create_subscription(HardwareCommandResult, gp("hand_result_topic").value,
                                 self._on_command_result, qos)
        self.create_subscription(ControlStatus, gp("control_status_topic").value,
                                 self._on_control_status, qos)

        # One timer paces the trajectory; the step period is the v3 step_delay.
        # Using a timer rather than v3's blocking `time.sleep` loop keeps the
        # motion profile identical while leaving the executor free.
        self._tick_period = min(self.ur_step_delay_s, self.sharpa_step_delay_s)
        self.create_timer(self._tick_period, self._tick)
        status_hz = max(0.5, float(gp("status_rate_hz").value))
        self.create_timer(1.0 / status_hz, self.publish_status)

        self.get_logger().info(
            f"bias_node up. ur={self.ur_steps} steps @ {self.ur_step_delay_s}s "
            f"tol={self.ur_tolerance_rad} rad; sharpa={self.sharpa_steps} steps; "
            f"safety jump_gate={self.safety_config.enable_jump_gate} "
            f"flip_hold={self.safety_config.enable_flip_hold} "
            f"joint_clip={self.safety_config.enable_joint_clip}"
        )

    # ------------------------------------------------------------------ inputs
    def _on_ur_state(self, msg: UrStateFrame) -> None:
        with self._lock:
            self._ur_state = msg
            self._ur_state_time = time.monotonic()

    def _on_hand_state(self, msg: HandStateFrame) -> None:
        with self._lock:
            self._hand_state = msg
            self._hand_state_time = time.monotonic()

    def _on_control_status(self, msg: ControlStatus) -> None:
        with self._lock:
            self._ur_active_state = int(msg.ur_state)
            self._sharpa_active_state = int(msg.sharpa_state)

    def _on_command_result(self, msg: HardwareCommandResult) -> None:
        """Advance only after every device ACKs this run's exact step."""
        if int(msg.source) != ArmCommand.SOURCE_BIAS:
            return
        with self._lock:
            run = self._run
            if run is None or run.pending_step is None:
                return
            if (
                str(msg.action_id) != run.action_id
                or int(msg.revision) != 0
                or int(msg.step_index) != run.pending_step
                or str(msg.device) not in run.pending_devices
            ):
                return
            device = str(msg.device)
            if not msg.success:
                self._counts["nacks"] += 1
                self._last_nack = f"{device}:{msg.failure_code}"
                self._fail_locked(
                    f"{device} rejected bias step {run.pending_step}: {msg.failure_code} "
                    f"{msg.failure_message}".strip()
                )
                return
            run.pending_devices.remove(device)
            if run.pending_devices:
                run.detail = (
                    f"step {run.pending_step} waiting for "
                    f"{','.join(sorted(run.pending_devices))} ACK"
                )
                return
            run.step += 1
            run.steps_sent += 1
            run.pending_step = None
            run.pending_since = 0.0
            run.detail = ""
            delays = []
            if run.arm_plan is not None:
                delays.append(self.ur_step_delay_s)
            if run.hand_plan is not None:
                delays.append(self.sharpa_step_delay_s)
            run.next_step_at = time.monotonic() + max(delays or [self._tick_period])
            if run.step >= run.total_steps:
                run.phase = PHASE_VERIFY
                run.verify_deadline = time.monotonic() + self.arrival_timeout_s
            elif run.arm_plan is not None and run.step >= len(run.arm_plan):
                run.phase = PHASE_SHARPA

    def _measured_arm_joints(self) -> dict | None:
        """Fresh measured joints, or None when the state is stale/invalid."""
        if self._ur_state is None or self._ur_state_time is None:
            return None
        if time.monotonic() - self._ur_state_time > self.state_timeout_s:
            return None
        if not self._ur_state.valid:
            return None
        return {
            "left": np.asarray(self._ur_state.left_joint, dtype=np.float64),
            "right": np.asarray(self._ur_state.right_joint, dtype=np.float64),
        }

    def _measured_hand_joints(self) -> np.ndarray | None:
        if self._hand_state is None or self._hand_state_time is None:
            return None
        if time.monotonic() - self._hand_state_time > self.state_timeout_s:
            return None
        if not (self._hand_state.left_valid and self._hand_state.right_valid):
            return None
        return np.asarray(self._hand_state.q, dtype=np.float64)

    # ----------------------------------------------------------------- request
    def _on_request(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {"request": raw}
        if isinstance(payload, str):
            payload = {"request": payload}
        self.handle_request(
            str(payload.get("request", "")).strip(),
            action_id=str(payload.get("action_id", "")).strip() or None,
        )

    def handle_request(self, request: str, action_id: str | None = None) -> bool:
        """Start a homing run. Returns False and reports why when refused."""
        with self._lock:
            self._counts["requests"] += 1
            if request == "save_current":
                try:
                    self._save_current_bias_locked()
                except (RuntimeError, ValueError, OSError) as exc:
                    self._counts["rejected"] += 1
                    self._record_result(request, PHASE_FAILED, f"save failed: {exc}")
                    self.publish_status()
                    return False
                self._record_result(request, PHASE_DONE, "current measured pose saved")
                self.publish_status()
                return True
            if request not in VALID_REQUESTS:
                self._counts["rejected"] += 1
                self._record_result(request, PHASE_FAILED,
                                    f"unknown bias request {request!r}; "
                                    f"expected one of {'/'.join(VALID_REQUESTS)}")
                self.publish_status()
                return False
            if self._run is not None and self._run.phase in (
                    PHASE_PREPARING, PHASE_UR, PHASE_SHARPA, PHASE_VERIFY):
                self._counts["rejected"] += 1
                self._record_result(request, PHASE_FAILED,
                                    f"a {self._run.request} run is already active")
                self.publish_status()
                return False

            run = BiasRun(
                request=request,
                action_id=action_id or uuid.uuid4().hex,
                started_at=time.monotonic(),
                phase=PHASE_PREPARING,
                detail="waiting for reset state and fresh hardware readiness",
            )
            self._run = run
            self.get_logger().info(
                f"bias {request} [{run.action_id}]: preparing"
            )
            self.publish_status()
            return True

    def _build_plans_locked(self, run: BiasRun) -> None:
        """Resolve home targets and build the interpolated ramps."""
        if run.wants_ur:
            measured = self._measured_arm_joints()
            if measured is None:
                # No start pose means no ramp. Guessing the start would make the
                # first commanded step an unbounded jump from wherever the arm
                # actually is -- precisely the motion homing exists to avoid.
                raise RuntimeError(
                    "no fresh /ur/state; cannot build a ramp without a start pose"
                )
            targets = self._load_bias_targets_locked() or both_home_joints(self.kinematics, seed=measured)
            run.arm_target = targets
            left = interpolate(measured["left"], targets["left"], self.ur_steps)
            right = interpolate(measured["right"], targets["right"], self.ur_steps)
            run.arm_plan = np.concatenate((left, right), axis=1)
            # Homing invalidates the teleop IK reference: after this move the
            # previous per-frame target is meaningless, so seed it with home.
            self.kinematics.reset(ref_joint=None)

        if run.wants_sharpa:
            measured = self._measured_hand_joints()
            if measured is None:
                raise RuntimeError(
                    "no fresh /sharpa/state; cannot prepare a safe hand hold"
                )
            start = measured
            run.hand_target = self._load_hand_bias_target_locked()
            run.hand_plan = interpolate(start, run.hand_target, self.sharpa_steps)

    def _prepare_locked(self, run: BiasRun) -> None:
        """Wait for lifecycle readiness, then sample starts and build ramps."""
        if time.monotonic() - run.started_at > self.prepare_timeout_s:
            self._fail_locked(
                "reset preparation timeout: " + (run.detail or "hardware not ready")
            )
            return
        waiting = []
        if run.wants_ur:
            if self._ur_active_state != int(ControlStatus.UR_HOMING):
                waiting.append("UR HOMING state")
            if (
                self._ur_state_time is None
                or self._ur_state_time < run.started_at
                or int(getattr(self._ur_state, "timestamp_ns", 0)) <= run.started_wall_ns
                or self._measured_arm_joints() is None
            ):
                waiting.append("fresh valid UR state")
        if run.wants_sharpa:
            if self._sharpa_active_state != int(ControlStatus.SHARPA_HOMING):
                waiting.append("SharpA HOMING state")
            hand = self._hand_state
            hand_after_request = bool(
                hand is not None
                and self._hand_state_time is not None
                and self._hand_state_time >= run.started_at
                and int(getattr(hand, "timestamp_ns", 0)) > run.started_wall_ns
            )
            if hand_after_request and str(hand.control_error):
                self._fail_locked(
                    f"SharpA POSITION transition failed: {hand.control_error}"
                )
                return
            if not hand_after_request or self._measured_hand_joints() is None:
                waiting.append("fresh valid two-hand state")
            if not hand_after_request or not bool(hand.position_control_ready):
                waiting.append("SharpA POSITION readiness")
        if waiting:
            run.detail = "waiting for " + ", ".join(waiting)
            return
        try:
            self._build_plans_locked(run)
        except (HomePoseError, ValueError, RuntimeError) as exc:
            self._fail_locked(f"plan failed: {exc}")
            return
        run.phase = PHASE_UR if run.arm_plan is not None else PHASE_SHARPA
        run.detail = ""
        run.next_step_at = time.monotonic()
        self.get_logger().info(
            f"bias {run.request} [{run.action_id}]: {run.total_steps} steps"
        )

    def _save_current_bias_locked(self) -> None:
        measured = self._measured_arm_joints()
        if measured is None:
            raise RuntimeError("no fresh /ur/state; cannot save bias")
        measured_hand = self._measured_hand_joints()
        if measured_hand is None:
            raise RuntimeError("no fresh /sharpa/state; cannot save bias")
        payload = {
            "schema": "ur_sharpa_bias.v1",
            "saved_at": time.time(),
            "left_joint": [float(v) for v in measured["left"]],
            "right_joint": [float(v) for v in measured["right"]],
            "left_eef": [float(v) for v in getattr(self._ur_state, "left_eef", [])],
            "right_eef": [float(v) for v in getattr(self._ur_state, "right_eef", [])],
            "eef_frame": str(getattr(self._ur_state, "eef_frame", "canonical")),
            "left_hand_joint": [float(v) for v in measured_hand[:22]],
            "right_hand_joint": [float(v) for v in measured_hand[22:]],
        }
        if len(payload["left_eef"]) != 9 or len(payload["right_eef"]) != 9:
            raise ValueError("/ur/state canonical EEF pose is incomplete")
        self.bias_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.bias_file.with_name(self.bias_file.name + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, self.bias_file)
        self._last_result = {"request": "save_current", "phase": PHASE_DONE,
                             "detail": f"saved {self.bias_file}",
                             "max_error_rad": None, "finished_at": time.time()}

    def _load_bias_targets_locked(self) -> dict | None:
        try:
            payload = json.loads(self.bias_file.read_text(encoding="utf-8"))
            targets = {side: np.asarray(payload[f"{side}_joint"], dtype=np.float64)
                       for side in ("left", "right")}
            if any(value.shape != (6,) or not np.all(np.isfinite(value)) for value in targets.values()):
                raise ValueError("saved bias joints must be finite 6D vectors")
            return targets
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"ignoring invalid bias file {self.bias_file}: {exc}")
            return None

    def _load_hand_bias_target_locked(self) -> np.ndarray:
        """Load the configured 44D SharpA reset pose; never assume numeric zero."""
        try:
            payload = json.loads(self.bias_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"SharpA Bias target is missing: {self.bias_file}; "
                "save the intended arm and hand pose before reset"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read Bias file {self.bias_file}: {exc}") from exc

        try:
            left = np.asarray(payload["left_hand_joint"], dtype=np.float64)
            right = np.asarray(payload["right_hand_joint"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Bias file {self.bias_file} must contain left_hand_joint and "
                "right_hand_joint"
            ) from exc
        if (
            left.shape != (22,)
            or right.shape != (22,)
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            raise RuntimeError(
                "saved SharpA Bias joints must be finite 22D vectors per hand"
            )
        return np.concatenate((left, right))

    # -------------------------------------------------------------- trajectory
    def _tick(self) -> None:
        with self._lock:
            run = self._run
            if run is None or run.phase in (PHASE_IDLE, PHASE_DONE, PHASE_FAILED):
                return
            if run.phase == PHASE_PREPARING:
                self._prepare_locked(run)
                return
            if run.phase == PHASE_VERIFY:
                self._verify_locked(run)
                return
            try:
                self._stream_step_locked(run)
            except Exception as exc:
                self._fail_locked(f"failed to publish bias step: {exc}")

    def _stream_step_locked(self, run: BiasRun) -> None:
        """Publish one paired step and wait for its complete ACK barrier."""
        now = time.monotonic()
        if run.pending_step is not None:
            if now - run.pending_since > self.command_ack_timeout_s:
                self._fail_locked(
                    f"bias step {run.pending_step} ACK timeout waiting for "
                    f"{','.join(sorted(run.pending_devices))}"
                )
            return
        if now < run.next_step_at:
            return

        idx = run.step
        devices = set()
        if run.arm_plan is not None and idx < len(run.arm_plan):
            devices.add("ur")
        if run.hand_plan is not None and idx < len(run.hand_plan):
            devices.add("sharpa")
        if not devices:
            run.phase = PHASE_VERIFY
            run.verify_deadline = now + self.arrival_timeout_s
            return

        # Install the barrier before publishing. Result callbacks can arrive
        # immediately on a multi-threaded executor and must see this step.
        run.pending_step = idx
        run.pending_devices = devices
        run.pending_since = now
        run.detail = f"step {idx} waiting for {','.join(sorted(devices))} ACK"

        if "ur" in devices:
            row = run.arm_plan[idx]
            self._publish_arm_step(row[:6], row[6:], idx, run.action_id)
            self._counts["arm_steps"] += 1

        if "sharpa" in devices:
            self._publish_hand_step(run.hand_plan[idx], idx, run.action_id)
            self._counts["hand_steps"] += 1

    def _publish_arm_step(
        self, left: Any, right: Any, step_index: int, action_id: str
    ) -> None:
        """The ONE place an ArmCommand leaves this node.

        `source = ArmCommand.SOURCE_BIAS` is set here and nowhere else, so the
        arbitration monopoly in the `homed` state cannot be lost to a second code
        path drifting. `action_id` identifies the reset transaction so stale
        ACKs from an older run cannot advance a new one.
        """
        msg = ArmCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = ArmCommand.SOURCE_BIAS
        msg.action_id = action_id
        msg.revision = 0
        msg.step_index = int(step_index)
        msg.period_s = float(self.ur_step_delay_s)
        msg.left_joint = [float(v) for v in np.asarray(left, dtype=np.float64)[:6]]
        msg.right_joint = [float(v) for v in np.asarray(right, dtype=np.float64)[:6]]
        msg.left_enable = True
        msg.right_enable = True
        self.arm_pub.publish(msg)

    def _publish_hand_step(self, q44: Any, step_index: int, action_id: str) -> None:
        msg = HandCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = HandCommand.SOURCE_BIAS
        msg.action_id = action_id
        msg.revision = 0
        msg.step_index = int(step_index)
        q = np.asarray(q44, dtype=np.float64).reshape(-1)
        if q.shape[0] != SHARPA_JOINTS_TOTAL:
            raise ValueError(f"hand command must be {SHARPA_JOINTS_TOTAL}D, got {q.shape[0]}")
        msg.q_cmd = [float(v) for v in q]
        msg.left_enable = True
        msg.right_enable = True
        self.hand_pub.publish(msg)

    def _verify_locked(self, run: BiasRun) -> None:
        """Check arrival. Running out of time is a FAILURE, not a success."""
        problems: list = []
        worst = 0.0
        errors: dict = {}

        if run.arm_plan is not None:
            measured = self._measured_arm_joints()
            result = check_arm_arrival(
                measured or {}, run.arm_target, self.ur_tolerance_rad)
            if not result.reached:
                problems.append(result.reason)
            worst = max(worst, result.max_error_rad if np.isfinite(result.max_error_rad) else 0.0)
            errors["ur"] = result.per_side_error

        if run.hand_plan is not None:
            result = check_hand_arrival(
                self._measured_hand_joints(), run.hand_target, self.sharpa_tolerance_rad)
            if not result.reached:
                problems.append(result.reason)
            worst = max(worst, result.max_error_rad if np.isfinite(result.max_error_rad) else 0.0)
            errors["sharpa"] = result.per_side_error

        if not problems:
            run.phase = PHASE_DONE
            self._counts["completed"] += 1
            self._record_result(run.request, PHASE_DONE, "", worst, errors)
            self.get_logger().info(
                f"bias {run.request} done: {run.steps_sent} steps, "
                f"max error {worst:.4f} rad"
            )
            self.publish_status()
            return

        if time.monotonic() < run.verify_deadline:
            # Still inside the grace period: the arm may simply be settling.
            run.detail = "; ".join(problems)
            return

        self._fail_locked("arrival timeout: " + "; ".join(problems), worst, errors)

    def _fail_locked(self, detail: str, max_error: float | None = None,
                     errors: dict | None = None) -> None:
        run = self._run
        request = run.request if run is not None else ""
        if run is not None:
            run.phase = PHASE_FAILED
            run.detail = detail
            run.pending_step = None
            run.pending_devices.clear()
            run.pending_since = 0.0
        self._counts["failed"] += 1
        self._record_result(request, PHASE_FAILED, detail, max_error, errors)
        # Explicit and loud. A silent drop makes a stuck arm indistinguishable
        # from a disconnected producer (see ur_node/arbitration.py).
        self.get_logger().error(f"bias {request} FAILED: {detail}")
        self.publish_status()

    def _record_result(self, request: str, phase: str, detail: str,
                       max_error: float | None = None,
                       errors: dict | None = None) -> None:
        self._last_result = {
            "request": request,
            "phase": phase,
            "detail": detail,
            "max_error_rad": (None if max_error is None or not np.isfinite(max_error)
                              else round(float(max_error), 6)),
            "per_side_error_rad": errors or {},
            "finished_at": time.time(),
        }

    # ------------------------------------------------------------------ status
    def status_payload(self) -> dict:
        with self._lock:
            run = self._run
            now = time.monotonic()
            active = run is not None and run.phase in (
                PHASE_PREPARING, PHASE_UR, PHASE_SHARPA, PHASE_VERIFY
            )
            return {
                "node": "bias_node",
                "active": active,
                "request": run.request if run is not None else "",
                "action_id": run.action_id if run is not None else "",
                "phase": run.phase if run is not None else PHASE_IDLE,
                "detail": run.detail if run is not None else "",
                "step": run.step if run is not None else 0,
                "total_steps": run.total_steps if run is not None else 0,
                "progress": (round(min(1.0, run.step / run.total_steps), 4)
                             if run is not None and run.total_steps else 0.0),
                "elapsed_s": round(now - run.started_at, 3) if run is not None else 0.0,
                "ur_state_fresh": self._measured_arm_joints() is not None,
                "sharpa_state_fresh": self._measured_hand_joints() is not None,
                "ur_state_from_control_status": self._ur_active_state,
                "sharpa_state_from_control_status": self._sharpa_active_state,
                "pending_step": run.pending_step if run is not None else None,
                "pending_devices": (
                    sorted(run.pending_devices) if run is not None else []
                ),
                "last_nack": self._last_nack,
                "last_result": dict(self._last_result),
                "counts": dict(self._counts),
                "config": {
                    "bias_file": str(self.bias_file),
                    "sharpa_target": "saved_hand_bias",
                    "ur_steps": self.ur_steps,
                    "ur_step_delay_s": self.ur_step_delay_s,
                    "ur_tolerance_rad": self.ur_tolerance_rad,
                    "sharpa_steps": self.sharpa_steps,
                    "sharpa_tolerance_rad": self.sharpa_tolerance_rad,
                    "arrival_timeout_s": self.arrival_timeout_s,
                    "prepare_timeout_s": self.prepare_timeout_s,
                    "command_ack_timeout_s": self.command_ack_timeout_s,
                },
                "safety": {
                    "enable_jump_gate": self.safety_config.enable_jump_gate,
                    "enable_flip_hold": self.safety_config.enable_flip_hold,
                    "enable_joint_clip": self.safety_config.enable_joint_clip,
                },
            }

    def publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self.status_payload(), separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False, default=float)
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BiasNode()
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
