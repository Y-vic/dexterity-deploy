"""Dual UR5 hardware node -- the single command outlet for both arms.

`/ur/command` has three publishers (bias, vive, actor). This node is the only
process holding RTDE control, so it arbitrates between them using
`arbitration.arbitrate` against the `/control/status` state machine, and it
publishes a `HardwareCommandResult` for EVERY command it sees -- accepted,
rejected by arbitration, or failed in hardware. A silently dropped command
makes a stalled arm indistinguishable from a dead producer.

Two deliberate departures from the deployed v3 `ur_node.py`:

1. **No homing on startup.** v3 ramped both arms to a hardcoded joint pose in
   its constructor, which meant `ros2 run` moved the robot. `bias_node` owns
   homing now; it publishes to `/ur/command` with `SOURCE_BIAS`, which is the
   only source arbitration admits before running. This node comes up read-only
   with the RTDE control program loaded; motion starts only when the operator
   requests the bias trajectory, then enables running.
2. **Joint-space commands.** `ArmCommand` carries joint targets, so there is no
   IK here at all. `actor_ik_node` / `vive_node` own IK via `ur_common`, which
   is what keeps FK (observation) and IK (execution) on one backend.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from std_msgs.msg import Bool, String

from ur_common.drivers import (
    HardwareConnectionError,
    HardwareDependencyError,
    UrRtdePair,
    format_ur_pair_diagnostics,
)
from ur_common.safety import SafetyConfig
from ur_common.kinematics import UrCommonKinematics
from ur_common.geometry import CANONICAL_EEF_FRAME
from ur_common.execution_log import execution_event_json

from ur_msgs.msg import ArmCommand, ControlStatus, HardwareCommandResult, UrStateFrame

from .arbitration import Source, UrState, arbitrate


# UR joints per arm; ArmCommand fixes this at 6 by hardware.
UR5_NUM_JOINTS = 6
DEVICE = "ur"


class UrNode(Node):
    """RTDE outlet. Publishes `/ur/state` at 30Hz, consumes `/ur/command`."""

    def __init__(self, driver: Any = None) -> None:
        super().__init__("ur_node")

        # --- hardware ---
        self.declare_parameter("left_ip", "192.168.56.20")
        self.declare_parameter("right_ip", "192.168.56.10")
        self.declare_parameter("rtde_frequency_hz", 125.0)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("reconnect_interval_s", 5.0)
        # The status/arbitration state machine remains the execution gate for
        # arm commands; these guards constrain every command that passes it.
        self.declare_parameter("enable_motion_limits", True)
        self.declare_parameter("enable_joint_step_limit", True)
        self.declare_parameter("max_joint_step_rad", 0.35)
        self.declare_parameter("max_abs_joint_rad", 6.28)
        self.declare_parameter("enable_payload_configuration", True)
        self.declare_parameter("left_payload_mass_kg", 1.3)
        self.declare_parameter("left_payload_cog_m", [0.0, 0.0, 0.08])
        self.declare_parameter("right_payload_mass_kg", 1.3)
        self.declare_parameter("right_payload_cog_m", [0.0, 0.0, 0.08])

        # --- final joint command limits ---
        self.declare_parameter("enable_joint_clip", True)
        self.declare_parameter("joint_vel_limit", 0.35)

        # --- topics: defaults MUST match docs/ur_topics.md ---
        self.declare_parameter("state_topic", "/ur/state")
        self.declare_parameter("command_topic", "/ur/command")
        self.declare_parameter("result_topic", "/ur/command_result")
        self.declare_parameter("status_topic", "/control/status")
        self.declare_parameter("stop_topic", "/hardware/stop")

        publish_hz = float(self.get_parameter("publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")

        self.enable_motion_limits = bool(self.get_parameter("enable_motion_limits").value)
        self.enable_joint_step_limit = bool(
            self.get_parameter("enable_joint_step_limit").value
        )
        self.enable_joint_clip = bool(self.get_parameter("enable_joint_clip").value)
        self.max_joint_step_rad = float(
            self.get_parameter("max_joint_step_rad").value
        )
        self.max_abs_joint_rad = float(
            self.get_parameter("max_abs_joint_rad").value
        )
        for name, value in (
            ("max_joint_step_rad", self.max_joint_step_rad),
            ("max_abs_joint_rad", self.max_abs_joint_rad),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        # JointLimiter bounds per-frame delta and absolute range. It does not
        # replace the wrist-flip hold, which lives upstream in the IK solver.
        self.kinematics = UrCommonKinematics(
            config=SafetyConfig(
                enable_joint_clip=self.enable_joint_clip,
                joint_vel_limit=float(self.get_parameter("joint_vel_limit").value),
                joint_pos_limit=(self.max_abs_joint_rad,) * UR5_NUM_JOINTS,
            )
        )

        self.driver = driver
        self.driver_error = ""
        self._driver_lock = threading.RLock()
        self._reconnect_lock = threading.Lock()
        self._reconnecting = False
        self._rtde_recovery_needed = False
        self._rtde_recovery_ready = False
        self._destroying = False
        self.reconnect_interval_s = max(
            1.0, float(self.get_parameter("reconnect_interval_s").value)
        )
        if self.driver is None:
            self.driver = UrRtdePair(
                left_ip=str(self.get_parameter("left_ip").value),
                right_ip=str(self.get_parameter("right_ip").value),
                frequency_hz=float(self.get_parameter("rtde_frequency_hz").value),
                enable_control=True,
                enable_motion_limits=self.enable_motion_limits,
                enable_joint_step_limit=self.enable_joint_step_limit,
                max_joint_step_rad=self.max_joint_step_rad,
                max_abs_joint_rad=self.max_abs_joint_rad,
                enable_payload_configuration=bool(
                    self.get_parameter("enable_payload_configuration").value
                ),
                left_payload_mass_kg=float(
                    self.get_parameter("left_payload_mass_kg").value
                ),
                left_payload_cog_m=self.get_parameter("left_payload_cog_m").value,
                right_payload_mass_kg=float(
                    self.get_parameter("right_payload_mass_kg").value
                ),
                right_payload_cog_m=self.get_parameter("right_payload_cog_m").value,
            )
            try:
                self.driver.connect()
            except (HardwareDependencyError, HardwareConnectionError) as exc:
                # Degrade instead of crashing: the node stays up, publishes
                # invalid frames carrying this reason, and refuses commands with
                # it. An operator sees the cause on /ur/state and
                # /ur/command_result instead of a traceback in a dead process.
                self.driver_error = str(exc)
                self.get_logger().error(f"UR hardware unavailable: {exc}")

        # NO startup homing. bias_node owns it; see the module docstring.
        self.ur_state = UrState.STOPPED
        self.active_source = Source.UNKNOWN
        self.stopped = False
        self.last_command_acked = True
        self.last_command_message = ""
        self._last_safety_signature: tuple[Any, ...] | None = None

        self.state_pub = self.create_publisher(
            UrStateFrame, str(self.get_parameter("state_topic").value), 10
        )
        self.result_pub = self.create_publisher(
            HardwareCommandResult, str(self.get_parameter("result_topic").value), 10
        )
        self.execution_event_pub = self.create_publisher(
            String, "/deploy/execution_event", 100
        )
        self.command_sub = self.create_subscription(
            ArmCommand,
            str(self.get_parameter("command_topic").value),
            self.on_command,
            10,
        )
        self.status_sub = self.create_subscription(
            ControlStatus,
            str(self.get_parameter("status_topic").value),
            self.on_status,
            10,
        )
        self.stop_sub = self.create_subscription(
            Bool, str(self.get_parameter("stop_topic").value), self.on_stop, 10
        )
        self.timer = self.create_timer(1.0 / publish_hz, self.publish_state)
        self.reconnect_timer = self.create_timer(
            self.reconnect_interval_s, self.schedule_reconnect
        )
        self.get_logger().info(
            f"ur_node up: state={publish_hz:g}Hz execution=true "
            f"motion_limits={self.enable_motion_limits} "
            f"joint_step_limit={self.enable_joint_step_limit} "
            f"joint_clip={self.enable_joint_clip} (no startup homing)"
        )

    # ---- inbound status ------------------------------------------------
    def on_status(self, message: ControlStatus) -> None:
        """Track the gate. `status_node` owns these states; we only read them."""
        try:
            self.ur_state = UrState(int(message.ur_state))
        except ValueError:
            self.get_logger().warning(
                f"unknown ur_state {message.ur_state}; treating as stopped"
            )
            self.ur_state = UrState.STOPPED
        try:
            self.active_source = Source(int(message.active_source))
        except ValueError:
            self.active_source = Source.UNKNOWN

    def on_stop(self, message: Bool) -> None:
        """`/hardware/stop` latch. True stops servo motion and blocks commands.

        The latch clears on False so the operator can resume from the web UI
        without restarting the node, but it never clears itself.
        """
        if message.data:
            self.stopped = True
            self.last_command_acked = False
            self.last_command_message = "hardware stop asserted"
            with self._driver_lock:
                self.driver.safe_stop()
            self.get_logger().warning("hardware stop asserted; UR servo halted")
        elif self.stopped:
            self.stopped = False
            self.last_command_message = ""
            self.get_logger().info("hardware stop cleared")

    def schedule_reconnect(self) -> None:
        if self._destroying or not (
            self.driver_error or self._rtde_recovery_ready
        ):
            return
        with self._reconnect_lock:
            if self._reconnecting:
                return
            self._reconnecting = True
        threading.Thread(
            target=self._reconnect,
            daemon=True,
            name="ur-rtde-reconnect",
        ).start()

    def _reconnect(self) -> None:
        try:
            with self._driver_lock:
                self.driver.close()
                self.driver.connect()
                if self.stopped:
                    # Re-apply the operator stop latch to every new control
                    # session, including one that reconnects before NORMAL.
                    self.driver.safe_stop()
                snapshot = self.driver.read()
                if not snapshot.normal_mode:
                    self.driver_error = ""
                    self._rtde_recovery_needed = True
                    self._rtde_recovery_ready = False
                    self.last_command_message = (
                        "UR RTDE connected; waiting for NORMAL mode"
                    )
                    return
                synchronize = getattr(
                    self.driver, "synchronize_command_state", None
                )
                if callable(synchronize):
                    synchronize(snapshot)
            self.driver_error = ""
            self._rtde_recovery_needed = False
            self._rtde_recovery_ready = False
            self.last_command_message = ""
            self.get_logger().info(
                "UR RTDE pair reconnected in NORMAL mode; command state synchronized"
            )
        except (HardwareDependencyError, HardwareConnectionError) as exc:
            self.driver_error = str(exc)
        except Exception as exc:
            self.driver_error = f"UR reconnect failed: {exc}"
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    # ---- outbound state ------------------------------------------------
    def publish_state(self) -> None:
        message = UrStateFrame()
        message.header.stamp = self.get_clock().now().to_msg()
        message.eef_frame = CANONICAL_EEF_FRAME
        message.last_command_acked = self.last_command_acked
        if self._reconnecting:
            # This is an expected, transient read outage. Do not turn it into
            # driver_error: doing so can schedule a second reconnect after the
            # first thread has deliberately returned to wait for NORMAL.
            message.timestamp_ns = time.time_ns()
            message.valid = False
            message.normal_mode = False
            message.status_message = "UR RTDE reconnecting"
            self.state_pub.publish(message)
            return
        try:
            if self.driver_error:
                raise HardwareConnectionError(self.driver_error)
            with self._driver_lock:
                snapshot = self.driver.read()
        except Exception as exc:
            # Publish an explicitly invalid frame rather than nothing: a
            # consumer needs to distinguish "arm is unreadable" from "ur_node
            # is not running", and only a frame with valid=false does that.
            message.timestamp_ns = time.time_ns()
            message.valid = False
            message.normal_mode = False
            message.status_message = self.driver_error or f"UR state read failed: {exc}"
            if not self._destroying:
                self.driver_error = message.status_message
                self.schedule_reconnect()
            self.state_pub.publish(message)
            return
        message.timestamp_ns = int(snapshot.timestamp_ns)
        message.left_joint = np.asarray(snapshot.left_joint, dtype=np.float32).tolist()
        message.right_joint = np.asarray(snapshot.right_joint, dtype=np.float32).tolist()
        message.left_tcp_pose6 = np.asarray(snapshot.left_rtde_pose, dtype=np.float32).tolist()
        message.right_tcp_pose6 = np.asarray(snapshot.right_rtde_pose, dtype=np.float32).tolist()
        try:
            message.left_eef = np.asarray(
                self.kinematics.arm_fk_eef9(snapshot.left_joint, "left"),
                dtype=np.float32,
            ).tolist()
            message.right_eef = np.asarray(
                self.kinematics.arm_fk_eef9(snapshot.right_joint, "right"),
                dtype=np.float32,
            ).tolist()
        except Exception as exc:
            message.valid = False
            message.normal_mode = False
            message.status_message = f"canonical FK failed: {exc}"
            self.state_pub.publish(message)
            return
        message.normal_mode = bool(snapshot.normal_mode)
        message.valid = True
        if snapshot.normal_mode:
            if self._rtde_recovery_needed:
                # The controller has recovered. Reconnect only now, so the old
                # RTDE control session is not torn down while protection is
                # still active.
                self._rtde_recovery_ready = True
                self.schedule_reconnect()
            if self._last_safety_signature is not None:
                self.get_logger().info("UR pair returned to normal running mode")
            self._last_safety_signature = None
            message.status_message = self.last_command_message
        else:
            safety_message = "UR safety state abnormal: " + format_ur_pair_diagnostics(
                snapshot
            )
            protective_stop = any(
                diagnostic.protective_stopped is True
                or diagnostic.safety_mode == 3
                for diagnostic in (
                    snapshot.left_diagnostic,
                    snapshot.right_diagnostic,
                )
            )
            if protective_stop:
                self._rtde_recovery_needed = True
            signature = self._safety_signature(snapshot)
            if signature != self._last_safety_signature:
                self.get_logger().error(safety_message)
                self._last_safety_signature = signature
            message.status_message = safety_message
        self.state_pub.publish(message)

    @staticmethod
    def _safety_signature(snapshot: Any) -> tuple[Any, ...]:
        result: list[Any] = []
        for diagnostic in (snapshot.left_diagnostic, snapshot.right_diagnostic):
            result.extend(
                (
                    diagnostic.normal_mode,
                    diagnostic.safety_mode,
                    diagnostic.safety_status_bits,
                    diagnostic.robot_mode,
                    diagnostic.protective_stopped,
                    diagnostic.emergency_stopped,
                    diagnostic.probe_errors,
                )
            )
        return tuple(result)

    # ---- commands ------------------------------------------------------
    def execution_event(self, event: str, message: ArmCommand, **fields: Any) -> None:
        publisher = getattr(self, "execution_event_pub", None)
        if publisher is None:
            return
        try:
            output = String()
            output.data = execution_event_json(
                "ur_node",
                event,
                action_id=str(getattr(message, "action_id", "")),
                revision=int(getattr(message, "revision", 0)),
                step_index=int(getattr(message, "step_index", 0)),
                source=int(getattr(message, "source", 0)),
                dry_run=bool(getattr(message, "dry_run", False)),
                **fields,
            )
            publisher.publish(output)
        except Exception as exc:
            self.get_logger().warning(f"failed to publish execution event: {exc}")

    def on_command(self, message: ArmCommand) -> None:
        """Arbitrate, then execute. Always publishes exactly one result."""
        self.execution_event("hardware_command_received", message, device=DEVICE)
        if bool(message.dry_run):
            if int(message.source) != int(ArmCommand.SOURCE_ACTOR):
                self.publish_result(
                    message,
                    False,
                    "dry_run_source_rejected",
                    "only Actor commands may use dry_run",
                )
                return
            try:
                period_s = float(message.period_s)
                if not np.isfinite(period_s) or period_s <= 0.0:
                    raise ValueError("period_s must be finite and positive")
                with self._driver_lock:
                    self.resolve_targets(message)
            except Exception as exc:
                self.publish_result(
                    message, False, "ur_dry_run_validation_failed", str(exc)
                )
                return
            self.publish_result(message, True, "", "")
            return
        if self.stopped:
            self.publish_result(message, False, "hardware_stopped", "hardware stop asserted")
            return
        if self.driver_error or self._reconnecting or self._rtde_recovery_needed:
            self.publish_result(
                message,
                False,
                "ur_recovering",
                self.driver_error or "UR RTDE recovery in progress",
            )
            return

        decision = arbitrate(self.ur_state, self.active_source, Source(int(message.source)))
        if not decision:
            # The reason string comes straight from arbitration so the producer
            # learns WHY it was refused, not just that it was.
            self.publish_result(message, False, "arbitration_rejected", decision.reason)
            self.get_logger().warning(f"command rejected: {decision.reason}")
            return

        try:
            sdk_started_ns = time.monotonic_ns()
            self.execution_event("hardware_sdk_command_started", message, device=DEVICE)
            with self._driver_lock:
                left, right = self.resolve_targets(message)
                self.driver.command_joint_step(
                    left, right, float(message.period_s),
                    left_enable=bool(message.left_enable),
                    right_enable=bool(message.right_enable),
                )
            self.execution_event(
                "hardware_sdk_command_completed",
                message,
                device=DEVICE,
                duration_ms=(time.monotonic_ns() - sdk_started_ns) / 1_000_000.0,
                success=True,
            )
        except Exception as exc:
            self.execution_event(
                "hardware_sdk_command_failed",
                message,
                device=DEVICE,
                success=False,
                error=str(exc),
            )
            with self._driver_lock:
                enabled = [
                    side for side in ("left", "right")
                    if bool(getattr(message, f"{side}_enable"))
                ]
                try:
                    for side in enabled:
                        self.driver.safe_stop(side)
                except TypeError:
                    self.driver.safe_stop()
            self.publish_result(message, False, "ur_command_failed", str(exc))
            self.get_logger().error(f"UR command failed: {exc}")
            return
        self.publish_result(message, True, "", "")

    def resolve_targets(self, message: ArmCommand) -> tuple[np.ndarray, np.ndarray]:
        """Validate the joint payload and apply SWITCH 3 if it is on.

        Clipping needs the measured joints as the reference, so it happens here
        (where a snapshot is available) rather than inside the driver.
        """
        left = np.asarray(message.left_joint, dtype=np.float64).reshape(-1)
        right = np.asarray(message.right_joint, dtype=np.float64).reshape(-1)
        enables = {
            "left": bool(message.left_enable),
            "right": bool(message.right_enable),
        }
        if not any(enables.values()):
            raise ValueError("at least one UR arm must be enabled")
        for side, value in (("left", left), ("right", right)):
            if not enables[side]:
                continue
            if value.shape != (UR5_NUM_JOINTS,) or not np.isfinite(value).all():
                raise ValueError(f"{side}_joint must contain {UR5_NUM_JOINTS} finite angles")
        if not (
            self.enable_motion_limits
            or self.enable_joint_clip
            or self.enable_joint_step_limit
        ):
            return left, right
        snapshot = self.driver.read()
        current_left = np.asarray(snapshot.left_joint, dtype=np.float64).reshape(-1)
        current_right = np.asarray(snapshot.right_joint, dtype=np.float64).reshape(-1)
        current = {"left": current_left, "right": current_right}
        targets = {"left": left, "right": right}
        for side in ("left", "right"):
            if not enables[side]:
                continue
            targets[side] = targets[side] + np.round(
                (current[side] - targets[side]) / (2.0 * np.pi)
            ) * (2.0 * np.pi)
            if self.enable_motion_limits and np.any(
                np.abs(targets[side]) > self.max_abs_joint_rad + 1.0e-9
            ):
                raise ValueError(
                    f"{side} joint target exceeds absolute limit "
                    f"{self.max_abs_joint_rad:.3f} rad"
                )
            if (
                self.enable_joint_step_limit
                and np.any(
                    np.abs(targets[side] - current[side])
                    > self.max_joint_step_rad + 1.0e-9
                )
            ):
                raise ValueError(
                    f"{side} joint target step exceeds "
                    f"{self.max_joint_step_rad:.3f} rad"
                )
        if not self.enable_joint_clip:
            return targets["left"], targets["right"]
        return (
            self.kinematics.limit_joints(targets["left"], current_left),
            self.kinematics.limit_joints(targets["right"], current_right),
        )

    def publish_result(
        self,
        message: ArmCommand,
        success: bool,
        failure_code: str,
        failure_message: str,
    ) -> None:
        result = HardwareCommandResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.source = message.source
        result.action_id = message.action_id
        result.revision = message.revision
        result.step_index = message.step_index
        result.device = DEVICE
        result.success = bool(success)
        result.failure_code = failure_code
        result.failure_message = failure_message
        result.dry_run = bool(message.dry_run)
        result.timestamp_ns = time.time_ns()
        self.result_pub.publish(result)
        self.execution_event(
            "hardware_command_result_published",
            message,
            device=DEVICE,
            success=bool(success),
            failure_code=str(failure_code),
            failure_message=str(failure_message),
        )
        self.last_command_acked = bool(success)
        self.last_command_message = "" if success else failure_message

    def destroy_node(self) -> bool:
        self._destroying = True
        try:
            with self._driver_lock:
                self.driver.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node: UrNode | None = None
    try:
        node = UrNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
