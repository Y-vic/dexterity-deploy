#!/usr/bin/env python3
"""Adam lowstate bridge and final command publisher."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import rclpy
from pnd_adam.msg import LowState
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from adam_node.body_joints import (
    ADAM_COMMAND_JOINTS_19,
    ADAM_CONTROL_JOINTS_32,
    ADAM_PHYSICAL_JOINTS_31,
    ADAM_ROBOT_STATE_JOINTS_31,
    CONTROL_HAND_JOINTS_12,
    LOWSTATE_INDEX_BY_JOINT,
    LOWSTATE_REAL_JOINTS_31,
    PND_CONTROL_HAND_PLACEHOLDER_VALUE,
    PND_CONTROL_ROOT_HEIGHT_VALUE,
    ROBOT_STATE_HAND_JOINTS_12,
    canonical_body_name,
)


@dataclass
class TimedCommand:
    values: dict[str, float]
    stamp: float
    count: int
    source: str


@dataclass
class TimedLowState:
    stamp: float
    tick: int
    motor_count: int


BIAS_CONTROL_STATES = {"t_init", "t_init_sharpa"}
COMMAND_CONTROL_STATES = {"t_adam", "t_adam_sharpa"}


class AdamNode(Node):
    """Own Adam physical state conversion and final command output.

    Topic data contracts:

    * /lowstate is pnd_adam/msg/LowState. It has no joint names. Its
      motor_state[31] order is LOWSTATE_REAL_JOINTS_31:
      legs 6+6, waist 3, neck 2, arms 7+7. Each MotorState carries q, dq,
      ddq, tau_est, mode, state. sensor_msgs/JointState has no acceleration
      field, so ddq is intentionally not represented on the JointState topics.

    * /adam_physical_joint_states is sensor_msgs/msg/JointState and keeps the
      real lowstate 31-joint order. position=q, velocity=dq, effort=tau_est.
      This is the topic Bias and Noitom should read when they need the current
      real Adam pose.

    * /robot_states is sensor_msgs/msg/JointState but uses the upper ROS
      contract: waist 3, neck 2, arms 7+7, then 12 virtual hand joints. The
      first 19 values are mapped from /lowstate by name; the virtual hand
      slots are non-existent Adam joints and are always zero.

    * /adam_bias_command_joint_states and /adam_command_joint_states are
      sensor_msgs/msg/JointState command candidates. Each must contain exactly
      ADAM_COMMAND_JOINTS_19: waist 3, neck 2, arms 7+7. They do not contain
      legs or virtual hands. Adam selects bias vs teleop command from
      /control_status.

    * /joint_states is the final command stream that Adam publishes downstream
      to PndControl. It uses PndControl's 32-field SPPRO external-retarget
      contract: 19 command joints, root_pos/z, then 12 virtual hand
      placeholders. The root height is 1.0 and the hand placeholder values are
      1000.0, matching PndControl's expected contract.
    """

    def __init__(self) -> None:
        super().__init__("adam")

        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter(
            "bias_joint_states_topic", "/adam_bias_command_joint_states"
        )
        self.declare_parameter(
            "command_joint_states_topic", "/adam_command_joint_states"
        )
        self.declare_parameter(
            "physical_joint_states_topic", "/adam_physical_joint_states"
        )
        self.declare_parameter("robot_states_topic", "/robot_states")
        self.declare_parameter("control_joint_states_topic", "/joint_states")
        self.declare_parameter("control_status_topic", "/control_status")
        self.declare_parameter("status_topic", "/adam/status")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("command_timeout", 0.25)
        self.declare_parameter("lowstate_timeout", 0.5)
        self.declare_parameter("control_status_timeout", 1.5)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("require_control_subscriber", True)
        self.declare_parameter("warn_period", 1.0)

        self.lowstate_topic = str(self.get_parameter("lowstate_topic").value)
        self.bias_topic = str(
            self.get_parameter("bias_joint_states_topic").value
        )
        self.command_topic = str(
            self.get_parameter("command_joint_states_topic").value
        )
        self.physical_topic = str(
            self.get_parameter("physical_joint_states_topic").value
        )
        self.robot_states_topic = str(self.get_parameter("robot_states_topic").value)
        self.control_topic = str(
            self.get_parameter("control_joint_states_topic").value
        )
        self.control_status_topic = str(
            self.get_parameter("control_status_topic").value
        )
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.lowstate_timeout = float(self.get_parameter("lowstate_timeout").value)
        self.control_status_timeout = float(
            self.get_parameter("control_status_timeout").value
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.require_control_subscriber = bool(
            self.get_parameter("require_control_subscriber").value
        )
        self.warn_period = float(self.get_parameter("warn_period").value)

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        if self.command_timeout <= 0.0:
            raise ValueError("command_timeout must be positive")
        if self.lowstate_timeout <= 0.0:
            raise ValueError("lowstate_timeout must be positive")
        if self.control_status_timeout <= 0.0:
            raise ValueError("control_status_timeout must be positive")

        self.bias_command: TimedCommand | None = None
        self.teleop_command: TimedCommand | None = None
        self.lowstate: TimedLowState | None = None
        self.last_command_values = {name: 0.0 for name in ADAM_COMMAND_JOINTS_19}
        self.control_state = "damping"
        self.control_status_count = 0
        self.last_control_status_time: float | None = None
        self.active_source = "idle_no_command"
        self.last_error = ""
        self.last_warn_time = 0.0
        self.bias_logged_for_state = False
        self.control_logged_for_state = False
        self.last_bias_frame_id = ""

        self.lowstate_received = 0
        self.physical_published = 0
        self.robot_states_published = 0
        self.bias_command_received = 0
        self.teleop_command_received = 0
        self.command_frames_processed = 0
        self.control_published = 0
        self.control_skipped_dry_run = 0
        self.control_blocked_no_subscriber = 0
        self.dropped = 0
        self.idle_ticks = 0
        self.status_window_control = 0
        self.status_window_time = time.monotonic()
        self.output_hz = 0.0

        self.physical_pub = self.create_publisher(JointState, self.physical_topic, 10)
        self.robot_states_pub = self.create_publisher(
            JointState, self.robot_states_topic, 10
        )
        self.control_pub = self.create_publisher(JointState, self.control_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(LowState, self.lowstate_topic, self._on_lowstate, 10)
        self.create_subscription(
            JointState,
            self.bias_topic,
            lambda msg: self._on_command(msg, "bias"),
            10,
        )
        self.create_subscription(
            JointState,
            self.command_topic,
            lambda msg: self._on_command(msg, "command"),
            10,
        )
        self.create_subscription(String, self.control_status_topic, self._on_status, 10)
        self.create_timer(1.0 / self.publish_rate, self._publish_command_outputs)
        self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            "Adam node: "
            f"lowstate={self.lowstate_topic}, bias={self.bias_topic}, "
            f"command={self.command_topic}, status={self.control_status_topic}, "
            f"physical={self.physical_topic}, robot_states={self.robot_states_topic}, "
            f"control={self.control_topic}, dry_run={self.dry_run}, "
            f"require_control_subscriber={self.require_control_subscriber}"
        )

    def _on_lowstate(self, msg: LowState) -> None:
        motor_count = len(msg.motor_state)
        if motor_count < len(LOWSTATE_REAL_JOINTS_31):
            self._drop(
                "ignored short LowState: "
                f"motor_state has {motor_count}, expected {len(LOWSTATE_REAL_JOINTS_31)}"
            )
            return

        try:
            physical = self._make_physical_msg(msg)
            robot_states = self._make_robot_states_msg(msg)
        except Exception as exc:  # noqa: BLE001 - keep node alive on malformed samples.
            self._drop(f"ignored bad LowState: {exc}")
            return

        self.physical_pub.publish(physical)
        self.robot_states_pub.publish(robot_states)

        self.lowstate_received += 1
        self.physical_published += 1
        self.robot_states_published += 1
        self.lowstate = TimedLowState(
            stamp=time.monotonic(),
            tick=int(msg.tick),
            motor_count=motor_count,
        )

    def _on_command(self, msg: JointState, source: str) -> None:
        try:
            values = self._command_positions_from_msg(msg)
        except Exception as exc:  # noqa: BLE001 - keep last valid command active.
            self._drop(f"ignored bad Adam {source} JointState: {exc}")
            return
        if source == "bias":
            self.bias_command_received += 1
            count = self.bias_command_received
        elif source == "command":
            self.teleop_command_received += 1
            count = self.teleop_command_received
        else:
            self._drop(f"ignored command from unknown source: {source}")
            return
        command = TimedCommand(
            values=values,
            stamp=time.monotonic(),
            count=count,
            source=f"{source}:{msg.header.frame_id or ''}".rstrip(":"),
        )
        if source == "bias":
            self.bias_command = command
            frame_id = msg.header.frame_id or ""
            if self.control_state in BIAS_CONTROL_STATES and (
                not self.bias_logged_for_state
                or frame_id != self.last_bias_frame_id
            ):
                self.bias_logged_for_state = True
                self.get_logger().info(
                    "Adam received Bias phase in "
                    f"{self.control_state}: count={count}, "
                    f"frame_id={frame_id!r}, "
                    f"names={len(msg.name)}, positions={len(msg.position)}"
                )
            self.last_bias_frame_id = frame_id
        else:
            self.teleop_command = command

    def _on_status(self, msg: String) -> None:
        state = self._state_from_status(msg.data)
        if not state:
            self._drop(f"ignored bad control status: {msg.data!r}")
            return
        previous_state = self.control_state
        self.control_state = state
        self.control_status_count += 1
        self.last_control_status_time = time.monotonic()
        if state != previous_state:
            self.bias_logged_for_state = False
            self.control_logged_for_state = False
            self.get_logger().info(
                "Adam control state: "
                f"{previous_state} -> {state}; "
                f"control_output_subscribers={self.control_pub.get_subscription_count()}"
            )

    def _publish_command_outputs(self) -> None:
        now = time.monotonic()
        command = self._selected_command(now)
        if command is None:
            self.idle_ticks += 1
            self.active_source = self._inactive_reason(now)
            self.last_error = ""
            return

        control_msg = self._make_control_msg(command)
        control_subscribers = self.control_pub.get_subscription_count()
        if self.dry_run:
            self.control_skipped_dry_run += 1
        elif self.require_control_subscriber and control_subscribers == 0:
            self.control_blocked_no_subscriber += 1
            self.active_source = "blocked_no_control_subscriber"
            self.last_error = (
                "blocked control output: /joint_states has no PndControl "
                "subscriber; run teleoperation as the same OS user and with "
                "the same RMW_IMPLEMENTATION as PndControl"
            )
            self._warn_throttled(self.last_error)
            return
        else:
            self.control_pub.publish(control_msg)
            self.control_published += 1
            self.status_window_control += 1
        if (
            self.control_state in BIAS_CONTROL_STATES
            and not self.control_logged_for_state
        ):
            self.control_logged_for_state = True
            output_action = "prepared" if self.dry_run else "published"
            self.get_logger().info(
                f"Adam {output_action} first Bias control frame: "
                f"state={self.control_state}, source={command.source}, "
                f"names={len(control_msg.name)}, "
                f"positions={len(control_msg.position)}, "
                f"frame_id={control_msg.header.frame_id!r}, "
                "root_pos/z="
                f"{control_msg.position[len(ADAM_COMMAND_JOINTS_19)]:.3f}, "
                f"control_output_subscribers={control_subscribers}"
            )

        self.last_command_values = dict(command.values)
        self.command_frames_processed += 1
        self.active_source = command.source
        self.last_error = ""

    def _selected_command(self, now: float) -> TimedCommand | None:
        if not self._status_is_fresh(now):
            return None
        if self.control_state in BIAS_CONTROL_STATES:
            return (
                self.bias_command
                if self._is_fresh(self.bias_command, self.command_timeout, now)
                else None
            )
        if self.control_state in COMMAND_CONTROL_STATES:
            return (
                self.teleop_command
                if self._is_fresh(self.teleop_command, self.command_timeout, now)
                else None
            )
        return None

    def _inactive_reason(self, now: float) -> str:
        if not self._status_is_fresh(now):
            return "idle_stale_or_missing_status"
        if self.control_state in BIAS_CONTROL_STATES:
            return "idle_waiting_for_bias"
        if self.control_state in COMMAND_CONTROL_STATES:
            return "idle_waiting_for_command"
        return f"idle_status_{self.control_state}"

    def _make_physical_msg(self, msg: LowState) -> JointState:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "adam_lowstate"
        out.name = list(ADAM_PHYSICAL_JOINTS_31)
        out.position = [
            self._finite_motor_field(msg, index, "q")
            for index in range(len(ADAM_PHYSICAL_JOINTS_31))
        ]
        out.velocity = [
            self._finite_motor_field(msg, index, "dq")
            for index in range(len(ADAM_PHYSICAL_JOINTS_31))
        ]
        out.effort = [
            self._finite_motor_field(msg, index, "tau_est")
            for index in range(len(ADAM_PHYSICAL_JOINTS_31))
        ]
        return out

    def _make_robot_states_msg(self, msg: LowState) -> JointState:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "adam_robot_states_from_lowstate"
        out.name = list(ADAM_ROBOT_STATE_JOINTS_31)
        out.position = [
            self._finite_motor_field(msg, LOWSTATE_INDEX_BY_JOINT[name], "q")
            for name in ADAM_COMMAND_JOINTS_19
        ] + [0.0] * len(ROBOT_STATE_HAND_JOINTS_12)
        out.velocity = [
            self._finite_motor_field(msg, LOWSTATE_INDEX_BY_JOINT[name], "dq")
            for name in ADAM_COMMAND_JOINTS_19
        ] + [0.0] * len(ROBOT_STATE_HAND_JOINTS_12)
        out.effort = [
            self._finite_motor_field(msg, LOWSTATE_INDEX_BY_JOINT[name], "tau_est")
            for name in ADAM_COMMAND_JOINTS_19
        ] + [0.0] * len(ROBOT_STATE_HAND_JOINTS_12)
        return out

    def _make_control_msg(self, command: TimedCommand) -> JointState:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = f"adam_control_from:{command.source}"
        out.name = list(ADAM_CONTROL_JOINTS_32)
        out.position = [
            float(command.values[name]) for name in ADAM_COMMAND_JOINTS_19
        ] + [PND_CONTROL_ROOT_HEIGHT_VALUE] + [
            PND_CONTROL_HAND_PLACEHOLDER_VALUE
        ] * len(CONTROL_HAND_JOINTS_12)
        out.velocity = [0.0] * len(out.name)
        out.effort = [0.0] * len(out.name)
        return out

    @staticmethod
    def _finite_motor_field(msg: LowState, index: int, field: str) -> float:
        motor = msg.motor_state[index]
        value = getattr(motor, field)
        return AdamNode._finite_or_raise(value, f"motor_state[{index}].{field}")

    @staticmethod
    def _finite_or_raise(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite value for {name}: {value!r}")
        return number

    @staticmethod
    def _command_positions_from_msg(msg: JointState) -> dict[str, float]:
        values: dict[str, float] = {}
        forbidden: list[str] = []
        for idx, name in enumerate(msg.name):
            if not name:
                continue
            canonical = canonical_body_name(name)
            if canonical in CONTROL_HAND_JOINTS_12 or canonical in ROBOT_STATE_HAND_JOINTS_12:
                forbidden.append(canonical)
                continue
            if canonical not in ADAM_COMMAND_JOINTS_19:
                continue
            if idx >= len(msg.position):
                raise ValueError(f"JointState position is missing for {canonical}")
            values[canonical] = AdamNode._finite_or_raise(
                msg.position[idx],
                canonical,
            )
        if forbidden:
            raise ValueError(
                "Adam command JointState must not contain virtual hand joints: "
                f"{forbidden}"
            )
        missing = [name for name in ADAM_COMMAND_JOINTS_19 if name not in values]
        if missing:
            raise ValueError(
                "Adam command JointState must contain exactly the 19 upper-body "
                f"command joints; missing={missing}"
            )
        return {name: values[name] for name in ADAM_COMMAND_JOINTS_19}

    def _drop(self, reason: str) -> None:
        self.dropped += 1
        self.last_error = reason
        self._warn_throttled(reason)

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_warn_time < self.warn_period:
            return
        self.last_warn_time = now
        self.get_logger().warning(message)

    def _publish_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.status_window_time
        if elapsed > 0.0:
            self.output_hz = self.status_window_control / elapsed
        self.status_window_control = 0
        self.status_window_time = now

        status = {
            "node": "adam",
            "mode": "lowstate_bridge_and_command_output",
            "dry_run": self.dry_run,
            "active_source": self.active_source,
            "topics": {
                "lowstate": self.lowstate_topic,
                "bias_command": self.bias_topic,
                "command": self.command_topic,
                "control_status": self.control_status_topic,
                "physical_state": self.physical_topic,
                "robot_states": self.robot_states_topic,
                "control_output": self.control_topic,
            },
            "fresh": {
                "lowstate": self._is_fresh(self.lowstate, self.lowstate_timeout, now),
                "bias_command": self._is_fresh(
                    self.bias_command, self.command_timeout, now
                ),
                "command": self._is_fresh(
                    self.teleop_command, self.command_timeout, now
                ),
                "control_status": self._status_is_fresh(now),
            },
            "age_ms": {
                "lowstate": self._age_ms(self.lowstate, now),
                "bias_command": self._age_ms(self.bias_command, now),
                "command": self._age_ms(self.teleop_command, now),
                "control_status": self._age_ms_from_stamp(
                    self.last_control_status_time, now
                ),
            },
            "counts": {
                "lowstate_received": self.lowstate_received,
                "physical_published": self.physical_published,
                "robot_states_published": self.robot_states_published,
                "bias_command_received": self.bias_command_received,
                "command_received": self.teleop_command_received,
                "control_status_received": self.control_status_count,
                "command_frames_processed": self.command_frames_processed,
                "control_published": self.control_published,
                "control_skipped_dry_run": self.control_skipped_dry_run,
                "control_blocked_no_subscriber": self.control_blocked_no_subscriber,
                "dropped": self.dropped,
                "idle_ticks": self.idle_ticks,
            },
            "contracts": {
                "lowstate_real_joint_count": len(LOWSTATE_REAL_JOINTS_31),
                "adam_command_joint_count": len(ADAM_COMMAND_JOINTS_19),
                "control_joint_count": len(ADAM_CONTROL_JOINTS_32),
                "control_root_height_value": PND_CONTROL_ROOT_HEIGHT_VALUE,
                "robot_states_joint_count": len(ADAM_ROBOT_STATE_JOINTS_31),
                "control_hand_placeholder_count": len(CONTROL_HAND_JOINTS_12),
                "control_hand_placeholder_value": PND_CONTROL_HAND_PLACEHOLDER_VALUE,
            },
            "last_lowstate": (
                {
                    "tick": self.lowstate.tick,
                    "motor_count": self.lowstate.motor_count,
                }
                if self.lowstate is not None
                else None
            ),
            "control_state": self.control_state,
            "selected_source": self._selected_source_name(now),
            "output_hz": round(self.output_hz, 2),
            "control_output_subscribers": self.control_pub.get_subscription_count(),
            "require_control_subscriber": self.require_control_subscriber,
            "last_error": self.last_error,
        }
        msg = String()
        msg.data = json.dumps(
            status,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self.status_pub.publish(msg)

    def _status_is_fresh(self, now: float) -> bool:
        return (
            self.last_control_status_time is not None
            and now - self.last_control_status_time <= self.control_status_timeout
        )

    def _selected_source_name(self, now: float) -> str:
        if not self._status_is_fresh(now):
            return "none"
        if self.control_state in BIAS_CONTROL_STATES:
            return "bias"
        if self.control_state in COMMAND_CONTROL_STATES:
            return "command"
        return "none"

    @staticmethod
    def _state_from_status(data: str) -> str:
        raw = (data or "").strip()
        if not raw:
            return ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw.lower().replace("-", "_")
        if isinstance(payload, str):
            return payload.strip().lower().replace("-", "_")
        if not isinstance(payload, dict):
            return ""
        for key in ("state", "mode", "teleop_state"):
            value = payload.get(key)
            if value:
                return str(value).strip().lower().replace("-", "_")
        return ""

    @staticmethod
    def _is_fresh(
        timed: TimedCommand | TimedLowState | None,
        timeout: float,
        now: float | None = None,
    ) -> bool:
        if timed is None:
            return False
        if now is None:
            now = time.monotonic()
        return now - timed.stamp <= timeout

    @staticmethod
    def _age_ms(timed: TimedCommand | TimedLowState | None, now: float) -> float | None:
        if timed is None:
            return None
        return AdamNode._age_ms_from_stamp(timed.stamp, now)

    @staticmethod
    def _age_ms_from_stamp(stamp: float | None, now: float) -> float | None:
        if stamp is None:
            return None
        return round((now - stamp) * 1000.0, 1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AdamNode()
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
