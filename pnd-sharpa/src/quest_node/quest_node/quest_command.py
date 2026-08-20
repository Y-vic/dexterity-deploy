"""Safety gate from Quest retarget output to Adam's 19D command topic."""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from quest_node.command_gate import (
    ADAM_COMMAND_JOINTS_19,
    NECK_WAIST_JOINTS,
    TrackingWatchdog,
    make_command_positions,
    parse_tracking_status,
    positions_from_joint_arrays,
)


class QuestCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("quest")
        self.declare_parameter(
            "input_topic",
            "/_quest/retargeted_joint_states_raw",
        )
        self.declare_parameter("tracking_status_topic", "/quest/webvr_status")
        self.declare_parameter("output_topic", "/adam_command_joint_states")
        self.declare_parameter(
            "bias_joint_states_topic",
            "/adam_bias_command_joint_states",
        )
        self.declare_parameter("command_status_topic", "/quest/command_status")
        self.declare_parameter("tracking_timeout", 0.2)
        self.declare_parameter("bias_state_timeout", 0.5)
        self.declare_parameter("fix_neck_waist", True)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.tracking_status_topic = str(
            self.get_parameter("tracking_status_topic").value
        )
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.bias_topic = str(self.get_parameter("bias_joint_states_topic").value)
        self.command_status_topic = str(
            self.get_parameter("command_status_topic").value
        )
        self.bias_state_timeout = float(self.get_parameter("bias_state_timeout").value)
        self.fix_neck_waist = bool(self.get_parameter("fix_neck_waist").value)
        if self.bias_state_timeout <= 0.0:
            raise ValueError("bias_state_timeout must be positive")
        self.tracking_watchdog = TrackingWatchdog(
            float(self.get_parameter("tracking_timeout").value)
        )

        self.bias_positions: dict[str, float] = {}
        self.last_bias_time: float | None = None
        self.raw_received = 0
        self.forwarded = 0
        self.dropped = 0
        self.tracking_status_received = 0
        self.last_drop_reason = "waiting_for_quest_tracking"
        self.neck_waist_source = "zero_fallback"

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.status_publisher = self.create_publisher(
            String,
            self.command_status_topic,
            10,
        )
        self.create_subscription(
            JointState,
            self.input_topic,
            self._on_retargeted_state,
            10,
        )
        self.create_subscription(
            JointState,
            self.bias_topic,
            self._on_bias_state,
            10,
        )
        self.create_subscription(
            String,
            self.tracking_status_topic,
            self._on_tracking_status,
            10,
        )
        self.create_timer(2.0, self._publish_status)
        self.get_logger().info(
            "Quest command gate ready: "
            f"input={self.input_topic}, output={self.output_topic}, "
            f"tracking_timeout={self.tracking_watchdog.timeout:.3f}s, "
            f"fix_neck_waist={self.fix_neck_waist}"
        )

    def _on_tracking_status(self, msg: String) -> None:
        self.tracking_status_received += 1
        try:
            status = parse_tracking_status(msg.data)
        except ValueError as exc:
            self.tracking_watchdog.status_valid = False
            self.last_drop_reason = f"bad_tracking_status:{exc}"
            return
        self.tracking_watchdog.observe(status, time.monotonic())
        if not self.tracking_watchdog.status_valid:
            self.last_drop_reason = "quest_tracking_not_ready"

    def _on_bias_state(self, msg: JointState) -> None:
        try:
            positions = positions_from_joint_arrays(msg.name, msg.position)
        except ValueError as exc:
            self.last_drop_reason = f"bad_bias_state:{exc}"
            return
        self.bias_positions = positions
        self.last_bias_time = time.monotonic()

    def _on_retargeted_state(self, msg: JointState) -> None:
        self.raw_received += 1
        now = time.monotonic()
        if not self.tracking_watchdog.is_fresh(now):
            self._drop("quest_tracking_stale_or_uncalibrated")
            return

        try:
            raw_positions = positions_from_joint_arrays(msg.name, msg.position)
            bias_age = (
                None if self.last_bias_time is None else now - self.last_bias_time
            )
            command_positions, self.neck_waist_source = make_command_positions(
                raw_positions,
                fix_neck_waist=self.fix_neck_waist,
                bias_positions=self.bias_positions,
                bias_fresh=(
                    bias_age is not None and bias_age <= self.bias_state_timeout
                ),
            )
        except ValueError as exc:
            self._drop(str(exc))
            return

        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = (
            "quest_command:fix_neck_waist" if self.fix_neck_waist else "quest_command"
        )
        command.name = list(ADAM_COMMAND_JOINTS_19)
        command.position = [command_positions[name] for name in ADAM_COMMAND_JOINTS_19]
        command.velocity = [0.0] * len(command.name)
        command.effort = [0.0] * len(command.name)
        self.publisher.publish(command)
        self.forwarded += 1
        self.last_drop_reason = ""

    def _drop(self, reason: str) -> None:
        self.dropped += 1
        self.last_drop_reason = reason

    def _publish_status(self) -> None:
        now = time.monotonic()
        tracking_age = self.tracking_watchdog.age(now)
        bias_age = None if self.last_bias_time is None else now - self.last_bias_time
        payload = {
            "tracking_ready": self.tracking_watchdog.is_fresh(now),
            "tracking_age_ms": (
                None if tracking_age is None else tracking_age * 1000.0
            ),
            "bias_age_ms": None if bias_age is None else bias_age * 1000.0,
            "fix_neck_waist": self.fix_neck_waist,
            "neck_waist_source": self.neck_waist_source,
            "raw_received": self.raw_received,
            "forwarded": self.forwarded,
            "dropped": self.dropped,
            "tracking_status_received": self.tracking_status_received,
            "last_drop_reason": self.last_drop_reason,
            "required_neck_waist_joints": list(NECK_WAIST_JOINTS),
        }
        status = String()
        status.data = json.dumps(payload, separators=(",", ":"))
        self.status_publisher.publish(status)
        self.get_logger().info(
            "Quest status: "
            f"tracking_ready={payload['tracking_ready']}, "
            f"tracking_age_ms={payload['tracking_age_ms']}, "
            f"forwarded={self.forwarded}, dropped={self.dropped}, "
            f"neck_waist_source={self.neck_waist_source}, "
            f"last_drop_reason={self.last_drop_reason!r}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = QuestCommandNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
