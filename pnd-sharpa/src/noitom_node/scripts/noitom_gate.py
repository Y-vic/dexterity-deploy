#!/usr/bin/env python3
"""Single visible Noitom node for retargeted Adam 19-joint commands."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

WAIST_JOINTS = [
    "dof_pos/waistRoll",
    "dof_pos/waistPitch",
    "dof_pos/waistYaw",
]
NECK_JOINTS = [
    "dof_pos/neckYaw",
    "dof_pos/neckPitch",
]
LEFT_ARM_JOINTS = [
    "dof_pos/shoulderPitch_Left",
    "dof_pos/shoulderRoll_Left",
    "dof_pos/shoulderYaw_Left",
    "dof_pos/elbow_Left",
    "dof_pos/wristYaw_Left",
    "dof_pos/wristPitch_Left",
    "dof_pos/wristRoll_Left",
]
RIGHT_ARM_JOINTS = [
    "dof_pos/shoulderPitch_Right",
    "dof_pos/shoulderRoll_Right",
    "dof_pos/shoulderYaw_Right",
    "dof_pos/elbow_Right",
    "dof_pos/wristYaw_Right",
    "dof_pos/wristPitch_Right",
    "dof_pos/wristRoll_Right",
]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
NECK_WAIST_JOINTS = WAIST_JOINTS + NECK_JOINTS
ADAM_COMMAND_JOINTS_19 = NECK_WAIST_JOINTS + ARM_JOINTS


def canonical_body_name(name: str) -> str:
    known = set(ADAM_COMMAND_JOINTS_19)
    if name in known:
        return name
    prefixed = f"dof_pos/{name}"
    if prefixed in known:
        return prefixed
    return name


class NoitomNode(Node):
    def __init__(self) -> None:
        super().__init__("noitom")
        self.declare_parameter("input_topic", "/_noitom/retargeted_joint_states_raw")
        self.declare_parameter("output_topic", "/adam_command_joint_states")
        self.declare_parameter("bias_joint_states_topic", "/adam_bias_command_joint_states")
        self.declare_parameter("bias_state_timeout", 0.5)
        self.declare_parameter("fix_neck_waist", True)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.bias_topic = str(self.get_parameter("bias_joint_states_topic").value)
        self.bias_state_timeout = float(
            self.get_parameter("bias_state_timeout").value
        )
        self.fix_neck_waist = bool(self.get_parameter("fix_neck_waist").value)
        if self.bias_state_timeout <= 0.0:
            raise ValueError("bias_state_timeout must be positive")

        self.bias_positions: dict[str, float] = {}
        self.last_bias_time: float | None = None
        self.received = 0
        self.forwarded = 0
        self.dropped = 0
        self.bias_received = 0
        self.neck_waist_zero_fallbacks = 0
        self.neck_waist_source = "zero_fallback"
        self.last_drop_reason = ""

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.create_subscription(JointState, self.input_topic, self._on_joint_state, 10)
        self.create_subscription(
            JointState,
            self.bias_topic,
            self._on_bias_state,
            10,
        )
        self.create_timer(2.0, self._log_status)

        self.get_logger().info(
            f"Noitom node: input={self.input_topic}, output={self.output_topic}, "
            f"bias={self.bias_topic}, fix_neck_waist={self.fix_neck_waist}"
        )

    def _on_bias_state(self, msg: JointState) -> None:
        try:
            positions = self._positions_from_msg(msg, allowed=set(ADAM_COMMAND_JOINTS_19))
        except Exception as exc:  # noqa: BLE001 - keep last valid bias state.
            self.last_drop_reason = f"bad_bias_state:{exc}"
            return
        self.bias_positions = positions
        self.last_bias_time = time.monotonic()
        self.bias_received += 1

    def _on_joint_state(self, msg: JointState) -> None:
        self.received += 1
        now = time.monotonic()

        try:
            raw_positions = self._positions_from_msg(msg, allowed=set(ADAM_COMMAND_JOINTS_19))
            command_positions = self._make_command_positions(raw_positions, now)
            command_msg = self._make_command_msg(command_positions)
        except Exception as exc:  # noqa: BLE001 - malformed retarget frame.
            self._drop(str(exc))
            return

        self.publisher.publish(command_msg)
        self.forwarded += 1
        self.last_drop_reason = ""

    def _make_command_positions(
        self,
        raw_positions: dict[str, float],
        now: float,
    ) -> dict[str, float]:
        missing_arms = [name for name in ARM_JOINTS if name not in raw_positions]
        if missing_arms:
            raise ValueError(f"missing_noitom_arm_joints:{missing_arms}")

        if self.fix_neck_waist:
            bias_age = (
                None if self.last_bias_time is None else now - self.last_bias_time
            )
            use_bias = (
                bias_age is not None
                and bias_age <= self.bias_state_timeout
                and all(name in self.bias_positions for name in NECK_WAIST_JOINTS)
            )
            if use_bias:
                neck_waist = {
                    name: self.bias_positions[name] for name in NECK_WAIST_JOINTS
                }
                self.neck_waist_source = "bias_command"
            else:
                neck_waist = {name: 0.0 for name in NECK_WAIST_JOINTS}
                self.neck_waist_zero_fallbacks += 1
                self.neck_waist_source = "zero_fallback"
            return {
                **neck_waist,
                **{name: raw_positions[name] for name in ARM_JOINTS},
            }

        missing_command = [
            name for name in ADAM_COMMAND_JOINTS_19 if name not in raw_positions
        ]
        if missing_command:
            raise ValueError(f"missing_noitom_command_joints:{missing_command}")
        return {name: raw_positions[name] for name in ADAM_COMMAND_JOINTS_19}

    def _make_command_msg(self, positions: dict[str, float]) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            "noitom_command:fix_neck_waist"
            if self.fix_neck_waist
            else "noitom_command"
        )
        msg.name = list(ADAM_COMMAND_JOINTS_19)
        msg.position = [float(positions[name]) for name in ADAM_COMMAND_JOINTS_19]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        return msg

    @staticmethod
    def _positions_from_msg(
        msg: JointState,
        *,
        allowed: set[str],
    ) -> dict[str, float]:
        positions: dict[str, float] = {}
        for idx, name in enumerate(msg.name):
            if not name:
                continue
            canonical = canonical_body_name(name)
            if canonical not in allowed:
                continue
            if idx >= len(msg.position):
                raise ValueError(f"JointState position is missing for {canonical}")
            try:
                value = float(msg.position[idx])
            except (TypeError, ValueError):
                raise ValueError(
                    f"non-finite joint value for {canonical}: {msg.position[idx]!r}"
                ) from None
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite joint value for {canonical}: {msg.position[idx]!r}"
                )
            positions[canonical] = value
        return positions

    def _drop(self, reason: str) -> None:
        self.dropped += 1
        self.last_drop_reason = reason

    def _log_status(self) -> None:
        now = time.monotonic()
        bias_age = (
            round((now - self.last_bias_time) * 1000.0, 1)
            if self.last_bias_time is not None
            else None
        )
        self.get_logger().info(
            "Noitom status: "
            f"fix_neck_waist={self.fix_neck_waist}, "
            f"received={self.received}, forwarded={self.forwarded}, "
            f"dropped={self.dropped}, bias_received={self.bias_received}, "
            f"bias_age_ms={bias_age}, neck_waist_source={self.neck_waist_source}, "
            f"neck_waist_zero_fallbacks={self.neck_waist_zero_fallbacks}, "
            f"last_drop_reason={self.last_drop_reason!r}"
        )


def main() -> None:
    rclpy.init()
    node = NoitomNode()
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
