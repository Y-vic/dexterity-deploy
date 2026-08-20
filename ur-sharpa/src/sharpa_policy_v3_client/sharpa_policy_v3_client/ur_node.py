from __future__ import annotations

import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from sharpa_policy_v3_interfaces.msg import (
    ArmCommand,
    HardwareCommandResult,
    UrStateFrame,
)

from .hardware_drivers import UrRtdePair


EXECUTION_CONFIRMATION = "ENABLE_UR_POLICY_EXECUTION"
DEFAULT_LEFT_INITIAL_JOINT = [
    1.8484487533569336,
    -1.8753947019577026,
    2.3183090686798096,
    -0.7468836903572083,
    2.0163028240203857,
    -0.46528369188308716,
]
DEFAULT_RIGHT_INITIAL_JOINT = [
    4.617546558380127,
    -1.1520198583602905,
    -2.2634379863739014,
    -2.5849359035491943,
    -1.868857741355896,
    2.060574531555176,
]


class UrNode(Node):
    def __init__(self) -> None:
        super().__init__("ur_node")
        self.declare_parameter("left_ip", "192.168.56.20")
        self.declare_parameter("right_ip", "192.168.56.10")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("rtde_frequency_hz", 125.0)
        self.declare_parameter("enable_execution", False)
        self.declare_parameter("execution_confirmation", "")
        self.declare_parameter("initialize_on_startup", True)
        self.declare_parameter("initial_left_joint", DEFAULT_LEFT_INITIAL_JOINT)
        self.declare_parameter("initial_right_joint", DEFAULT_RIGHT_INITIAL_JOINT)
        self.declare_parameter("initialization_steps", 120)
        self.declare_parameter("initialization_step_delay_s", 0.02)
        self.declare_parameter("initialization_tolerance_rad", 0.08)
        self.declare_parameter("state_topic", "/ur_position")
        self.declare_parameter("command_topic", "/sharpa/v3/hardware/ur/command")
        self.declare_parameter("result_topic", "/sharpa/v3/hardware/command_result")
        self.declare_parameter("stop_topic", "/sharpa/v3/hardware/stop")

        enabled = bool(self.get_parameter("enable_execution").value)
        confirmation = str(self.get_parameter("execution_confirmation").value)
        if enabled and confirmation != EXECUTION_CONFIRMATION:
            raise ValueError(
                "UR execution requested without the exact execution confirmation"
            )
        self.driver = UrRtdePair(
            left_ip=str(self.get_parameter("left_ip").value),
            right_ip=str(self.get_parameter("right_ip").value),
            frequency_hz=float(self.get_parameter("rtde_frequency_hz").value),
            enable_control=enabled,
        )
        self.driver.connect()
        initialize = bool(self.get_parameter("initialize_on_startup").value)
        if enabled and initialize:
            self.get_logger().info("initializing UR pair to fixed startup joints")
            try:
                self.driver.initialize_joints(
                    self._joint_parameter("initial_left_joint"),
                    self._joint_parameter("initial_right_joint"),
                    steps=int(self.get_parameter("initialization_steps").value),
                    step_delay_s=float(
                        self.get_parameter("initialization_step_delay_s").value
                    ),
                    tolerance_rad=float(
                        self.get_parameter("initialization_tolerance_rad").value
                    ),
                )
            except Exception:
                self.driver.close()
                raise
            self.get_logger().info("UR startup joint initialization complete")
        self.state_pub = self.create_publisher(
            UrStateFrame,
            str(self.get_parameter("state_topic").value),
            10,
        )
        self.result_pub = self.create_publisher(
            HardwareCommandResult,
            str(self.get_parameter("result_topic").value),
            10,
        )
        self.command_sub = self.create_subscription(
            ArmCommand,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
        )
        self.stop_sub = self.create_subscription(
            Bool,
            str(self.get_parameter("stop_topic").value),
            self._on_stop,
            10,
        )
        publish_hz = float(self.get_parameter("publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.timer = self.create_timer(1.0 / publish_hz, self._publish_state)
        self.get_logger().info(
            f"UR pair connected; state={publish_hz:g}Hz execution={enabled}"
        )

    def _joint_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain 6 finite joint angles")
        return value

    def _publish_state(self) -> None:
        try:
            snapshot = self.driver.read()
        except Exception as exc:
            self.get_logger().error(f"UR state read failed: {exc}")
            return
        message = UrStateFrame()
        message.timestamp_ns = snapshot.timestamp_ns
        message.joint_dimension = 6
        message.left_joint = snapshot.left_joint.tolist()
        message.right_joint = snapshot.right_joint.tolist()
        message.eef_dimension = 9
        message.left_eef = snapshot.left_wire_eef.tolist()
        message.right_eef = snapshot.right_wire_eef.tolist()
        message.eef_frame = "robot_base"
        message.normal_mode = snapshot.normal_mode
        message.valid = True
        self.state_pub.publish(message)

    def _on_command(self, message: ArmCommand) -> None:
        result = HardwareCommandResult()
        result.action_id = message.action_id
        result.revision = message.revision
        result.step_index = message.step_index
        result.device = "ur"
        result.timestamp_ns = time.time_ns()
        try:
            if message.eef_dimension != 9:
                raise ValueError("EEF dimension must be 9")
            if len(message.left_eef) != 9 or len(message.right_eef) != 9:
                raise ValueError("left and right EEF commands must contain 9 values")
            self.driver.command_eef_step(
                np.asarray(message.left_eef, dtype=np.float32),
                np.asarray(message.right_eef, dtype=np.float32),
                float(message.period_s),
            )
            result.success = True
        except Exception as exc:
            self.driver.safe_stop()
            result.success = False
            result.failure_code = "ur_command_failed"
            result.failure_message = str(exc)
        self.result_pub.publish(result)

    def _on_stop(self, message: Bool) -> None:
        if message.data:
            self.driver.safe_stop()

    def destroy_node(self) -> bool:
        self.driver.close()
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node: UrNode | None = None
    try:
        node = UrNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
