from __future__ import annotations

import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from sharpa_policy_v3_interfaces.msg import (
    DeformationFrame,
    HandCommand,
    HandStateFrame,
    HardwareCommandResult,
    TauFrame,
    WrenchFrame,
)

from .hardware_drivers import SharpADevicePair


EXECUTION_CONFIRMATION = "ENABLE_SHARPA_POLICY_EXECUTION"


class SharpaNode(Node):
    def __init__(self) -> None:
        super().__init__("sharpa_node")
        self.declare_parameter(
            "sdk_root",
            "/home/fqx/Sharpa/SDK/SharpaWaveSDK_4_3_12",
        )
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("discovery_timeout_s", 5.0)
        self.declare_parameter("enable_execution", False)
        self.declare_parameter("execution_confirmation", "")
        self.declare_parameter("max_hand_step_rad", 0.5)
        self.declare_parameter("initialize_on_startup", True)
        self.declare_parameter("initial_left_joint", [0.0] * 22)
        self.declare_parameter("initial_right_joint", [0.0] * 22)
        self.declare_parameter("initialization_steps", 60)
        self.declare_parameter("initialization_step_delay_s", 0.02)
        self.declare_parameter("initialization_tolerance_rad", 0.08)
        self.declare_parameter("hand_state_topic", "/sharpa/hand_state")
        self.declare_parameter("tau_topic", "/sharpa/v3/source/tau")
        self.declare_parameter("wrench_topic", "/sharpa/v3/source/wrench")
        self.declare_parameter("deformation_topic", "/sharpa/v3/source/deformation")
        self.declare_parameter("command_topic", "/sharpa/v3/hardware/sharpa/command")
        self.declare_parameter("result_topic", "/sharpa/v3/hardware/command_result")
        self.declare_parameter("stop_topic", "/sharpa/v3/hardware/stop")

        enabled = bool(self.get_parameter("enable_execution").value)
        confirmation = str(self.get_parameter("execution_confirmation").value)
        if enabled and confirmation != EXECUTION_CONFIRMATION:
            raise ValueError(
                "SharpA execution requested without the exact execution confirmation"
            )
        self.driver = SharpADevicePair(
            sdk_root=str(self.get_parameter("sdk_root").value),
            discovery_timeout_s=float(
                self.get_parameter("discovery_timeout_s").value
            ),
            enable_control=enabled,
            max_hand_step_rad=float(
                self.get_parameter("max_hand_step_rad").value
            ),
        )
        self.driver.connect()
        initialize = bool(self.get_parameter("initialize_on_startup").value)
        if enabled and initialize:
            self.get_logger().info("initializing SharpA pair to configured startup joints")
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
            self.get_logger().info("SharpA startup joint initialization complete")
        self.hand_state_pub = self.create_publisher(
            HandStateFrame,
            str(self.get_parameter("hand_state_topic").value),
            10,
        )
        self.tau_pub = self.create_publisher(
            TauFrame,
            str(self.get_parameter("tau_topic").value),
            10,
        )
        self.wrench_pub = self.create_publisher(
            WrenchFrame,
            str(self.get_parameter("wrench_topic").value),
            10,
        )
        self.deformation_pub = self.create_publisher(
            DeformationFrame,
            str(self.get_parameter("deformation_topic").value),
            10,
        )
        self.result_pub = self.create_publisher(
            HardwareCommandResult,
            str(self.get_parameter("result_topic").value),
            10,
        )
        self.command_sub = self.create_subscription(
            HandCommand,
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
            f"SharpA pair connected; state={publish_hz:g}Hz execution={enabled}"
        )

    def _joint_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (22,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain 22 finite joint angles")
        return value

    def _publish_state(self) -> None:
        try:
            snapshot = self.driver.read()
        except Exception as exc:
            self.get_logger().error(f"SharpA state read failed: {exc}")
            return
        hand = HandStateFrame()
        hand.timestamp_ns = snapshot.timestamp_ns
        hand.joint_dimension = 22
        hand.left_joint = snapshot.left_joint.tolist()
        hand.right_joint = snapshot.right_joint.tolist()
        hand.left_valid = snapshot.left_joint_valid
        hand.right_valid = snapshot.right_joint_valid
        self.hand_state_pub.publish(hand)

        tau = TauFrame()
        tau.timestamp_ns = snapshot.timestamp_ns
        tau.joint_dimension = 22
        tau.left = snapshot.left_tau.tolist()
        tau.right = snapshot.right_tau.tolist()
        tau.left_valid = snapshot.left_tau_valid.tolist()
        tau.right_valid = snapshot.right_tau_valid.tolist()
        self.tau_pub.publish(tau)

        wrench = WrenchFrame()
        wrench.timestamp_ns = snapshot.timestamp_ns
        wrench.fingertip_count = 5
        wrench.wrench_dimension = 6
        wrench.left = snapshot.left_wrench.reshape(-1).tolist()
        wrench.right = snapshot.right_wrench.reshape(-1).tolist()
        wrench.left_valid = snapshot.left_wrench_valid.tolist()
        wrench.right_valid = snapshot.right_wrench_valid.tolist()
        self.wrench_pub.publish(wrench)

        deformation = DeformationFrame()
        deformation.timestamp_ns = snapshot.timestamp_ns
        deformation.fingertip_count = 5
        deformation.height = 240
        deformation.width = 240
        deformation.left = snapshot.left_deformation.tobytes()
        deformation.right = snapshot.right_deformation.tobytes()
        deformation.left_valid = snapshot.left_deformation_valid.tolist()
        deformation.right_valid = snapshot.right_deformation_valid.tolist()
        self.deformation_pub.publish(deformation)

    def _on_command(self, message: HandCommand) -> None:
        result = HardwareCommandResult()
        result.action_id = message.action_id
        result.revision = message.revision
        result.step_index = message.step_index
        result.device = "sharpa"
        result.timestamp_ns = time.time_ns()
        try:
            if message.joint_dimension != 22:
                raise ValueError("hand joint dimension must be 22")
            if len(message.left_joint) != 22 or len(message.right_joint) != 22:
                raise ValueError("left and right hand commands must contain 22 values")
            self.driver.command(
                np.asarray(message.left_joint, dtype=np.float64),
                np.asarray(message.right_joint, dtype=np.float64),
            )
            result.success = True
        except Exception as exc:
            self.driver.hold()
            result.success = False
            result.failure_code = "sharpa_command_failed"
            result.failure_message = str(exc)
        self.result_pub.publish(result)

    def _on_stop(self, message: Bool) -> None:
        if message.data:
            self.driver.hold()

    def destroy_node(self) -> bool:
        self.driver.close()
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node: SharpaNode | None = None
    try:
        node = SharpaNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
