#!/usr/bin/env python3
"""Merge command JointState streams for Foxglove-only visualization."""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def default_urdf_path() -> str:
    try:
        share = get_package_share_directory("adam_sharpa_description")
    except PackageNotFoundError:
        return os.path.abspath(
            os.path.join(
                os.getcwd(),
                "src",
                "adam_sharpa_description",
                "urdf",
                "adam_pro_sharpa",
                "adam_pro_sharpa.urdf",
            )
        )
    return os.path.join(share, "urdf", "adam_pro_sharpa", "adam_pro_sharpa.urdf")


def movable_joint_names_from_urdf(path: str) -> list[str]:
    root = ET.parse(path).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "") == "fixed":
            continue
        name = joint.attrib.get("name", "").strip()
        if name:
            names.append(name)
    return names


class FoxgloveJointStateMerge(Node):
    def __init__(self) -> None:
        super().__init__("foxglove_joint_state_merge")

        self.declare_parameter(
            "adam_joint_states_topic", "/adam_command_joint_states"
        )
        self.declare_parameter("sharpa_joint_states_topic", "/sharpa_command_joint_states")
        self.declare_parameter("output_topic", "/foxglove/joint_states")
        self.declare_parameter("status_topic", "/foxglove/status")
        self.declare_parameter("urdf_path", default_urdf_path())
        self.declare_parameter("publish_rate", 60.0)

        self.adam_topic = str(self.get_parameter("adam_joint_states_topic").value)
        self.sharpa_topic = str(self.get_parameter("sharpa_joint_states_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")

        self.joint_names = movable_joint_names_from_urdf(self.urdf_path)
        if not self.joint_names:
            raise ValueError(f"URDF has no movable joints: {self.urdf_path}")
        self.positions = {name: 0.0 for name in self.joint_names}
        self.joint_name_set = set(self.joint_names)

        self.adam_received = 0
        self.sharpa_received = 0
        self.published = 0
        self.status_window_publish = 0
        self.status_window_time = time.monotonic()
        self.output_hz = 0.0
        self.last_adam_time: float | None = None
        self.last_sharpa_time: float | None = None
        self.last_unknown_adam: list[str] = []
        self.last_unknown_sharpa: list[str] = []

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(JointState, self.adam_topic, self._on_adam, 10)
        self.create_subscription(JointState, self.sharpa_topic, self._on_sharpa, 10)
        self.create_timer(1.0 / self.publish_rate, self._publish)
        self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            f"Foxglove JointState merge: adam={self.adam_topic}, "
            f"sharpa={self.sharpa_topic}, output={self.output_topic}, "
            f"urdf={self.urdf_path}, joints={len(self.joint_names)}"
        )

    def _on_adam(self, msg: JointState) -> None:
        unknown = self._merge_msg(msg, strip_dof_pos=True)
        self.adam_received += 1
        self.last_adam_time = time.monotonic()
        self.last_unknown_adam = unknown[:20]

    def _on_sharpa(self, msg: JointState) -> None:
        unknown = self._merge_msg(msg, strip_dof_pos=False)
        self.sharpa_received += 1
        self.last_sharpa_time = time.monotonic()
        self.last_unknown_sharpa = unknown[:20]

    def _merge_msg(self, msg: JointState, *, strip_dof_pos: bool) -> list[str]:
        unknown: list[str] = []
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position) or not name:
                continue
            target_name = name.removeprefix("dof_pos/") if strip_dof_pos else name
            if target_name not in self.joint_name_set:
                unknown.append(name)
                continue
            try:
                value = float(msg.position[idx])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                self.positions[target_name] = value
        return unknown

    def _publish(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "foxglove_visualization"
        msg.name = list(self.joint_names)
        msg.position = [self.positions[name] for name in self.joint_names]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        self.publisher.publish(msg)
        self.published += 1
        self.status_window_publish += 1

    def _publish_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self.status_window_time
        if elapsed > 0.0:
            self.output_hz = self.status_window_publish / elapsed
        self.status_window_publish = 0
        self.status_window_time = now
        payload = {
            "node": "foxglove_joint_state_merge",
            "topics": {
                "adam": self.adam_topic,
                "sharpa": self.sharpa_topic,
                "output": self.output_topic,
            },
            "counts": {
                "adam_received": self.adam_received,
                "sharpa_received": self.sharpa_received,
                "published": self.published,
            },
            "age_ms": {
                "adam": self._age_ms(self.last_adam_time, now),
                "sharpa": self._age_ms(self.last_sharpa_time, now),
            },
            "joint_count": len(self.joint_names),
            "output_hz": round(self.output_hz, 2),
            "unknown_adam_joints": self.last_unknown_adam,
            "unknown_sharpa_joints": self.last_unknown_sharpa,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        self.status_pub.publish(msg)

    @staticmethod
    def _age_ms(stamp: float | None, now: float) -> float | None:
        if stamp is None:
            return None
        return round((now - stamp) * 1000.0, 1)


def main() -> None:
    rclpy.init()
    node = FoxgloveJointStateMerge()
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
