from __future__ import annotations

from typing import Any

import rclpy
from rclpy.node import Node

from sharpa_policy_v3_interfaces.msg import CameraFrame

from .hardware_drivers import ZedLeftCamera


class ZedNode(Node):
    def __init__(self) -> None:
        super().__init__("zed_node")
        self.declare_parameter("resolution", "HD720")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("serial_number", 0)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("topic", "/sharpa/v3/source/ego_cam")
        publish_hz = float(self.get_parameter("publish_hz").value)
        serial_number = int(self.get_parameter("serial_number").value)
        self.camera = ZedLeftCamera(
            resolution=str(self.get_parameter("resolution").value),
            frequency_hz=int(round(publish_hz)),
            serial_number=serial_number or None,
            jpeg_quality=int(self.get_parameter("jpeg_quality").value),
        )
        self.camera.connect()
        self.camera.start()
        self.publisher = self.create_publisher(
            CameraFrame,
            str(self.get_parameter("topic").value),
            10,
        )
        self._last_timestamp_ns = -1
        self.timer = self.create_timer(1.0 / publish_hz, self._publish_frame)
        self.get_logger().info(f"ZED left-eye publisher started at {publish_hz:g}Hz")

    def _publish_frame(self) -> None:
        snapshot = self.camera.read()
        if snapshot is None or snapshot.timestamp_ns == self._last_timestamp_ns:
            return
        self._last_timestamp_ns = snapshot.timestamp_ns
        message = CameraFrame()
        message.timestamp_ns = snapshot.timestamp_ns
        message.encoding = "jpeg"
        message.data = snapshot.jpeg
        message.valid = True
        self.publisher.publish(message)

    def destroy_node(self) -> bool:
        self.camera.close()
        return super().destroy_node()


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node: ZedNode | None = None
    try:
        node = ZedNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
