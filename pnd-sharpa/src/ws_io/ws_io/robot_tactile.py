#!/usr/bin/env python3
"""Receive TACTILE_BULK PND1 frames and publish workstation tactile messages."""

from __future__ import annotations

import time
from array import array
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import RobotTactile, Status

from deploy_common.protocol import FRAME_TYPE_TACTILE_BULK, unpack_tactile_bulk_payload

from ws_io.tcp_server import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    TcpFrameServer,
    compact_json,
    make_status_msg,
)


class RobotTactileNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_tactile")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 15021)
        self.declare_parameter("tactile_topic", "/ws/robot_tactile")
        self.declare_parameter("status_topic", "/ws/robot_tactile/status")
        self.declare_parameter("status_period", 0.5)
        self.declare_parameter("max_payload_bytes", DEFAULT_MAX_PAYLOAD_BYTES)

        self.host = str(self.get_parameter("host").value)
        self.port = self._valid_port("port", self.get_parameter("port").value)
        self.tactile_topic = str(self.get_parameter("tactile_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_period = float(self.get_parameter("status_period").value)
        self.max_payload_bytes = int(self.get_parameter("max_payload_bytes").value)
        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")

        self.tactile_pub = self.create_publisher(RobotTactile, self.tactile_topic, 10)
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.server = TcpFrameServer(
            name="robot_tactile",
            host=self.host,
            port=self.port,
            expected_frame_type=FRAME_TYPE_TACTILE_BULK,
            handler=self._handle_frame,
            logger=self.get_logger(),
            max_payload_bytes=self.max_payload_bytes,
        )
        self.server.start()
        self.status_timer = self.create_timer(self.status_period, self._publish_status)

    def _handle_frame(self, frame: Any, recv_time_ns: int) -> None:
        metadata, raw = unpack_tactile_bulk_payload(frame.payload)
        msg = RobotTactile()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.seq = int(frame.seq)
        msg.nearest_obs_seq = self._uint64(metadata.get("nearest_obs_seq", 0))
        msg.stamp_ns = int(frame.stamp_ns)
        msg.recv_time_ns = int(recv_time_ns)
        msg.metadata_json = compact_json(metadata)
        msg.data = array("B", raw)
        self.tactile_pub.publish(msg)

    def _publish_status(self) -> None:
        payload = {
            "node": "robot_tactile",
            "topic": self.tactile_topic,
            "endpoint": self.server.snapshot(),
            "time_ns": time.time_ns(),
        }
        msg = make_status_msg(
            "robot_tactile",
            self.server.listening,
            payload,
            self.get_clock().now().to_msg(),
        )
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.server.close()
        return super().destroy_node()

    @staticmethod
    def _valid_port(name: str, value: Any) -> int:
        port = int(value)
        if port <= 0 or port > 65535:
            raise ValueError(f"{name} must be in [1, 65535]")
        return port

    @staticmethod
    def _uint64(value: Any) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, min(number, 18446744073709551615))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotTactileNode()
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
