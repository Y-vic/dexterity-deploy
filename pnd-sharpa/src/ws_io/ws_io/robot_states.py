#!/usr/bin/env python3
"""Receive OBS_STATE PND1 frames and publish workstation robot state messages."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import RobotState, Status

from deploy_common.protocol import FRAME_TYPE_OBS_STATE

from ws_io.tcp_server import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    TcpFrameServer,
    make_status_msg,
)


BIAS_JOINTS = (
    "dof_pos/waistRoll",
    "dof_pos/waistPitch",
    "dof_pos/waistYaw",
    "dof_pos/neckYaw",
    "dof_pos/neckPitch",
)


class RobotStatesNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_states")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 15020)
        self.declare_parameter("state_topic", "/ws/robot_states")
        self.declare_parameter("raw_state_topic", "/ws/robot_states/raw")
        self.declare_parameter("status_topic", "/ws/robot_states/status")
        self.declare_parameter("status_period", 0.5)
        self.declare_parameter("max_payload_bytes", DEFAULT_MAX_PAYLOAD_BYTES)
        self.declare_parameter("pnd_bias_ssh_host", "pnd")
        self.declare_parameter(
            "pnd_bias_remote_path",
            "/home/pnd-humanoid/.adam/joint/bias_joints_set_with_init.json",
        )
        self.declare_parameter(
            "pnd_bias_local_path",
            "/home/ps/Deploy-v2/pnd-sharpa/deploy/runtime/pnd_bias/"
            "bias_joints_set_with_init.json",
        )

        self.host = str(self.get_parameter("host").value)
        self.port = self._valid_port("port", self.get_parameter("port").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.raw_state_topic = str(self.get_parameter("raw_state_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_period = float(self.get_parameter("status_period").value)
        self.max_payload_bytes = int(self.get_parameter("max_payload_bytes").value)
        self.pnd_bias_ssh_host = str(
            self.get_parameter("pnd_bias_ssh_host").value
        )
        self.pnd_bias_remote_path = str(
            self.get_parameter("pnd_bias_remote_path").value
        )
        self.pnd_bias_local_path = Path(
            str(self.get_parameter("pnd_bias_local_path").value)
        ).expanduser().resolve()
        if self.status_period <= 0.0:
            raise ValueError("status_period must be positive")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if not self.raw_state_topic or self.raw_state_topic == self.state_topic:
            raise ValueError("raw_state_topic must differ from state_topic")

        self.bias_positions = self._sync_pnd_bias()
        self.state_pub = self.create_publisher(RobotState, self.state_topic, 10)
        self.raw_state_pub = self.create_publisher(
            RobotState, self.raw_state_topic, 10
        )
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.server = TcpFrameServer(
            name="robot_states",
            host=self.host,
            port=self.port,
            expected_frame_type=FRAME_TYPE_OBS_STATE,
            handler=self._handle_frame,
            logger=self.get_logger(),
            max_payload_bytes=self.max_payload_bytes,
        )
        self.server.start()
        self.status_timer = self.create_timer(self.status_period, self._publish_status)

    def _handle_frame(self, frame: Any, recv_time_ns: int) -> None:
        raw_payload_json = frame.payload.decode("utf-8")
        normalized_payload_json = self._normalized_payload_json(raw_payload_json)
        stamp = self.get_clock().now().to_msg()
        self.raw_state_pub.publish(
            self._robot_state_msg(frame, recv_time_ns, stamp, raw_payload_json)
        )
        self.state_pub.publish(
            self._robot_state_msg(
                frame,
                recv_time_ns,
                stamp,
                normalized_payload_json,
            )
        )

    def _publish_status(self) -> None:
        payload = {
            "node": "robot_states",
            "topic": self.state_topic,
            "raw_topic": self.raw_state_topic,
            "bias": {
                "source": (
                    f"{self.pnd_bias_ssh_host}:{self.pnd_bias_remote_path}"
                ),
                "local_path": str(self.pnd_bias_local_path),
                "replaced_joints": list(BIAS_JOINTS),
                "positions": self.bias_positions,
            },
            "endpoint": self.server.snapshot(),
            "time_ns": time.time_ns(),
        }
        msg = make_status_msg(
            "robot_states",
            self.server.listening,
            payload,
            self.get_clock().now().to_msg(),
        )
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.server.close()
        return super().destroy_node()

    @staticmethod
    def _robot_state_msg(
        frame: Any,
        recv_time_ns: int,
        stamp: Any,
        payload_json: str,
    ) -> RobotState:
        msg = RobotState()
        msg.header.stamp = stamp
        msg.seq = int(frame.seq)
        msg.stamp_ns = int(frame.stamp_ns)
        msg.recv_time_ns = int(recv_time_ns)
        msg.payload_json = payload_json
        return msg

    def _normalized_payload_json(self, payload_json: str) -> str:
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("OBS_STATE payload must be a JSON object")
        adam = payload.get("adam")
        if not isinstance(adam, dict):
            raise ValueError("OBS_STATE payload is missing adam section")
        names = adam.get("name")
        positions = adam.get("q")
        if not isinstance(names, list) or not isinstance(positions, list):
            raise ValueError("OBS_STATE adam section is missing name/q")
        if len(names) != len(positions):
            raise ValueError("OBS_STATE adam name/q lengths differ")
        indices = {str(name): index for index, name in enumerate(names)}
        missing = [name for name in BIAS_JOINTS if name not in indices]
        if missing:
            raise ValueError(f"OBS_STATE adam section is missing bias joints: {missing}")
        normalized = list(positions)
        for name, value in self.bias_positions.items():
            normalized[indices[name]] = value
        adam["q"] = normalized
        payload["state_override"] = {
            "source": "pnd_bias",
            "joints": list(BIAS_JOINTS),
        }
        return json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def _sync_pnd_bias(self) -> dict[str, float]:
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            self.pnd_bias_ssh_host,
            f"cat -- {shlex.quote(self.pnd_bias_remote_path)}",
        ]
        environment = os.environ.copy()
        for variable in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "OPENSSL_CONF",
            "OPENSSL_MODULES",
        ):
            environment.pop(variable, None)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=10.0,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"PND bias sync failed: {error}")
        try:
            payload = json.loads(result.stdout)
            positions = self._bias_positions(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"PND bias validation failed: {exc}") from exc
        self.pnd_bias_local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.pnd_bias_local_path.with_suffix(
            self.pnd_bias_local_path.suffix + ".tmp"
        )
        try:
            temporary_path.write_bytes(result.stdout)
            os.replace(temporary_path, self.pnd_bias_local_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self.get_logger().info(
            "PND bias synced: "
            f"{self.pnd_bias_ssh_host}:{self.pnd_bias_remote_path} -> "
            f"{self.pnd_bias_local_path}; replacing {list(BIAS_JOINTS)}"
        )
        return positions

    @staticmethod
    def _bias_positions(payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            raise ValueError("PND bias file must be a JSON object")
        raw = payload.get("joints", payload.get("bias"))
        if not isinstance(raw, dict):
            names = payload.get("names")
            positions = payload.get("positions")
            if isinstance(names, list) and isinstance(positions, list):
                raw = dict(zip(names, positions, strict=False))
        if not isinstance(raw, dict):
            raise ValueError("PND bias file has no joints mapping")
        result: dict[str, float] = {}
        for name in BIAS_JOINTS:
            if name not in raw:
                raise ValueError(f"PND bias file is missing {name}")
            value = float(raw[name])
            if not math.isfinite(value):
                raise ValueError(f"PND bias contains non-finite {name}")
            result[name] = value
        return result

    @staticmethod
    def _valid_port(name: str, value: Any) -> int:
        port = int(value)
        if port <= 0 or port > 65535:
            raise ValueError(f"{name} must be in [1, 65535]")
        return port


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotStatesNode()
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
