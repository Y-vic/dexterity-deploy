#!/usr/bin/env python3
"""Receive deploy action frames from the inference device and publish commands."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Any

import rclpy
from adam_node.body_joints import ADAM_COMMAND_JOINTS_19
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sharpa_node.common import SHARPA_JOINT_NAMES
from std_msgs.msg import String

from deploy_common.protocol import (
    FRAME_TYPE_ACTION,
    configure_tcp,
    json_from_bytes,
    now_ns,
    recv_frame,
)


class ActorNode(Node):
    def __init__(self) -> None:
        super().__init__("actor_node")

        self.declare_parameter("server_host", "10.10.20.110")
        self.declare_parameter("action_port", 15010)
        self.declare_parameter("adam_command_topic", "/adam_command_joint_states")
        self.declare_parameter("sharpa_command_topic", "/sharpa_command_joint_states")
        self.declare_parameter("status_topic", "/actor_node/status")
        self.declare_parameter("connect_timeout_s", 1.0)
        self.declare_parameter("reconnect_initial_s", 0.2)
        self.declare_parameter("reconnect_max_s", 5.0)
        self.declare_parameter("default_ttl_ms", 120)
        self.declare_parameter("max_payload_bytes", 1024 * 1024)
        self.declare_parameter("default_sharpa_control_mode", "position")

        self.server_host = str(self.get_parameter("server_host").value)
        self.action_port = int(self.get_parameter("action_port").value)
        self.adam_topic = str(self.get_parameter("adam_command_topic").value)
        self.sharpa_topic = str(self.get_parameter("sharpa_command_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.connect_timeout_s = float(self.get_parameter("connect_timeout_s").value)
        self.reconnect_initial_s = float(
            self.get_parameter("reconnect_initial_s").value
        )
        self.reconnect_max_s = float(self.get_parameter("reconnect_max_s").value)
        self.default_ttl_ms = int(self.get_parameter("default_ttl_ms").value)
        self.max_payload_bytes = int(self.get_parameter("max_payload_bytes").value)
        self.default_sharpa_control_mode = self._normalize_sharpa_mode(
            self.get_parameter("default_sharpa_control_mode").value
        )

        if self.action_port <= 0 or self.action_port > 65535:
            raise ValueError("action_port must be in [1, 65535]")
        if self.connect_timeout_s <= 0.0:
            raise ValueError("connect_timeout_s must be positive")
        if self.reconnect_initial_s <= 0.0 or self.reconnect_max_s <= 0.0:
            raise ValueError("reconnect delays must be positive")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")

        self.adam_pub = self.create_publisher(JointState, self.adam_topic, 10)
        self.sharpa_pub = self.create_publisher(JointState, self.sharpa_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.sock: socket.socket | None = None
        self.connected = False
        self.connect_attempts = 0
        self.frames_received = 0
        self.actions_applied = 0
        self.adam_published = 0
        self.sharpa_published = 0
        self.ignored_frames = 0
        self.dropped_actions = 0
        self.last_frame_seq: int | None = None
        self.last_action_seq: int | None = None
        self.last_action_stamp_ns: int | None = None
        self.clock_offset_baseline_ns: int | None = None
        self.last_raw_action_age_ns: int | None = None
        self.last_transport_age_ns: int | None = None
        self.last_connect_time: float | None = None
        self.last_action_time: float | None = None
        self.last_error = ""

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            "Actor node: "
            f"action=tcp://{self.server_host}:{self.action_port}, "
            f"adam={self.adam_topic}, sharpa={self.sharpa_topic}"
        )

    def _run(self) -> None:
        delay = self.reconnect_initial_s
        while not self.stop_event.is_set():
            sock: socket.socket | None = None
            try:
                with self.lock:
                    self.connect_attempts += 1
                sock = socket.create_connection(
                    (self.server_host, self.action_port),
                    timeout=self.connect_timeout_s,
                )
                configure_tcp(sock, timeout_s=None)
                with self.lock:
                    self.sock = sock
                    self.connected = True
                    self.last_connect_time = time.monotonic()
                    self.clock_offset_baseline_ns = None
                    self.last_raw_action_age_ns = None
                    self.last_transport_age_ns = None
                    self.last_error = ""
                delay = self.reconnect_initial_s
                while not self.stop_event.is_set():
                    frame = recv_frame(sock, self.max_payload_bytes)
                    with self.lock:
                        self.frames_received += 1
                        self.last_frame_seq = frame.seq
                    if frame.frame_type != FRAME_TYPE_ACTION:
                        with self.lock:
                            self.ignored_frames += 1
                        continue
                    self._handle_action(frame.seq, frame.stamp_ns, frame.payload)
            except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
                if not self.stop_event.is_set():
                    with self.lock:
                        self.last_error = str(exc)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                with self.lock:
                    if self.sock is sock:
                        self.sock = None
                    self.connected = False
            if not self.stop_event.wait(delay):
                delay = min(self.reconnect_max_s, max(delay * 1.7, delay))

    def _handle_action(self, seq: int, frame_stamp_ns: int, payload: bytes) -> None:
        try:
            data = json_from_bytes(payload)
            ttl_ms = int(data.get("ttl_ms", self.default_ttl_ms))
            stamp_ns = int(data.get("stamp_ns", frame_stamp_ns))
            current_ns = now_ns()
            if ttl_ms > 0 and stamp_ns > 0:
                raw_age_ns = current_ns - stamp_ns
                with self.lock:
                    if (
                        self.clock_offset_baseline_ns is None
                        or raw_age_ns < self.clock_offset_baseline_ns
                    ):
                        self.clock_offset_baseline_ns = raw_age_ns
                    baseline_ns = self.clock_offset_baseline_ns
                    transport_age_ns = max(0, raw_age_ns - baseline_ns)
                    self.last_raw_action_age_ns = raw_age_ns
                    self.last_transport_age_ns = transport_age_ns
                if transport_age_ns > ttl_ms * 1_000_000:
                    raise ValueError(
                        "expired action: "
                        f"transport_age_ms={transport_age_ns / 1_000_000:.1f}"
                    )
            adam_sent = self._publish_adam_if_valid(data)
            sharpa_sent = self._publish_sharpa_if_valid(data)
        except Exception as exc:  # noqa: BLE001 - keep TCP reader alive.
            with self.lock:
                self.dropped_actions += 1
                self.last_error = str(exc)
            return

        with self.lock:
            self.actions_applied += 1
            self.last_action_seq = seq
            self.last_action_stamp_ns = stamp_ns
            self.last_action_time = time.monotonic()
            if adam_sent:
                self.adam_published += 1
            if sharpa_sent:
                self.sharpa_published += 1
            self.last_error = ""

    def _publish_adam_if_valid(self, data: dict[str, Any]) -> bool:
        adam = data.get("adam") if isinstance(data.get("adam"), dict) else {}
        valid = bool(data.get("adam_valid", adam.get("valid", False)))
        if not valid:
            return False
        q = self._vector(
            adam.get("q", data.get("adam_q")),
            len(ADAM_COMMAND_JOINTS_19),
            "adam.q",
        )
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "actor_node"
        msg.name = list(ADAM_COMMAND_JOINTS_19)
        msg.position = q
        self.adam_pub.publish(msg)
        return True

    def _publish_sharpa_if_valid(self, data: dict[str, Any]) -> bool:
        sharpa = data.get("sharpa") if isinstance(data.get("sharpa"), dict) else {}
        valid = bool(data.get("sharpa_valid", sharpa.get("valid", False)))
        if not valid:
            return False
        mode = self._normalize_sharpa_mode(
            data.get(
                "sharpa_control_mode",
                sharpa.get("control_mode", self.default_sharpa_control_mode),
            )
        )
        q = self._vector(
            sharpa.get("q", data.get("sharpa_q")),
            len(SHARPA_JOINT_NAMES),
            "sharpa.q",
        )
        dq = self._vector(
            sharpa.get("dq", data.get("sharpa_dq")),
            len(SHARPA_JOINT_NAMES),
            "sharpa.dq",
            default=0.0,
        )
        tau = self._vector(
            sharpa.get("tau", data.get("sharpa_tau")),
            len(SHARPA_JOINT_NAMES),
            "sharpa.tau",
            default=0.0,
        )
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"actor_node:{mode}"
        msg.name = list(SHARPA_JOINT_NAMES)
        msg.position = q
        msg.velocity = dq
        msg.effort = tau
        self.sharpa_pub.publish(msg)
        return True

    @staticmethod
    def _vector(
        value: Any,
        length: int,
        label: str,
        *,
        default: float | None = None,
    ) -> list[float]:
        if value is None:
            if default is None:
                raise ValueError(f"{label} is required")
            return [float(default)] * length
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{label} must be a list")
        if len(value) != length:
            raise ValueError(f"{label} length must be {length}, got {len(value)}")
        output: list[float] = []
        for idx, raw in enumerate(value):
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"{label}[{idx}] is non-finite")
            output.append(number)
        return output

    @staticmethod
    def _normalize_sharpa_mode(value: Any) -> str:
        if isinstance(value, int):
            return "mit" if value == 2 else "position"
        text = str(value or "").strip().lower().replace("-", "_")
        if text in {"2", "mit", "torque", "position_velocity_torque"}:
            return "mit"
        return "position"

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            payload = {
                "node": "actor_node",
                "action_endpoint": {
                    "role": "tcp_client",
                    "host": self.server_host,
                    "port": self.action_port,
                    "connected": self.connected,
                    "connect_attempts": self.connect_attempts,
                    "last_connect_age_ms": self._age_ms(
                        self.last_connect_time, now
                    ),
                },
                "topics": {
                    "adam_command": self.adam_topic,
                    "sharpa_command": self.sharpa_topic,
                },
                "counts": {
                    "frames_received": self.frames_received,
                    "ignored_frames": self.ignored_frames,
                    "actions_applied": self.actions_applied,
                    "dropped_actions": self.dropped_actions,
                    "adam_published": self.adam_published,
                    "sharpa_published": self.sharpa_published,
                },
                "last": {
                    "frame_seq": self.last_frame_seq,
                    "action_seq": self.last_action_seq,
                    "action_stamp_ns": self.last_action_stamp_ns,
                    "action_age_ms": self._age_ms(self.last_action_time, now),
                    "raw_clock_age_ms": self._ns_to_ms(self.last_raw_action_age_ns),
                    "clock_offset_baseline_ms": self._ns_to_ms(
                        self.clock_offset_baseline_ns
                    ),
                    "transport_age_ms": self._ns_to_ms(self.last_transport_age_ns),
                },
                "schema": {
                    "action": "pnd.deploy.action.v1",
                    "adam_q": len(ADAM_COMMAND_JOINTS_19),
                    "sharpa_q_dq_tau": len(SHARPA_JOINT_NAMES),
                },
                "last_error": self.last_error,
            }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        self.status_pub.publish(msg)

    @staticmethod
    def _age_ms(stamp: float | None, now: float) -> float | None:
        if stamp is None:
            return None
        return round((now - stamp) * 1000.0, 1)

    @staticmethod
    def _ns_to_ms(value: int | None) -> float | None:
        if value is None:
            return None
        return round(value / 1_000_000.0, 1)

    def destroy_node(self) -> bool:
        self.stop_event.set()
        with self.lock:
            sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self.worker.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActorNode()
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
