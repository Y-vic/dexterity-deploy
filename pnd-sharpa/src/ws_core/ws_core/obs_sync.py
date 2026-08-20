#!/usr/bin/env python3
"""Aggregate workstation robot inputs into PolicyObs messages."""

from __future__ import annotations

import threading
import time
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import ModelImage, PolicyObs, RobotState, RobotTactile, Status

from ws_core.common import (
    age_ms,
    header_stamp_ns,
    json_dumps,
    json_or_raw,
    make_status,
    now_ns,
    set_header,
    uint8_array,
)
from ws_core.kinematics import KinematicsError, PndKinematics


class ObsSync(Node):
    def __init__(self) -> None:
        super().__init__("obs_sync")

        self.declare_parameter("robot_states_topic", "/ws/robot_states")
        self.declare_parameter("robot_tactile_topic", "/ws/robot_tactile")
        self.declare_parameter("model_image_topic", "/ws/robot_vision")
        self.declare_parameter("obs_topic", "/ws/obs")
        self.declare_parameter("debug_topic", "/ws/obs/debug")
        self.declare_parameter("status_topic", "/ws/obs_sync/status")
        self.declare_parameter("provider", "ws_core.obs_sync")
        self.declare_parameter("obs_rate_hz", 30.0)
        self.declare_parameter("model_xml", "")
        self.declare_parameter("require_fk", True)

        self.robot_states_topic = str(self.get_parameter("robot_states_topic").value)
        self.robot_tactile_topic = str(self.get_parameter("robot_tactile_topic").value)
        self.model_image_topic = str(self.get_parameter("model_image_topic").value)
        self.obs_topic = str(self.get_parameter("obs_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.provider = str(self.get_parameter("provider").value)
        self.obs_rate_hz = float(self.get_parameter("obs_rate_hz").value)
        self.model_xml = str(self.get_parameter("model_xml").value)
        self.require_fk = bool(self.get_parameter("require_fk").value)
        if self.obs_rate_hz <= 0.0:
            raise ValueError("obs_rate_hz must be positive")

        self.lock = threading.Lock()
        self.robot_state: RobotState | None = None
        self.robot_state_time: float | None = None
        self.robot_tactile: RobotTactile | None = None
        self.robot_tactile_time: float | None = None
        self.model_image: ModelImage | None = None
        self.model_image_time: float | None = None

        self.obs_seq = 0
        self.robot_states_received = 0
        self.robot_tactile_received = 0
        self.model_images_received = 0
        self.obs_published = 0
        self.debug_published = 0
        self.last_error = ""
        self.kinematics: PndKinematics | None = None
        self.kinematics_error = ""
        self.last_kinematics_attempt = 0.0
        self._ensure_kinematics(force=True)

        self.create_subscription(
            RobotState,
            self.robot_states_topic,
            self._on_robot_state,
            10,
        )
        self.create_subscription(
            RobotTactile,
            self.robot_tactile_topic,
            self._on_robot_tactile,
            10,
        )
        self.create_subscription(
            ModelImage,
            self.model_image_topic,
            self._on_model_image,
            10,
        )
        self.obs_pub = self.create_publisher(PolicyObs, self.obs_topic, 10)
        self.debug_pub = self.create_publisher(Status, self.debug_topic, 10)
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)

        self.create_timer(1.0 / self.obs_rate_hz, self._publish_obs)

        self.get_logger().info(
            "obs_sync: "
            f"state={self.robot_states_topic}, tactile={self.robot_tactile_topic}, "
            f"image={self.model_image_topic}, obs={self.obs_topic}"
        )

    def _on_robot_state(self, msg: RobotState) -> None:
        with self.lock:
            self.robot_state = msg
            self.robot_state_time = time.monotonic()
            self.robot_states_received += 1

    def _on_robot_tactile(self, msg: RobotTactile) -> None:
        with self.lock:
            self.robot_tactile = msg
            self.robot_tactile_time = time.monotonic()
            self.robot_tactile_received += 1

    def _on_model_image(self, msg: ModelImage) -> None:
        with self.lock:
            self.model_image = msg
            self.model_image_time = time.monotonic()
            self.model_images_received += 1

    def _publish_obs(self) -> None:
        stamp_ns = now_ns()
        with self.lock:
            self.obs_seq += 1
            seq = self.obs_seq
            robot_state = self.robot_state
            robot_state_time = self.robot_state_time
            robot_tactile = self.robot_tactile
            robot_tactile_time = self.robot_tactile_time
            model_image = self.model_image
            model_image_time = self.model_image_time
            counts = {
                "robot_states_received": self.robot_states_received,
                "robot_tactile_received": self.robot_tactile_received,
                "model_images_received": self.model_images_received,
                "obs_published": self.obs_published,
                "debug_published": self.debug_published,
            }

        now = time.monotonic()
        policy_input = self._policy_input_payload(robot_state)
        payload = {
            "schema": "ws.policy_obs.v1",
            "seq": seq,
            "stamp_ns": stamp_ns,
            "provider": self.provider,
            "mode": "payload_passthrough_summary",
            "topics": {
                "robot_state": self.robot_states_topic,
                "robot_tactile": self.robot_tactile_topic,
                "model_image": self.model_image_topic,
            },
            "robot_state": self._robot_state_payload(
                robot_state,
                robot_state_time,
                now,
            ),
            "robot_tactile": self._robot_tactile_payload(
                robot_tactile,
                robot_tactile_time,
                now,
            ),
            "model_image": self._model_image_payload(
                model_image,
                model_image_time,
                now,
            ),
            "policy_input": policy_input,
            "implementation": {
                "fk": "mujoco",
                "model_xml": self.model_xml,
                "require_fk": self.require_fk,
            },
        }

        try:
            obs_msg = PolicyObs()
            set_header(obs_msg, "obs_sync", self.get_clock())
            obs_msg.seq = seq
            obs_msg.provider = self.provider
            obs_msg.payload_json = json_dumps(payload)
            obs_msg.image_rgb = uint8_array(
                model_image.data if model_image is not None else None
            )
            obs_msg.tactile_data = uint8_array(
                robot_tactile.data if robot_tactile is not None else None
            )
            self.obs_pub.publish(obs_msg)
            self.last_error = ""
            with self.lock:
                self.obs_published += 1
        except Exception as exc:  # noqa: BLE001 - keep timer alive.
            self.last_error = f"obs publish failed: {exc}"

        ready = (
            robot_state is not None
            and robot_tactile is not None
            and model_image is not None
            and (not self.require_fk or bool(policy_input.get("valid")))
        )
        debug_payload = {
            "schema": "ws.obs_sync.debug.v1",
            "seq": seq,
            "ready": ready,
            "input_ages_ms": {
                "robot_state": age_ms(robot_state_time, now),
                "robot_tactile": age_ms(robot_tactile_time, now),
                "model_image": age_ms(model_image_time, now),
            },
            "kinematics": {
                "ready": self.kinematics is not None,
                "model_xml": self.model_xml,
                "last_error": self.kinematics_error,
            },
            "policy_input": {
                "valid": bool(policy_input.get("valid")),
                "reason": policy_input.get("reason"),
                "hand_pose_source": policy_input.get("hand_pose_source"),
            },
            "counts": counts,
            "last_error": self.last_error,
        }
        self.debug_pub.publish(
            make_status(
                self.get_clock(),
                "obs_sync",
                ready and not self.last_error,
                debug_payload,
            )
        )
        self.status_pub.publish(
            make_status(
                self.get_clock(),
                "obs_sync",
                ready and not self.last_error,
                debug_payload,
            )
        )
        with self.lock:
            self.debug_published += 1

    def _robot_state_payload(
        self,
        msg: RobotState | None,
        recv_time: float | None,
        now: float,
    ) -> dict[str, Any]:
        if msg is None:
            return {"valid": False, "age_ms": None}
        return {
            "valid": True,
            "seq": int(msg.seq),
            "stamp_ns": int(msg.stamp_ns),
            "recv_time_ns": int(msg.recv_time_ns),
            "header_stamp_ns": header_stamp_ns(msg),
            "age_ms": age_ms(recv_time, now),
            "payload": json_or_raw(msg.payload_json),
        }

    def _robot_tactile_payload(
        self,
        msg: RobotTactile | None,
        recv_time: float | None,
        now: float,
    ) -> dict[str, Any]:
        if msg is None:
            return {"valid": False, "age_ms": None, "data_len": 0}
        return {
            "valid": True,
            "seq": int(msg.seq),
            "nearest_obs_seq": int(msg.nearest_obs_seq),
            "stamp_ns": int(msg.stamp_ns),
            "recv_time_ns": int(msg.recv_time_ns),
            "header_stamp_ns": header_stamp_ns(msg),
            "age_ms": age_ms(recv_time, now),
            "metadata": json_or_raw(msg.metadata_json),
            "data_len": len(msg.data),
        }

    def _model_image_payload(
        self,
        msg: ModelImage | None,
        recv_time: float | None,
        now: float,
    ) -> dict[str, Any]:
        if msg is None:
            return {"valid": False, "age_ms": None, "data_len": 0}
        return {
            "valid": True,
            "frame_seq": int(msg.frame_seq),
            "stamp_ns": int(msg.stamp_ns),
            "header_stamp_ns": header_stamp_ns(msg),
            "age_ms": age_ms(recv_time, now),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": msg.encoding,
            "data_len": len(msg.data),
        }

    def _ensure_kinematics(self, *, force: bool = False) -> bool:
        if self.kinematics is not None:
            return True
        now = time.monotonic()
        if not force and now - self.last_kinematics_attempt < 2.0:
            return False
        self.last_kinematics_attempt = now
        try:
            self.kinematics = PndKinematics(self.model_xml)
            self.kinematics_error = ""
            self.get_logger().info(
                f"obs_sync FK loaded MuJoCo model: {self.kinematics.model_xml}"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - keep node alive for status.
            self.kinematics = None
            self.kinematics_error = str(exc)
            self.get_logger().error(f"obs_sync FK unavailable: {exc}")
            return False

    def _policy_input_payload(self, robot_state: RobotState | None) -> dict[str, Any]:
        if robot_state is None:
            return {
                "schema": "ws.policy_input.sharpa62.v1",
                "valid": False,
                "hand_pose_62d": None,
                "hand_pose_source": "missing",
                "fallback": False,
                "reason": "missing_robot_state",
            }
        if not self._ensure_kinematics():
            return {
                "schema": "ws.policy_input.sharpa62.v1",
                "valid": False,
                "hand_pose_62d": None,
                "hand_pose_source": "mujoco_fk_unavailable",
                "fallback": False,
                "reason": self.kinematics_error or "mujoco_fk_unavailable",
            }
        parsed = json_or_raw(robot_state.payload_json)
        state = parsed.get("json") if parsed.get("valid") else None
        if not isinstance(state, dict):
            return {
                "schema": "ws.policy_input.sharpa62.v1",
                "valid": False,
                "hand_pose_62d": None,
                "hand_pose_source": "bad_robot_state_payload",
                "fallback": False,
                "reason": "bad_robot_state_payload",
            }
        try:
            assert self.kinematics is not None
            converted = self.kinematics.convert_state(state)
        except KinematicsError as exc:
            return {
                "schema": "ws.policy_input.sharpa62.v1",
                "valid": False,
                "hand_pose_62d": None,
                "hand_pose_source": "mujoco_fk_robot_state",
                "fallback": False,
                "reason": str(exc),
            }
        return {
            "schema": "ws.policy_input.sharpa62.v1",
            "valid": True,
            "hand_pose_62d": converted.hand_pose_62d.tolist(),
            "hand_pose_source": "mujoco_fk_robot_state",
            "fallback": False,
            "layout": "left_wrist_9d,right_wrist_9d,sharpa_q44",
            "frame": "current_robot_hip",
            "fk": converted.report,
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObsSync()
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
