#!/usr/bin/env python3
"""Publish teleop head targets from Quest/WebVR pose input."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, Joy
from std_msgs.msg import String

from quest_node.joy_indices import QuestJoyButton

try:
    from rclpy._rclpy_pybind11 import RCLError
except ImportError:  # pragma: no cover - depends on rclpy build layout
    RCLError = RuntimeError

try:
    import tf2_ros
except Exception as exc:  # pragma: no cover - exercised on minimal installs
    tf2_ros = None
    TF2_IMPORT_ERROR: Exception | None = exc
else:
    TF2_IMPORT_ERROR = None

try:
    from geometry_msgs.msg import PoseStamped, QuaternionStamped
except Exception as exc:  # pragma: no cover - exercised on minimal installs
    PoseStamped = None
    QuaternionStamped = None
    GEOMETRY_IMPORT_ERROR: Exception | None = exc
else:
    GEOMETRY_IMPORT_ERROR = None


@dataclass
class QuaternionSample:
    source: str
    stamp: float
    w: float
    x: float
    y: float
    z: float


class QuestNode(Node):
    """Convert Quest/WebVR head pose into neck yaw/pitch JointState targets."""

    VALID_SOURCES = {"auto", "tf", "pose", "orientation", "imu"}

    def __init__(self) -> None:
        super().__init__("quest")

        self.declare_parameter("enabled", True)
        self.declare_parameter("input_source", "auto")
        self.declare_parameter("input_priority", ["tf", "pose", "orientation", "imu"])
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("head_frame", "Head")
        self.declare_parameter("fallback_head_frame", "Head_uncalibrated")
        self.declare_parameter("pose_topic", "/quest/head_pose")
        self.declare_parameter("orientation_topic", "/quest/head_orientation")
        self.declare_parameter("imu_topic", "/quest/head_imu")
        self.declare_parameter("joy_topic", "vr/joy")
        self.declare_parameter("output_topic", "/quest/head_joint_states")
        self.declare_parameter("status_topic", "/quest/head_status")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("status_rate", 2.0)
        self.declare_parameter("input_timeout", 0.5)
        self.declare_parameter("warn_period", 2.0)
        self.declare_parameter("auto_calibrate", False)
        self.declare_parameter("calibrate_button", int(QuestJoyButton.R_A))
        self.declare_parameter("decalibrate_button", int(QuestJoyButton.R_B))
        self.declare_parameter("yaw_sign", 1.0)
        self.declare_parameter("pitch_sign", 1.0)
        self.declare_parameter("yaw_limit", 1.571)
        self.declare_parameter("pitch_limit", 0.873)
        self.declare_parameter("yaw_joint_name", "dof_pos/neckYaw")
        self.declare_parameter("pitch_joint_name", "dof_pos/neckPitch")
        self.declare_parameter("frame_id", "quest_head")
        self.declare_parameter("publish_zero_without_input", False)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.input_source = str(self.get_parameter("input_source").value).strip()
        if self.input_source not in self.VALID_SOURCES:
            self.get_logger().warning(
                f"Invalid input_source={self.input_source!r}; falling back to 'auto'"
            )
            self.input_source = "auto"
        self.input_priority = self._configured_input_priority()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.head_frame = str(self.get_parameter("head_frame").value)
        self.fallback_head_frame = str(
            self.get_parameter("fallback_head_frame").value
        )
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.orientation_topic = str(self.get_parameter("orientation_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.status_rate = float(self.get_parameter("status_rate").value)
        self.input_timeout = float(self.get_parameter("input_timeout").value)
        self.warn_period = float(self.get_parameter("warn_period").value)
        self.auto_calibrate = bool(self.get_parameter("auto_calibrate").value)
        self.calibrate_button = int(self.get_parameter("calibrate_button").value)
        self.decalibrate_button = int(self.get_parameter("decalibrate_button").value)
        self.yaw_sign = float(self.get_parameter("yaw_sign").value)
        self.pitch_sign = float(self.get_parameter("pitch_sign").value)
        self.yaw_limit = float(self.get_parameter("yaw_limit").value)
        self.pitch_limit = float(self.get_parameter("pitch_limit").value)
        self.yaw_joint_name = str(self.get_parameter("yaw_joint_name").value)
        self.pitch_joint_name = str(self.get_parameter("pitch_joint_name").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_zero_without_input = bool(
            self.get_parameter("publish_zero_without_input").value
        )

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        if self.status_rate < 0.0:
            raise ValueError("status_rate must be non-negative")
        if self.input_timeout <= 0.0:
            raise ValueError("input_timeout must be positive")
        if self.warn_period <= 0.0:
            raise ValueError("warn_period must be positive")

        self.calibrated = False
        self.calibration_pending = False
        self.zero_yaw = 0.0
        self.zero_pitch = 0.0
        self.latest_yaw = 0.0
        self.latest_pitch = 0.0
        self.last_output_yaw = 0.0
        self.last_output_pitch = 0.0
        self.last_warning_time = 0.0
        self.last_error = ""
        self.last_input_source = ""
        self.last_tf_error = ""
        self.received = {"pose": 0, "orientation": 0, "imu": 0, "joy": 0}
        self.published = 0
        self.samples: dict[str, QuaternionSample] = {}

        self.tf_buffer: Any = None
        self.tf_listener: Any = None
        self.joint_state_publishers = self._create_joint_state_publishers()
        self.status_pub = (
            self.create_publisher(String, self.status_topic, 10)
            if self.status_topic
            else None
        )

        self._setup_inputs()
        self.create_timer(1.0 / self.publish_rate, self._publish)
        if self.status_rate > 0.0:
            self.create_timer(1.0 / self.status_rate, self._publish_status)

        self.get_logger().info(
            "quest: "
            f"enabled={self.enabled}, input_source={self.input_source}, "
            f"priority={self.input_priority}, "
            f"output_topics={list(self.joint_state_publishers)}, "
            f"status_topic={self.status_topic or '<disabled>'}"
        )

    def _configured_input_priority(self) -> list[str]:
        if self.input_source != "auto":
            return [self.input_source]

        configured = [
            str(source).strip()
            for source in self.get_parameter("input_priority").value
            if str(source).strip()
        ]
        priority = [
            source
            for source in configured
            if source in self.VALID_SOURCES and source != "auto"
        ]
        return priority or ["tf", "pose", "orientation", "imu"]

    def _create_joint_state_publishers(self) -> dict[str, Any]:
        topics: list[str] = []
        if self.output_topic:
            topics.append(self.output_topic)

        publishers: dict[str, Any] = {}
        for topic in topics:
            publishers[topic] = self.create_publisher(JointState, topic, 10)

        if not publishers:
            self.get_logger().warning(
                "No JointState output topic configured; node will only publish status"
            )
        return publishers

    def _setup_inputs(self) -> None:
        if self._source_configured("tf"):
            if tf2_ros is None:
                self.get_logger().error(
                    "input_source includes tf but tf2_ros is unavailable: "
                    f"{TF2_IMPORT_ERROR}; waiting for non-TF inputs"
                )
            else:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        if self._source_configured("pose"):
            if PoseStamped is None:
                self.get_logger().error(
                    "input_source includes pose but geometry_msgs PoseStamped is "
                    f"unavailable: {GEOMETRY_IMPORT_ERROR}"
                )
            elif self.pose_topic:
                self.create_subscription(PoseStamped, self.pose_topic, self._on_pose, 10)

        if self._source_configured("orientation"):
            if QuaternionStamped is None:
                self.get_logger().error(
                    "input_source includes orientation but geometry_msgs "
                    f"QuaternionStamped is unavailable: {GEOMETRY_IMPORT_ERROR}"
                )
            elif self.orientation_topic:
                self.create_subscription(
                    QuaternionStamped,
                    self.orientation_topic,
                    self._on_orientation,
                    10,
                )

        if self._source_configured("imu") and self.imu_topic:
            self.create_subscription(Imu, self.imu_topic, self._on_imu, 10)

        if self.joy_topic:
            self.create_subscription(Joy, self.joy_topic, self._on_joy, 10)

    def _source_configured(self, source: str) -> bool:
        return source in self.input_priority

    def _on_pose(self, msg: Any) -> None:
        self.received["pose"] += 1
        q = msg.pose.orientation
        self._store_sample("pose", q.w, q.x, q.y, q.z)

    def _on_orientation(self, msg: Any) -> None:
        self.received["orientation"] += 1
        q = msg.quaternion
        self._store_sample("orientation", q.w, q.x, q.y, q.z)

    def _on_imu(self, msg: Imu) -> None:
        self.received["imu"] += 1
        q = msg.orientation
        self._store_sample("imu", q.w, q.x, q.y, q.z)

    def _on_joy(self, msg: Joy) -> None:
        self.received["joy"] += 1
        if self._button_pressed(msg, self.calibrate_button):
            self.calibration_pending = True

        if self._button_pressed(msg, self.decalibrate_button) and self.calibrated:
            self.calibrated = False
            self.calibration_pending = False
            self.get_logger().info("Quest head calibration deactivated")

    def _button_pressed(self, msg: Joy, index: int) -> bool:
        return index >= 0 and len(msg.buttons) > index and msg.buttons[index] == 1

    def _store_sample(
        self, source: str, w: float, x: float, y: float, z: float
    ) -> None:
        self.samples[source] = QuaternionSample(
            source=source,
            stamp=time.monotonic(),
            w=float(w),
            x=float(x),
            y=float(y),
            z=float(z),
        )

    def _publish(self) -> None:
        if not rclpy.ok():
            return
        if not self.enabled:
            self.last_error = "disabled"
            return

        sample = self._select_input()
        if sample is None:
            self.last_error = "waiting for Quest/WebVR head pose input"
            self._log_waiting_for_input()
            if self.publish_zero_without_input:
                self._publish_joint_state(0.0, 0.0)
            return

        angles = self._sample_to_yaw_pitch(sample)
        if angles is None:
            self.last_error = f"invalid quaternion from {sample.source}"
            self._log_waiting_for_input()
            return

        self.last_input_source = sample.source
        self.latest_yaw, self.latest_pitch = angles

        if self.calibration_pending or (self.auto_calibrate and not self.calibrated):
            self.zero_yaw = self.latest_yaw
            self.zero_pitch = self.latest_pitch
            self.calibrated = True
            self.calibration_pending = False
            self.get_logger().info(
                f"Quest head calibration activated from {sample.source} input"
            )

        yaw = 0.0
        pitch = 0.0
        if self.calibrated:
            yaw = self._wrap_pi(self.latest_yaw - self.zero_yaw) * self.yaw_sign
            pitch = self._wrap_pi(self.latest_pitch - self.zero_pitch) * self.pitch_sign
            yaw = self._clamp(yaw, -self.yaw_limit, self.yaw_limit)
            pitch = self._clamp(pitch, -self.pitch_limit, self.pitch_limit)

        self.last_error = ""
        self._publish_joint_state(yaw, pitch)

    def _select_input(self) -> QuaternionSample | None:
        now = time.monotonic()
        for source in self.input_priority:
            if source == "tf":
                sample = self._lookup_tf_sample()
            else:
                sample = self.samples.get(source)
                if sample is not None and now - sample.stamp > self.input_timeout:
                    sample = None
            if sample is not None:
                return sample
        return None

    def _lookup_tf_sample(self) -> QuaternionSample | None:
        if self.tf_buffer is None:
            return None

        transform = None
        for frame_name in self._tf_frame_candidates():
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, frame_name, rclpy.time.Time()
                )
                break
            except Exception as exc:
                self.last_tf_error = str(exc)
                continue

        if transform is None:
            return None

        q = transform.transform.rotation
        return QuaternionSample(
            source="tf",
            stamp=time.monotonic(),
            w=float(q.w),
            x=float(q.x),
            y=float(q.y),
            z=float(q.z),
        )

    def _tf_frame_candidates(self) -> list[str]:
        frames = [self.head_frame]
        if self.fallback_head_frame and self.fallback_head_frame not in frames:
            frames.append(self.fallback_head_frame)
        return frames

    def _sample_to_yaw_pitch(
        self, sample: QuaternionSample
    ) -> tuple[float, float] | None:
        norm = math.sqrt(
            sample.w * sample.w
            + sample.x * sample.x
            + sample.y * sample.y
            + sample.z * sample.z
        )
        if norm <= 1e-9:
            return None
        return self._quat_to_yaw_pitch(
            sample.w / norm,
            sample.x / norm,
            sample.y / norm,
            sample.z / norm,
        )

    def _publish_joint_state(self, yaw: float, pitch: float) -> None:
        if not self.joint_state_publishers:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.name = [self.yaw_joint_name, self.pitch_joint_name]
        msg.position = [float(yaw), float(pitch)]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]

        for publisher in self.joint_state_publishers.values():
            publisher.publish(msg)

        self.last_output_yaw = float(yaw)
        self.last_output_pitch = float(pitch)
        self.published += 1

    def _publish_status(self) -> None:
        if self.status_pub is None:
            return

        status = {
            "enabled": self.enabled,
            "calibrated": self.calibrated,
            "calibration_pending": self.calibration_pending,
            "input_source": self.input_source,
            "input_priority": self.input_priority,
            "active_source": self.last_input_source,
            "base_frame": self.base_frame,
            "head_frame": self.head_frame,
            "fallback_head_frame": self.fallback_head_frame,
            "output_topics": list(self.joint_state_publishers),
            "received": self.received,
            "published": self.published,
            "latest_yaw": self.latest_yaw,
            "latest_pitch": self.latest_pitch,
            "output_yaw": self.last_output_yaw,
            "output_pitch": self.last_output_pitch,
            "last_error": self.last_error,
        }
        if self.last_tf_error:
            status["last_tf_error"] = self.last_tf_error

        msg = String()
        msg.data = json.dumps(status, sort_keys=True, separators=(",", ":"))
        self.status_pub.publish(msg)

    def _log_waiting_for_input(self) -> None:
        now = time.monotonic()
        if now - self.last_warning_time < self.warn_period:
            return
        self.last_warning_time = now

        details = self.last_error
        if self._source_configured("tf"):
            frames = "/".join(self._tf_frame_candidates())
            details += f"; tf={self.base_frame}->{frames}"
            if self.last_tf_error:
                details += f"; last_tf_error={self.last_tf_error}"
        self.get_logger().warning(details)

    @staticmethod
    def _quat_to_yaw_pitch(w: float, x: float, y: float, z: float) -> tuple[float, float]:
        sin_pitch = 2.0 * (w * y - z * x)
        sin_pitch = QuestNode._clamp(sin_pitch, -1.0, 1.0)
        pitch = math.asin(sin_pitch)

        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw, pitch

    @staticmethod
    def _wrap_pi(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QuestNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        node.get_logger().info("Shutting down quest")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
