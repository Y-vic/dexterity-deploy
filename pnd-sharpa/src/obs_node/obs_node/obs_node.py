#!/usr/bin/env python3
"""Send robot observations to the inference device over TCP."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import rclpy
from adam_node.body_joints import ADAM_PHYSICAL_JOINTS_31
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from sharpa_node.common import SHARPA_JOINT_NAMES
from std_msgs.msg import String
from teleop_interfaces.msg import (
    SharpaJointState,
    TactileContactPointsArray,
    TactileDeformImageArray,
    TactileForce6DArray,
)

from deploy_common.protocol import (
    FRAME_TYPE_OBS_STATE,
    FRAME_TYPE_TACTILE_BULK,
    json_bytes,
    now_ns,
    pack_tactile_bulk_payload,
)
from obs_node.tcp_client import TcpFrameSender


TACTILE_ORDER = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb",
)

SHARPA_JOINT_LAYOUT = "sharpa_joint_order.v1"
TACTILE_LAYOUT = "sharpa_tactile_right_then_left_pinky_to_thumb.v1"


class ObsNode(Node):
    def __init__(self) -> None:
        super().__init__("obs_node")

        self.declare_parameter("server_host", "10.10.20.110")
        self.declare_parameter("state_port", 15020)
        self.declare_parameter("tactile_bulk_port", 15021)
        self.declare_parameter("state_rate_hz", 60.0)
        self.declare_parameter("connect_timeout_s", 0.2)
        self.declare_parameter("socket_timeout_s", 0.2)
        self.declare_parameter("reconnect_initial_s", 0.2)
        self.declare_parameter("reconnect_max_s", 5.0)
        self.declare_parameter("adam_state_topic", "/adam_physical_joint_states")
        self.declare_parameter("sharpa_state_topic", "/sharpa_physical_joint_states")
        self.declare_parameter(
            "tactile_topic_prefix", "/sharpa_physical_tactile"
        )
        self.declare_parameter("zed_status_topic", "/zed/status")
        self.declare_parameter("status_topic", "/obs_node/status")

        self.server_host = str(self.get_parameter("server_host").value)
        self.state_port = int(self.get_parameter("state_port").value)
        self.tactile_bulk_port = int(
            self.get_parameter("tactile_bulk_port").value
        )
        self.state_rate_hz = float(self.get_parameter("state_rate_hz").value)
        self.connect_timeout_s = float(self.get_parameter("connect_timeout_s").value)
        self.socket_timeout_s = float(self.get_parameter("socket_timeout_s").value)
        self.reconnect_initial_s = float(
            self.get_parameter("reconnect_initial_s").value
        )
        self.reconnect_max_s = float(self.get_parameter("reconnect_max_s").value)
        self.adam_topic = str(self.get_parameter("adam_state_topic").value)
        self.sharpa_topic = str(self.get_parameter("sharpa_state_topic").value)
        self.tactile_prefix = str(
            self.get_parameter("tactile_topic_prefix").value
        ).rstrip("/")
        self.zed_status_topic = str(self.get_parameter("zed_status_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        if self.state_rate_hz <= 0.0:
            raise ValueError("state_rate_hz must be positive")
        for name, port in (
            ("state_port", self.state_port),
            ("tactile_bulk_port", self.tactile_bulk_port),
        ):
            if port <= 0 or port > 65535:
                raise ValueError(f"{name} must be in [1, 65535]")

        sender_kwargs = {
            "connect_timeout_s": self.connect_timeout_s,
            "socket_timeout_s": self.socket_timeout_s,
            "reconnect_initial_s": self.reconnect_initial_s,
            "reconnect_max_s": self.reconnect_max_s,
        }
        self.state_sender = TcpFrameSender(
            self.server_host, self.state_port, **sender_kwargs
        )
        self.tactile_sender = TcpFrameSender(
            self.server_host, self.tactile_bulk_port, **sender_kwargs
        )

        self.lock = threading.Lock()
        self.adam_msg: JointState | None = None
        self.adam_time: float | None = None
        # Keep the wrapper, rather than only ``msg.joint_state``: q_cmd and its
        # source validity bit live on SharpaJointState itself.
        self.sharpa_msg: SharpaJointState | None = None
        self.sharpa_time: float | None = None
        self.force_msg: TactileForce6DArray | None = None
        self.force_time: float | None = None
        self.contact_msg: TactileContactPointsArray | None = None
        self.contact_time: float | None = None
        self.deform_time: float | None = None
        self.zed_status: dict[str, Any] | None = None
        self.zed_time: float | None = None

        self.state_seq = 0
        self.tactile_seq = 0
        self.state_send_attempts = 0
        self.state_sent = 0
        self.tactile_send_attempts = 0
        self.tactile_sent = 0
        self.adam_received = 0
        self.sharpa_received = 0
        self.force_received = 0
        self.contact_received = 0
        self.deform_received = 0
        self.zed_status_received = 0
        self.last_error = ""

        tactile_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(JointState, self.adam_topic, self._on_adam, 10)
        self.create_subscription(
            SharpaJointState, self.sharpa_topic, self._on_sharpa, 10
        )
        self.create_subscription(
            TactileForce6DArray,
            f"{self.tactile_prefix}/force6d",
            self._on_force,
            tactile_qos,
        )
        self.create_subscription(
            TactileContactPointsArray,
            f"{self.tactile_prefix}/contact_points",
            self._on_contact,
            tactile_qos,
        )
        self.create_subscription(
            TactileDeformImageArray,
            f"{self.tactile_prefix}/deform_images",
            self._on_deform,
            tactile_qos,
        )
        self.create_subscription(String, self.zed_status_topic, self._on_zed_status, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_timer(1.0 / self.state_rate_hz, self._send_state)
        self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            "Obs node: "
            f"state=tcp://{self.server_host}:{self.state_port}, "
            f"tactile=tcp://{self.server_host}:{self.tactile_bulk_port}, "
            f"adam={self.adam_topic}, sharpa={self.sharpa_topic}, "
            f"tactile_prefix={self.tactile_prefix}, zed={self.zed_status_topic}"
        )

    def _on_adam(self, msg: JointState) -> None:
        with self.lock:
            self.adam_msg = msg
            self.adam_time = time.monotonic()
            self.adam_received += 1

    def _on_sharpa(self, msg: SharpaJointState) -> None:
        with self.lock:
            self.sharpa_msg = msg
            self.sharpa_time = time.monotonic()
            self.sharpa_received += 1

    def _on_force(self, msg: TactileForce6DArray) -> None:
        with self.lock:
            self.force_msg = msg
            self.force_time = time.monotonic()
            self.force_received += 1

    def _on_contact(self, msg: TactileContactPointsArray) -> None:
        with self.lock:
            self.contact_msg = msg
            self.contact_time = time.monotonic()
            self.contact_received += 1

    def _on_zed_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("ZED status must be a JSON object")
        except Exception as exc:  # noqa: BLE001 - keep last valid status.
            self.last_error = f"bad zed status: {exc}"
            return
        with self.lock:
            self.zed_status = payload
            self.zed_time = time.monotonic()
            self.zed_status_received += 1

    def _on_deform(self, msg: TactileDeformImageArray) -> None:
        stamp_ns = now_ns()
        with self.lock:
            self.deform_time = time.monotonic()
            self.deform_received += 1
            self.tactile_seq += 1
            seq = self.tactile_seq
            nearest_obs_seq = self.state_seq
        try:
            metadata, raw = self._tactile_bulk_metadata(msg, seq, stamp_ns, nearest_obs_seq)
            payload = pack_tactile_bulk_payload(metadata, raw)
            self.tactile_send_attempts += 1
            if self.tactile_sender.send(
                FRAME_TYPE_TACTILE_BULK,
                payload,
                seq,
                stamp_ns,
            ):
                self.tactile_sent += 1
                self.last_error = ""
        except Exception as exc:  # noqa: BLE001 - keep callback alive.
            self.last_error = f"tactile bulk send failed: {exc}"

    def _send_state(self) -> None:
        stamp_ns = now_ns()
        with self.lock:
            self.state_seq += 1
            seq = self.state_seq
            adam_msg = self.adam_msg
            adam_time = self.adam_time
            sharpa_msg = self.sharpa_msg
            sharpa_time = self.sharpa_time
            force_msg = self.force_msg
            force_time = self.force_time
            contact_msg = self.contact_msg
            contact_time = self.contact_time
            zed_status = self.zed_status
            zed_time = self.zed_time

        now = time.monotonic()
        payload = {
            "schema": "pnd.deploy.obs_state.v1",
            "seq": seq,
            "stamp_ns": stamp_ns,
            "topics": {
                "adam_state": self.adam_topic,
                "sharpa_state": self.sharpa_topic,
                "tactile_force6d": f"{self.tactile_prefix}/force6d",
                "tactile_contact_points": f"{self.tactile_prefix}/contact_points",
                "tactile_deform_bulk": {
                    "host": self.server_host,
                    "port": self.tactile_bulk_port,
                    "frame_type": FRAME_TYPE_TACTILE_BULK,
                },
                "zed_status": self.zed_status_topic,
            },
            "adam": self._joint_payload(
                adam_msg, ADAM_PHYSICAL_JOINTS_31, adam_time, now
            ),
            "sharpa": self._sharpa_joint_payload(sharpa_msg, sharpa_time, now),
            "tactile": self._tactile_state_payload(
                force_msg, force_time, contact_msg, contact_time, now
            ),
            "zed": {
                "valid": zed_status is not None,
                "age_ms": self._age_ms(zed_time, now),
                "status": zed_status or {},
            },
        }
        self.state_send_attempts += 1
        try:
            if self.state_sender.send(
                FRAME_TYPE_OBS_STATE,
                json_bytes(payload),
                seq,
                stamp_ns,
            ):
                self.state_sent += 1
                self.last_error = ""
        except Exception as exc:  # noqa: BLE001 - keep timer alive.
            self.last_error = f"state send failed: {exc}"

    def _joint_payload(
        self,
        msg: JointState | None,
        names: list[str] | tuple[str, ...],
        recv_time: float | None,
        now: float,
    ) -> dict[str, Any]:
        if msg is None:
            return {
                "valid": False,
                "stamp_ns": None,
                "age_ms": None,
                "name": list(names),
                "q": [0.0] * len(names),
                "dq": [0.0] * len(names),
                "tau": [0.0] * len(names),
            }
        return {
            "valid": True,
            "stamp_ns": self._header_stamp_ns(msg),
            "age_ms": self._age_ms(recv_time, now),
            "name": list(names),
            "q": self._joint_vector(msg, names, "position"),
            "dq": self._joint_vector(msg, names, "velocity"),
            "tau": self._joint_vector(msg, names, "effort"),
        }

    def _sharpa_joint_payload(
        self,
        msg: SharpaJointState | None,
        recv_time: float | None,
        now: float,
    ) -> dict[str, Any]:
        """Return the model-neutral 44-joint SharpA feedback fact.

        The legacy q/dq/tau/name/stamp_ns fields remain present for existing
        consumers.  The explicitly named fields and element masks prevent
        missing or non-finite values from silently becoming valid zeros.
        ``joint_velocity`` is measured q-dot; it is deliberately not named
        delta_q, which GCC defines as q_cmd - q_exe.
        """
        names = SHARPA_JOINT_NAMES
        if msg is None:
            zeros = [0.0] * len(names)
            invalid = [False] * len(names)
            return {
                "valid": False,
                "stamp_ns": None,
                "feedback_stamp_ns": None,
                "age_ms": None,
                "name": list(names),
                "joint_order": list(names),
                "joint_layout": SHARPA_JOINT_LAYOUT,
                "q": list(zeros),
                "dq": list(zeros),
                "tau": list(zeros),
                "q_exe": list(zeros),
                "q_exe_valid": list(invalid),
                "q_cmd": list(zeros),
                "q_cmd_valid": list(invalid),
                "q_cmd_message_valid": False,
                "joint_velocity": list(zeros),
                "joint_velocity_valid": list(invalid),
                "tau_valid": list(invalid),
            }

        joint_state = msg.joint_state
        q_exe, q_exe_valid = self._joint_vector_with_valid(
            joint_state, names, "position"
        )
        joint_velocity, joint_velocity_valid = self._joint_vector_with_valid(
            joint_state, names, "velocity"
        )
        tau, tau_valid = self._joint_vector_with_valid(
            joint_state, names, "effort"
        )
        q_cmd_message_valid = bool(msg.q_cmd_valid)
        q_cmd, q_cmd_valid = self._fixed_vector_with_valid(
            msg.q_cmd,
            len(names),
            message_valid=q_cmd_message_valid,
        )
        feedback_stamp_ns = self._header_stamp_ns(joint_state)
        return {
            # Backward-compatible fields.
            "valid": True,
            "stamp_ns": feedback_stamp_ns,
            "age_ms": self._age_ms(recv_time, now),
            "name": list(names),
            "q": q_exe,
            "dq": joint_velocity,
            "tau": tau,
            # Explicit unified observation fields.
            "feedback_stamp_ns": feedback_stamp_ns,
            "joint_order": list(names),
            "joint_layout": SHARPA_JOINT_LAYOUT,
            "q_exe": q_exe,
            "q_exe_valid": q_exe_valid,
            "q_cmd": q_cmd,
            "q_cmd_valid": q_cmd_valid,
            "q_cmd_message_valid": q_cmd_message_valid,
            "joint_velocity": joint_velocity,
            "joint_velocity_valid": joint_velocity_valid,
            "tau_valid": tau_valid,
        }

    def _joint_vector(
        self,
        msg: JointState,
        names: list[str] | tuple[str, ...],
        field: str,
    ) -> list[float]:
        values = getattr(msg, field)
        index = {name: idx for idx, name in enumerate(msg.name)}
        output = [0.0] * len(names)
        for out_idx, name in enumerate(names):
            source_idx = index.get(name)
            if source_idx is None or source_idx >= len(values):
                continue
            output[out_idx] = self._finite_float(values[source_idx])
        return output

    def _joint_vector_with_valid(
        self,
        msg: JointState,
        names: list[str] | tuple[str, ...],
        field: str,
    ) -> tuple[list[float], list[bool]]:
        values = getattr(msg, field)
        index = {name: idx for idx, name in enumerate(msg.name)}
        output = [0.0] * len(names)
        valid = [False] * len(names)
        for out_idx, name in enumerate(names):
            source_idx = index.get(name)
            if source_idx is None or source_idx >= len(values):
                continue
            value, value_valid = self._finite_float_with_valid(values[source_idx])
            output[out_idx] = value
            valid[out_idx] = value_valid
        return output, valid

    def _fixed_vector_with_valid(
        self,
        values: Any,
        size: int,
        *,
        message_valid: bool,
    ) -> tuple[list[float], list[bool]]:
        output = [0.0] * size
        valid = [False] * size
        if not message_valid:
            return output, valid
        try:
            value_count = len(values)
        except TypeError:
            return output, valid
        for idx in range(min(value_count, size)):
            value, value_valid = self._finite_float_with_valid(values[idx])
            output[idx] = value
            valid[idx] = value_valid
        return output, valid

    def _tactile_state_payload(
        self,
        force_msg: TactileForce6DArray | None,
        force_time: float | None,
        contact_msg: TactileContactPointsArray | None,
        contact_time: float | None,
        now: float,
    ) -> dict[str, Any]:
        forces = [[0.0, 0.0, 0.0] for _ in TACTILE_ORDER]
        torques = [[0.0, 0.0, 0.0] for _ in TACTILE_ORDER]
        force_frame_id = [0 for _ in TACTILE_ORDER]
        force_sensor_time = [0.0 for _ in TACTILE_ORDER]
        force_valid = [False for _ in TACTILE_ORDER]
        if force_msg is not None:
            for source_idx, entry in enumerate(force_msg.forces):
                idx = self._tactile_index(entry, source_idx)
                if idx is None:
                    continue
                force, force_values_valid = self._fixed_vector_with_valid(
                    entry.force, 3, message_valid=True
                )
                torque, torque_values_valid = self._fixed_vector_with_valid(
                    entry.torque, 3, message_valid=True
                )
                forces[idx] = force
                torques[idx] = torque
                force_frame_id[idx] = int(entry.frame_id)
                force_sensor_time[idx] = self._finite_float(entry.sensor_time)
                force_valid[idx] = (
                    self._tactile_metadata_valid(
                        force_frame_id[idx],
                        force_sensor_time[idx],
                    )
                    and all(force_values_valid)
                    and all(torque_values_valid)
                )
                if not force_valid[idx]:
                    forces[idx] = [0.0, 0.0, 0.0]
                    torques[idx] = [0.0, 0.0, 0.0]

        contact_offsets: list[int] = []
        contact_counts: list[int] = []
        contact_xyz: list[float] = []
        contact_frame_id = [0 for _ in TACTILE_ORDER]
        contact_sensor_time = [0.0 for _ in TACTILE_ORDER]
        contact_valid = [False for _ in TACTILE_ORDER]
        contact_entries: dict[int, Any] = {}
        if contact_msg is not None:
            for source_idx, entry in enumerate(contact_msg.contacts):
                idx = self._tactile_index(entry, source_idx)
                if idx is not None:
                    contact_entries[idx] = entry
        for idx in range(len(TACTILE_ORDER)):
            contact_offsets.append(len(contact_xyz))
            values: list[float] = []
            entry = contact_entries.get(idx)
            if entry is not None:
                raw_values = [
                    self._finite_float_with_valid(value) for value in entry.points
                ]
                raw = [value for value, _valid in raw_values]
                point_values = (len(raw) // 3) * 3
                values = raw[:point_values]
                contact_frame_id[idx] = int(entry.frame_id)
                contact_sensor_time[idx] = self._finite_float(entry.sensor_time)
                contact_valid[idx] = (
                    self._tactile_metadata_valid(
                        contact_frame_id[idx],
                        contact_sensor_time[idx],
                    )
                    and len(raw) % 3 == 0
                    and all(valid for _value, valid in raw_values)
                )
            contact_counts.append(len(values) // 3)
            contact_xyz.extend(values)

        return {
            "valid": any(force_valid) or any(contact_valid),
            "order": list(TACTILE_ORDER),
            "tactile_layout": TACTILE_LAYOUT,
            "force_age_ms": self._age_ms(force_time, now),
            "force_stamp_ns": (
                self._header_stamp_ns(force_msg) if force_msg is not None else None
            ),
            "contact_age_ms": self._age_ms(contact_time, now),
            "contact_stamp_ns": (
                self._header_stamp_ns(contact_msg) if contact_msg is not None else None
            ),
            "force": forces,
            "torque": torques,
            "force_frame_id": force_frame_id,
            "force_sensor_time": force_sensor_time,
            "force_valid": force_valid,
            "wrench": [
                [*force, *torque] for force, torque in zip(forces, torques)
            ],
            "wrench_frame_id": force_frame_id,
            "wrench_sensor_timestamp": force_sensor_time,
            "wrench_valid": force_valid,
            "contact_offset": contact_offsets,
            "contact_count": contact_counts,
            "contact_xyz": contact_xyz,
            "contact_frame_id": contact_frame_id,
            "contact_sensor_time": contact_sensor_time,
            "contact_valid": contact_valid,
        }

    def _tactile_bulk_metadata(
        self,
        msg: TactileDeformImageArray,
        seq: int,
        stamp_ns: int,
        nearest_obs_seq: int,
    ) -> tuple[dict[str, Any], bytes]:
        entries: list[dict[str, Any]] = []
        chunks: list[bytes] = []
        offset = 0
        images_by_index: dict[int, Any] = {}
        for source_idx, image in enumerate(msg.images):
            idx = self._tactile_index(image, source_idx)
            if idx is not None:
                images_by_index[idx] = image
        for idx, tactile_name in enumerate(TACTILE_ORDER):
            image = images_by_index.get(idx)
            if image is None:
                side, finger = tactile_name.split("_", 1)
                entries.append(
                    {
                        "index": idx,
                        "side": side,
                        "finger": finger,
                        "channel": idx,
                        "frame_id": 0,
                        "sensor_time": 0.0,
                        "sensor_timestamp": 0.0,
                        "height": 0,
                        "width": 0,
                        "valid": False,
                        "offset": offset,
                        "raw_offset": offset,
                        "length": 0,
                        "raw_length": 0,
                    }
                )
                continue
            data = bytes(image.data)
            sensor_time = self._finite_float(image.sensor_time)
            valid = (
                self._tactile_metadata_valid(int(image.frame_id), sensor_time)
                and int(image.height) > 0
                and int(image.width) > 0
                and len(data) == int(image.height) * int(image.width)
            )
            entries.append(
                {
                    "index": idx,
                    "side": image.side,
                    "finger": image.finger,
                    "channel": int(image.channel),
                    "frame_id": int(image.frame_id),
                    "sensor_time": sensor_time,
                    "sensor_timestamp": sensor_time,
                    "height": int(image.height),
                    "width": int(image.width),
                    "valid": valid,
                    "offset": offset,
                    "raw_offset": offset,
                    "length": len(data),
                    "raw_length": len(data),
                }
            )
            chunks.append(data)
            offset += len(data)
        metadata = {
            "schema": "pnd.deploy.tactile_bulk.v1",
            "seq": seq,
            "stamp_ns": stamp_ns,
            "ros_stamp_ns": self._header_stamp_ns(msg),
            "nearest_obs_seq": nearest_obs_seq,
            "encoding": "uint8",
            "layout": "metadata_entries_with_raw_offsets",
            "tactile_layout": TACTILE_LAYOUT,
            "order": list(TACTILE_ORDER),
            "image_count": len(entries),
            "valid": [bool(entry["valid"]) for entry in entries],
            "frame_id": [int(entry["frame_id"]) for entry in entries],
            "sensor_timestamp": [
                float(entry["sensor_timestamp"]) for entry in entries
            ],
            "raw_offset": [int(entry["raw_offset"]) for entry in entries],
            "raw_length": [int(entry["raw_length"]) for entry in entries],
            "entries": entries,
        }
        return metadata, b"".join(chunks)

    @staticmethod
    def _header_stamp_ns(msg: Any) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _finite_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    @staticmethod
    def _finite_float_with_valid(value: Any) -> tuple[float, bool]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0, False
        if not math.isfinite(number):
            return 0.0, False
        return number, True

    @staticmethod
    def _tactile_index(entry: Any, fallback_index: int) -> int | None:
        side = str(getattr(entry, "side", ""))
        finger = str(getattr(entry, "finger", ""))
        try:
            return TACTILE_ORDER.index(f"{side}_{finger}")
        except ValueError:
            # Preserve compatibility with older ordered producers that did not
            # populate the side/finger metadata.
            if 0 <= fallback_index < len(TACTILE_ORDER):
                return fallback_index
            return None

    @staticmethod
    def _tactile_metadata_valid(frame_id: int, sensor_time: float) -> bool:
        return frame_id > 0 and math.isfinite(sensor_time) and sensor_time > 0.0

    @staticmethod
    def _age_ms(stamp: float | None, now: float) -> float | None:
        if stamp is None:
            return None
        return round((now - stamp) * 1000.0, 1)

    def _sender_payload(self, sender: TcpFrameSender, now: float) -> dict[str, Any]:
        status = sender.status()
        return {
            "connected": status.connected,
            "connect_attempts": status.connect_attempts,
            "sent_frames": status.sent_frames,
            "dropped_frames": status.dropped_frames,
            "last_connect_age_ms": self._age_ms(status.last_connect_time, now),
            "last_send_age_ms": self._age_ms(status.last_send_time, now),
            "last_error": status.last_error,
        }

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            payload = {
                "node": "obs_node",
                "endpoints": {
                    "state": {
                        "role": "tcp_client",
                        "host": self.server_host,
                        "port": self.state_port,
                        **self._sender_payload(self.state_sender, now),
                    },
                    "tactile_bulk": {
                        "role": "tcp_client",
                        "host": self.server_host,
                        "port": self.tactile_bulk_port,
                        **self._sender_payload(self.tactile_sender, now),
                    },
                },
                "topics": {
                    "adam_state": self.adam_topic,
                    "sharpa_state": self.sharpa_topic,
                    "tactile_prefix": self.tactile_prefix,
                    "zed_status": self.zed_status_topic,
                },
                "counts": {
                    "state_send_attempts": self.state_send_attempts,
                    "state_sent": self.state_sent,
                    "tactile_send_attempts": self.tactile_send_attempts,
                    "tactile_sent": self.tactile_sent,
                    "adam_received": self.adam_received,
                    "sharpa_received": self.sharpa_received,
                    "force_received": self.force_received,
                    "contact_received": self.contact_received,
                    "deform_received": self.deform_received,
                    "zed_status_received": self.zed_status_received,
                },
                "age_ms": {
                    "adam": self._age_ms(self.adam_time, now),
                    "sharpa": self._age_ms(self.sharpa_time, now),
                    "force": self._age_ms(self.force_time, now),
                    "contact": self._age_ms(self.contact_time, now),
                    "deform": self._age_ms(self.deform_time, now),
                    "zed_status": self._age_ms(self.zed_time, now),
                },
                "schema": {
                    "obs_state": "pnd.deploy.obs_state.v1",
                    "tactile_bulk": "pnd.deploy.tactile_bulk.v1",
                    "adam_q_dq_tau": len(ADAM_PHYSICAL_JOINTS_31),
                    "sharpa_q_dq_tau": len(SHARPA_JOINT_NAMES),
                    "sharpa_joint_layout": SHARPA_JOINT_LAYOUT,
                    "tactile_order": list(TACTILE_ORDER),
                    "tactile_layout": TACTILE_LAYOUT,
                },
                "last_error": self.last_error,
            }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.state_sender.close()
        self.tactile_sender.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObsNode()
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
