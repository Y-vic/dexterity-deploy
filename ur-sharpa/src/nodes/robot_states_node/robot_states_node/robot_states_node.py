"""Join hardware facts and resolve the UR embodiment EEF representation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ur_common.geometry import CANONICAL_EEF_FRAME


class ObservationNotReady(RuntimeError):
    """A required hardware modality has not arrived yet."""


def buffer_status(buffer: Any) -> dict[str, int]:
    timestamps = buffer.timestamps
    frame_count = len(buffer)
    return {
        "frames": frame_count,
        "history_frames": max(frame_count - 1, 0),
        "current_frames": 1 if frame_count else 0,
        "capacity": int(buffer.frame_capacity),
        "bytes": int(buffer.total_bytes),
        "oldest_timestamp_ns": int(timestamps[0]) if timestamps else 0,
        "newest_timestamp_ns": int(timestamps[-1]) if timestamps else 0,
    }


@dataclass(frozen=True)
class RobotSnapshot:
    timestamp_ns: int
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_eef: np.ndarray
    right_eef: np.ndarray
    hand_joint: np.ndarray
    eef_frame: str = CANONICAL_EEF_FRAME


class RobotStateAggregator:
    """Hardware-independent state join used by the ROS node and tests."""

    def __init__(self, kinematics: Any):
        self.kinematics = kinematics
        self.buffers = None
        self._ur: tuple[int, np.ndarray, np.ndarray] | None = None
        self._hand: tuple[int, np.ndarray] | None = None
        self._seq = 0

    def update_ur(self, timestamp_ns: int, left_joint: Any, right_joint: Any, valid: bool = True) -> None:
        left = np.asarray(left_joint, dtype=np.float32).reshape(-1)
        right = np.asarray(right_joint, dtype=np.float32).reshape(-1)
        if not valid or left.shape != (6,) or right.shape != (6,) or not np.isfinite(left).all() or not np.isfinite(right).all():
            self._ur = None
            return
        self._ur = (int(timestamp_ns), left.copy(), right.copy())

    def update_hand(self, timestamp_ns: int, joints: Any, left_valid: bool = True, right_valid: bool = True) -> None:
        hand = np.asarray(joints, dtype=np.float32).reshape(-1)
        if not left_valid or not right_valid or hand.shape != (44,) or not np.isfinite(hand).all():
            self._hand = None
            return
        self._hand = (int(timestamp_ns), hand.copy())

    def snapshot(self) -> RobotSnapshot:
        if self._ur is None:
            raise ObservationNotReady("UR state is not ready")
        if self._hand is None:
            raise ObservationNotReady("SharpA state is not ready")
        ur_stamp, left, right = self._ur
        hand_stamp, hand = self._hand
        left_eef = np.asarray(self.kinematics.arm_fk_eef9(left, "left"), dtype=np.float32)
        right_eef = np.asarray(self.kinematics.arm_fk_eef9(right, "right"), dtype=np.float32)
        if left_eef.shape != (9,) or right_eef.shape != (9,):
            raise ObservationNotReady("FK did not return EEF9")
        return RobotSnapshot(
            timestamp_ns=max(ur_stamp, hand_stamp),
            left_joint=left.copy(), right_joint=right.copy(),
            left_eef=left_eef, right_eef=right_eef, hand_joint=hand.copy(),
        )

    def policy_obs(self, *, session_id: str = "", request_id: int = 0, prompt: str = "") -> dict[str, Any]:
        snap = self.snapshot()
        self._seq += 1
        return {
            "schema": "sharpa_policy_observation.v3",
            "metadata_format_id": "ur_state_only",
            "session_id": session_id,
            "request_id": int(request_id),
            "timestamp_ns": snap.timestamp_ns,
            "prompt": prompt,
            "image": {},
            "state": {
                "history": None,
                "current": {
                    "timestamp_ns": snap.timestamp_ns,
                    "left_wrist": {"joint": snap.left_joint, "eef": snap.left_eef, "eef_def": "absolute"},
                    "right_wrist": {"joint": snap.right_joint, "eef": snap.right_eef, "eef_def": "absolute"},
                    "hand_joint": {"left": snap.hand_joint[:22], "right": snap.hand_joint[22:]},
                    "valid": True,
                },
            },
            "sensor": {},
            "execution_feedback": {"last_action_id": None, "executed_steps": 0, "success": True},
        }

    def push_state_buffer(self, timestamp_ns: int, left_joint: Any, right_joint: Any,
                          left_eef: Any, right_eef: Any, hand_joint: Any, valid: bool) -> None:
        if self.buffers is None:
            return
        from ur_common.policy_buffers import StateFrame
        hand = np.asarray(hand_joint, dtype=np.float32)
        self.buffers.push_state(StateFrame(
            timestamp_ns=int(timestamp_ns), left_joint=np.asarray(left_joint, dtype=np.float32),
            right_joint=np.asarray(right_joint, dtype=np.float32), left_eef=np.asarray(left_eef, dtype=np.float32),
            right_eef=np.asarray(right_eef, dtype=np.float32), left_eef_frame=CANONICAL_EEF_FRAME,
            right_eef_frame=CANONICAL_EEF_FRAME, left_hand_joint=hand[:22], right_hand_joint=hand[22:], valid=bool(valid)))


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import CameraSet, DeformationFrame, HandStateFrame, PolicyObs, TauFrame, UrStateFrame, WrenchFrame
    from ur_msgs.srv import BuildObservation
    from ur_common.kinematics import UrCommonKinematics
    from sharpa_interface.server.serialization import packb
    from ur_common.policy_buffers import CameraFrame, DeformationFrame as BDeformationFrame, ObservationBuffers, TauFrame as BTauFrame, WrenchFrame as BWrenchFrame
    from ur_common.policy_observation import ObservationBuilder, ObservationNotReady as BufferObservationNotReady

    class RosRobotStatesNode(Node):
        def __init__(self):
            super().__init__("robot_states_node")
            self._agg = RobotStateAggregator(UrCommonKinematics())
            self._agg.buffers = ObservationBuffers()
            self._builder = ObservationBuilder(self._agg.buffers)
            self._last_ur = None; self._last_hand = None
            self._session_id = ""
            self._prompt = ""
            self._metadata_format = {}
            self._obs_pub = self.create_publisher(PolicyObs, "/policy/obs", 10)
            self._status_pub = self.create_publisher(String, "/robot_states/status", 10)
            self.create_subscription(UrStateFrame, "/ur/state", self._ur, 10)
            self.create_subscription(HandStateFrame, "/sharpa/state", self._hand, 10)
            self.create_subscription(CameraSet, "/camera/set", self._camera, 10)
            self.create_subscription(TauFrame, "/sharpa/tau", self._tau, 10)
            self.create_subscription(WrenchFrame, "/sharpa/wrench", self._wrench, 10)
            self.create_subscription(DeformationFrame, "/sharpa/deformation", self._deformation, 1)
            self.create_service(BuildObservation, "/policy/build_observation", self._build)
            self.create_timer(0.5, self._publish_status)

        def _ur(self, message):
            self._last_ur = message
            ur_valid = bool(
                message.valid and message.eef_frame == CANONICAL_EEF_FRAME
            )
            self._agg.update_ur(
                message.timestamp_ns,
                message.left_joint,
                message.right_joint,
                ur_valid,
            )
            if not ur_valid:
                return
            if self._last_hand is not None:
                h = self._last_hand
                self._agg.update_hand(h.timestamp_ns, h.q, h.left_valid, h.right_valid)
                try:
                    self._agg.push_state_buffer(max(message.timestamp_ns, h.timestamp_ns), message.left_joint, message.right_joint, message.left_eef, message.right_eef, h.q, bool(ur_valid and h.left_valid and h.right_valid))
                except Exception as exc: self.get_logger().warning(f"state buffer rejected: {exc}")

        def _hand(self, message):
            self._last_hand = message
            self._agg.update_hand(message.timestamp_ns, message.q, message.left_valid, message.right_valid)
            if self._last_ur is not None:
                u = self._last_ur
                ur_valid = bool(
                    u.valid and u.eef_frame == CANONICAL_EEF_FRAME
                )
                self._agg.update_ur(
                    u.timestamp_ns,
                    u.left_joint,
                    u.right_joint,
                    ur_valid,
                )
                if not ur_valid:
                    return
                try:
                    self._agg.push_state_buffer(max(message.timestamp_ns, u.timestamp_ns), u.left_joint, u.right_joint, u.left_eef, u.right_eef, message.q, bool(ur_valid and message.left_valid and message.right_valid))
                except Exception as exc: self.get_logger().warning(f"state buffer rejected: {exc}")

        def _camera(self, message):
            for name, data, stamp, valid in (("ego_cam", message.ego, message.ego_timestamp_ns, message.ego_valid), ("left_wrist_cam", message.left_wrist, message.left_wrist_timestamp_ns, message.left_wrist_valid), ("right_wrist_cam", message.right_wrist, message.right_wrist_timestamp_ns, message.right_wrist_valid)):
                try: self._agg.buffers.push_camera(name, CameraFrame(int(stamp), "jpeg", bytes(data) if valid else b"", bool(valid)))
                except Exception as exc: self.get_logger().warning(f"camera buffer rejected: {exc}")

        def _tau(self, message):
            try: self._agg.buffers.push_tau(BTauFrame(
                int(message.timestamp_ns),
                np.asarray(message.left, dtype=np.float32),
                np.asarray(message.right, dtype=np.float32),
                np.full(22, bool(message.left_valid), dtype=np.bool_),
                np.full(22, bool(message.right_valid), dtype=np.bool_),
            ))
            except Exception as exc: self.get_logger().warning(f"tau buffer rejected: {exc}")

        def _wrench(self, message):
            try: self._agg.buffers.push_wrench(BWrenchFrame(int(message.timestamp_ns), np.asarray(message.left, dtype=np.float32).reshape(5, 6), np.asarray(message.right, dtype=np.float32).reshape(5, 6), np.asarray(message.left_valid, dtype=bool), np.asarray(message.right_valid, dtype=bool)))
            except Exception as exc: self.get_logger().warning(f"wrench buffer rejected: {exc}")

        def _deformation(self, message):
            try: self._agg.buffers.push_deformation(BDeformationFrame(int(message.timestamp_ns), np.asarray(message.left, dtype=np.uint8).reshape(5, 240, 240), np.asarray(message.right, dtype=np.uint8).reshape(5, 240, 240), np.asarray(message.left_valid, dtype=bool), np.asarray(message.right_valid, dtype=bool)))
            except Exception as exc: self.get_logger().warning(f"deformation buffer rejected: {exc}")

        def _publish_status(self):
            payload = {
                "buffers": {
                    "ego_cam": buffer_status(self._agg.buffers.camera("ego_cam")),
                    "left_wrist_cam": buffer_status(self._agg.buffers.camera("left_wrist_cam")),
                    "right_wrist_cam": buffer_status(self._agg.buffers.camera("right_wrist_cam")),
                    "state": buffer_status(self._agg.buffers.state),
                    "tau": buffer_status(self._agg.buffers.tau),
                    "wrench": buffer_status(self._agg.buffers.wrench),
                    "deformation": buffer_status(self._agg.buffers.deformation),
                },
                "metadata_format_id": str(self._metadata_format.get("format_id") or ""),
            }
            message = String()
            message.data = json.dumps(payload, separators=(",", ":"))
            self._status_pub.publish(message)

        def _build(self, request, response):
            try:
                self._session_id = request.session_id
                metadata = json.loads(request.metadata_format_json) if request.metadata_format_json else None
                self._metadata_format = metadata or {}
                payload = (self._builder.build(metadata, session_id=request.session_id, request_id=request.request_id, timestamp_ns=request.timestamp_ns or time.time_ns(), prompt=request.prompt, execution_feedback=json.loads(request.execution_feedback_json) if request.execution_feedback_json else None, max_message_size=int(request.max_message_size) if request.max_message_size else None) if metadata else self._agg.policy_obs(session_id=request.session_id, request_id=request.request_id, prompt=request.prompt))
                encoded = packb(payload, max_size=int(request.max_message_size or 64 * 1024 * 1024))
                snap = self._agg.snapshot()
                message = PolicyObs()
                message.seq = self._agg._seq
                message.session_id = request.session_id
                message.request_id = request.request_id
                message.metadata_format_id = str(payload.get("metadata_format_id", "ur_state_only"))
                message.timestamp_ns = snap.timestamp_ns
                message.prompt = request.prompt
                message.left_joint = snap.left_joint.tolist(); message.right_joint = snap.right_joint.tolist()
                message.left_eef = snap.left_eef.tolist(); message.right_eef = snap.right_eef.tolist()
                message.eef_frame = CANONICAL_EEF_FRAME; message.hand_joint = snap.hand_joint.tolist()
                message.complete = True
                self._obs_pub.publish(message)
                response.success = True; response.retryable = False; response.error_code = ""; response.error_message = ""
                response.observation_msgpack = list(encoded); response.obs_seq = self._agg._seq
            except (ObservationNotReady, BufferObservationNotReady) as exc:
                response.success = False; response.retryable = True; response.error_code = "state_not_ready"; response.error_message = str(exc)
                response.observation_msgpack = []; response.obs_seq = self._agg._seq
            except Exception as exc:
                response.success = False; response.retryable = False; response.error_code = "observation_failed"; response.error_message = str(exc)
                response.observation_msgpack = []; response.obs_seq = self._agg._seq
            return response

    rclpy.init(args=args)
    node = RosRobotStatesNode()
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
