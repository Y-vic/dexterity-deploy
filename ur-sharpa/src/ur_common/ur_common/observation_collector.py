"""Collect ROS hardware facts into policy observation buffers.

The collector is deliberately ROS-independent: the policy ROS adapter converts
messages to plain values at its boundary, while this module owns state joining,
buffer validation, status reporting, and observation construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from sharpa_interface.sharpa import CANONICAL_EEF_FRAME

from .policy_buffers import (
    CameraFrame,
    DeformationFrame,
    ObservationBuffers,
    StateFrame,
    TauFrame,
    WrenchFrame,
)
from .policy_observation import ObservationBuilder


def buffer_status(buffer: Any) -> dict[str, int]:
    """Return the compact buffer health payload consumed by the dashboard."""

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


@dataclass(frozen=True, slots=True)
class _UrSample:
    timestamp_ns: int
    left_joint: np.ndarray
    right_joint: np.ndarray
    left_eef: np.ndarray
    right_eef: np.ndarray
    eef_frame: str
    valid: bool


@dataclass(frozen=True, slots=True)
class _HandSample:
    timestamp_ns: int
    joints: np.ndarray
    left_valid: bool
    right_valid: bool


class ObservationCollector:
    """Join the latest arm/hand facts and maintain all policy buffers."""

    def __init__(self, buffers: ObservationBuffers | None = None) -> None:
        self.buffers = buffers or ObservationBuffers()
        self.builder = ObservationBuilder(self.buffers)
        self._join_lock = threading.RLock()
        self._latest_ur: _UrSample | None = None
        self._latest_hand: _HandSample | None = None

    def push_ur(
        self,
        *,
        timestamp_ns: int,
        left_joint: Any,
        right_joint: Any,
        left_eef: Any,
        right_eef: Any,
        eef_frame: str,
        valid: bool,
    ) -> bool:
        with self._join_lock:
            self._latest_ur = _UrSample(
                timestamp_ns=int(timestamp_ns),
                left_joint=np.asarray(left_joint, dtype=np.float32).reshape(
                    -1
                ),
                right_joint=np.asarray(right_joint, dtype=np.float32).reshape(
                    -1
                ),
                left_eef=np.asarray(left_eef, dtype=np.float32).reshape(-1),
                right_eef=np.asarray(right_eef, dtype=np.float32).reshape(-1),
                eef_frame=str(eef_frame),
                valid=bool(valid),
            )
            return self._push_joined_state()

    def push_hand(
        self,
        *,
        timestamp_ns: int,
        joints: Any,
        left_valid: bool,
        right_valid: bool,
    ) -> bool:
        with self._join_lock:
            self._latest_hand = _HandSample(
                timestamp_ns=int(timestamp_ns),
                joints=np.asarray(joints, dtype=np.float32).reshape(-1),
                left_valid=bool(left_valid),
                right_valid=bool(right_valid),
            )
            return self._push_joined_state()

    def _push_joined_state(self) -> bool:
        ur = self._latest_ur
        hand = self._latest_hand
        if ur is None or hand is None:
            return False
        if not ur.valid or ur.eef_frame != CANONICAL_EEF_FRAME:
            return False
        frame = StateFrame(
            timestamp_ns=max(ur.timestamp_ns, hand.timestamp_ns),
            left_joint=ur.left_joint,
            right_joint=ur.right_joint,
            left_eef=ur.left_eef,
            right_eef=ur.right_eef,
            left_eef_frame=CANONICAL_EEF_FRAME,
            right_eef_frame=CANONICAL_EEF_FRAME,
            left_hand_joint=hand.joints[:22],
            right_hand_joint=hand.joints[22:],
            valid=bool(hand.left_valid and hand.right_valid),
        )
        return self.buffers.push_state(frame)

    def push_camera(
        self,
        name: str,
        *,
        timestamp_ns: int,
        data: Any,
        valid: bool,
    ) -> bool:
        return self.buffers.push_camera(
            name,
            CameraFrame(
                timestamp_ns=int(timestamp_ns),
                encoding="jpeg",
                data=bytes(data) if valid else b"",
                valid=bool(valid),
            ),
        )

    def push_tau(
        self,
        *,
        timestamp_ns: int,
        left: Any,
        right: Any,
        left_valid: bool,
        right_valid: bool,
    ) -> bool:
        return self.buffers.push_tau(
            TauFrame(
                timestamp_ns=int(timestamp_ns),
                left=np.asarray(left, dtype=np.float32),
                right=np.asarray(right, dtype=np.float32),
                left_valid=np.full(22, bool(left_valid), dtype=np.bool_),
                right_valid=np.full(22, bool(right_valid), dtype=np.bool_),
            )
        )

    def push_wrench(
        self,
        *,
        timestamp_ns: int,
        left: Any,
        right: Any,
        left_valid: Any,
        right_valid: Any,
    ) -> bool:
        return self.buffers.push_wrench(
            WrenchFrame(
                timestamp_ns=int(timestamp_ns),
                left=np.asarray(left, dtype=np.float32).reshape(5, 6),
                right=np.asarray(right, dtype=np.float32).reshape(5, 6),
                left_valid=np.asarray(left_valid, dtype=np.bool_),
                right_valid=np.asarray(right_valid, dtype=np.bool_),
            )
        )

    def push_deformation(
        self,
        *,
        timestamp_ns: int,
        left: Any,
        right: Any,
        left_valid: Any,
        right_valid: Any,
    ) -> bool:
        return self.buffers.push_deformation(
            DeformationFrame(
                timestamp_ns=int(timestamp_ns),
                left=np.asarray(left, dtype=np.uint8).reshape(5, 240, 240),
                right=np.asarray(right, dtype=np.uint8).reshape(5, 240, 240),
                left_valid=np.asarray(left_valid, dtype=np.bool_),
                right_valid=np.asarray(right_valid, dtype=np.bool_),
            )
        )

    def build(
        self,
        metadata_format: object,
        *,
        session_id: str,
        request_id: int,
        timestamp_ns: int,
        prompt: str = "",
        execution_feedback: object | None = None,
        max_message_size: int | None = None,
    ) -> dict[str, Any]:
        return self.builder.build(
            metadata_format,
            session_id=session_id,
            request_id=request_id,
            timestamp_ns=timestamp_ns,
            prompt=prompt,
            execution_feedback=execution_feedback,
            max_message_size=max_message_size,
        )

    def status(self) -> dict[str, dict[str, int]]:
        return {
            "ego_cam": buffer_status(self.buffers.camera("ego_cam")),
            "left_wrist_cam": buffer_status(
                self.buffers.camera("left_wrist_cam")
            ),
            "right_wrist_cam": buffer_status(
                self.buffers.camera("right_wrist_cam")
            ),
            "state": buffer_status(self.buffers.state),
            "tau": buffer_status(self.buffers.tau),
            "wrench": buffer_status(self.buffers.wrench),
            "deformation": buffer_status(self.buffers.deformation),
        }
