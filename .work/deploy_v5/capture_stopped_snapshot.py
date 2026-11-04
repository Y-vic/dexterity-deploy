"""Passively capture the stopped robot's latest ROS state and camera frames.

This tool has no publishers and performs no model RPC.  It refuses to save a
snapshot unless the live execution gate is closed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def main() -> None:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from ur_msgs.msg import CameraSet, ControlStatus, HandStateFrame, UrStateFrame

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "stopped_snapshots"),
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    class SnapshotNode(Node):
        def __init__(self) -> None:
            super().__init__("stopped_robot_snapshot")
            self.control: Any | None = None
            self.ur: Any | None = None
            self.hand: Any | None = None
            self.camera: Any | None = None
            self.create_subscription(ControlStatus, "/control/status", self._control, 10)
            self.create_subscription(UrStateFrame, "/ur/state", self._ur, 10)
            self.create_subscription(HandStateFrame, "/sharpa/state", self._hand, 10)
            camera_qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.create_subscription(CameraSet, "/camera/set", self._camera, camera_qos)

        def _control(self, message: Any) -> None:
            self.control = message

        def _ur(self, message: Any) -> None:
            if message.valid:
                self.ur = message

        def _hand(self, message: Any) -> None:
            if message.left_valid and message.right_valid:
                self.hand = message

        def _camera(self, message: Any) -> None:
            if message.ego_valid:
                self.camera = message

    rclpy.init()
    node = SnapshotNode()
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if all(x is not None for x in (node.control, node.ur, node.hand, node.camera)):
                break
        if node.control is None:
            raise RuntimeError("timed out waiting for /control/status")
        gate_open = (
            int(node.control.active_source) == 3
            and not bool(node.control.execution_dry_run)
            and int(node.control.ur_state) == 2
            and int(node.control.sharpa_state) == 1
            and not bool(node.control.execution_paused)
        )
        if gate_open:
            raise RuntimeError("refusing capture because the live execution gate is open")
        if node.ur is None or node.hand is None or node.camera is None:
            raise RuntimeError("timed out waiting for valid robot, hand, and camera samples")

        output = Path(args.output_root).resolve() / time.strftime("%Y-%m-%d_%H-%M-%S")
        output.mkdir(parents=True, exist_ok=False)
        arrays = {
            "ur_timestamp_ns": np.int64(node.ur.timestamp_ns),
            "hand_timestamp_ns": np.int64(node.hand.timestamp_ns),
            "camera_timestamp_ns": np.int64(node.camera.ego_timestamp_ns),
            "left_eef": np.asarray(node.ur.left_eef),
            "right_eef": np.asarray(node.ur.right_eef),
            "left_joint": np.asarray(node.ur.left_joint),
            "right_joint": np.asarray(node.ur.right_joint),
            "left_hand": np.asarray(node.hand.q[:22]),
            "right_hand": np.asarray(node.hand.q[22:]),
        }
        np.savez_compressed(output / "state_current.npz", **arrays)
        (output / "ego_current.jpg").write_bytes(bytes(node.camera.ego))
        if node.camera.left_wrist_valid:
            (output / "left_wrist_current.jpg").write_bytes(bytes(node.camera.left_wrist))
        if node.camera.right_wrist_valid:
            (output / "right_wrist_current.jpg").write_bytes(bytes(node.camera.right_wrist))
        manifest = {
            "schema": "ur_sharpa.stopped_snapshot.v1",
            "created_at_ns": time.time_ns(),
            "eef_frame": str(node.ur.eef_frame),
            "control": {
                "ur_state": int(node.control.ur_state),
                "sharpa_state": int(node.control.sharpa_state),
                "active_source": int(node.control.active_source),
                "execution_paused": bool(node.control.execution_paused),
                "execution_dry_run": bool(node.control.execution_dry_run),
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(output)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
