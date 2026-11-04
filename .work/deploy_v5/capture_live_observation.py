"""Capture one exact online policy observation without publishing an action.

The script subscribes to the same ROS facts as ``policy_client_node``, builds
the same v3 payload, performs one inference in an isolated diagnostic session,
and writes both wire messages to disk.  It deliberately creates no ROS
publisher and refuses to run while the live machine execution gate is open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_to_probe(control: Any) -> bool:
    """A diagnostic inference is allowed only with the hardware gate closed."""

    if control is None:
        return False
    live_gate_open = (
        int(control.active_source) == 3
        and not bool(control.execution_dry_run)
        and int(control.ur_state) == 2
        and int(control.sharpa_state) == 1
        and not bool(control.execution_paused)
    )
    return not live_gate_open


def _action_summary(parsed: Any) -> dict[str, Any]:
    return {
        "session_id": parsed.session_id,
        "request_id": parsed.request_id,
        "action_id": parsed.action_id,
        "revision": parsed.revision,
        "timestamp_ns": parsed.timestamp_ns,
        "execution": {
            "frequency_hz": parsed.execution.frequency_hz,
            "action_length": parsed.execution.action_length,
            "execute_start": parsed.execution.execute_start,
            "execute_length": parsed.execution.execute_length,
            "rtc": parsed.execution.rtc,
        },
        "left_eef": parsed.left_wrist.eef,
        "right_eef": parsed.right_wrist.eef,
        "left_hand": parsed.hand_joint.left,
        "right_hand": parsed.hand_joint.right,
        "diagnostics": parsed.diagnostics,
    }


def main() -> None:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from ur_msgs.msg import CameraSet, ControlStatus, HandStateFrame, UrStateFrame

    from sharpa_interface.server.action import parse_policy_action
    from sharpa_interface.server.client import ServerClient, ServerConfig
    from sharpa_interface.server.serialization import packb
    from ur_common.observation_collector import ObservationCollector
    from ur_common.policy_observation import ObservationNotReady

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "captures"),
        help="Directory under which a timestamped capture is created.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("policy_observation_capture")
            self.collector = ObservationCollector()
            self.control = None
            self.model: dict[str, Any] | None = None
            self.create_subscription(ControlStatus, "/control/status", self._control, 10)
            self.create_subscription(String, "/model/status", self._model, 10)
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

        def _model(self, message: Any) -> None:
            try:
                value = json.loads(message.data)
            except (TypeError, ValueError):
                return
            if isinstance(value, dict):
                self.model = value

        def _ur(self, message: Any) -> None:
            try:
                self.collector.push_ur(
                    timestamp_ns=message.timestamp_ns,
                    left_joint=message.left_joint,
                    right_joint=message.right_joint,
                    left_eef=message.left_eef,
                    right_eef=message.right_eef,
                    eef_frame=message.eef_frame,
                    valid=message.valid,
                )
            except Exception as exc:
                self.get_logger().warning(f"UR sample rejected: {exc}")

        def _hand(self, message: Any) -> None:
            try:
                self.collector.push_hand(
                    timestamp_ns=message.timestamp_ns,
                    joints=message.q,
                    left_valid=message.left_valid,
                    right_valid=message.right_valid,
                )
            except Exception as exc:
                self.get_logger().warning(f"SharpA sample rejected: {exc}")

        def _camera(self, message: Any) -> None:
            for name, data, timestamp_ns, valid in (
                ("ego_cam", message.ego, message.ego_timestamp_ns, message.ego_valid),
                (
                    "left_wrist_cam",
                    message.left_wrist,
                    message.left_wrist_timestamp_ns,
                    message.left_wrist_valid,
                ),
                (
                    "right_wrist_cam",
                    message.right_wrist,
                    message.right_wrist_timestamp_ns,
                    message.right_wrist_valid,
                ),
            ):
                try:
                    self.collector.push_camera(
                        name,
                        timestamp_ns=timestamp_ns,
                        data=data,
                        valid=valid,
                    )
                except Exception as exc:
                    self.get_logger().warning(f"{name} sample rejected: {exc}")

    rclpy.init()
    node = CaptureNode()
    client = None
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.control is not None and node.model is not None:
                break
        if node.control is None:
            raise RuntimeError("timed out waiting for /control/status")
        if not _safe_to_probe(node.control):
            raise RuntimeError(
                "refusing diagnostic inference while the live machine execution gate is open; "
                "press Stop first"
            )
        model = node.model or {}
        if str(model.get("phase") or "") != "running":
            raise RuntimeError("model is not running")

        config = ServerConfig(
            url=str(model.get("server_url") or "ws://127.0.0.1:5500/infer"),
            configured_prompt="",
            ssh_host=str(model.get("ssh_host") or ""),
            ssh_remote_host=str(model.get("remote_host") or "127.0.0.1"),
            ssh_remote_port=int(model.get("port") or 5500),
        )
        client = ServerClient(config)
        metadata = client.connect()
        session_id = f"ur-sharpa-diagnostic-{uuid.uuid4().hex[:12]}"
        reset = client.reset(session_id, request_id=0)
        active_metadata = dict(metadata)
        active_metadata["metadata_format"] = reset["metadata_format"]
        prompt = str(model.get("prompt") or metadata.get("prompt") or "")

        payload = None
        last_error = "observation buffers not ready"
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if not _safe_to_probe(node.control):
                raise RuntimeError("live machine execution gate opened during capture")
            try:
                payload = node.collector.build(
                    active_metadata["metadata_format"],
                    session_id=session_id,
                    request_id=0,
                    timestamp_ns=node.get_clock().now().nanoseconds,
                    prompt=prompt,
                    execution_feedback=None,
                    max_message_size=int(
                        active_metadata.get("max_message_size", 64 * 1024 * 1024)
                    ),
                )
                break
            except ObservationNotReady as exc:
                last_error = str(exc)
        if payload is None:
            raise RuntimeError(f"timed out building observation: {last_error}")

        observation_wire = packb(payload)
        rpc_result = client.infer(payload)
        parsed = parse_policy_action(
            rpc_result.payload,
            expected_session_id=session_id,
            expected_request_id=0,
            expected_schema=str(metadata["action_schema"]),
        )
        action_wire = packb(rpc_result.payload)

        root = Path(args.output_root).expanduser().resolve()
        output = root / time.strftime("%Y-%m-%d_%H-%M-%S")
        output.mkdir(parents=True, exist_ok=False)
        (output / "observation.msgpack").write_bytes(observation_wire)
        (output / "action.msgpack").write_bytes(action_wire)
        ego = payload["image"]["ego_cam"]["current"]
        if ego is not None:
            (output / "ego_current.jpg").write_bytes(bytes(ego["data"]))
        (output / "metadata.json").write_text(
            json.dumps(active_metadata, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (output / "action_summary.json").write_text(
            json.dumps(_action_summary(parsed), indent=2, default=_json_default),
            encoding="utf-8",
        )
        manifest = {
            "schema": "ur_sharpa.diagnostic_capture.v1",
            "created_at_ns": time.time_ns(),
            "session_id": session_id,
            "request_id": 0,
            "inference_latency_s": rpc_result.latency_s,
            "observation_sha256": _sha256(observation_wire),
            "action_sha256": _sha256(action_wire),
            "model": model,
            "server": {
                "url": config.url,
                "ssh_host": config.ssh_host,
                "ssh_remote_host": config.ssh_remote_host,
                "ssh_remote_port": config.ssh_remote_port,
            },
            "control": {
                "ur_state": int(node.control.ur_state),
                "sharpa_state": int(node.control.sharpa_state),
                "active_source": int(node.control.active_source),
                "execution_paused": bool(node.control.execution_paused),
                "execution_dry_run": bool(node.control.execution_dry_run),
            },
            "buffers": node.collector.status(),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default),
            encoding="utf-8",
        )
        current = payload["state"]["current"]
        state_arrays = {
            "timestamp_ns": np.int64(current["timestamp_ns"]),
        }
        for name, value in (
            ("left_eef", current["left_wrist"]["eef"]),
            ("right_eef", current["right_wrist"]["eef"]),
            ("left_joint", current["left_wrist"]["joint"]),
            ("right_joint", current["right_wrist"]["joint"]),
            ("left_hand", current["hand_joint"]["left"]),
            ("right_hand", current["hand_joint"]["right"]),
        ):
            if value is not None:
                state_arrays[name] = np.asarray(value)
        np.savez_compressed(output / "state_current.npz", **state_arrays)
        print(output)
    finally:
        if client is not None:
            client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
