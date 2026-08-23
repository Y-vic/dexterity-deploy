"""Operator dashboard for machine, model and policy observability."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model_manager import ModelManagerError, RemoteModelManager


DASHBOARD_ACTIONS = frozenset({"reset_both", "start_deploy", "stop_deploy"})


class DashboardState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {}

    def update(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)


def forward_action(url: str, action: str, timeout_s: float = 2.0) -> dict[str, Any]:
    if action not in DASHBOARD_ACTIONS:
        raise ValueError(f"unsupported dashboard action: {action}")
    payload = json.dumps({"action": action}, separators=(",", ":")).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"status API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"status API unavailable: {exc.reason}") from exc


def action_visualization(message: Any) -> dict[str, Any]:
    count = int(message.execute_length)

    def rows(values: Any, width: int) -> list[list[float]]:
        data = list(values)
        if count <= 0 or len(data) != count * width:
            return []
        return [
            [float(value) for value in data[index * width : (index + 1) * width]]
            for index in range(count)
        ]

    left_eef = rows(message.left_eef, 9)
    right_eef = rows(message.right_eef, 9)
    hand = rows(message.hand_joint, 44)
    frames = []
    for index in range(count):
        hand_frame = hand[index] if hand else []
        frames.append(
            {
                "step": int(message.execute_start) + index,
                "left_eef": left_eef[index] if left_eef else [],
                "right_eef": right_eef[index] if right_eef else [],
                "left_hand": hand_frame[:22] if hand_frame else [],
                "right_hand": hand_frame[22:] if hand_frame else [],
            }
        )
    return {
        "session_id": message.session_id,
        "request_id": int(message.request_id),
        "action_id": message.action_id,
        "revision": int(message.revision),
        "timestamp_ns": int(message.timestamp_ns),
        "frequency_hz": float(message.frequency_hz),
        "action_length": int(message.action_length),
        "execute_start": int(message.execute_start),
        "execute_length": count,
        "eef_frame": message.eef_frame,
        "frames": frames,
    }


def hand_motion_visualization(
    left_eef: Any,
    right_eef: Any,
    left_hand: Any,
    right_hand: Any,
    *,
    keypoint_builder: Any = None,
) -> dict[str, Any]:
    import numpy as np
    from ur_common.geometry import column_pose9_to_transform

    eef = {
        "left": np.asarray(left_eef, dtype=np.float64),
        "right": np.asarray(right_eef, dtype=np.float64),
    }
    hands = {
        "left": np.asarray(left_hand, dtype=np.float64),
        "right": np.asarray(right_hand, dtype=np.float64),
    }
    for side in ("left", "right"):
        if eef[side].shape != (9,) or not np.isfinite(eef[side]).all():
            raise ValueError(f"{side} canonical wrist must be finite pose9")
        if hands[side].shape != (22,) or not np.isfinite(hands[side]).all():
            raise ValueError(f"{side} hand must contain 22 finite joints")

    if keypoint_builder is None:
        from ur_common.sharpa_fk import hand_joints_to_keypoints

        keypoint_builder = hand_joints_to_keypoints
    local = keypoint_builder(hands["left"], hands["right"])
    result: dict[str, Any] = {
        "frame": "canonical",
        "kinematics": "sharpa_urdf",
    }
    for side in ("left", "right"):
        points = np.asarray(local[side], dtype=np.float64)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError(f"{side} FK must return 21 finite keypoints")
        root_from_wrist = column_pose9_to_transform(eef[side])
        canonical = (
            points @ root_from_wrist[:3, :3].T
            + root_from_wrist[:3, 3]
        )
        result[f"{side}_keypoints"] = canonical.tolist()
        result[f"{side}_wrist"] = eef[side].tolist()
    return result


def make_handler(
    state: DashboardState,
    status_api_url: str,
    model_manager: RemoteModelManager,
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/status":
                payload = state.snapshot()
                payload["model"] = model_manager.snapshot()
                self._send_json(payload)
                return
            if self.path == "/api/models":
                try:
                    self._send_json(model_manager.catalog())
                except ModelManagerError as exc:
                    self._send_json({"error": str(exc)}, 502)
                return
            if self.path == "/" or self.path.startswith("/index"):
                body = Path(__file__).with_name("dashboard.html").read_bytes()
                self._send(body, "text/html; charset=utf-8")
                return
            self.send_error(404)

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/api/action":
                    action = str(payload.get("action", ""))
                    if action == "start_deploy":
                        if model_manager.snapshot()["phase"] != "running":
                            raise ModelManagerError("请先启动并等待模型服务就绪")
                        policy = state.snapshot().get("policy") or {}
                        if not policy.get("model_identity_ok"):
                            raise ModelManagerError(
                                str(policy.get("model_identity_error") or "正在校验模型 task / run")
                            )
                    result = forward_action(
                        status_api_url, action
                    )
                    self._send_json(result)
                    return
                if self.path == "/api/model/start":
                    result = model_manager.start(
                        str(payload.get("task", "")),
                        str(payload.get("family", "")),
                        str(payload.get("run_id", "")),
                    )
                    self._send_json({"accepted": True, "model": result}, 202)
                    return
                if self.path == "/api/model/stop":
                    machine_warning = ""
                    try:
                        forward_action(status_api_url, "stop_deploy")
                    except RuntimeError as exc:
                        machine_warning = str(exc)
                    result = model_manager.stop()
                    self._send_json(
                        {
                            "accepted": True,
                            "model": result,
                            "machine_warning": machine_warning,
                        },
                        202,
                    )
                    return
                self.send_error(404)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self._send_json({"accepted": False, "gate_reason": str(exc)}, 400)
            except ModelManagerError as exc:
                self._send_json({"accepted": False, "gate_reason": str(exc)}, 409)
            except RuntimeError as exc:
                self._send_json({"accepted": False, "gate_reason": str(exc)}, 502)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            return payload

        def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
            self._send(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode(),
                "application/json; charset=utf-8",
                code,
            )

        def _send(self, body: bytes, content_type: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            pass

    return Handler


def main(args=None) -> None:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import (
        CameraSet,
        ControlStatus,
        ExecutableAction,
        HandStateFrame,
        UrStateFrame,
    )

    state = DashboardState()

    class DashboardNode(Node):
        def __init__(self) -> None:
            super().__init__("dashboard_node")
            self.declare_parameter("port", 8088)
            self.declare_parameter(
                "status_api_url", "http://127.0.0.1:8080/api/action"
            )
            self.declare_parameter("model_ssh_host", "BAAI")
            self.declare_parameter(
                "model_remote_root",
                "/share/project/yenaisheng/dreamzero",
            )
            self._model_status_pub = self.create_publisher(
                String, "/model/status", 10
            )
            self.model_manager = RemoteModelManager(
                ssh_host=str(self.get_parameter("model_ssh_host").value),
                remote_root=str(self.get_parameter("model_remote_root").value),
                event_callback=self._model_event,
            )
            self.create_timer(
                0.5,
                lambda: self._model_event(self.model_manager.snapshot()),
            )
            handler = make_handler(
                state,
                str(self.get_parameter("status_api_url").value),
                self.model_manager,
            )
            self.server = ThreadingHTTPServer(
                ("127.0.0.1", int(self.get_parameter("port").value)), handler
            )
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True,
                name="dashboard-http",
            )
            self.thread.start()
            self.create_subscription(
                ControlStatus, "/control/status", self._control, 10
            )
            self.create_subscription(String, "/policy/status", self._policy, 10)
            self.create_subscription(
                String, "/robot_states/status", self._buffers, 10
            )
            self.create_subscription(UrStateFrame, "/ur/state", self._ur, 10)
            self.create_subscription(
                HandStateFrame, "/sharpa/state", self._sharpa, 10
            )
            self.create_subscription(
                CameraSet, "/camera/set", self._camera, 10
            )
            self.create_subscription(
                String, "/control/bias_status", self._bias, 10
            )
            self.create_subscription(
                ExecutableAction,
                "/policy/executable_action",
                self._prediction,
                10,
            )
            self._visual_ur: dict[str, Any] = {}
            self._visual_hand: dict[str, Any] = {}
            self._motion_sample_count = 0
            self._last_motion_update = 0.0

        def _model_event(self, payload: dict[str, Any]) -> None:
            state.update("model", payload)
            message = String()
            message.data = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
            self._model_status_pub.publish(message)

        @staticmethod
        def _json(message: String) -> Any:
            try:
                return json.loads(message.data)
            except json.JSONDecodeError:
                return {"raw": message.data}

        def _control(self, message: ControlStatus) -> None:
            state.update(
                "control",
                {
                    "ur_state": int(message.ur_state),
                    "sharpa_state": int(message.sharpa_state),
                    "record_state": int(message.record_state),
                    "active_source": int(message.active_source),
                    "gate_reason": message.gate_reason,
                },
            )

        def _policy(self, message: String) -> None:
            state.update("policy", self._json(message))

        def _buffers(self, message: String) -> None:
            state.update("buffers", self._json(message))

        def _bias(self, message: String) -> None:
            state.update("bias", self._json(message))

        def _ur(self, message: UrStateFrame) -> None:
            payload = {
                "valid": bool(message.valid),
                "normal_mode": bool(message.normal_mode),
                "timestamp_ns": int(message.timestamp_ns),
                "status": message.status_message,
                "left_eef": [float(value) for value in message.left_eef],
                "right_eef": [float(value) for value in message.right_eef],
                "eef_frame": message.eef_frame,
            }
            self._visual_ur = payload
            state.update("ur", payload)
            self._update_measured_motion()

        def _sharpa(self, message: HandStateFrame) -> None:
            joints = [float(value) for value in message.q]
            payload = {
                "left_valid": bool(message.left_valid),
                "right_valid": bool(message.right_valid),
                "tau_valid": bool(message.tau_valid),
                "timestamp_ns": int(message.timestamp_ns),
                "left_hand": joints[:22],
                "right_hand": joints[22:],
            }
            self._visual_hand = payload
            state.update("sharpa", payload)
            self._update_measured_motion()

        def _prediction(self, message: ExecutableAction) -> None:
            payload = action_visualization(message)
            payload["received_at_unix_ms"] = int(time.time() * 1000)
            for frame in payload["frames"]:
                try:
                    frame.update(
                        hand_motion_visualization(
                            frame["left_eef"],
                            frame["right_eef"],
                            frame["left_hand"],
                            frame["right_hand"],
                        )
                    )
                except Exception as exc:
                    frame["visualization_error"] = str(exc)
            state.update("prediction", payload)

        def _update_measured_motion(self) -> None:
            if not self._visual_ur or not self._visual_hand:
                return
            if not self._visual_ur.get("valid"):
                return
            if not (
                self._visual_hand.get("left_valid")
                and self._visual_hand.get("right_valid")
            ):
                return
            now = time.monotonic()
            if now - self._last_motion_update < 0.1:
                return
            self._last_motion_update = now
            try:
                payload = hand_motion_visualization(
                    self._visual_ur.get("left_eef", []),
                    self._visual_ur.get("right_eef", []),
                    self._visual_hand.get("left_hand", []),
                    self._visual_hand.get("right_hand", []),
                )
                self._motion_sample_count += 1
                payload.update(
                    sample_count=self._motion_sample_count,
                    timestamp_ns=max(
                        int(self._visual_ur.get("timestamp_ns", 0)),
                        int(self._visual_hand.get("timestamp_ns", 0)),
                    ),
                )
            except Exception as exc:
                payload = {"error": str(exc), "frame": "canonical"}
            state.update("measured_motion", payload)

        def _camera(self, message: CameraSet) -> None:
            state.update(
                "camera",
                {
                    "ego": bool(message.ego_valid),
                    "left_wrist": bool(message.left_wrist_valid),
                    "right_wrist": bool(message.right_wrist_valid),
                },
            )

        def destroy_node(self):
            self.server.shutdown()
            self.server.server_close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
