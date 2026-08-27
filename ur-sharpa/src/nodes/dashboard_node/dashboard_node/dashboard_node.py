"""Operator dashboard for machine, model and policy observability."""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model_manager import ModelManagerError, RemoteModelManager


DASHBOARD_ACTIONS = frozenset(
    {
        "reset_both",
        "start_deploy",
        "continue_deploy",
        "preview_deploy",
        "pause_deploy",
        "stop_deploy",
        "select_policy",
        "select_replay",
    }
)
REPLAY_ACTIONS = frozenset(
    {"load", "start", "pause", "resume", "stop", "unload"}
)
LIVE_REPLAY_TIMEOUT_S = 6.0


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


def wait_for_dashboard_state(
    state: DashboardState,
    predicate: Any,
    *,
    timeout_s: float = LIVE_REPLAY_TIMEOUT_S,
    failure: Any = None,
) -> dict[str, Any]:
    """Wait for an asynchronous ROS status transition from an HTTP worker."""
    deadline = time.monotonic() + timeout_s
    snapshot = state.snapshot()
    while time.monotonic() < deadline:
        snapshot = state.snapshot()
        if failure is not None:
            reason = failure(snapshot)
            if reason:
                raise ModelManagerError(str(reason))
        if predicate(snapshot):
            return snapshot
        time.sleep(0.05)
    raise ModelManagerError("Replay 状态切换超时，请检查节点日志和硬件状态")


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


def stop_policy_deployment(
    status_api_url: str,
    model_manager: RemoteModelManager,
    *,
    forward: Any = forward_action,
) -> dict[str, Any]:
    """Stop the current execution while keeping the loaded model resident."""
    stopped = forward(status_api_url, "stop_deploy")
    if not stopped.get("accepted"):
        raise ModelManagerError(
            str(stopped.get("gate_reason") or "部署停止门未关闭")
        )
    return {
        **stopped,
        "accepted": True,
        "model": model_manager.snapshot(),
        "model_unload_requested": False,
    }


def actor_execution_active(snapshot: dict[str, Any]) -> bool:
    control = snapshot.get("control") or {}
    return bool(
        (
            int(control.get("ur_state", 0)) == 2
            and int(control.get("sharpa_state", 0)) == 1
        )
        or control.get("execution_dry_run")
        or control.get("execution_paused")
    )


def validate_replay_command(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action", "")).strip().lower()
    if action not in REPLAY_ACTIONS:
        raise ValueError(f"unsupported replay action: {action}")
    result: dict[str, Any] = {"action": action}
    if action != "load":
        return result
    sample_dir = str(payload.get("sample_dir", "")).strip()
    if not sample_dir:
        raise ValueError("replay sample_dir is required")
    execution_steps = payload.get("execution_steps")
    frequency_hz = payload.get("frequency_hz")
    if isinstance(execution_steps, bool) or not isinstance(execution_steps, int):
        raise ValueError("replay execution_steps must be an integer")
    if execution_steps <= 0 or execution_steps > 1000:
        raise ValueError("replay execution_steps must be in [1, 1000]")
    if isinstance(frequency_hz, bool):
        raise ValueError("replay frequency_hz must be a number")
    try:
        frequency_hz = float(frequency_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("replay frequency_hz must be a number") from exc
    if not math.isfinite(frequency_hz) or not 5.0 <= frequency_hz <= 125.0:
        raise ValueError("replay frequency_hz must be in [5, 125]")
    joint_source = str(payload.get("joint_source", "measured")).strip().lower()
    if joint_source not in {"measured", "commanded"}:
        raise ValueError("replay joint_source must be measured or commanded")
    for field in ("loop", "dry_run"):
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"replay {field} must be boolean")
    if payload["loop"]:
        raise ValueError("replay loop is disabled; clear progress to run again")
    result.update(
        sample_dir=sample_dir,
        execution_steps=execution_steps,
        frequency_hz=frequency_hz,
        joint_source=joint_source,
        loop=payload["loop"],
        dry_run=payload["dry_run"],
    )
    return result


def validate_live_replay_command(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a live launch and force it out of preview mode."""
    return validate_replay_command(
        {
            **payload,
            "action": "load",
            "dry_run": False,
        }
    )


def replay_config_matches_status(
    replay: dict[str, Any], command: dict[str, Any]
) -> bool:
    """Whether a loaded sample can resume without resetting its cursor."""
    try:
        return bool(
            replay.get("state") in {"ready", "paused"}
            and bool(replay.get("dry_run"))
            == bool(command.get("dry_run", False))
            and str(replay.get("sample_dir") or "")
            == str(command.get("sample_dir") or "")
            and int(replay.get("execution_steps"))
            == int(command.get("execution_steps"))
            and math.isclose(
                float(replay.get("frequency_hz")),
                float(command.get("frequency_hz")),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and str(replay.get("joint_source") or "")
            == str(command.get("joint_source") or "")
            and bool(replay.get("loop")) == bool(command.get("loop"))
        )
    except (TypeError, ValueError):
        return False


def launch_live_replay(
    state: DashboardState,
    status_api_url: str,
    publish_replay_command: Any,
    command: dict[str, Any],
    *,
    forward: Any = forward_action,
) -> dict[str, Any]:
    """Synchronously arm and start Replay while observing every async gate."""
    before = state.snapshot()
    replay_before = before.get("replay") or {}
    if replay_before.get("waiting_action_id") or replay_before.get("preview_active"):
        raise ModelManagerError("Replay 仍在执行或预览，请先停止")
    if (
        replay_before.get("state") == "complete"
        and str(replay_before.get("sample_dir") or "")
        == str(command.get("sample_dir") or "")
    ):
        raise ModelManagerError("Replay 已完成，请先点击停止重置执行进度")

    previous_execution_id = str(replay_before.get("execution_id") or "")
    gate_opened = False
    try:
        if replay_config_matches_status(replay_before, command):
            loaded = before
        else:
            publish_replay_command(command)

            def replay_load_failure(snapshot: dict[str, Any]) -> str:
                replay = snapshot.get("replay") or {}
                if (
                    str(replay.get("execution_id") or "")
                    != previous_execution_id
                    and replay.get("state") == "fault"
                ):
                    return str(replay.get("error") or "Replay 加载失败")
                return ""

            loaded = wait_for_dashboard_state(
                state,
                lambda snapshot: (
                    str((snapshot.get("replay") or {}).get("execution_id") or "")
                    != previous_execution_id
                    and (snapshot.get("replay") or {}).get("state") == "ready"
                    and (snapshot.get("replay") or {}).get("dry_run") is False
                ),
                failure=replay_load_failure,
            )
        execution_id = str(
            (loaded.get("replay") or {}).get("execution_id") or ""
        )

        selected = forward(status_api_url, "select_replay")
        if not selected.get("accepted"):
            raise ModelManagerError(
                str(selected.get("gate_reason") or "无法切换到 Replay 执行来源")
            )
        state.update("execution_provider", "replay")
        wait_for_dashboard_state(
            state,
            lambda snapshot: (
                (snapshot.get("replay") or {}).get("provider") == "replay"
            ),
        )

        started = forward(status_api_url, "start_deploy")
        if not started.get("accepted"):
            raise ModelManagerError(
                str(started.get("gate_reason") or "机器执行门无法开启")
            )
        gate_opened = True
        wait_for_dashboard_state(
            state,
            lambda snapshot: (
                (snapshot.get("replay") or {}).get("machine_running") is True
                and (snapshot.get("replay") or {}).get("provider") == "replay"
            ),
        )

        publish_replay_command({"action": "start"})

        def replay_start_failure(snapshot: dict[str, Any]) -> str:
            replay = snapshot.get("replay") or {}
            if (
                str(replay.get("execution_id") or "") == execution_id
                and replay.get("state") == "fault"
            ):
                return str(replay.get("error") or "Replay 启动失败")
            return ""

        running = wait_for_dashboard_state(
            state,
            lambda snapshot: (
                str((snapshot.get("replay") or {}).get("execution_id") or "")
                == execution_id
                and (snapshot.get("replay") or {}).get("state")
                in {"aligning", "playing", "complete"}
            ),
            failure=replay_start_failure,
        )
    except Exception:
        if gate_opened:
            publish_replay_command({"action": "stop"})
            try:
                forward(status_api_url, "stop_deploy")
            except RuntimeError:
                pass
        raise
    return {
        "accepted": True,
        "replay_action": "live_start",
        "execution_id": execution_id,
        "replay": running.get("replay") or {},
    }


def launch_replay_preview(
    state: DashboardState,
    status_api_url: str,
    publish_replay_command: Any,
    command: dict[str, Any],
    *,
    forward: Any = forward_action,
) -> dict[str, Any]:
    """Run Replay through Actor and dry-run hardware acknowledgements."""
    before = state.snapshot()
    control = before.get("control") or {}
    replay_before = before.get("replay") or {}
    if int(control.get("ur_state", 0)) == 2 or control.get("execution_dry_run"):
        raise ModelManagerError("机器运行时不能启动 Replay 预览")
    if replay_before.get("waiting_action_id") or replay_before.get("preview_active"):
        raise ModelManagerError("Replay 仍在执行或预览，请先停止")
    if (
        replay_before.get("state") == "complete"
        and str(replay_before.get("sample_dir") or "")
        == str(command.get("sample_dir") or "")
    ):
        raise ModelManagerError("Replay 已完成，请先点击停止重置执行进度")
    previous_execution_id = str(replay_before.get("execution_id") or "")
    gate_opened = False
    try:
        if replay_config_matches_status(replay_before, command):
            loaded = before
        else:
            publish_replay_command(command)
            loaded = wait_for_dashboard_state(
                state,
                lambda snapshot: (
                    str((snapshot.get("replay") or {}).get("execution_id") or "")
                    != previous_execution_id
                    and (snapshot.get("replay") or {}).get("state") == "ready"
                    and (snapshot.get("replay") or {}).get("dry_run") is True
                ),
                failure=lambda snapshot: (
                    str((snapshot.get("replay") or {}).get("error") or "Replay 加载失败")
                    if (snapshot.get("replay") or {}).get("state") == "fault"
                    else ""
                ),
            )
        execution_id = str((loaded.get("replay") or {}).get("execution_id") or "")
        selected = forward(status_api_url, "select_replay")
        if not selected.get("accepted"):
            raise ModelManagerError(
                str(selected.get("gate_reason") or "无法切换到 Replay 执行来源")
            )
        state.update("execution_provider", "replay")
        wait_for_dashboard_state(
            state,
            lambda snapshot: (
                (snapshot.get("replay") or {}).get("provider") == "replay"
            ),
        )
        started = forward(status_api_url, "preview_deploy")
        if not started.get("accepted"):
            raise ModelManagerError(
                str(started.get("gate_reason") or "Actor 预览门无法开启")
            )
        gate_opened = True
        wait_for_dashboard_state(
            state,
            lambda snapshot: (
                (snapshot.get("replay") or {}).get("machine_running") is True
            ),
        )
        publish_replay_command({"action": "start"})
        running = wait_for_dashboard_state(
            state,
            lambda snapshot: (
                str((snapshot.get("replay") or {}).get("execution_id") or "")
                == execution_id
                and (
                    (snapshot.get("replay") or {}).get("state") == "complete"
                    or (
                        (snapshot.get("replay") or {}).get("preview_active") is True
                        and (snapshot.get("replay") or {}).get("state")
                        in {"aligning", "playing"}
                    )
                )
            ),
            failure=lambda snapshot: (
                str((snapshot.get("replay") or {}).get("error") or "Replay 预览失败")
                if (snapshot.get("replay") or {}).get("state") == "fault"
                else ""
            ),
        )
    except Exception:
        if gate_opened:
            publish_replay_command({"action": "stop"})
            try:
                forward(status_api_url, "stop_deploy")
            except RuntimeError:
                pass
        raise
    return {
        "accepted": True,
        "replay_action": "preview_start",
        "execution_id": execution_id,
        "replay": running.get("replay") or {},
    }


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

    left_eef = rows(getattr(message, "left_eef", []), 9)
    right_eef = rows(getattr(message, "right_eef", []), 9)
    left_joint = rows(getattr(message, "left_arm_joint", []), 6)
    right_joint = rows(getattr(message, "right_arm_joint", []), 6)
    hand = rows(getattr(message, "hand_joint", []), 44)
    eef_complete = bool(left_eef and right_eef and hand)
    joint_complete = bool(left_joint and right_joint and hand)
    frames = []
    for index in range(count):
        hand_frame = hand[index] if hand else []
        frames.append(
            {
                "step": int(message.execute_start) + index,
                "left_eef": left_eef[index] if eef_complete else [],
                "right_eef": right_eef[index] if eef_complete else [],
                "left_arm_joint": left_joint[index] if joint_complete else [],
                "right_arm_joint": right_joint[index] if joint_complete else [],
                "left_hand": hand_frame[:22] if (eef_complete or joint_complete) else [],
                "right_hand": hand_frame[22:] if (eef_complete or joint_complete) else [],
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
    publish_replay_command: Any,
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
            if self.path == "/api/servers":
                self._send_json(model_manager.servers())
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
                    if action in {"start_deploy", "preview_deploy"}:
                        snapshot = state.snapshot()
                        provider = snapshot.get("execution_provider", "policy")
                        if provider == "policy":
                            if model_manager.snapshot()["phase"] != "running":
                                raise ModelManagerError("请先启动并等待模型服务就绪")
                            policy = snapshot.get("policy") or {}
                            if not policy.get("model_identity_ok"):
                                raise ModelManagerError(
                                    str(policy.get("model_identity_error") or "正在校验模型 task / run")
                                )
                        else:
                            replay = snapshot.get("replay") or {}
                            if replay.get("state") not in {"ready", "paused"}:
                                raise ModelManagerError("请先加载有效的 replay 样本")
                            if action == "start_deploy" and replay.get("dry_run", True):
                                raise ModelManagerError("Replay 仍处于 Dry run；取消勾选后重新加载")
                            if action == "preview_deploy" and not replay.get("dry_run", False):
                                raise ModelManagerError("Replay 尚未以预览模式加载")
                    if (
                        action == "stop_deploy"
                        and state.snapshot().get("execution_provider", "policy")
                        == "policy"
                    ):
                        result = stop_policy_deployment(
                            status_api_url, model_manager
                        )
                    else:
                        result = forward_action(status_api_url, action)
                    if result.get("accepted") and action in {
                        "select_policy",
                        "select_replay",
                    }:
                        state.update(
                            "execution_provider",
                            "policy" if action == "select_policy" else "replay",
                        )
                    self._send_json(result)
                    return
                if self.path == "/api/replay":
                    command = validate_replay_command(payload)
                    publish_replay_command(command)
                    self._send_json(
                        {
                            "accepted": True,
                            "replay_action": command["action"],
                        },
                        202,
                    )
                    return
                if self.path == "/api/replay/live/start":
                    command = validate_live_replay_command(payload)
                    self._send_json(
                        launch_live_replay(
                            state,
                            status_api_url,
                            publish_replay_command,
                            command,
                        )
                    )
                    return
                if self.path == "/api/replay/preview/start":
                    command = validate_replay_command(
                        {
                            **payload,
                            "action": "load",
                            "dry_run": True,
                        }
                    )
                    self._send_json(
                        launch_replay_preview(
                            state,
                            status_api_url,
                            publish_replay_command,
                            command,
                        )
                    )
                    return
                if self.path == "/api/model/start":
                    result = model_manager.start(
                        str(payload.get("task", "")),
                        str(payload.get("run_id", "")),
                    )
                    self._send_json({"accepted": True, "model": result}, 202)
                    return
                if self.path == "/api/model/stop":
                    snapshot = state.snapshot()
                    if actor_execution_active(snapshot):
                        stop_policy_deployment(
                            status_api_url, model_manager
                        )
                    result = model_manager.stop()
                    self._send_json(
                        {
                            "accepted": True,
                            "model": result,
                        },
                        202,
                    )
                    return
                if self.path == "/api/server/select":
                    result = model_manager.select_server(
                        str(payload.get("server_id", ""))
                    )
                    self._send_json({"accepted": True, "model": result})
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
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from ur_msgs.msg import (
        ActorTrajectory,
        CameraSet,
        ControlStatus,
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
                "/share/project/yenaisheng/sharpa-deploy",
            )
            self.declare_parameter("model_server_config", "")
            self._model_status_pub = self.create_publisher(
                String, "/model/status", 10
            )
            self._replay_command_pub = self.create_publisher(
                String, "/replay/command", 10
            )
            server_config = str(
                self.get_parameter("model_server_config").value
            ).strip()
            manager_options: dict[str, Any] = {
                "event_callback": self._model_event,
            }
            if server_config:
                manager_options["profile_path"] = server_config
            else:
                manager_options.update(
                    ssh_host=str(self.get_parameter("model_ssh_host").value),
                    remote_root=str(self.get_parameter("model_remote_root").value),
                )
            self.model_manager = RemoteModelManager(**manager_options)
            self.create_timer(
                0.5,
                lambda: self._model_event(self.model_manager.snapshot()),
            )
            handler = make_handler(
                state,
                str(self.get_parameter("status_api_url").value),
                self.model_manager,
                self._publish_replay_command,
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
            self.create_subscription(String, "/replay/status", self._replay, 10)
            self.create_subscription(
                String,
                "/actor_ik/latency_status",
                self._actor_latency,
                10,
            )
            self.create_subscription(
                String,
                "/control/execution_provider",
                self._provider,
                10,
            )
            self.create_subscription(UrStateFrame, "/ur/state", self._ur, 10)
            self.create_subscription(
                HandStateFrame, "/sharpa/state", self._sharpa, 10
            )
            camera_qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.create_subscription(
                CameraSet, "/camera/set", self._camera, camera_qos
            )
            self.create_subscription(
                String, "/control/bias_status", self._bias, 10
            )
            self.create_subscription(
                ActorTrajectory,
                "/actor_ik/trajectory",
                self._prediction,
                10,
            )
            self._visual_ur: dict[str, Any] = {}
            self._visual_hand: dict[str, Any] = {}
            self._motion_sample_count = 0
            self._last_motion_update = 0.0
            self._prediction_kin = None

        def _publish_replay_command(self, payload: dict[str, Any]) -> None:
            message = String()
            message.data = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
            self._replay_command_pub.publish(message)

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
                    "execution_paused": bool(message.execution_paused),
                    "execution_dry_run": bool(message.execution_dry_run),
                    "gate_reason": message.gate_reason,
                },
            )

        def _policy(self, message: String) -> None:
            payload = self._json(message)
            state.update("policy", payload)
            if isinstance(payload, dict) and isinstance(
                payload.get("buffers"), dict
            ):
                state.update(
                    "buffers",
                    {
                        "buffers": payload["buffers"],
                        "metadata_format_id": str(
                            (payload.get("metadata") or {}).get("format_id")
                            or ""
                        ),
                    },
                )

        def _actor_latency(self, message: String) -> None:
            state.update("actor_latency", self._json(message))

        def _replay(self, message: String) -> None:
            state.update("replay", self._json(message))

        def _provider(self, message: String) -> None:
            state.update("execution_provider", message.data.strip().lower() or "policy")

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
                "control_mode": str(message.control_mode),
                "position_control_ready": bool(message.position_control_ready),
                "control_error": str(message.control_error),
                "timestamp_ns": int(message.timestamp_ns),
                "left_hand": joints[:22],
                "right_hand": joints[22:],
            }
            self._visual_hand = payload
            state.update("sharpa", payload)
            self._update_measured_motion()

        def _prediction(self, message: ActorTrajectory) -> None:
            payload = action_visualization(message)
            payload.update(
                provider=str(message.provider),
                dry_run=bool(message.dry_run),
                trace_dir=str(message.trace_dir),
                ik_applied=bool(message.ik_applied),
            )
            try:
                payload["diagnostics"] = json.loads(message.diagnostics_json or "{}")
            except json.JSONDecodeError:
                payload["diagnostics"] = {"raw": message.diagnostics_json}
            payload["received_at_unix_ms"] = int(time.time() * 1000)
            for frame in payload["frames"]:
                try:
                    if (
                        not frame["left_eef"]
                        and frame["left_arm_joint"]
                        and frame["right_arm_joint"]
                    ):
                        if self._prediction_kin is None:
                            from ur_common.kinematics import UrCommonKinematics

                            self._prediction_kin = UrCommonKinematics()
                        frame["left_eef"] = self._prediction_kin.arm_fk_eef9(
                            frame["left_arm_joint"], "left"
                        ).tolist()
                        frame["right_eef"] = self._prediction_kin.arm_fk_eef9(
                            frame["right_arm_joint"], "right"
                        ).tolist()
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
                    "ego_timestamp_ns": int(message.ego_timestamp_ns),
                    "left_wrist_timestamp_ns": int(
                        message.left_wrist_timestamp_ns
                    ),
                    "right_wrist_timestamp_ns": int(
                        message.right_wrist_timestamp_ns
                    ),
                    "publish_timestamp_ns": (
                        int(message.header.stamp.sec) * 1_000_000_000
                        + int(message.header.stamp.nanosec)
                    ),
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
