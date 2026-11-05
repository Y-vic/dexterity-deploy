"""Hidden ROS control state machine for teleop and deploy."""

from __future__ import annotations

from ur_common.protection_config import setting
import json
import threading
import time
from zipfile import BadZipFile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Bool, String
    from ur_msgs.msg import (
        CameraSet,
        ControlStatus,
        HandStateFrame,
        HardwareCommandResult,
        UrStateFrame,
    )
    ROS_AVAILABLE = True
except Exception:  # pragma: no cover
    ROS_AVAILABLE = False
    Node = object  # type: ignore

from .gates import GateMachine
from ur_common.reset_target import episode_initial_target


def resolve_reset_target(body):
    choice = body.get("reset_target", {"mode": "bias"})
    if not isinstance(choice, dict):
        raise ValueError("reset_target must be an object")
    mode = choice.get("mode")
    if mode == "bias":
        return None
    if mode != "episode":
        raise ValueError("reset_target mode must be bias or episode")
    root, episode = choice.get("root"), choice.get("episode")
    if not isinstance(root, str) or not isinstance(episode, str) or not root or not episode:
        raise ValueError("select a reference dataset and episode before reset")
    return episode_initial_target(root, episode)


if ROS_AVAILABLE:
    class StatusNode(Node):
        """ROS-only shell around :class:`GateMachine`; no HTTP server."""

        def __init__(self):
            super().__init__("status_node")
            self.declare_parameter("mode", "teleop")
            self.declare_parameter("allow_mode_switch", False)
            self.declare_parameter("enable_hil", False)
            self._hil_enabled = bool(self.get_parameter("enable_hil").value)
            self.declare_parameter("camera_max_age_s", setting("status", "camera_max_age_s"))
            self.declare_parameter("status_rate_hz", 10.0)
            self.declare_parameter("web_enabled", False)
            self.declare_parameter("web_port", 8080)
            self.machine = GateMachine(
                mode=str(self.get_parameter("mode").value),
                allow_mode_switch=bool(self.get_parameter("allow_mode_switch").value),
                camera_max_age_s=float(self.get_parameter("camera_max_age_s").value),
            )
            qos = QoSProfile(depth=10)
            camera_qos = QoSProfile(
                depth=1, history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.status_pub = self.create_publisher(ControlStatus, "/control/status", qos)
            self.provider_pub = self.create_publisher(
                String, "/control/execution_provider", qos
            )
            self.policy_recording_pub = self.create_publisher(Bool, '/control/policy_recording', qos)
            self.bias_pub = self.create_publisher(String, "/control/bias_request", qos)
            self.record_pub = self.create_publisher(String, "/control/record_command", qos)
            self.stop_pub = self.create_publisher(Bool, "/hardware/stop", qos)
            self.create_subscription(String, "/control/ui_command", self._command, qos)
            self.create_subscription(CameraSet, "/camera/set", self._camera, camera_qos)
            self.create_subscription(String, "/control/bias_status", self._bias, qos)
            self.create_subscription(String, "/control/record_status", self._record, qos)
            self.create_subscription(String, "/teleop/vive_status", self._upstream("vive_status"), qos)
            self.create_subscription(String, "/teleop/glove_status", self._upstream("glove_status"), qos)
            self.create_subscription(UrStateFrame, "/ur/state", self._ur_state, qos)
            self.create_subscription(HandStateFrame, "/sharpa/state", self._sharpa_state, qos)
            self.create_subscription(HardwareCommandResult, "/ur/command_result", self._result("ur"), qos)
            self.create_subscription(HardwareCommandResult, "/sharpa/command_result", self._result("sharpa"), qos)
            self.upstream = {}
            self._record_seen = 0.0
            self._action_lock = threading.RLock()
            self.http = None
            if bool(self.get_parameter("web_enabled").value):
                self._start_compat_api(int(self.get_parameter("web_port").value))
            self.create_timer(1.0 / max(.1, float(self.get_parameter("status_rate_hz").value)), self._publish)

        def _start_compat_api(self, port):
            node = self
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args): pass
                def _json(self, payload, code=200):
                    data = json.dumps(payload, separators=(",", ":")).encode()
                    self.send_response(code); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                def do_GET(self):
                    if self.path.startswith("/api/status"):
                        self._json(node._snapshot()); return
                    self.send_error(404)
                def do_POST(self):
                    if not self.path.startswith("/api/action"): self.send_error(404); return
                    try:
                        count = int(self.headers.get("content-length", "0"))
                        self._json(node._apply_action(json.loads(self.rfile.read(count) or b"{}")))
                    except Exception as exc:
                        self._json({"accepted": False, "gate_reason": str(exc)}, 400)
            self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            threading.Thread(target=self.http.serve_forever, daemon=True, name="status-api").start()

        def _upstream(self, key):
            def callback(msg):
                try:
                    self.upstream[key] = json.loads(msg.data)
                except Exception:
                    self.upstream[key] = msg.data
            return callback

        def _result(self, key):
            def callback(msg):
                payload = {
                    "timestamp_ns": int(msg.timestamp_ns),
                    "source": int(msg.source),
                    "action_id": str(msg.action_id),
                    "revision": int(msg.revision),
                    "step_index": int(msg.step_index),
                    "device": str(msg.device),
                    "dry_run": bool(msg.dry_run),
                    "success": bool(msg.success),
                    "failure_code": str(msg.failure_code),
                    "failure_message": str(msg.failure_message),
                }
                self.upstream[f"{key}_command"] = payload
                history_key = f"{key}_command_history"
                history = list(self.upstream.get(history_key) or [])
                history.append(payload)
                self.upstream[history_key] = history[-200:]
            return callback

        def _camera(self, msg):
            self.machine.update_cameras(
                {"ego": msg.ego_timestamp_ns, "left_wrist": msg.left_wrist_timestamp_ns,
                 "right_wrist": msg.right_wrist_timestamp_ns},
                {"ego": msg.ego_valid, "left_wrist": msg.left_wrist_valid,
                 "right_wrist": msg.right_wrist_valid}, time.time(),
            )

        def _ur_state(self, msg):
            self.machine.update_hardware_readiness(
                ur_ready=bool(msg.valid and msg.normal_mode)
            )

        def _sharpa_state(self, msg):
            self.machine.update_hardware_readiness(
                sharpa_ready=bool(msg.left_valid and msg.right_valid)
            )
            self.upstream["sharpa_sides"] = {
                "left_valid": bool(msg.left_valid),
                "right_valid": bool(msg.right_valid),
                "control_mode": str(msg.control_mode),
                "position_control_ready": bool(msg.position_control_ready),
                "control_error": str(msg.control_error),
            }
            outcome = self.machine.on_execution_start_readiness(
                position_ready=bool(msg.position_control_ready),
                control_error=str(msg.control_error),
                state_timestamp_ns=int(msg.timestamp_ns),
            )
            if outcome is not None:
                self._publish()
                if outcome.hardware_stop is not None:
                    stop = Bool()
                    stop.data = bool(outcome.hardware_stop)
                    self.stop_pub.publish(stop)

        def _bias(self, msg):
            try:
                payload = json.loads(msg.data) if msg.data else {}
            except Exception:
                payload = {"phase": "invalid", "detail": str(msg.data)}
            outcome = self.machine.on_bias_status(payload)
            self.upstream["bias_status"] = payload
            self._publish()
            if outcome.hardware_stop is not None:
                stop = Bool()
                stop.data = bool(outcome.hardware_stop)
                self.stop_pub.publish(stop)

        def _record(self, msg):
            try:
                self.upstream["record_status"] = json.loads(msg.data)
                self._record_seen = time.monotonic()
            except Exception:
                self.upstream["record_status"] = msg.data

        def _command(self, msg):
            try:
                body = json.loads(msg.data) if msg.data.startswith("{") else {"action": msg.data}
            except Exception:
                body = {"action": msg.data.strip()}
            self._apply_action(body)

        def _apply_action(self, body):
            with self._action_lock:
                return self._apply_action_locked(body)

        def _apply_action_locked(self, body):
            action = str(body.get("action", ""))
            if action == "select_hil" and not self._hil_enabled:
                reason = "HIL 未启用；请以 enable_hil:=true 启动部署"
                self.machine._set_reason(reason)
                return {**self._snapshot(), "accepted": False, "gate_reason": reason}
            if action == "reset_both":
                record = self.upstream.get("record_status") or {}
                # Once recording=False is acknowledged, the saver only drains
                # captured data. Bias motion may overlap that disk/video work.
                if self.machine.execution_paused or (isinstance(record, dict) and record.get("saving")):
                    if (not isinstance(record, dict) or time.monotonic() - self._record_seen > 2.0
                            or record.get("recording") is not False or record.get("rollout_requested")):
                        reason = "复位需等待数采在线且当前样本采样结束；后台保存无需等待"
                        self.machine._set_reason(reason)
                        return {**self._snapshot(), "accepted": False, "gate_reason": reason}
            if action == "start_teleop_record":
                record = self.upstream.get("record_status") or {}
                if (not isinstance(record, dict) or time.monotonic() - self._record_seen > 2.0
                        or record.get("capture_mode") != "teleop" or record.get("recording")
                        or record.get("saving") or record.get("rollout_requested")):
                    reason = "等待数采节点在线、确认数采模式且录制保存完成"
                    self.machine._set_reason(reason)
                    return {**self._snapshot(), "accepted": False, "gate_reason": reason}
            if self.machine.allow_mode_switch:
                record = self.upstream.get("record_status") or {}
                selecting = action in {"select_teleop", "select_policy", "select_hil", "select_replay"}
                reason = ""
                if selecting and (time.monotonic() - self._record_seen > 2.0
                                  or record.get("recording") or record.get("saving")
                                  or record.get("rollout_requested")):
                    reason = "等待数采在线且录制保存完成后才能切换模式"
                if action in {"run_ur", "start_record"} and self.machine.mode != "teleop":
                    reason = "手动遥操与录制仅在数采模式可用"
                if action == "start_record" and (record.get("capture_mode") != "teleop"
                        or time.monotonic() - self._record_seen > 2.0 or record.get("saving")):
                    reason = "等待数采节点确认模式或完成保存"
                if action in {"start_deploy", "start_hands_deploy", "preview_deploy", "preview_hands_deploy"} and (
                        record.get("capture_mode") != "rollout" or time.monotonic() - self._record_seen > 2.0
                        or record.get("saving")):
                    reason = "等待数采节点确认部署模式或完成保存"
                if reason:
                    self.machine._set_reason(reason)
                    return {**self._snapshot(), "accepted": False, "gate_reason": reason}
                if selecting:
                    outcome = self.machine.attempt_set_workbench_mode(action.removeprefix("select_"))
                    self._publish()
                    return {**self._snapshot(), "accepted": bool(outcome), "gate_reason": outcome.reason}
            actions = {
                "start_ur": self.machine.attempt_start_ur,
                "run_ur": self.machine.attempt_run_ur,
                "reset_ur": self.machine.attempt_reset_ur,
                "start_sharpa": self.machine.attempt_start_sharpa,
                "reset_both": self.machine.attempt_reset_both,
                "start_record": self.machine.attempt_start_recording,
                "start_teleop_record": self.machine.attempt_start_teleop_recording,
                "stop_record": self.machine.attempt_stop_recording,
                "discard_record": self.machine.attempt_discard_recording,
                "start_deploy": self.machine.attempt_start_deploy,
                "start_hands_deploy": self.machine.attempt_start_hands_deploy,
                "continue_deploy": self.machine.attempt_continue_deploy,
                "preview_deploy": self.machine.attempt_preview_deploy,
                "preview_hands_deploy": self.machine.attempt_preview_hands_deploy,
                "pause_deploy": self.machine.attempt_pause_deploy,
                "stop_deploy": self.machine.attempt_stop_deploy,
                "select_policy": lambda: self.machine.attempt_set_execution_provider("policy"),
                "select_replay": lambda: self.machine.attempt_set_execution_provider("replay"),
                "select_hil": lambda: self.machine.attempt_set_execution_provider("hil"),
                "enable_policy_recording": lambda: self.machine.attempt_set_policy_recording(True),
                "disable_policy_recording": lambda: self.machine.attempt_set_policy_recording(False),
                "pause": self.machine.attempt_pause,
                "stop": self.machine.attempt_stop,
            }
            reset_target = None
            if action == "reset_both":
                try:
                    reset_target = resolve_reset_target(body)
                except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
                    outcome = self.machine._set_reason(f"machine reset blocked: {exc}")
                    result = self._snapshot()
                    result["accepted"] = False
                    result["gate_reason"] = outcome.reason
                    return result
            outcome = actions.get(action, lambda: self.machine._set_reason(f"unknown action: {action}"))()
            # State is the sole owner of hardware lifecycle. Publish the new
            # state before releasing the stop latch or announcing a reset;
            # bias_node then waits for explicit device readiness.
            self._publish()
            if outcome.hardware_stop is not None:
                msg = Bool(); msg.data = bool(outcome.hardware_stop); self.stop_pub.publish(msg)
            if outcome.bias_request:
                self._publish_string(
                    self.bias_pub,
                    json.dumps({
                        "request": outcome.bias_request,
                        "action_id": self.machine.bias_action_id,
                        "target": reset_target,
                    }, separators=(",", ":")),
                )
            if outcome.record_command:
                self._publish_string(self.record_pub, outcome.record_command)
            self.upstream["last_action"] = {"action": action, "accepted": bool(outcome), "reason": outcome.reason}
            result = self._snapshot()
            result["accepted"] = bool(outcome)
            result["gate_reason"] = outcome.reason
            return result

        def _snapshot(self):
            return {"status": self.machine.status_payload(), "upstream": dict(self.upstream)}

        @staticmethod
        def _publish_string(pub, value):
            msg = String(); msg.data = str(value); pub.publish(msg)

        def _publish(self):
            state = self.machine
            asynchronous = state.poll_async_transitions()
            msg = ControlStatus()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.timestamp_ns = time.time_ns()
            msg.ur_state = int(state.ur_state)
            msg.sharpa_state = int(state.sharpa_state)
            msg.record_state = int(state.record_state)
            msg.active_source = int(state.active_source)
            msg.execution_paused = bool(state.execution_paused)
            msg.execution_dry_run = bool(state.execution_dry_run)
            msg.execution_scope = int(state.execution_scope)
            msg.gate_reason = state.gate_reason or state.camera_gate_reason()
            msg.session_id = state.session_id
            self.status_pub.publish(msg)
            self._publish_string(self.provider_pub, state.execution_provider)
            recording = Bool(); recording.data = state.policy_recording
            self.policy_recording_pub.publish(recording)
            if asynchronous is not None and asynchronous.hardware_stop is not None:
                stop = Bool()
                stop.data = bool(asynchronous.hardware_stop)
                self.stop_pub.publish(stop)

        def destroy_node(self):
            if self.http is not None:
                self.http.shutdown(); self.http.server_close()
            return super().destroy_node()

else:
    class StatusNode:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("rclpy is unavailable")


def main(args=None):
    if not ROS_AVAILABLE:
        raise RuntimeError("rclpy is unavailable")
    rclpy.init(args=args)
    node = StatusNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
