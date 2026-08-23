"""ROS-facing adapter around the frozen interface policy loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PendingRequest:
    request_id: int
    feedback: dict[str, Any]


class LegacyServerPoseAdapter:
    """Keep the remote physical-pose adapter outside the canonical ROS ABI."""

    def __init__(self, geometry: Any):
        self.geometry = geometry

    def observation_to_server(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        state = dict(result.get("state") or {})
        for name in ("history", "current"):
            frame = state.get(name)
            if frame is not None:
                state[name] = self._convert_frame(
                    frame, self.geometry.canonical_pose_to_legacy_server_pose
                )
        result["state"] = state
        return result

    def action_to_canonical(self, command: dict[str, Any]) -> dict[str, Any]:
        result = dict(command)
        action = dict(result["action"])
        result["action"] = self._convert_frame(
            action, self.geometry.legacy_server_pose_to_canonical_pose
        )
        return result

    @staticmethod
    def _pose_array(value: Any, side: str, converter: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != 9 or not np.isfinite(array).all():
            raise ValueError(f"{side} wrist EEF must be finite with last dimension 9")
        rows = array.reshape(-1, 9)
        converted = np.stack([converter(row, side) for row in rows])
        return converted.reshape(array.shape).astype(np.float32)

    def _convert_frame(self, frame: dict[str, Any], converter: Any) -> dict[str, Any]:
        result = dict(frame)
        for side in ("left", "right"):
            key = f"{side}_wrist"
            wrist = dict(result[key])
            wrist["eef"] = self._pose_array(wrist.get("eef"), side, converter)
            result[key] = wrist
        return result


class PolicyClientCore:
    """Keep transport-independent inference/action/done identity state."""

    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id must be nonempty")
        from sharpa_interface.server.execution import SyncExecutionGate
        self.session_id = session_id
        self.gate = SyncExecutionGate()

    def begin(self) -> PendingRequest:
        request_id, feedback = self.gate.begin_inference()
        return PendingRequest(request_id, feedback)

    @property
    def ready(self) -> bool:
        return self.gate.phase.value == "ready"

    @property
    def inference_pending(self) -> bool:
        return self.gate.phase.value == "inference"

    def accept_action(self, action_payload: dict[str, Any]) -> dict[str, Any]:
        from sharpa_interface.server.action import parse_policy_action
        action = parse_policy_action(
            action_payload,
            expected_session_id=self.session_id,
            expected_request_id=self.gate.active_request_id,
        )
        return self.gate.accept_action(action)

    def accept_parsed_action(self, action: Any) -> dict[str, Any]:
        return self.gate.accept_action(action)

    def cancel_request(self) -> None:
        self.gate.cancel_inference()

    def complete(self, done_payload: dict[str, Any]) -> dict[str, Any]:
        return self.gate.complete(done_payload)

    def reset(self) -> None:
        self.gate.reset()


def deployment_ready(ur_state: int, sharpa_state: int, active_source: int) -> bool:
    return int(ur_state) == 2 and int(sharpa_state) == 1 and int(active_source) == 3


def metadata_identity_error(
    metadata: dict[str, Any] | None,
    *,
    expected_task_id: str,
    expected_run_id: str,
    expected_prompt: str = "",
) -> str:
    if not isinstance(metadata, dict):
        return "模型 metadata 缺失"
    actual_task_id = str(metadata.get("task_id") or "")
    actual_run_id = str(metadata.get("run_id") or "")
    actual_prompt = str(metadata.get("prompt") or "")
    mismatches = []
    if not expected_task_id or actual_task_id != expected_task_id:
        mismatches.append(f"task {actual_task_id or '<missing>'} != {expected_task_id or '<missing>'}")
    if not expected_run_id or actual_run_id != expected_run_id:
        mismatches.append(f"run {actual_run_id or '<missing>'} != {expected_run_id or '<missing>'}")
    if expected_prompt and actual_prompt and actual_prompt != expected_prompt:
        mismatches.append("task prompt 不一致")
    if mismatches:
        return "模型身份不匹配：" + "；".join(mismatches)
    return ""


def summarize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    active_format = metadata.get("metadata_format")
    if not isinstance(active_format, dict):
        active_format = {}
    requirements = []

    def add(name: str, selector: Any) -> None:
        if not isinstance(selector, dict):
            return
        history = int(selector.get("history_len", 0))
        current = bool(selector.get("current", False))
        if history or current:
            requirements.append(
                {"name": name, "history": history, "current": current}
            )

    images = active_format.get("image") or {}
    for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
        add(name, images.get(name))
    add("state", active_format.get("state"))
    sensors = active_format.get("sensor") or {}
    for name in ("tau", "wrench", "deformation"):
        add(name, sensors.get(name))
    return {
        "policy_family": str(metadata.get("policy_family") or ""),
        "task_id": str(metadata.get("task_id") or ""),
        "run_id": str(metadata.get("run_id") or ""),
        "checkpoint_id": str(metadata.get("checkpoint_id") or ""),
        "prompt": str(metadata.get("prompt") or ""),
        "format_id": str(active_format.get("format_id") or ""),
        "requirements": requirements,
    }


def main(args=None) -> None:
    import json
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from std_msgs.msg import String
    from ur_msgs.msg import ControlStatus, ExecutableAction, ExecutionDone
    from ur_msgs.srv import BuildObservation
    from sharpa_interface.server.client import ServerClient, ServerConfig
    from sharpa_interface.server.serialization import unpackb
    from sharpa_interface.sharpa import CANONICAL_EEF_FRAME
    from ur_common.geometry import UrSharpAWireGeometry

    class RosPolicyClientNode(Node):
        def __init__(self):
            super().__init__("policy_client_node")
            self.declare_parameter("session_id", "ur-sharpa")
            self.declare_parameter("server_url", "ws://127.0.0.1:5500/infer")
            self.declare_parameter("prompt", "")
            self.declare_parameter("ssh_host", "BAAI")
            self.declare_parameter("ssh_remote_host", "127.0.0.1")
            self.declare_parameter("ssh_remote_port", 5500)
            self._core = PolicyClientCore(str(self.get_parameter("session_id").value))
            self._server_pose = LegacyServerPoseAdapter(UrSharpAWireGeometry())
            self._fallback_prompt = str(self.get_parameter("prompt").value)
            self._server_config = ServerConfig(
                url=str(self.get_parameter("server_url").value),
                configured_prompt="",
                ssh_host=str(self.get_parameter("ssh_host").value),
                ssh_remote_host=str(self.get_parameter("ssh_remote_host").value),
                ssh_remote_port=int(self.get_parameter("ssh_remote_port").value),
            )
            self._default_remote_port = self._server_config.ssh_remote_port
            self._client = ServerClient(self._server_config)
            self._metadata = None; self._started = False; self._busy = False; self._pending = None
            self._control_ready = False
            self._model_available = False
            self._model_generation = -1
            self._model_phase = "stopped"
            self._model_port = self._default_remote_port
            self._expected_task_id = ""
            self._expected_run_id = ""
            self._expected_prompt = ""
            self._model_identity_ok = False
            self._model_identity_error = ""
            self._model_status_signature = None
            self._active_prompt = ""
            self._phase = "waiting"
            self._reason = "等待模型启动"
            self._last_error = ""
            self._inference_count = 0
            self._last_request_id = -1
            self._last_latency_s = None
            self._last_waiting_reason = ""
            self._status = self.create_publisher(String, "/policy/status", 10)
            self._action_pub = self.create_publisher(ExecutableAction, "/policy/executable_action", 10)
            self._build_client = self.create_client(BuildObservation, "/policy/build_observation")
            self.create_subscription(String, "/policy/action_result_json", self._action, 10)
            self.create_subscription(String, "/policy/execution_done_json", self._done, 10)
            self.create_subscription(ExecutionDone, "/policy/execution_done", self._done_msg, 10)
            self.create_subscription(ControlStatus, "/control/status", self._control_status, 10)
            self.create_subscription(String, "/model/status", self._model_status, 10)
            self._timer = self.create_timer(0.1, self._tick)
            self._status_timer = self.create_timer(1.0, lambda: self._publish({}))

        def _publish(self, payload):
            if payload.get("phase"):
                self._phase = str(payload["phase"])
            if "reason" in payload:
                self._reason = str(payload.get("reason") or "")
            if "error" in payload:
                self._last_error = str(payload.get("error") or "")
            result = {
                "phase": self._phase,
                "reason": self._reason,
                "error": self._last_error,
                "session_id": self._core.session_id,
                "model_phase": self._model_phase,
                "model_generation": self._model_generation,
                "model_port": self._model_port,
                "model_identity_ok": self._model_identity_ok,
                "model_identity_error": self._model_identity_error,
                "inference_count": self._inference_count,
                "last_request_id": self._last_request_id,
                "latency_s": self._last_latency_s,
                "metadata": summarize_metadata(self._metadata),
            }
            result.update(payload)
            message = String(); message.data = json.dumps(result, separators=(",", ":")); self._status.publish(message)

        def _model_status(self, message):
            try:
                payload = json.loads(message.data)
                generation = int(payload.get("generation", -1))
                phase = str(payload.get("phase", "stopped"))
                remote_port = int(payload.get("port") or self._default_remote_port)
                expected_task_id = str(payload.get("task_id") or "")
                expected_run_id = str(payload.get("checkpoint_run_id") or "")
                expected_prompt = str(payload.get("prompt") or "")
                signature = (
                    generation,
                    phase,
                    remote_port,
                    expected_task_id,
                    expected_run_id,
                    expected_prompt,
                )
                if signature == self._model_status_signature:
                    return
                self._model_status_signature = signature
                self._model_generation = generation
                self._model_phase = phase
                self._model_port = remote_port
                self._expected_task_id = expected_task_id
                self._expected_run_id = expected_run_id
                self._expected_prompt = expected_prompt
                self._model_available = phase == "running"
                self._client.close()
                self._server_config = replace(
                    self._server_config,
                    ssh_remote_port=remote_port,
                )
                self._client = ServerClient(self._server_config)
                self._metadata = None
                self._model_identity_ok = False
                self._model_identity_error = ""
                self._active_prompt = ""
                self._started = False
                self._busy = False
                self._pending = None
                self._core.reset()
                self._reason = "" if self._model_available else f"模型状态: {phase}"
                self._last_error = ""
                self._publish({"phase": "waiting", "reason": self._reason})
            except Exception as exc:
                self._publish({"phase": "fault", "error": str(exc)})

        def _control_status(self, message):
            self._control_ready = deployment_ready(
                message.ur_state, message.sharpa_state, message.active_source
            )
            reason = "" if self._control_ready else "waiting for UR and SharpA actor control"
            if reason and reason != self._last_waiting_reason:
                self._publish({"phase": "waiting", "reason": reason})
            self._last_waiting_reason = reason

        def _tick(self):
            if self._busy or not self._core.ready: return
            try:
                if not self._model_available: return
                if not self._started:
                    self._metadata = self._client.connect()
                    identity_error = metadata_identity_error(
                        self._metadata,
                        expected_task_id=self._expected_task_id,
                        expected_run_id=self._expected_run_id,
                        expected_prompt=self._expected_prompt,
                    )
                    if identity_error:
                        self._model_identity_ok = False
                        self._model_identity_error = identity_error
                        self._model_available = False
                        self._client.close()
                        self._publish({"phase": "fault", "reason": "", "error": identity_error})
                        return
                    self._model_identity_ok = True
                    self._model_identity_error = ""
                    self._active_prompt = (
                        self._expected_prompt
                        or str(self._metadata.get("prompt") or "")
                        or self._fallback_prompt
                    )
                    self._client.reset(self._core.session_id, request_id=0); self._core.reset(); self._started = True
                    self._reason = ""
                    self._last_error = ""
                    self._publish({"phase": "connected", "metadata_format_id": self._metadata["metadata_format"]["format_id"]}); return
                if not self._control_ready: return
                if not self._build_client.service_is_ready(): return
                pending = self._core.begin(); self._pending = pending
                request = BuildObservation.Request(); request.metadata_format_json = json.dumps(self._metadata["metadata_format"], separators=(",", ":")); request.session_id = self._core.session_id; request.request_id = pending.request_id; request.timestamp_ns = self.get_clock().now().nanoseconds; request.prompt = self._active_prompt; request.execution_feedback_json = json.dumps(pending.feedback, separators=(",", ":")); request.max_message_size = int(self._metadata.get("max_message_size", 64 * 1024 * 1024)); self._busy = True
                self._build_client.call_async(request).add_done_callback(self._observation_done)
            except Exception as exc: self._publish({"phase": "fault", "error": str(exc)})

        def _observation_done(self, future):
            try:
                response = future.result()
                if not response.success:
                    self._core.cancel_request(); self._publish({"phase": "waiting", "error": response.error_message}); return
                if not self._control_ready:
                    self._core.cancel_request(); self._publish({"phase": "waiting", "reason": "control state changed before inference"}); return
                payload = unpackb(bytes(response.observation_msgpack), max_size=int(self._metadata.get("max_message_size", 64 * 1024 * 1024)))
                server_payload = self._server_pose.observation_to_server(payload)
                action, latency = self._client.infer_action(server_payload, expected_session_id=self._core.session_id, expected_request_id=self._pending.request_id)
                if action.next_metadata_format is not None:
                    self._metadata = dict(self._metadata); self._metadata["metadata_format"] = action.next_metadata_format
                command = self._server_pose.action_to_canonical(self._core.accept_parsed_action(action)); execution = command["execution"]; values = command["action"]
                self._inference_count += 1
                self._last_request_id = int(command["request_id"])
                self._last_latency_s = float(latency)
                message = ExecutableAction(); message.session_id = command["session_id"]; message.request_id = command["request_id"]; message.action_id = command["action_id"]; message.revision = command["revision"]; message.timestamp_ns = command["timestamp_ns"]; message.frequency_hz = execution["frequency_hz"]; message.action_length = execution["action_length"]; message.execute_start = execution["execute_start"]; message.execute_length = execution["execute_length"]; message.eef_frame = CANONICAL_EEF_FRAME
                for field, source in (("left_eef", values["left_wrist"]["eef"]), ("right_eef", values["right_wrist"]["eef"])):
                    if source is not None: setattr(message, field, np.asarray(source, dtype=np.float32).reshape(-1).tolist())
                for field, source in (("left_arm_joint", values["left_wrist"]["joint"]), ("right_arm_joint", values["right_wrist"]["joint"])):
                    if source is not None: setattr(message, field, np.asarray(source, dtype=np.float32).reshape(-1).tolist())
                left, right = values["hand_joint"]["left"], values["hand_joint"]["right"]
                if left is not None and right is not None: message.hand_joint = np.concatenate((left, right), axis=-1).reshape(-1).tolist()
                self._action_pub.publish(message); self._publish({"phase": "executing", "action_id": message.action_id, "latency_s": latency, "reason": "", "error": ""})
            except Exception as exc:
                if self._core.inference_pending:
                    self._core.cancel_request()
                self._publish({"phase": "fault", "error": str(exc)})
            finally: self._busy = False

        def _action(self, message):
            try:
                command = self._core.accept_action(json.loads(message.data))
                self._publish({"phase": "executing", "action": command})
            except Exception as exc:
                self._publish({"phase": "fault", "error": str(exc)})

        def _done(self, message):
            try:
                feedback = self._core.complete(json.loads(message.data))
                self._publish({"phase": "ready", "execution_feedback": feedback})
            except Exception as exc:
                self._publish({"phase": "fault", "error": str(exc)})

        def _done_msg(self, message):
            self._done(String(data=json.dumps({"schema": "sharpa_execution_done.v1", "session_id": message.session_id, "request_id": int(message.request_id), "action_id": message.action_id, "revision": int(message.revision), "execute_start": int(message.execute_start), "execute_length": int(message.execute_length), "executed_steps": int(message.executed_steps), "success": bool(message.success), "done": True, "error": message.error or None})))

    rclpy.init(args=args); node = RosPolicyClientNode()
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        node._client.close()
        if rclpy.ok():
            rclpy.shutdown()
