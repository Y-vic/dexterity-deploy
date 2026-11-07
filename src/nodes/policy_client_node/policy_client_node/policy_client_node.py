"""ROS-facing adapter around the frozen interface policy loop."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Any, Mapping

import numpy as np

from .request_metrics import PolicyRequestMetrics


DIRECT_CANONICAL_EEF_SCHEMA = "sharpa62_direct_absolute_eef.v1"
CANONICAL_EEF_POSE_LAYOUT = "xyz,r00,r10,r20,r01,r11,r21"


def activate_reset_metadata(
    server_metadata: Mapping[str, Any], reset_response: Mapping[str, Any]
) -> dict[str, Any]:
    """Activate the metadata format returned by a policy session reset.

    Stateful policies such as T-Rex may switch formats while a rollout is
    running. A reset always returns the format for request 0, which is the
    only authoritative value for the new session.
    """

    metadata_format = reset_response.get("metadata_format")
    if not isinstance(metadata_format, Mapping):
        raise ValueError("policy reset response is missing metadata_format")
    format_id = metadata_format.get("format_id")
    if not isinstance(format_id, str) or not format_id:
        raise ValueError("policy reset metadata_format.format_id must be nonempty")
    activated = dict(server_metadata)
    activated["metadata_format"] = deepcopy(dict(metadata_format))
    return activated


@dataclass(frozen=True)
class PendingRequest:
    request_id: int
    feedback: dict[str, Any]


def validate_server_eef_contract(metadata: dict[str, Any]) -> None:
    """Reject an explicitly non-canonical server contract.

    Every policy uses the same public absolute-EEF ABI. Model-specific
    relative/legacy representations must be resolved inside the model server,
    never in this embodiment client.
    """
    interface = metadata.get("interface")
    if not isinstance(interface, dict) or "eef_contract" not in interface:
        return
    eef_contract = interface["eef_contract"]
    if not isinstance(eef_contract, dict):
        raise ValueError("server interface.eef_contract must be a mapping")
    expected = {
        "schema": DIRECT_CANONICAL_EEF_SCHEMA,
        "observation_eef_def": "absolute",
        "action_eef_def": "absolute",
        "eef_pose_layout": CANONICAL_EEF_POSE_LAYOUT,
        "server_xyz_direction_transform": "none",
    }
    mismatches = [
        f"{key}={eef_contract.get(key)!r}"
        for key, value in expected.items()
        if eef_contract.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "server EEF contract is not the canonical absolute ABI: "
            + ", ".join(mismatches)
        )


class PolicyClientCore:
    """Keep transport-independent inference/action/done identity state."""

    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id must be nonempty")
        from sharpa_interface.server.execution import SyncExecutionGate
        self.session_id = session_id
        self.gate = SyncExecutionGate()
        self._actor_gate_running = False
        self._restart_acknowledged = False

    def begin(self) -> PendingRequest:
        request_id, feedback = self.gate.begin_inference()
        return PendingRequest(request_id, feedback)

    @property
    def ready(self) -> bool:
        return self.gate.phase.value == "ready"

    @property
    def inference_pending(self) -> bool:
        return self.gate.phase.value in {"inference", "prefetching"}

    @property
    def executing(self) -> bool:
        return self.gate.active_action is not None

    @property
    def failed(self) -> bool:
        return self.gate.phase.value == "failed"

    @property
    def phase(self) -> str:
        return self.gate.phase.value

    @property
    def waiting_for_restart(self) -> bool:
        return self._restart_acknowledged and self.failed

    def accept_action(
        self,
        action_payload: dict[str, Any],
        *,
        expected_schema: str = "sharpa_policy_action.v5",
    ) -> dict[str, Any]:
        from sharpa_interface.server.action import parse_policy_action
        action = parse_policy_action(
            action_payload,
            expected_session_id=self.session_id,
            expected_request_id=self.gate.active_request_id,
            expected_schema=expected_schema,
        )
        return self.gate.accept_action(action)

    def accept_parsed_action(self, action: Any) -> dict[str, Any]:
        return self.gate.accept_action(action)

    def prefetch(self, message: Any) -> PendingRequest:
        request_id, feedback = self.gate.begin_prefetch(
            request_id=int(message.request_id),
            action_id=str(message.action_id),
            revision=int(message.revision),
            execute_start=int(message.execute_start),
            execute_length=int(message.execute_length),
            executed_steps=int(message.executed_steps),
        )
        return PendingRequest(request_id, feedback)

    def cancel_request(self) -> None:
        self.gate.cancel_inference()

    def complete(self, done_payload: dict[str, Any]) -> dict[str, Any]:
        feedback = self.gate.complete(done_payload)
        if feedback["success"]:
            self._restart_acknowledged = False
        elif self._restart_acknowledged and self._actor_gate_running:
            # Handles a fast stop -> start sequence where the Actor's failed
            # terminal event arrives after the gate has already reopened.
            self.gate.resume_after_failure()
            self._restart_acknowledged = False
        return feedback

    def update_actor_gate(self, running: bool) -> bool:
        """Track explicit stop/start edges and recover only after a restart.

        Pause deliberately keeps this gate running, so it never enters this
        path and the active chunk remains owned by the synchronous gate.
        """
        running = bool(running)
        if self._actor_gate_running and not running and (
            self.executing or self.failed
        ):
            self._restart_acknowledged = True
        self._actor_gate_running = running
        if running and self._restart_acknowledged and self.failed:
            self.gate.resume_after_failure()
            self._restart_acknowledged = False
            return True
        return False

    def reset(self) -> None:
        self.gate.reset()
        self._actor_gate_running = False
        self._restart_acknowledged = False


def completion_belongs_to_session(message: Any, session_id: str) -> bool:
    """Keep another execution provider's terminal events out of the policy gate."""
    return bool(session_id) and str(message.session_id) == str(session_id)


def action_period_deadline(published_at: float, frequency_hz: float) -> float:
    """Earliest time a published action's final native step is complete.

    Hardware ACKs only confirm that ``servoJ``/the hand command was accepted;
    they arrive in a few milliseconds even though the command owns a full
    ``1 / frequency_hz`` motion period.  Sampling the next observation at ACK
    time therefore observes a step that is still moving and turns a nominally
    synchronous one-step rollout into a delayed feedback loop.
    """
    published = float(published_at)
    frequency = float(frequency_hz)
    if not np.isfinite(published):
        raise ValueError("published_at must be finite")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    return published + 1.0 / frequency


def deployment_ready(
    ur_state: int,
    sharpa_state: int,
    active_source: int,
    execution_paused: bool = False,
    execution_dry_run: bool = False,
) -> bool:
    return (
        int(active_source) == 3
        and (
            bool(execution_dry_run)
            or (int(ur_state) == 2 and int(sharpa_state) == 1)
        )
        and not execution_paused
    )


def metadata_identity_error(
    metadata: dict[str, Any] | None,
    *,
    expected_model: str = "",
    expected_task_id: str,
    expected_run_id: str,
    expected_prompt: str = "",
    expected_checkpoint_path: str = "",
) -> str:
    if not isinstance(metadata, dict):
        return "模型 metadata 缺失"
    actual_model = str(
        metadata.get("model_id") or metadata.get("model_name") or ""
    )
    actual_task_id = str(metadata.get("task_id") or "")
    actual_run_id = str(metadata.get("run_id") or "")
    actual_prompt = str(metadata.get("prompt") or "")
    mismatches = []
    if not expected_model or actual_model != expected_model:
        mismatches.append(
            "model " f"{actual_model or '<missing>'} != {expected_model or '<missing>'}"
        )
    if not expected_task_id or actual_task_id != expected_task_id:
        mismatches.append(f"task {actual_task_id or '<missing>'} != {expected_task_id or '<missing>'}")
    if not expected_run_id or actual_run_id != expected_run_id:
        mismatches.append(f"run {actual_run_id or '<missing>'} != {expected_run_id or '<missing>'}")
    if expected_prompt and actual_prompt != expected_prompt:
        mismatches.append("task prompt 不一致")
    if expected_checkpoint_path and str(metadata.get("checkpoint_path") or "") != expected_checkpoint_path:
        mismatches.append("checkpoint_path 不一致")
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
        "model_id": str(metadata.get("model_id") or metadata.get("model_name") or ""),
        "model_name": str(metadata.get("model_name") or ""),
        "task_id": str(metadata.get("task_id") or ""),
        "run_id": str(metadata.get("run_id") or ""),
        "checkpoint_id": str(metadata.get("checkpoint_id") or ""),
        "prompt": str(metadata.get("prompt") or ""),
        "format_id": str(active_format.get("format_id") or ""),
        "requirements": requirements,
        "policy_transport": str(metadata.get("policy_transport") or "legacy"),
        "native_interface": deepcopy(metadata.get("native_interface")),
        "interface_revision": metadata.get("interface_revision"),
        "pending_interface_update": deepcopy(metadata.get("pending_interface_update")),
    }


def main(args=None) -> None:
    import json
    import time
    import uuid
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String, Bool
    from rclpy.qos import QoSProfile
    from ur_msgs.msg import (
        ControlStatus,
        DeformationFrame,
        ExecutableAction,
        ExecutionDone,
        HandStateFrame,
        TauFrame,
        UrStateFrame,
        WrenchFrame,
    )
    from sharpa_interface.server.client import ServerClient, ServerConfig
    from sharpa_interface.sharpa import CANONICAL_EEF_FRAME
    from ur_common.protection_config import setting
    from ur_common.policy_inference import inference_json
    from ur_common.execution_log import execution_event_json
    from ur_common.observation_collector import ObservationCollector
    from ur_common.policy_observation import ObservationNotReady

    class RosPolicyClientNode(Node):
        def __init__(self):
            super().__init__("policy_client_node")
            self.declare_parameter("session_id", "ur-sharpa")
            self.declare_parameter("server_url", "ws://127.0.0.1:5500/infer")
            self.declare_parameter("prompt", "")
            self.declare_parameter("ssh_host", "BAAI")
            self.declare_parameter("ssh_remote_host", "127.0.0.1")
            self.declare_parameter("ssh_remote_port", 5500)
            self.declare_parameter("dfm_python", "/home/yns/deploy/runtime/venvs/dfm-serving/bin/python")
            self.declare_parameter("dfm_source_root", "/home/yns/deploy/runtime/dfm-source")
            self.declare_parameter("dfm_calibration_path", "/home/yns/deploy/runtime/config/dfm_calibration.json")
            self.declare_parameter("dfm_data_contract_path", "/home/yns/deploy/runtime/config/real101_data_contract.json")
            from concurrent.futures import ThreadPoolExecutor
            from ur_common.camera_transport import SharedCameraReader
            from ur_common.raw_camera import RAW_ENCODING
            from ur_common.remote_observation import RemoteCameraEncoder
            self.declare_parameter("camera_shm_prefix", "ur_sharpa_camera_v1")
            self.declare_parameter("enable_hil", False)
            self._hil = {}
            self._policy_recording = False
            self._execution_dry_run = False
            self._recording_status_seen = 0.0
            self._recording_epoch_floor = 0
            self._recording_rollout_id = ''
            self._retired_recording_ids = set()
            self._epoch = 0
            self._inference_future = None
            self._service_future = None
            self._rpc_faulted = False
            self._inference_context = None
            self._completion_lock = Lock()
            self._completion_closed = False
            self._inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='remote-inference')
            self._raw_reader = SharedCameraReader(prefix=str(self.get_parameter('camera_shm_prefix').value)+'_raw',encoding=RAW_ENCODING)
            self._camera_encoder = RemoteCameraEncoder()
            self._base_session_id = str(self.get_parameter("session_id").value)
            self._core = PolicyClientCore(self._base_session_id)
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
            self._policy_transport = "legacy"
            self._interface_update_error = ""
            self._metadata = None; self._started = False; self._busy = False; self._pending = None
            self._control_ready = False
            self._machine_control_ready = False
            self._actor_gate_running = False
            self._rollout_reset_pending = False
            self._rollout_generation = 0
            self._execution_provider = "policy"
            self._model_available = False
            self._model_generation = -1
            self._model_phase = "stopped"
            self._model_port = self._default_remote_port
            self._expected_model = ""
            self._expected_task_id = ""
            self._expected_run_id = ""
            self._expected_prompt = ""
            self._expected_checkpoint_path = ""
            self._model_identity_ok = False
            self._model_identity_error = ""
            self._model_status_signature = None
            self._active_prompt = ""
            self._phase = "waiting"
            self._reason = "等待模型启动"
            self._last_error = ""
            self._request_metrics = PolicyRequestMetrics()
            self._last_status_published_at = 0.0
            self._last_waiting_reason = ""
            # One-step actions are not physically complete when the hardware
            # ACK arrives: the ACK only says the asynchronous command was
            # accepted.  Track the end of the native action period so the next
            # observation is sampled after the step, not while it is moving.
            self._action_period_deadlines = {}
            self._next_observation_not_before = 0.0
            self._observations = ObservationCollector()
            self._observation_group = MutuallyExclusiveCallbackGroup()
            self._status = self.create_publisher(String, "/policy/status", 10)
            self._execution_event_pub = self.create_publisher(
                String, "/deploy/execution_event", 100
            )
            self._inference_pub = self.create_publisher(String, "/policy/inference", 256)
            self._action_pub = self.create_publisher(ExecutableAction, "/policy/executable_action", 10)
            self.create_subscription(String, "/policy/action_result_json", self._action, 10)
            self.create_subscription(String, "/policy/execution_done_json", self._done, 10)
            self.create_subscription(ExecutionDone, "/policy/execution_done", self._done_msg, 10)
            self.create_subscription(ControlStatus, "/control/status", self._control_status, 10)
            self.create_subscription(
                String,
                "/control/execution_provider",
                self._provider_status,
                10,
            )
            self.create_subscription(String, "/model/status", self._model_status, 10)
            self.create_subscription(String, "/policy/interface_update", self._interface_update, 10)
            self.create_subscription(
                UrStateFrame,
                "/ur/state",
                self._ur,
                10,
                callback_group=self._observation_group,
            )
            self.create_subscription(
                HandStateFrame,
                "/sharpa/state",
                self._hand,
                10,
                callback_group=self._observation_group,
            )
            self.create_subscription(
                TauFrame,
                "/sharpa/tau",
                self._tau,
                10,
                callback_group=self._observation_group,
            )
            self.create_subscription(
                WrenchFrame,
                "/sharpa/wrench",
                self._wrench,
                10,
                callback_group=self._observation_group,
            )
            self.create_subscription(
                DeformationFrame,
                "/sharpa/deformation",
                self._deformation,
                1,
                callback_group=self._observation_group,
            )
            self._completion_guard = self.create_guard_condition(self._tick)
            self._timer = self.create_timer(0.1, self._tick)
            self.create_timer(1/30, self._raw_cameras, callback_group=self._observation_group)
            self.create_subscription(String, '/hil/state', self._hil_status, 10)
            self.create_subscription(Bool, '/control/policy_recording', self._policy_recording_status,
                                     QoSProfile(depth=1))
            self._status_timer = self.create_timer(0.1, self._publish_timing_status)

        def _submit_rpc(self, call, *args, **kwargs):
            future = self._inference_pool.submit(call, *args, **kwargs)
            future.add_done_callback(self._rpc_completed)
            return future

        def _rpc_completed(self, future):
            # Worker threads only wake the ROS executor. Existing _tick checks
            # still own epoch/session validation and all publication.
            with self._completion_lock:
                if not self._completion_closed:
                    self._completion_guard.trigger()

        def _close_completion_wakeups(self):
            with self._completion_lock:
                self._completion_closed = True

        def _execution_event(self, event, **fields):
            try:
                message = String()
                message.data = execution_event_json(
                    "policy_client_node", event, **fields
                )
                self._execution_event_pub.publish(message)
            except Exception as exc:
                self.get_logger().warning(f"failed to publish execution event: {exc}")

        def _inference_event(self, event, **fields):
            if self._inference_context is None:
                return
            payload = dict(self._inference_context, event=event, **fields)
            if event != 'requested':
                payload['completed_unix_ns'] = time.time_ns()
            self._inference_pub.publish(String(data=json.dumps(inference_json(payload))))
            if event != 'requested':
                self._inference_context = None

        def _invalidate_request(self, reason):
            self._epoch += 1
            self._request_metrics.cancel()
            if self._inference_future is not None:
                self._inference_event('cancelled', error=reason)
                # A stateful server may already have advanced its metadata or
                # history. Resume only after resetting once the active chunk ends.
                self._rollout_reset_pending = True
                self._control_ready = False
            if self._service_future is not None and self._service_context[0] == 'interface':
                # An accepted proposal may have reached the native owner even
                # when its acknowledgment belongs to a retired control epoch.
                self._rollout_reset_pending = True
                self._control_ready = False
            self._publish({})

        def _publish(self, payload):
            payload = dict(payload, hil=self._hil, rollout_recording=self._hil.get("rollout_recording", {}),
                           learning_command=self._hil.get("learning_command", {}))
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
                "execution_provider": self._execution_provider,
                "policy_transport": self._policy_transport,
                "interface_update_error": self._interface_update_error,
                "native_interface": (self._metadata or {}).get("native_interface"),
                "interface_revision": (self._metadata or {}).get("interface_revision"),
                "pending_interface_update": (self._metadata or {}).get("pending_interface_update"),
                **self._request_metrics.snapshot(),
                "metadata": summarize_metadata(self._metadata),
                "buffers": self._observations.status(),
            }
            result.update(payload)
            message = String(); message.data = json.dumps(result, separators=(",", ":")); self._status.publish(message)
            self._last_status_published_at = time.monotonic()

        def _publish_timing_status(self):
            if (
                self._request_metrics.inference_pending
                or time.monotonic() - self._last_status_published_at >= 1.0
            ):
                self._publish({})

        def _collect(self, source, operation, *args, **kwargs):
            try:
                operation(*args, **kwargs)
            except Exception as exc:
                self.get_logger().warning(f"{source} buffer rejected: {exc}")

        def _ur(self, message):
            self._collect(
                "state",
                self._observations.push_ur,
                timestamp_ns=message.timestamp_ns,
                left_joint=message.left_joint,
                right_joint=message.right_joint,
                left_eef=message.left_eef,
                right_eef=message.right_eef,
                eef_frame=message.eef_frame,
                valid=message.valid,
            )

        def _hand(self, message):
            self._collect(
                "state",
                self._observations.push_hand,
                timestamp_ns=message.timestamp_ns,
                joints=message.q,
                left_valid=message.left_valid,
                right_valid=message.right_valid,
            )

        def _tau(self, message):
            self._collect(
                "tau",
                self._observations.push_tau,
                timestamp_ns=message.timestamp_ns,
                left=message.left,
                right=message.right,
                left_valid=message.left_valid,
                right_valid=message.right_valid,
            )

        def _wrench(self, message):
            self._collect(
                "wrench",
                self._observations.push_wrench,
                timestamp_ns=message.timestamp_ns,
                left=message.left,
                right=message.right,
                left_valid=message.left_valid,
                right_valid=message.right_valid,
            )

        def _deformation(self, message):
            self._collect(
                "deformation",
                self._observations.push_deformation,
                timestamp_ns=message.timestamp_ns,
                left=message.left,
                right=message.right,
                left_valid=message.left_valid,
                right_valid=message.right_valid,
            )

        def _model_status(self, message):
            try:
                payload = json.loads(message.data)
                generation = int(payload.get("generation", -1))
                phase = str(payload.get("phase", "stopped"))
                remote_port = int(payload.get("port") or self._default_remote_port)
                expected_task_id = str(payload.get("task_id") or "")
                expected_model = str(payload.get("model") or "")
                expected_run_id = str(payload.get("run_id") or "")
                expected_prompt = str(payload.get("prompt") or "")
                expected_checkpoint_path = str(payload.get("checkpoint_path") or "")
                server_url = str(payload.get("server_url") or self._server_config.url)
                ssh_host = str(payload.get("ssh_host") or "")
                remote_host = str(payload.get("remote_host") or "127.0.0.1")
                policy_transport = str(payload.get("policy_transport") or "legacy")
                if policy_transport not in {"legacy", "dfm_native"}:
                    raise ValueError("Unsupported model policy_transport")
                signature = (
                    generation,
                    phase,
                    remote_port,
                    expected_model,
                    expected_task_id,
                    expected_run_id,
                    expected_prompt,
                    expected_checkpoint_path,
                    server_url,
                    ssh_host,
                    remote_host,
                    policy_transport,
                )
                if signature == self._model_status_signature:
                    return
                self._model_status_signature = signature
                self._model_generation = generation
                self._model_phase = phase
                self._model_port = remote_port
                self._expected_model = expected_model
                self._expected_task_id = expected_task_id
                self._expected_run_id = expected_run_id
                self._expected_prompt = expected_prompt
                self._expected_checkpoint_path = expected_checkpoint_path
                self._model_available = phase == "running"
                self._execution_event(
                    "model_status_changed",
                    generation=generation,
                    phase=phase,
                    model_port=remote_port,
                    model=expected_model,
                    server_id=str(payload.get("server_id") or ""),
                    task_id=expected_task_id,
                    run_id=expected_run_id,
                    detail=str(payload.get("detail") or ""),
                    log_tail=str(payload.get("log_tail") or "")[-12000:],
                    success=phase != "fault",
                )
                self._invalidate_request("model changed")
                self._client.close()
                self._server_config = replace(
                    self._server_config,
                    url=server_url,
                    ssh_host=ssh_host,
                    ssh_remote_host=remote_host,
                    ssh_remote_port=remote_port,
                )
                self._policy_transport = policy_transport
                if policy_transport == "dfm_native":
                    from sharpa_interface.dfm.client import NativePolicyClient
                    self._client = NativePolicyClient(self._server_config,
                        python=str(self.get_parameter("dfm_python").value),
                        source_root=str(self.get_parameter("dfm_source_root").value),
                        calibration_path=str(self.get_parameter("dfm_calibration_path").value),
                        data_contract_path=str(self.get_parameter("dfm_data_contract_path").value))
                else:
                    self._client = ServerClient(self._server_config)
                self._interface_update_error = ""
                self._metadata = None
                self._model_identity_ok = False
                self._model_identity_error = ""
                self._active_prompt = ""
                self._started = False
                self._rpc_faulted = False
                self._busy = False
                self._pending = None
                self._core = PolicyClientCore(self._base_session_id)
                self._actor_gate_running = False
                self._rollout_reset_pending = False
                self._rollout_generation = 0
                self._request_metrics.reset()
                self._reason = "" if self._model_available else f"模型状态: {phase}"
                self._last_error = ""
                self._publish({"phase": "waiting", "reason": self._reason})
            except Exception as exc:
                self._publish({"phase": "fault", "error": str(exc)})

        def _policy_recording_status(self, message):
            if bool(message.data) != self._policy_recording:
                self._invalidate_request('recording mode changed')
            self._policy_recording = bool(message.data)

        def _recording_ready(self):
            recording = self._hil.get('rollout_recording', {})
            required = not self._execution_dry_run and (
                self._policy_recording or self._execution_provider == 'hil' or recording.get('required'))
            if not required:
                return True
            rollout_id = str(recording.get('rollout_id') or '')
            return bool(
                recording.get('requested') and recording.get('recording') and rollout_id
                and rollout_id not in self._retired_recording_ids
                and int(self._hil.get('control_timestamp_ns') or 0) >= self._recording_epoch_floor
                and time.monotonic() - self._recording_status_seen < setting('recording', 'heartbeat_timeout_s'))

        def _control_status(self, message):
            machine_ready = deployment_ready(
                message.ur_state, message.sharpa_state, message.active_source,
                message.execution_paused, message.execution_dry_run)
            self._execution_dry_run = bool(message.execution_dry_run)
            if machine_ready and not self._machine_control_ready:
                self._recording_epoch_floor = int(message.timestamp_ns)
            elif self._machine_control_ready and not machine_ready:
                if self._recording_rollout_id:
                    self._retired_recording_ids.add(self._recording_rollout_id)
                self._recording_rollout_id = ''
            actor_gate_running = bool(
                int(message.active_source) == 3
                and (
                    bool(message.execution_dry_run)
                    or (
                        int(message.ur_state) == 2
                        and int(message.sharpa_state) == 1
                    )
                )
                and (self._execution_provider in {"policy", "hil"} and not self._hil.get("active"))
            )
            if actor_gate_running != self._actor_gate_running or bool(message.execution_paused) != getattr(self, '_paused', False):
                self._invalidate_request('Actor control changed')
            self._paused = bool(message.execution_paused)
            rollout_started = actor_gate_running and not self._actor_gate_running
            self._actor_gate_running = actor_gate_running
            recovered = self._core.update_actor_gate(actor_gate_running)
            self._machine_control_ready = deployment_ready(
                message.ur_state,
                message.sharpa_state,
                message.active_source,
                message.execution_paused,
                message.execution_dry_run,
            )
            self._control_ready = bool(
                self._machine_control_ready
                and (self._execution_provider in {"policy", "hil"} and not self._hil.get("active"))
                and not self._rollout_reset_pending
            )
            if rollout_started:
                self._rpc_faulted = False
                self._rollout_reset_pending = True
                self._control_ready = False
                self._publish(
                    {
                        "phase": "resetting",
                        "reason": "new rollout: resetting server history",
                    }
                )
            if self._rollout_reset_pending:
                reason = "new rollout: resetting server history"
            elif self._control_ready:
                reason = ""
            elif (self._execution_provider not in {"policy", "hil"} or self._hil.get("active")):
                reason = "policy disabled: replay provider selected"
            elif message.execution_paused:
                reason = "policy paused: waiting for continue"
            else:
                reason = "waiting for live or preview Actor control"
            if reason and reason != self._last_waiting_reason:
                self._publish({"phase": "waiting", "reason": reason})
            elif recovered:
                self._publish({"phase": "ready", "reason": "", "error": ""})
            self._last_waiting_reason = reason

        def _provider_status(self, message):
            provider = str(message.data).strip().lower()
            if provider not in {"policy", "replay", "hil"}:
                return
            if provider != self._execution_provider:
                self._invalidate_request('execution provider changed')
            self._execution_provider = provider
            self._control_ready = bool(
                self._machine_control_ready and (provider in {"policy", "hil"} and not self._hil.get("active"))
                and not self._rollout_reset_pending
            )
            if (provider not in {"policy", "hil"} or self._hil.get("active")):
                self._publish(
                    {"phase": "waiting", "reason": "policy disabled: replay provider selected"}
                )

        def _request_observation(self, pending):
            if self._busy:
                raise RuntimeError("another policy request is already in flight")
            self._pending = pending
            self._busy = True
            self._execution_event(
                "observation_build_requested",
                session_id=self._core.session_id,
                request_id=int(pending.request_id),
                model_generation=int(self._model_generation),
                model_port=int(self._model_port),
            )
            try:
                payload = self._observations.build(
                    self._metadata["metadata_format"],
                    session_id=self._core.session_id,
                    request_id=pending.request_id,
                    timestamp_ns=self.get_clock().now().nanoseconds,
                    prompt=self._active_prompt,
                    execution_feedback=pending.feedback,
                    max_message_size=int(
                        self._metadata.get(
                            "max_message_size", 64 * 1024 * 1024
                        )
                    ),
                )
            except ObservationNotReady as exc:
                self._execution_event(
                    "observation_build_failed",
                    session_id=self._core.session_id,
                    request_id=int(pending.request_id),
                    success=False,
                    retryable=True,
                    error=str(exc),
                )
                self._core.cancel_request()
                self._busy = False
                self._publish({"phase": "waiting", "error": str(exc)})
                return
            except Exception as exc:
                self._execution_event(
                    "observation_build_failed",
                    session_id=self._core.session_id,
                    request_id=int(pending.request_id),
                    success=False,
                    retryable=False,
                    error=str(exc),
                )
                self._core.cancel_request()
                self._busy = False
                self._publish({"phase": "waiting", "error": str(exc)})
                return
            self._execution_event(
                "observation_build_completed",
                session_id=self._core.session_id,
                request_id=int(pending.request_id),
                success=True,
            )
            self._request_epoch = self._epoch
            self._recording_rollout_id = str(self._hil.get('rollout_recording', {}).get('rollout_id') or '')
            self._inference_context = dict(
                schema='ur_sharpa_policy_inference.v1', request_id=uuid.uuid4().hex,
                session_id=self._core.session_id, policy_request_id=int(pending.request_id),
                model_generation=self._model_generation,
                rollout_id=str(self._hil.get('rollout_recording', {}).get('rollout_id') or ''),
                submitted_unix_ns=time.time_ns(),
                metadata_format=deepcopy(self._metadata['metadata_format']),
                execution_feedback=deepcopy(pending.feedback))
            self._inference_event('requested')
            self._execution_event('model_inference_started',
                session_id=self._core.session_id, request_id=int(pending.request_id),
                model_generation=int(self._model_generation), model_port=int(self._model_port))
            try:
                self._request_metrics.begin(
                    self._core.session_id, pending.request_id, self._request_epoch
                )
                self._inference_future = self._submit_rpc(
                    self._client.infer_action, payload,
                    expected_session_id=self._core.session_id,
                    expected_request_id=pending.request_id)
                self._publish({'phase': self._core.phase, 'reason': '', 'error': ''})
            except Exception as exc:
                self._request_metrics.cancel()
                self._inference_event('failed', error=str(exc))
                self._core.cancel_request()
                self._busy = False
                raise

        def _reset_rollout(self):
            self._rollout_generation += 1
            session_id = (
                f"{self._base_session_id}-m{self._model_generation}"
                f"-r{self._rollout_generation}"
            )
            self._service_context = ('reset', self._client, self._epoch, session_id)
            self._service_future = self._submit_rpc(self._client.reset, session_id, request_id=0)

        def _interface_update(self, message):
            try:
                if self._policy_transport != "dfm_native" or not self._started:
                    raise ValueError("Interface proposals require a connected native DFM policy")
                if self._inference_future is not None or self._service_future is not None:
                    raise ValueError("Wait for the current policy operation before proposing an interface")
                value = json.loads(message.data)
                if set(value) != {"model_generation", "session_id", "interface_revision", "interface"}:
                    raise ValueError("Proposal requires model_generation, session_id, interface_revision and full interface")
                if type(value["model_generation"]) is not int or value["model_generation"] != self._model_generation:
                    raise ValueError("Proposal belongs to a retired model generation")
                if value["session_id"] != self._core.session_id:
                    raise ValueError("Proposal belongs to a retired policy session")
                if type(value["interface_revision"]) is not int or value["interface_revision"] != self._metadata.get("interface_revision"):
                    raise ValueError("Proposal interface_revision differs from the active native declaration")
                self._interface_update_error = ""
                self._service_context = ('interface', self._client, self._epoch, self._core.session_id)
                self._interface_proposal_generation = self._model_generation
                self._service_future = self._submit_rpc(self._client.update_interface, value["interface"])
            except Exception as exc:
                self._interface_update_error = str(exc)
                self._publish({})

        def _finish_service(self):
            future, self._service_future = self._service_future, None
            kind, client, epoch, session_id = self._service_context
            if client is not self._client:
                return
            if kind == 'interface':
                if (epoch != self._epoch or session_id != self._core.session_id
                        or self._interface_proposal_generation != self._model_generation):
                    self._interface_update_error = "Interface acknowledgment belongs to a retired model or rollout"
                    self._rollout_reset_pending = True
                    self._control_ready = False
                    self._publish({})
                    return
                try:
                    future.result()
                    self._metadata = deepcopy(client.current_metadata)
                    self._interface_update_error = ""
                except Exception as exc:
                    # A rejected declaration does not invalidate model history.
                    self._interface_update_error = str(exc)
                self._publish({})
                return
            if kind == 'reset' and epoch != self._epoch:
                # Stop/pause/provider changes while reset was outstanding.
                self._rollout_reset_pending = True
                return
            result = future.result()
            if kind == 'connect':
                self._metadata = result
                validate_server_eef_contract(self._metadata)
                identity_error = metadata_identity_error(
                    self._metadata, expected_model=self._expected_model,
                    expected_task_id=self._expected_task_id, expected_run_id=self._expected_run_id,
                    expected_prompt=self._expected_prompt,
                    expected_checkpoint_path=(self._expected_checkpoint_path if self._policy_transport == "dfm_native" else ""))
                if self._policy_transport == "dfm_native" and (not self._expected_prompt or not self._expected_checkpoint_path):
                    identity_error = "Native model status requires exact prompt and checkpoint_path"
                if identity_error:
                    self._model_identity_ok = False
                    self._model_identity_error = identity_error
                    self._model_available = False
                    self._client.close()
                    self._publish({'phase': 'fault', 'reason': '', 'error': identity_error})
                    return
                self._model_identity_ok = True
                self._model_identity_error = ''
                self._active_prompt = self._expected_prompt or str(result.get('prompt') or '') or self._fallback_prompt
                self._started = True
                self._rollout_reset_pending = True
                self._control_ready = False
                self._publish({'phase': 'connected', 'reason': '', 'error': ''})
                return
            if self._policy_transport == "dfm_native":
                self._metadata = deepcopy(client.current_metadata)
            self._metadata = activate_reset_metadata(self._metadata, result)
            self._core = PolicyClientCore(session_id)
            self._core.update_actor_gate(self._actor_gate_running)
            self._pending = None
            self._busy = False
            self._action_period_deadlines.clear()
            self._next_observation_not_before = 0.0
            self._request_metrics.cancel()
            self._rollout_reset_pending = False
            self._control_ready = bool(
                self._machine_control_ready
                and (self._execution_provider in {'policy', 'hil'} and not self._hil.get('active')))
            self._execution_event('rollout_reset', session_id=session_id, request_id=0,
                model_generation=int(self._model_generation), rollout_generation=int(self._rollout_generation))
            self._publish({'phase': 'ready', 'reason': '', 'error': '',
                'rollout_generation': self._rollout_generation})

        def _raw_cameras(self):
            try:
                for name, stamp, data in self._camera_encoder.encode(self._raw_reader.read()):
                    self._collect(name, self._observations.push_camera, name,
                                  timestamp_ns=stamp, data=data, valid=True)
                if self._camera_encoder.errors != getattr(self, '_camera_errors', {}):
                    self._camera_errors = dict(self._camera_encoder.errors)
                    if self._camera_errors:
                        self.get_logger().warning('policy raw cameras rejected: ' + str(self._camera_errors))
            except Exception as exc:
                self.get_logger().warning('policy raw camera: ' + str(exc))

        def _hil_status(self, message):
            try:
                value = json.loads(message.data)
                if not isinstance(value, dict):
                    return
                changed = bool(value.get('active')) != bool(self._hil.get('active'))
                self._hil = value
                self._recording_status_seen = time.monotonic()
                if changed:
                    self._invalidate_request('human takeover changed')
                    self._control_ready = False
                    self._rollout_reset_pending = not value.get('active')
                    self._core = PolicyClientCore(self._base_session_id)
                    self._actor_gate_running = False
            except (ValueError, TypeError):
                pass

        def _tick(self):
            if self._service_future is not None:
                if not self._service_future.done():
                    return
                try:
                    self._finish_service()
                except Exception as exc:
                    self._rpc_faulted = True
                    self._publish({'phase': 'fault', 'error': str(exc)})
                return
            if self._inference_future is not None:
                if not self._inference_future.done(): return
                future, self._inference_future = self._inference_future, None
                if self._request_epoch != self._epoch or not self._control_ready:
                    self._request_metrics.cancel()
                    self._inference_event('cancelled', error='control state changed during inference')
                    self._busy = False
                    self._pending = None
                    if self._core.inference_pending:
                        self._core.cancel_request()
                    return
                self._inference_result = future
                self._infer_observation(None)
                return
            if self._busy or self._rpc_faulted: return
            try:
                if (self._execution_provider not in {"policy", "hil"} or self._hil.get("active")): return
                if not self._model_available: return
                if not self._started:
                    self._service_context = ('connect', self._client, self._epoch, self._core.session_id)
                    self._service_future = self._submit_rpc(self._client.connect)
                    return
                if self._rollout_reset_pending:
                    if not self._actor_gate_running or not self._machine_control_ready or self._core.executing:
                        return
                    self._reset_rollout()
                    return
                if not self._core.ready: return
                if not self._control_ready: return
                if not self._recording_ready():
                    self._publish({'phase': 'waiting', 'reason': 'waiting for rollout recording acknowledgement'})
                    return
                if time.monotonic() < self._next_observation_not_before: return
                self._request_observation(self._core.begin())
            except Exception as exc:
                self._rpc_faulted = True
                self._execution_event(
                    "policy_tick_failed",
                    request_id=(
                        int(self._pending.request_id)
                        if self._pending is not None
                        else None
                    ),
                    success=False,
                    error=str(exc),
                )
                self._publish({"phase": "fault", "error": str(exc)})

        def _infer_observation(self, payload):
            try:
                if not self._control_ready or (self._execution_provider not in {"policy", "hil"} or self._hil.get("active")):
                    self._execution_event(
                        "inference_cancelled_before_request",
                        session_id=self._core.session_id,
                        request_id=int(self._pending.request_id),
                        success=False,
                        error="control state changed before inference",
                    )
                    self._request_metrics.cancel()
                    self._core.cancel_request(); self._publish({"phase": "waiting", "reason": "control state changed before inference"}); return
                action, latency = self._inference_result.result()
                if self._policy_transport == "dfm_native":
                    self._metadata = deepcopy(self._client.current_metadata)
                self._inference_event('completed', result=asdict(action), latency_s=float(latency))
                self._execution_event(
                    "model_inference_completed",
                    session_id=self._core.session_id,
                    request_id=int(self._pending.request_id),
                    latency_ms=float(latency) * 1000.0,
                    diagnostics=inference_json(action.diagnostics),
                    success=True,
                )
                if action.next_metadata_format is not None:
                    self._metadata = dict(self._metadata); self._metadata["metadata_format"] = action.next_metadata_format
                command = self._core.accept_parsed_action(action); execution = command["execution"]; values = command["action"]
                self._request_metrics.complete(
                    command["session_id"], command["request_id"], self._request_epoch, latency
                )
                message = ExecutableAction(); message.session_id = command["session_id"]; message.request_id = command["request_id"]; message.action_id = command["action_id"]; message.revision = command["revision"]; message.timestamp_ns = command["timestamp_ns"]; message.frequency_hz = execution["frequency_hz"]; message.action_length = execution["action_length"]; message.execute_start = execution["execute_start"]; message.execute_length = execution["execute_length"]; message.rtc = execution["rtc"]; message.eef_frame = CANONICAL_EEF_FRAME
                for field, source in (("left_eef", values["left_wrist"]["eef"]), ("right_eef", values["right_wrist"]["eef"])):
                    if source is not None: setattr(message, field, np.asarray(source, dtype=np.float32).reshape(-1).tolist())
                for field, source in (("left_arm_joint", values["left_wrist"]["joint"]), ("right_arm_joint", values["right_wrist"]["joint"])):
                    if source is not None: setattr(message, field, np.asarray(source, dtype=np.float32).reshape(-1).tolist())
                left, right = values["hand_joint"]["left"], values["hand_joint"]["right"]
                if left is not None and right is not None: message.hand_joint = np.concatenate((left, right), axis=-1).reshape(-1).tolist()
                self._action_pub.publish(message)
                action_key = (
                    str(message.session_id),
                    str(message.action_id),
                    int(message.revision),
                )
                self._action_period_deadlines[action_key] = action_period_deadline(
                    time.monotonic(), float(message.frequency_hz)
                )
                self._execution_event(
                    "policy_action_published",
                    session_id=str(message.session_id),
                    request_id=int(message.request_id),
                    action_id=str(message.action_id),
                    revision=int(message.revision),
                    execute_start=int(message.execute_start),
                    execute_length=int(message.execute_length),
                    rtc=int(message.rtc),
                    frequency_hz=float(message.frequency_hz),
                )
                self._publish({"phase": "executing", "action_id": message.action_id, "latency_s": latency, "reason": "", "error": ""})
            except Exception as exc:
                self._rpc_faulted = True
                self._request_metrics.cancel()
                self._inference_event('failed', error=str(exc))
                self._execution_event(
                    "policy_inference_pipeline_failed",
                    session_id=self._core.session_id,
                    request_id=(
                        int(self._pending.request_id)
                        if self._pending is not None
                        else None
                    ),
                    success=False,
                    error=str(exc),
                )
                if self._core.inference_pending:
                    self._core.cancel_request()
                self._publish({"phase": "fault", "error": str(exc)})
            finally:
                self._request_metrics.cancel()
                self._busy = False

        def _action(self, message):
            try:
                command = self._core.accept_action(
                    json.loads(message.data),
                    expected_schema=str(
                        (self._metadata or {}).get("action_schema")
                        or "sharpa_policy_action.v5"
                    ),
                )
                self._publish({"phase": "executing", "action": command})
            except Exception as exc:
                self._publish({"phase": "fault", "error": str(exc)})

        def _done(self, message):
            try:
                payload = json.loads(message.data)
                # A full STOP may unload/reset the model before Actor's aborted
                # chunk terminal event reaches this callback.  That event is
                # stale once the synchronous gate no longer owns an execution.
                if not self._core.executing:
                    return
                feedback = self._core.complete(payload)
                waiting_for_restart = self._core.waiting_for_restart
                self._publish({
                    "phase": self._core.phase,
                    "reason": (
                        "execution stopped; start again to request a new chunk"
                        if waiting_for_restart
                        else ""
                    ),
                    "error": (
                        str(payload.get("error") or "execution failed")
                        if self._core.failed and not waiting_for_restart
                        else ""
                    ),
                    "execution_feedback": feedback,
                })
            except Exception as exc:
                self._publish({"phase": "fault", "error": str(exc)})

        def _done_msg(self, message):
            if not completion_belongs_to_session(message, self._core.session_id):
                return
            execute_fetch = bool(getattr(message, "execute_fetch", False))
            execute_done = bool(message.execute_done)
            if execute_done:
                action_key = (
                    str(message.session_id),
                    str(message.action_id),
                    int(message.revision),
                )
                period_deadline = self._action_period_deadlines.pop(action_key, None)
                if bool(message.success) and period_deadline is not None:
                    self._next_observation_not_before = max(
                        self._next_observation_not_before,
                        float(period_deadline),
                    )
                    self._execution_event(
                        "action_period_gate_armed",
                        session_id=str(message.session_id),
                        request_id=int(message.request_id),
                        action_id=str(message.action_id),
                        revision=int(message.revision),
                        wait_remaining_ms=max(
                            0.0,
                            (float(period_deadline) - time.monotonic()) * 1000.0,
                        ),
                    )
                self._done(String(data=json.dumps({"schema": "sharpa_execution_done.v1", "session_id": message.session_id, "request_id": int(message.request_id), "action_id": message.action_id, "revision": int(message.revision), "execute_start": int(message.execute_start), "execute_length": int(message.execute_length), "executed_steps": int(message.executed_steps), "success": bool(message.success), "done": True, "error": message.error or None})))
            if execute_fetch and not execute_done:
                try:
                    if not bool(message.success):
                        raise RuntimeError("execute_fetch requires successful progress")
                    if not self._control_ready or (self._execution_provider not in {"policy", "hil"} or self._hil.get("active")):
                        return
                    if not self._recording_ready():
                        return
                    pending = self._core.prefetch(message)
                    self._request_observation(pending)
                    self._publish(
                        {
                            "phase": self._core.phase,
                            "reason": "RTC next chunk requested",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    if self._core.inference_pending:
                        self._core.cancel_request()
                    self._publish({"phase": "fault", "error": str(exc)})

    rclpy.init(args=args); node = RosPolicyClientNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):
        pass
    finally:
        node._close_completion_wakeups()
        executor.shutdown()
        node._invalidate_request('policy client shutdown')
        node._client.close()
        node.destroy_node()
        node._inference_pool.shutdown(wait=False, cancel_futures=True)
        node._raw_reader.close()
        if rclpy.ok():
            rclpy.shutdown()
