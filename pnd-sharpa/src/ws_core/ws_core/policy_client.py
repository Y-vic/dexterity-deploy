#!/usr/bin/env python3
"""Single policy-server client for metadata-driven workstation inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import io
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
import uuid
from typing import Any

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ws_msgs.msg import ExecutionDone, InferenceRequest, PolicyObs, PolicyPred, Status

from ws_core.baseline_wire import (
    BASELINE_HISTORY_FRAMES,
    TREX_HISTORY_FRAMES,
    VITACFORMER_HISTORY_FRAMES,
    BaselineHistoryFrame,
    build_baseline_request as build_baseline_wire_request,
    extract_baseline_history_frame,
)
from ws_core.common import (
    age_ms,
    json_dumps,
    json_or_raw,
    make_status,
    now_ns,
    set_header,
)
from ws_core.gcc_wire import (
    GCC_HISTORY_FRAMES,
    GCC_REQUEST_SCHEMA,
    GccHistoryFrame,
    build_gcc_request as build_gcc_wire_request,
    extract_gcc_history_frame,
)
from ws_core.sharpa_v3 import (
    HISTORY_STREAM_MAX_CAPACITIES,
    SharpaV3Frame,
    SharpaV3History,
    SharpaV3HistorySnapshot,
    SharpaV3ProtocolError,
    action_to_policy_payload,
    build_observation as build_sharpa_v3_observation,
    extract_frame as extract_sharpa_v3_frame,
    required_stream_lengths as sharpa_v3_required_stream_lengths,
    validate_metadata_format as validate_sharpa_v3_metadata_format,
    validate_server_metadata as validate_sharpa_v3_server_metadata,
)
from ws_core.tactile_wire import build_mot_tactile_request
from ws_core.websocket_rpc import (
    MsgpackPolicyWsClient,
    PolicyRpcResult,
    SharpaV3PolicyClient,
)


DREAMZERO_PROVIDERS = {"dreamzero", "dz", "sharpa62"}
CGP_PROVIDERS = {"cgp", "cgp_n17", "cgp_n17_sharpa62"}
TREX_PROVIDERS = {"trex", "t_rex", "t-rex", "trex_sharpa62"}
VITACFORMER_PROVIDERS = {
    "vitacformer",
    "vitac",
    "vitac_former",
    "vitacformer_sharpa62",
}
BASELINE_PROVIDERS = TREX_PROVIDERS | VITACFORMER_PROVIDERS
GROOT_PROVIDERS = {"groot", "groot_n17", "groot_n17_sharpa62"}
GROOT_MOT_PROVIDERS = {"groot_n17_mot", "groot_n17_mot_sharpa62"}
GROOT_PROVIDERS |= GROOT_MOT_PROVIDERS
GCC_PROVIDERS = {"gcc", "gcc_n17", "gcc_n17_sharpa62"}
PACE_PROVIDERS = {"pace", "pace_n17", "pace_n17_sharpa62"}
GCC_WIRE_PROVIDERS = GCC_PROVIDERS | PACE_PROVIDERS
UNIFIED_SHARPA62_PROVIDERS = (
    GROOT_PROVIDERS | GCC_PROVIDERS | PACE_PROVIDERS
)
SHARPA62_PROVIDERS = (
    DREAMZERO_PROVIDERS | BASELINE_PROVIDERS | UNIFIED_SHARPA62_PROVIDERS
)
SHARPA_V3_POLICY_FAMILY_BY_PROVIDER = {
    **{name: "dreamzero" for name in DREAMZERO_PROVIDERS},
    **{name: "cgp" for name in CGP_PROVIDERS},
    **{name: "trex" for name in TREX_PROVIDERS},
    **{name: "vitacformer" for name in VITACFORMER_PROVIDERS},
    **{name: "groot_n17" for name in GROOT_PROVIDERS},
    **{name: "gcc_n17" for name in GCC_PROVIDERS},
    **{name: "pace_n17" for name in PACE_PROVIDERS},
}

BASELINE_ACTION_SCHEMA = "dreamzero_sharpa62_action.v1"
BASELINE_ACTION_HORIZON = 24
BASELINE_ACTION_HZ = 15.0
BASELINE_OBS_HZ = 30.0
BASELINE_ACTION_SPACE = "sharpa_dexretarget_position_62d"
BASELINE_WRIST_FRAME = "absolute_current_hip"
BASELINE_ACTION_LAYOUT = "left_wrist9,right_wrist9,sharpa_q44"
BASELINE_SERVER_SCHEMA = "dreamzero_sharpa62_server.v1"
BASELINE_REQUEST_SCHEMA = "dreamzero_sharpa62_observation.v1"

UNIFIED_ACTION_SCHEMA = "sharpa62_policy_action.v1"
UNIFIED_ACTION_HORIZON = 40
UNIFIED_ACTION_HZ = 30.0
UNIFIED_ACTION_SPACE = "sharpa_dexretarget_position_62d"
UNIFIED_WRIST_FRAME = "absolute_current_hip"
UNIFIED_ACTION_LAYOUT = "left_wrist9,right_wrist9,sharpa_q44"
UNIFIED_SERVER_SCHEMA = "sharpa62_policy_server.v1"
PACE_REQUEST_SCHEMA = "pace_n17_sharpa62_observation.v1"
PACE_POLICY_FAMILY = "pace_n17"
PACE_EXECUTE_JOINT_SOURCES = {
    "predicted_q_exe",
    "reconstructed_q_cmd",
}


@dataclass(frozen=True)
class ObsSample:
    seq: int
    provider: str
    payload_json: str
    image_rgb: bytes
    tactile_data: bytes
    recv_time: float
    timestamp_unix_s: float
    stamp_ns: int


@dataclass(frozen=True)
class PolicyInputSnapshot:
    window: tuple[ObsSample, ...]
    gcc_history: tuple[GccHistoryFrame, ...] = ()
    gcc_history_generation: int = 0
    baseline_history: tuple[BaselineHistoryFrame, ...] = ()
    baseline_history_generation: int = 0
    sharpa_v3: SharpaV3HistorySnapshot | None = None

    @property
    def latest(self) -> ObsSample:
        return self.window[-1]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prediction_video_path(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    containers = [payload]
    for key in ("action", "debug", "diagnostics"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in (
            "dashboard_video_path",
            "video_path",
            "video",
            "server_video_pred_path",
        ):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def validate_unified_sharpa62_response(
    response: Any,
    *,
    expected_policy_family: str | None = None,
) -> dict[str, Any]:
    """Strictly validate a common GROOT/GCC/PACE executable response."""

    if not isinstance(response, dict):
        raise ValueError("policy response must be an object")
    if response.get("schema") != UNIFIED_ACTION_SCHEMA:
        raise ValueError(
            f"response schema must be {UNIFIED_ACTION_SCHEMA}"
        )
    if response.get("action_space") != UNIFIED_ACTION_SPACE:
        raise ValueError(
            f"response action_space must be {UNIFIED_ACTION_SPACE}"
        )
    if response.get("wrist_frame") != UNIFIED_WRIST_FRAME:
        raise ValueError(
            f"response wrist_frame must be {UNIFIED_WRIST_FRAME}"
        )
    if response.get("layout") != UNIFIED_ACTION_LAYOUT:
        raise ValueError(
            f"response layout must be {UNIFIED_ACTION_LAYOUT}"
        )

    horizon = _required_scalar_number(
        response.get("action_horizon"),
        "action_horizon",
    )
    if not horizon.is_integer() or int(horizon) != UNIFIED_ACTION_HORIZON:
        raise ValueError(
            "response action_horizon must be "
            f"{UNIFIED_ACTION_HORIZON}"
        )
    action_hz = _required_scalar_number(
        response.get("action_hz"),
        "action_hz",
    )
    if not np.isclose(action_hz, UNIFIED_ACTION_HZ):
        raise ValueError(
            f"response action_hz must be {UNIFIED_ACTION_HZ:g}"
        )
    try:
        action = np.asarray(response.get("action_hand_pose_62d"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action_hand_pose_62d is not numeric") from exc
    if expected_policy_family is not None and action.dtype != np.float32:
        raise ValueError(
            "action_hand_pose_62d dtype is "
            f"{action.dtype}, expected float32"
        )
    try:
        action = action.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("action_hand_pose_62d is not numeric") from exc
    expected_shape = (UNIFIED_ACTION_HORIZON, 62)
    if action.shape != expected_shape:
        raise ValueError(
            f"action_hand_pose_62d shape is {action.shape}, "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("action_hand_pose_62d contains NaN or Inf")
    if expected_policy_family is not None:
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("response metadata must be an object")
        if metadata.get("policy_family") != expected_policy_family:
            raise ValueError(
                "response metadata policy_family must be "
                f"{expected_policy_family}"
            )
        if metadata.get("execute_joint_source") not in PACE_EXECUTE_JOINT_SOURCES:
            raise ValueError(
                "response metadata execute_joint_source must be a supported "
                "PACE source"
            )

    validated = dict(response)
    validated["action_hand_pose_62d"] = action
    return validated


def validate_pace_server_metadata(
    metadata: Any,
    provider: str,
) -> dict[str, Any]:
    """Reject a compatible GCC server accidentally selected for PACE."""

    if str(provider).strip().lower() not in PACE_PROVIDERS:
        raise ValueError(f"unsupported PACE provider: {provider!r}")
    if not isinstance(metadata, dict):
        raise ValueError("PACE server metadata must be an object")

    expected_strings = (
        ("schema", UNIFIED_SERVER_SCHEMA),
        ("policy_family", PACE_POLICY_FAMILY),
        ("request_schema", PACE_REQUEST_SCHEMA),
        ("response_schema", UNIFIED_ACTION_SCHEMA),
        ("action_space", UNIFIED_ACTION_SPACE),
        ("output_wrist_frame", UNIFIED_WRIST_FRAME),
        ("layout", UNIFIED_ACTION_LAYOUT),
    )
    for field, expected in expected_strings:
        if metadata.get(field) != expected:
            raise ValueError(f"PACE server metadata {field} must be {expected}")

    expected_numbers = (
        ("state_dim", 62),
        ("action_dim", 62),
        ("action_horizon", UNIFIED_ACTION_HORIZON),
        ("action_hz", UNIFIED_ACTION_HZ),
        ("history_length", GCC_HISTORY_FRAMES),
    )
    for field, expected in expected_numbers:
        value = _required_scalar_number(metadata.get(field), field)
        if not np.isclose(value, expected):
            raise ValueError(f"PACE server metadata {field} must be {expected}")

    accepted_schemas = metadata.get("accepted_request_schemas")
    if not isinstance(accepted_schemas, (list, tuple)) or (
        GCC_REQUEST_SCHEMA not in accepted_schemas
        or PACE_REQUEST_SCHEMA not in accepted_schemas
    ):
        raise ValueError(
            "PACE server metadata accepted_request_schemas must include the "
            "PACE and GCC-compatible schemas"
        )
    if metadata.get("execute_joint_source") not in PACE_EXECUTE_JOINT_SOURCES:
        raise ValueError(
            "PACE server metadata execute_joint_source must be one of "
            f"{sorted(PACE_EXECUTE_JOINT_SOURCES)}"
        )
    return dict(metadata)


def validate_baseline_sharpa62_response(response: Any) -> dict[str, Any]:
    """Strictly validate the executable response from a baseline server."""

    if not isinstance(response, dict):
        raise ValueError("policy response must be an object")
    expected_strings = (
        ("schema", BASELINE_ACTION_SCHEMA),
        ("action_space", BASELINE_ACTION_SPACE),
        ("wrist_frame", BASELINE_WRIST_FRAME),
        ("layout", BASELINE_ACTION_LAYOUT),
    )
    for field, expected in expected_strings:
        if response.get(field) != expected:
            raise ValueError(f"response {field} must be {expected}")

    horizon = _required_scalar_number(
        response.get("action_horizon"),
        "action_horizon",
    )
    if not horizon.is_integer() or int(horizon) != BASELINE_ACTION_HORIZON:
        raise ValueError(
            "response action_horizon must be "
            f"{BASELINE_ACTION_HORIZON}"
        )
    action_hz = _required_scalar_number(response.get("action_hz"), "action_hz")
    if not np.isclose(action_hz, BASELINE_ACTION_HZ):
        raise ValueError(
            f"response action_hz must be {BASELINE_ACTION_HZ:g}"
        )
    try:
        action = np.asarray(response.get("action_hand_pose_62d"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action_hand_pose_62d is not numeric") from exc
    if action.dtype != np.float32:
        raise ValueError(
            "action_hand_pose_62d dtype is "
            f"{action.dtype}, expected float32"
        )
    expected_shape = (BASELINE_ACTION_HORIZON, 62)
    if action.shape != expected_shape:
        raise ValueError(
            f"action_hand_pose_62d shape is {action.shape}, "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("action_hand_pose_62d contains NaN or Inf")

    validated = dict(response)
    validated["action_hand_pose_62d"] = action
    return validated


def validate_baseline_server_metadata(
    metadata: Any,
    provider: str,
) -> dict[str, Any]:
    """Reject a valid-looking baseline server of the wrong policy family."""

    provider_key = str(provider).strip().lower()
    if provider_key in TREX_PROVIDERS:
        policy_family = "trex"
        state_frames = 1
        wrench_frames = TREX_HISTORY_FRAMES
        require_deformation = True
    elif provider_key in VITACFORMER_PROVIDERS:
        policy_family = "vitacformer"
        state_frames = TREX_HISTORY_FRAMES
        wrench_frames = VITACFORMER_HISTORY_FRAMES
        require_deformation = False
    else:
        raise ValueError(f"unsupported baseline provider: {provider!r}")
    if not isinstance(metadata, dict):
        raise ValueError("baseline server metadata must be an object")

    expected_strings = (
        ("schema", BASELINE_SERVER_SCHEMA),
        ("policy_family", policy_family),
        ("request_schema", BASELINE_REQUEST_SCHEMA),
        ("response_schema", BASELINE_ACTION_SCHEMA),
        ("action_space", BASELINE_ACTION_SPACE),
        ("output_wrist_frame", BASELINE_WRIST_FRAME),
        ("layout", BASELINE_ACTION_LAYOUT),
    )
    for field, expected in expected_strings:
        if metadata.get(field) != expected:
            raise ValueError(
                f"baseline server metadata {field} must be {expected}"
            )
    expected_numbers = (
        ("action_horizon", BASELINE_ACTION_HORIZON),
        ("action_hz", BASELINE_ACTION_HZ),
        ("required_state_history", state_frames),
        ("required_wrench_history", wrench_frames),
    )
    for field, expected in expected_numbers:
        value = _required_scalar_number(metadata.get(field), field)
        if not np.isclose(value, expected):
            raise ValueError(
                f"baseline server metadata {field} must be {expected}"
            )
    if metadata.get("required_tactile_deformation") is not require_deformation:
        raise ValueError(
            "baseline server metadata required_tactile_deformation must be "
            f"{require_deformation}"
        )
    return dict(metadata)


def _required_scalar_number(value: Any, field: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"response {field} must be a scalar")
    try:
        number = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"response {field} is not numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"response {field} is not finite")
    return number


def _validate_sharpa_v3_format_capacity(value: Any) -> dict[str, Any]:
    metadata_format = validate_sharpa_v3_metadata_format(value)
    required = sharpa_v3_required_stream_lengths(metadata_format)
    for name, required_length in required.items():
        capacity = HISTORY_STREAM_MAX_CAPACITIES[name]
        if required_length > capacity:
            raise SharpaV3ProtocolError(
                f"v3 metadata stream {name} requires {required_length} frames, "
                f"fixed workstation capacity is {capacity}"
            )
    return metadata_format


class PolicyClient(Node):
    policy_protocol = "legacy"

    def __init__(self) -> None:
        super().__init__("policy_client")

        self.declare_parameter("obs_topic", "/ws/obs")
        self.declare_parameter("pred_topic", "/ws/pred")
        self.declare_parameter("execution_done_topic", "/ws/execution_done")
        self.declare_parameter("status_topic", "/ws/policy_client/status")
        self.declare_parameter("provider", "dreamzero")
        self.declare_parameter("policy_protocol", "sharpa_v3")
        self.declare_parameter("server_url", "")
        self.declare_parameter("ssh_host", "")
        self.declare_parameter("ssh_remote_host", "127.0.0.1")
        self.declare_parameter("ssh_remote_port", 5500)
        self.declare_parameter(
            "pred_video_dir", "deploy/runs/ws_dashboard/pred_videos"
        )
        self.declare_parameter("request_timeout_s", 90.0)
        self.declare_parameter("session_id", "")
        self.declare_parameter("prompt", "")
        self.declare_parameter("expect_initial_message", True)
        self.declare_parameter("allow_zero_wrist_fallback", False)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("dry_run_horizon", 24)
        self.declare_parameter("dry_run_action_dim", 62)
        self.declare_parameter("action_horizon", 24)
        self.declare_parameter("actor_send_hz", 15.0)
        self.declare_parameter("obs_rate_hz", 30.0)
        self.declare_parameter("policy_window_frames", 4)
        self.declare_parameter("policy_window_stride", 2)
        self.declare_parameter("initial_window_frames", 1)
        self.declare_parameter("obs_buffer_size", 128)
        self.declare_parameter("baseline_history_max_gap_s", 0.25)
        self.declare_parameter("baseline_image_max_age_ms", 150.0)
        self.declare_parameter("baseline_wrench_max_age_ms", 150.0)
        self.declare_parameter("baseline_deformation_max_age_ms", 150.0)
        self.declare_parameter("v3_history_max_gap_s", 0.25)
        self.declare_parameter("v3_image_max_age_ms", 150.0)
        self.declare_parameter("v3_joint_max_age_ms", 150.0)
        self.declare_parameter("v3_wrench_max_age_ms", 150.0)
        self.declare_parameter("v3_deformation_max_age_ms", 150.0)
        self.declare_parameter("gcc_history_frames", GCC_HISTORY_FRAMES)
        self.declare_parameter("gcc_history_max_gap_s", 0.25)
        self.declare_parameter("gcc_joint_max_age_ms", 150.0)
        self.declare_parameter("gcc_wrench_max_age_ms", 150.0)
        self.declare_parameter("gcc_deformation_max_age_ms", 150.0)

        self.obs_topic = str(self.get_parameter("obs_topic").value)
        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.execution_done_topic = str(
            self.get_parameter("execution_done_topic").value
        )
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.provider = str(self.get_parameter("provider").value)
        self.policy_protocol = str(
            self.get_parameter("policy_protocol").value
        ).strip().lower()
        self.server_url = str(self.get_parameter("server_url").value)
        self.ssh_host = str(self.get_parameter("ssh_host").value).strip()
        self.ssh_remote_host = str(
            self.get_parameter("ssh_remote_host").value
        ).strip()
        self.ssh_remote_port = int(self.get_parameter("ssh_remote_port").value)
        self.pred_video_dir = Path(
            str(self.get_parameter("pred_video_dir").value)
        ).resolve()
        self.request_timeout_s = float(self.get_parameter("request_timeout_s").value)
        configured_session = str(self.get_parameter("session_id").value)
        self.session_id = configured_session or f"ws-{uuid.uuid4()}"
        self.prompt = str(self.get_parameter("prompt").value)
        self.expect_initial_message = bool(
            self.get_parameter("expect_initial_message").value
        )
        self.allow_zero_wrist_fallback = bool(
            self.get_parameter("allow_zero_wrist_fallback").value
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.dry_run_horizon = int(self.get_parameter("dry_run_horizon").value)
        self.dry_run_action_dim = int(self.get_parameter("dry_run_action_dim").value)
        self.action_horizon = int(self.get_parameter("action_horizon").value)
        self.actor_send_hz = float(self.get_parameter("actor_send_hz").value)
        self.obs_rate_hz = float(self.get_parameter("obs_rate_hz").value)
        self.policy_window_frames = int(
            self.get_parameter("policy_window_frames").value
        )
        self.policy_window_stride = int(
            self.get_parameter("policy_window_stride").value
        )
        self.initial_window_frames = int(
            self.get_parameter("initial_window_frames").value
        )
        self.obs_buffer_size = int(self.get_parameter("obs_buffer_size").value)
        self.baseline_history_max_gap_s = float(
            self.get_parameter("baseline_history_max_gap_s").value
        )
        self.baseline_image_max_age_ms = float(
            self.get_parameter("baseline_image_max_age_ms").value
        )
        self.baseline_wrench_max_age_ms = float(
            self.get_parameter("baseline_wrench_max_age_ms").value
        )
        self.baseline_deformation_max_age_ms = float(
            self.get_parameter("baseline_deformation_max_age_ms").value
        )
        self.v3_history_max_gap_s = float(
            self.get_parameter("v3_history_max_gap_s").value
        )
        self.v3_image_max_age_ms = float(
            self.get_parameter("v3_image_max_age_ms").value
        )
        self.v3_joint_max_age_ms = float(
            self.get_parameter("v3_joint_max_age_ms").value
        )
        self.v3_wrench_max_age_ms = float(
            self.get_parameter("v3_wrench_max_age_ms").value
        )
        self.v3_deformation_max_age_ms = float(
            self.get_parameter("v3_deformation_max_age_ms").value
        )
        self.gcc_history_frames = int(
            self.get_parameter("gcc_history_frames").value
        )
        self.gcc_history_max_gap_s = float(
            self.get_parameter("gcc_history_max_gap_s").value
        )
        self.gcc_joint_max_age_ms = float(
            self.get_parameter("gcc_joint_max_age_ms").value
        )
        self.gcc_wrench_max_age_ms = float(
            self.get_parameter("gcc_wrench_max_age_ms").value
        )
        self.gcc_deformation_max_age_ms = float(
            self.get_parameter("gcc_deformation_max_age_ms").value
        )
        if self.dry_run_horizon <= 0 or self.dry_run_action_dim <= 0:
            raise ValueError("dry_run_horizon and dry_run_action_dim must be positive")
        if self.policy_protocol not in {"sharpa_v3", "v3", "legacy", "v1"}:
            raise ValueError(
                "policy_protocol must be sharpa_v3/v3 or legacy/v1"
            )
        if self.policy_protocol == "v3":
            self.policy_protocol = "sharpa_v3"
        if self.policy_protocol == "v1":
            self.policy_protocol = "legacy"
        if (
            self.policy_protocol == "sharpa_v3"
            and self.provider.strip().lower()
            not in SHARPA_V3_POLICY_FAMILY_BY_PROVIDER
        ):
            raise ValueError(
                f"unsupported sharpa_v3 provider: {self.provider!r}"
            )
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be positive")
        if self.ssh_host and not 1 <= self.ssh_remote_port <= 65535:
            raise ValueError("ssh_remote_port must be in [1, 65535]")
        if self.actor_send_hz <= 0.0:
            raise ValueError("actor_send_hz must be positive")
        if self.obs_rate_hz <= 0.0:
            raise ValueError("obs_rate_hz must be positive")
        if self.policy_window_frames <= 0 or self.policy_window_stride <= 0:
            raise ValueError("policy_window_frames/stride must be positive")
        if self.initial_window_frames <= 0:
            raise ValueError("initial_window_frames must be positive")
        if self.obs_buffer_size < self.policy_window_frames:
            raise ValueError("obs_buffer_size must fit at least one policy window")
        if (
            not np.isfinite(self.baseline_history_max_gap_s)
            or self.baseline_history_max_gap_s <= 0.0
        ):
            raise ValueError(
                "baseline_history_max_gap_s must be finite and positive"
            )
        for name, value in (
            ("baseline_image_max_age_ms", self.baseline_image_max_age_ms),
            ("baseline_wrench_max_age_ms", self.baseline_wrench_max_age_ms),
            (
                "baseline_deformation_max_age_ms",
                self.baseline_deformation_max_age_ms,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("v3_history_max_gap_s", self.v3_history_max_gap_s),
            ("v3_image_max_age_ms", self.v3_image_max_age_ms),
            ("v3_joint_max_age_ms", self.v3_joint_max_age_ms),
            ("v3_wrench_max_age_ms", self.v3_wrench_max_age_ms),
            ("v3_deformation_max_age_ms", self.v3_deformation_max_age_ms),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gcc_history_frames != GCC_HISTORY_FRAMES:
            raise ValueError(
                f"gcc_history_frames is fixed at {GCC_HISTORY_FRAMES}"
            )
        if self.gcc_history_max_gap_s <= 0.0:
            raise ValueError("gcc_history_max_gap_s must be positive")
        for name, value in (
            ("gcc_joint_max_age_ms", self.gcc_joint_max_age_ms),
            ("gcc_wrench_max_age_ms", self.gcc_wrench_max_age_ms),
            (
                "gcc_deformation_max_age_ms",
                self.gcc_deformation_max_age_ms,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.policy_protocol == "legacy"
            and self.provider.strip().lower() in UNIFIED_SHARPA62_PROVIDERS
        ):
            if self.action_horizon != UNIFIED_ACTION_HORIZON:
                raise ValueError(
                    "GROOT/GCC/PACE action_horizon is fixed at "
                    f"{UNIFIED_ACTION_HORIZON}"
                )
            if not np.isclose(self.actor_send_hz, UNIFIED_ACTION_HZ):
                raise ValueError(
                    "GROOT/GCC/PACE actor_send_hz is fixed at "
                    f"{UNIFIED_ACTION_HZ:g}"
                )
        if (
            self.policy_protocol == "legacy"
            and self.provider.strip().lower() in BASELINE_PROVIDERS
        ):
            if self.action_horizon != BASELINE_ACTION_HORIZON:
                raise ValueError(
                    "T-Rex/ViTacFormer action_horizon is fixed at "
                    f"{BASELINE_ACTION_HORIZON}"
                )
            if not np.isclose(self.actor_send_hz, BASELINE_ACTION_HZ):
                raise ValueError(
                    "T-Rex/ViTacFormer actor_send_hz is fixed at "
                    f"{BASELINE_ACTION_HZ:g}"
                )
            if not np.isclose(self.obs_rate_hz, BASELINE_OBS_HZ):
                raise ValueError(
                    "T-Rex/ViTacFormer obs_rate_hz is fixed at "
                    f"{BASELINE_OBS_HZ:g}"
                )
        self.lock = threading.Lock()
        self.client_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.request_event = threading.Event()
        self.client: MsgpackPolicyWsClient | SharpaV3PolicyClient | None = None
        self.obs_buffer: deque[ObsSample] = deque(maxlen=self.obs_buffer_size)
        self.latest_obs: ObsSample | None = None
        self.sharpa_v3_history = SharpaV3History()
        self.last_sharpa_v3_obs_seq: int | None = None
        self.last_sharpa_v3_obs_time: float | None = None
        self.sharpa_v3_history_resets = 0
        self.sharpa_v3_history_last_reset_reason = ""
        self.sharpa_v3_metadata: dict[str, Any] | None = None
        self.sharpa_v3_active_format: dict[str, Any] | None = None
        self.sharpa_v3_effective_prompt = ""
        self.sharpa_v3_reset_pending = False
        self.sharpa_v3_force_initial_feedback = True
        self.baseline_history: deque[BaselineHistoryFrame] = deque(
            maxlen=BASELINE_HISTORY_FRAMES
        )
        self.baseline_history_generation = 0
        self.last_baseline_obs_seq: int | None = None
        self.last_baseline_obs_time: float | None = None
        self.baseline_history_resets = 0
        self.baseline_history_last_reset_reason = ""
        self.gcc_history: deque[GccHistoryFrame] = deque(
            maxlen=self.gcc_history_frames
        )
        self.gcc_history_generation = 0
        self.last_gcc_obs_seq: int | None = None
        self.last_gcc_obs_time: float | None = None
        self.gcc_history_resets = 0
        self.gcc_history_last_reset_reason = ""
        self.pending_request: InferenceRequest | None = None
        self.last_request_id = 0
        self.last_action_id: str | None = None
        self.execution_done_received = 0
        self.last_pred_request_id: int | None = None
        self.last_pred_msg: PolicyPred | None = None
        self.request_index = 0
        self.request_inflight = False
        self.last_request_start_time: float | None = None
        self.last_request_started_ns: int | None = None
        self.last_response_received_ns: int | None = None
        self.last_published_pred_seq: int | None = None
        self.last_trigger_action_seq: int | None = None
        self.pred_seq = 0
        self.obs_received = 0
        self.pred_published = 0
        self.obs_dropped = 0
        self.policy_calls = 0
        self.policy_failures = 0
        self.last_obs_seq: int | None = None
        self.last_obs_time: float | None = None
        self.last_pred_time: float | None = None
        self.last_latency_s: float | None = None
        self.last_request_info: dict[str, Any] = {}
        self.last_error = ""

        self.create_subscription(PolicyObs, self.obs_topic, self._on_obs, 10)
        self.create_subscription(
            ExecutionDone,
            self.execution_done_topic,
            self._on_execution_done,
            10,
        )
        self.pred_pub = self.create_publisher(PolicyPred, self.pred_topic, 10)
        self.status_pub = self.create_publisher(Status, self.status_topic, 10)
        self.create_timer(0.5, self._publish_status)
        self.worker = threading.Thread(target=self._request_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            "policy_client: "
            f"provider={self.provider}, dry_run={self.dry_run}, obs={self.obs_topic}, "
            f"transport={'ssh:' + self.ssh_host if self.ssh_host else 'direct'}, "
            f"pred={self.pred_topic}, action_horizon={self.action_horizon}, "
            f"actor_send_hz={self.actor_send_hz}, "
            f"done={self.execution_done_topic}, "
            f"window={self.policy_window_frames}@stride{self.policy_window_stride}"
        )

    def _on_obs(self, msg: PolicyObs) -> None:
        now = time.monotonic()
        received_ns = now_ns()
        parsed = json_or_raw(msg.payload_json)
        parsed_json = parsed.get("json") if parsed.get("valid") else None
        obs_stamp_ns = (
            int(parsed_json.get("stamp_ns", received_ns))
            if isinstance(parsed_json, dict)
            else received_ns
        )
        sample = ObsSample(
            seq=int(msg.seq),
            provider=msg.provider,
            payload_json=msg.payload_json,
            image_rgb=bytes(msg.image_rgb),
            tactile_data=bytes(msg.tactile_data),
            recv_time=now,
            timestamp_unix_s=time.time(),
            stamp_ns=obs_stamp_ns,
        )
        provider = self.provider.strip().lower()
        is_v3 = getattr(self, "policy_protocol", "legacy") == "sharpa_v3"
        sharpa_v3_frame: SharpaV3Frame | None = None
        sharpa_v3_frame_error = ""
        if is_v3:
            try:
                wrapper = parsed_json if isinstance(parsed_json, dict) else {}
                model_image = wrapper.get("model_image")
                image_age_ms = (
                    float(model_image.get("age_ms"))
                    if isinstance(model_image, dict)
                    else float("inf")
                )
                image, image_info = self._image_from_obs(sample, wrapper)
                image_valid = bool(
                    isinstance(model_image, dict)
                    and model_image.get("valid") is True
                    and not image_info.get("fallback")
                    and np.isfinite(image_age_ms)
                    and 0.0 <= image_age_ms <= self.v3_image_max_age_ms
                )
                image_jpeg = self._encode_jpeg(image) if image_valid else b""
                sharpa_v3_frame = extract_sharpa_v3_frame(
                    wrapper,
                    image_jpeg=image_jpeg,
                    tactile_data=sample.tactile_data,
                    obs_seq=sample.seq,
                    timestamp_ns=sample.stamp_ns,
                    image_valid=image_valid,
                    joint_max_age_ms=self.v3_joint_max_age_ms,
                    wrench_max_age_ms=self.v3_wrench_max_age_ms,
                    deformation_max_age_ms=self.v3_deformation_max_age_ms,
                )
            except (TypeError, ValueError, SharpaV3ProtocolError) as exc:
                sharpa_v3_frame_error = str(exc)
        baseline_frame: BaselineHistoryFrame | None = None
        baseline_frame_error = ""
        if not is_v3 and provider in BASELINE_PROVIDERS:
            try:
                hand_pose = self._find_hand_pose_62d(parsed_json)
                if hand_pose is None:
                    raise ValueError("PolicyObs payload has no valid FK hand_pose_62d")
                baseline_frame = extract_baseline_history_frame(
                    parsed_json,
                    hand_pose_62d=hand_pose,
                    obs_seq=sample.seq,
                    obs_stamp_ns=sample.stamp_ns,
                    timestamp_unix_s=sample.timestamp_unix_s,
                    wrench_max_age_ms=self.baseline_wrench_max_age_ms,
                )
            except (TypeError, ValueError) as exc:
                baseline_frame_error = str(exc)
        gcc_frame: GccHistoryFrame | None = None
        gcc_frame_error = ""
        if not is_v3 and provider in GCC_WIRE_PROVIDERS:
            try:
                gcc_frame = extract_gcc_history_frame(
                    parsed_json,
                    obs_seq=sample.seq,
                    obs_stamp_ns=sample.stamp_ns,
                    timestamp_unix_s=sample.timestamp_unix_s,
                    joint_max_age_ms=self.gcc_joint_max_age_ms,
                    wrench_max_age_ms=self.gcc_wrench_max_age_ms,
                )
            except (TypeError, ValueError) as exc:
                gcc_frame_error = str(exc)
        with self.lock:
            self.latest_obs = sample
            # Dedicated history adapters own their retained data. Keep only the
            # latest raw observation for those providers.
            if (
                is_v3
                or provider in UNIFIED_SHARPA62_PROVIDERS | BASELINE_PROVIDERS
            ):
                self.obs_buffer.clear()
            self.obs_buffer.append(sample)
            if is_v3:
                reset_reason = ""
                if (
                    self.last_sharpa_v3_obs_seq is not None
                    and sample.seq <= self.last_sharpa_v3_obs_seq
                ):
                    reset_reason = "obs_seq_reset"
                elif (
                    self.sharpa_v3_history.last_timestamp_ns is not None
                    and sample.stamp_ns
                    <= self.sharpa_v3_history.last_timestamp_ns
                ):
                    reset_reason = "obs_stamp_reset"
                elif (
                    self.last_sharpa_v3_obs_time is not None
                    and now - self.last_sharpa_v3_obs_time
                    > self.v3_history_max_gap_s
                ):
                    reset_reason = "observation_gap"
                elif sharpa_v3_frame is None:
                    reset_reason = "invalid_observation_frame"
                if reset_reason:
                    self._clear_sharpa_v3_history_locked(reset_reason)
                if sharpa_v3_frame is not None:
                    self.sharpa_v3_history.append(sharpa_v3_frame)
                self.last_sharpa_v3_obs_seq = sample.seq
                self.last_sharpa_v3_obs_time = now
                if sharpa_v3_frame_error:
                    self.last_error = (
                        "SharpA v3 observation rejected: "
                        f"{sharpa_v3_frame_error}"
                    )
                elif self.last_error.startswith(
                    "SharpA v3 observation rejected:"
                ):
                    self.last_error = ""
            if not is_v3 and provider in BASELINE_PROVIDERS:
                reset_reason = ""
                if (
                    self.last_baseline_obs_seq is not None
                    and sample.seq <= self.last_baseline_obs_seq
                ):
                    reset_reason = "obs_seq_reset"
                elif (
                    self.baseline_history
                    and sample.stamp_ns
                    <= self.baseline_history[-1].obs_stamp_ns
                ):
                    reset_reason = "obs_stamp_reset"
                elif (
                    self.last_baseline_obs_time is not None
                    and now - self.last_baseline_obs_time
                    > self.baseline_history_max_gap_s
                ):
                    reset_reason = "observation_gap"
                elif baseline_frame is None:
                    reset_reason = "invalid_baseline_history_frame"
                if reset_reason:
                    self._clear_baseline_history_locked(reset_reason)
                if baseline_frame is not None:
                    self.baseline_history.append(baseline_frame)
                self.last_baseline_obs_seq = sample.seq
                self.last_baseline_obs_time = now
                if baseline_frame_error:
                    self.last_error = (
                        "Baseline observation rejected: "
                        f"{baseline_frame_error}"
                    )
                elif self.last_error.startswith(
                    "Baseline observation rejected:"
                ):
                    self.last_error = ""
            if not is_v3 and provider in GCC_WIRE_PROVIDERS:
                reset_reason = ""
                if (
                    self.last_gcc_obs_seq is not None
                    and sample.seq <= self.last_gcc_obs_seq
                ):
                    reset_reason = "obs_seq_reset"
                elif (
                    self.last_gcc_obs_time is not None
                    and now - self.last_gcc_obs_time
                    > self.gcc_history_max_gap_s
                ):
                    reset_reason = "observation_gap"
                elif gcc_frame is None:
                    reset_reason = "invalid_gcc_history_frame"
                if reset_reason:
                    self._clear_gcc_history_locked(reset_reason)
                if gcc_frame is not None:
                    # Each /ws/obs callback is one real 30 Hz sample.  Source
                    # timestamps are retained in the payload but never used to
                    # deduplicate or resample this history.
                    self.gcc_history.append(gcc_frame)
                self.last_gcc_obs_seq = sample.seq
                self.last_gcc_obs_time = now
                if gcc_frame_error:
                    self.last_error = (
                        "GCC-compatible observation rejected: "
                        f"{gcc_frame_error}"
                    )
                elif self.last_error.startswith(
                    "GCC-compatible observation rejected:"
                ):
                    self.last_error = ""
            self.obs_received += 1
            self.last_obs_seq = int(msg.seq)
            self.last_obs_time = now
            start_cycle = (
                hasattr(self, "_clock")
                and getattr(self, "last_request_id", 0) == 0
                and getattr(self, "pending_request", None) is None
                and not getattr(self, "request_inflight", False)
            )
        if start_cycle:
            self._queue_inference("history_ready")

    def _clear_baseline_history_locked(self, reason: str) -> None:
        self.baseline_history.clear()
        self.baseline_history_generation += 1
        self.baseline_history_resets += 1
        self.baseline_history_last_reset_reason = str(reason)

    def _clear_sharpa_v3_history_locked(self, reason: str) -> None:
        self.sharpa_v3_history.clear()
        self.sharpa_v3_history_resets += 1
        self.sharpa_v3_history_last_reset_reason = str(reason)

    def _sharpa_v3_snapshot_context_is_current_locked(
        self,
        snapshot: SharpaV3HistorySnapshot | None,
    ) -> bool:
        return bool(
            snapshot is not None
            and snapshot.generation == self.sharpa_v3_history.generation
            and snapshot.format_revision
            == self.sharpa_v3_history.format_revision
        )

    def _clear_gcc_history_locked(self, reason: str) -> None:
        self.gcc_history.clear()
        self.gcc_history_generation += 1
        self.gcc_history_resets += 1
        self.gcc_history_last_reset_reason = str(reason)

    def _queue_inference(
        self,
        reason: str,
        *,
        execution_feedback: dict[str, Any] | None = None,
        trigger_action_seq: int = 0,
    ) -> bool:
        with self.lock:
            if self.pending_request is not None or self.request_inflight:
                return False
            request_id = self.last_request_id + 1
        msg = InferenceRequest()
        set_header(msg, "policy_client", self.get_clock())
        msg.request_id = request_id
        msg.request_stamp_ns = now_ns()
        msg.trigger_action_seq = int(trigger_action_seq)
        msg.reason = str(reason)
        payload: dict[str, Any] = {
            "schema": "ws.inference_request.v1",
            "execution_mode": "synchronous",
        }
        if execution_feedback is not None:
            payload["execution_feedback"] = dict(execution_feedback)
        msg.payload_json = json_dumps(payload)
        self._on_inference_request(msg)
        return True

    def _on_execution_done(self, msg: ExecutionDone) -> None:
        if not bool(msg.done):
            return
        with self.lock:
            if int(msg.request_id) != self.last_pred_request_id:
                return
            if self.last_action_id is not None and msg.action_id != self.last_action_id:
                return
            self.execution_done_received += 1
        feedback = {
            "last_action_id": msg.action_id,
            "executed_steps": int(msg.executed_steps),
            "success": bool(msg.success),
        }
        self._queue_inference(
            "execute_done",
            execution_feedback=feedback,
            trigger_action_seq=int(msg.executed_steps),
        )

    def _on_inference_request(self, msg: InferenceRequest) -> None:
        request_id = int(msg.request_id)
        if request_id <= 0:
            return
        reset_history = self._is_pipeline_reset_request(msg)
        cached_pred: PolicyPred | None = None
        with self.lock:
            if request_id < self.last_request_id:
                return
            if request_id == self.last_request_id:
                if self.last_pred_request_id == request_id:
                    cached_pred = self.last_pred_msg
                elif self.pending_request is not None or self.request_inflight:
                    return
                else:
                    self.pending_request = msg
            else:
                if reset_history:
                    provider = self.provider.strip().lower()
                    if self.policy_protocol == "sharpa_v3":
                        self._clear_sharpa_v3_history_locked("pipeline_reset")
                        self.sharpa_v3_reset_pending = True
                        self.sharpa_v3_force_initial_feedback = True
                    elif provider in BASELINE_PROVIDERS:
                        self._clear_baseline_history_locked("pipeline_reset")
                    if (
                        self.policy_protocol != "sharpa_v3"
                        and provider in GCC_WIRE_PROVIDERS
                    ):
                        self._clear_gcc_history_locked("pipeline_reset")
                self.last_request_id = request_id
                self.last_trigger_action_seq = int(msg.trigger_action_seq)
                self.pending_request = msg
        if cached_pred is not None:
            self.pred_pub.publish(cached_pred)
            return
        self.request_event.set()

    @staticmethod
    def _is_pipeline_reset_request(msg: InferenceRequest) -> bool:
        if str(msg.reason).strip().lower() in {
            "reset",
            "pipeline_reset",
            "session_reset",
            "session_switch",
        }:
            return True
        parsed = json_or_raw(msg.payload_json)
        payload = parsed.get("json") if parsed.get("valid") else None
        return isinstance(payload, dict) and bool(
            payload.get("reset") or payload.get("pipeline_reset")
        )

    def _request_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.request_event.wait(0.02):
                continue
            with self.lock:
                request = self.pending_request
            if request is None:
                self.request_event.clear()
                continue
            if self.policy_protocol == "sharpa_v3" and not self.dry_run:
                try:
                    self._ensure_sharpa_v3_client()
                except Exception as exc:  # noqa: BLE001 - retry pending request.
                    with self.lock:
                        if self.pending_request is request:
                            self.last_error = f"v3 metadata/reset failed: {exc}"
                    self.stop_event.wait(0.1)
                    continue
            snapshot = self._select_obs_window()
            if snapshot is None:
                self.stop_event.wait(0.02)
                continue

            request_start = time.monotonic()
            request_started_ns = now_ns()
            with self.lock:
                if self.pending_request is not request:
                    continue
                if (
                    self.policy_protocol != "sharpa_v3"
                    and self.provider.strip().lower() in BASELINE_PROVIDERS
                ):
                    snapshot_is_current = (
                        snapshot.baseline_history_generation
                        == self.baseline_history_generation
                        and bool(snapshot.baseline_history)
                        and bool(self.baseline_history)
                        and snapshot.baseline_history[-1].obs_seq
                        == self.baseline_history[-1].obs_seq
                    )
                    if not snapshot_is_current:
                        continue
                if (
                    self.policy_protocol == "sharpa_v3"
                    and not self.dry_run
                    and (
                        snapshot.sharpa_v3 is None
                        or not self.sharpa_v3_history.is_current(
                            snapshot.sharpa_v3
                        )
                    )
                ):
                    continue
                if (
                    self.policy_protocol != "sharpa_v3"
                    and self.provider.strip().lower() in GCC_WIRE_PROVIDERS
                ):
                    snapshot_is_current = (
                        snapshot.gcc_history_generation
                        == self.gcc_history_generation
                        and bool(snapshot.gcc_history)
                        and bool(self.gcc_history)
                        and snapshot.gcc_history[-1].obs_seq
                        == self.gcc_history[-1].obs_seq
                    )
                    if not snapshot_is_current:
                        continue
                self.pending_request = None
                self.request_event.clear()
                self.request_inflight = True
                self.last_request_start_time = request_start
                self.last_request_started_ns = request_started_ns
            completed = False
            try:
                try:
                    completed = self._run_policy_request(
                        snapshot,
                        request,
                        request_started_ns,
                    )
                except RCLError:
                    if self.stop_event.is_set() or not rclpy.ok(context=self.context):
                        return
                    raise
            finally:
                with self.lock:
                    self.request_inflight = False
            if not completed and not self.stop_event.is_set():
                with self.lock:
                    if (
                        self.pending_request is None
                        and self.last_request_id == int(request.request_id)
                    ):
                        self.pending_request = request
                self.request_event.set()
                self.stop_event.wait(0.1)

    def _select_obs_window(self) -> PolicyInputSnapshot | None:
        provider = self.provider.strip().lower()
        with self.lock:
            request_index = self.request_index
            samples = list(self.obs_buffer)
            latest_obs = self.latest_obs
            if self.policy_protocol == "sharpa_v3":
                if latest_obs is None:
                    return None
                if self.dry_run:
                    if not self._obs_has_hand_pose(latest_obs):
                        return None
                    return PolicyInputSnapshot(window=(latest_obs,))
                history = self.sharpa_v3_history.snapshot()
                if (
                    history is None
                    or history.anchor_obs_seq != latest_obs.seq
                    or history.anchor_timestamp_ns != latest_obs.stamp_ns
                ):
                    return None
                return PolicyInputSnapshot(
                    window=(latest_obs,),
                    sharpa_v3=history,
                )
            baseline_history = (
                tuple(self.baseline_history)
                if provider in BASELINE_PROVIDERS
                else ()
            )
            baseline_history_generation = (
                self.baseline_history_generation
                if provider in BASELINE_PROVIDERS
                else 0
            )
            gcc_history = (
                tuple(self.gcc_history)
                if provider in GCC_WIRE_PROVIDERS
                else ()
            )
            gcc_history_generation = (
                self.gcc_history_generation
                if provider in GCC_WIRE_PROVIDERS
                else 0
            )
        if provider in BASELINE_PROVIDERS:
            required_frames = (
                TREX_HISTORY_FRAMES
                if provider in TREX_PROVIDERS
                else VITACFORMER_HISTORY_FRAMES
            )
            if (
                latest_obs is None
                or len(baseline_history) < required_frames
                or baseline_history[-1].obs_seq != latest_obs.seq
            ):
                return None
            return PolicyInputSnapshot(
                window=(latest_obs,),
                baseline_history=baseline_history[-required_frames:],
                baseline_history_generation=baseline_history_generation,
            )
        if provider in GCC_WIRE_PROVIDERS:
            if (
                latest_obs is None
                or len(gcc_history) != GCC_HISTORY_FRAMES
                or gcc_history[-1].obs_seq != latest_obs.seq
                or not np.all(gcc_history[-1].q_exe_valid)
                or not self._obs_has_hand_pose(latest_obs)
            ):
                return None
            return PolicyInputSnapshot(
                window=(latest_obs,),
                gcc_history=gcc_history,
                gcc_history_generation=gcc_history_generation,
            )
        if self.provider.strip().lower() in SHARPA62_PROVIDERS:
            samples = [
                sample
                for sample in samples
                if self._obs_has_hand_pose(sample)
            ]
        if not samples:
            return None
        if provider in GROOT_PROVIDERS:
            return PolicyInputSnapshot(window=(samples[-1],))
        if request_index == 0:
            count = min(self.initial_window_frames, len(samples))
            return PolicyInputSnapshot(window=tuple(samples[-count:]))

        needed = (
            (self.policy_window_frames - 1) * self.policy_window_stride + 1
        )
        if len(samples) < needed:
            return None
        recent = samples[-needed:]
        return PolicyInputSnapshot(
            window=tuple(recent[:: self.policy_window_stride])
        )

    def _obs_has_hand_pose(self, sample: ObsSample) -> bool:
        parsed = json_or_raw(sample.payload_json)
        wrapper = parsed.get("json") if parsed.get("valid") else None
        try:
            return self._find_hand_pose_62d(wrapper) is not None
        except (TypeError, ValueError):
            return False

    def _run_policy_request(
        self,
        snapshot: PolicyInputSnapshot,
        request: InferenceRequest,
        request_started_ns: int,
    ) -> bool:
        window = snapshot.window
        if not window:
            return False
        stamp_ns = request_started_ns
        with self.lock:
            self.pred_seq += 1
            pred_seq = self.pred_seq
            request_index = self.request_index
            self.request_index += 1

        if self.dry_run:
            payload = self._dry_run_payload(window[-1], pred_seq, stamp_ns)
            response_received_ns = now_ns()
            request_info = {"request_index": request_index, "mode": "dry_run"}
            payload["_ws_policy_client"] = self._result_metadata(
                window[-1],
                pred_seq,
                stamp_ns,
                0.0,
                request_info,
                request_started_ns,
                response_received_ns,
                request,
            )
        else:
            try:
                if self.policy_protocol == "sharpa_v3":
                    self._ensure_sharpa_v3_client()
                    with self.lock:
                        if not self._sharpa_v3_snapshot_context_is_current_locked(
                            snapshot.sharpa_v3
                        ):
                            return False
                request_payload, request_info = self._build_policy_request(
                    snapshot,
                    inference_request=request,
                )
                request_info["request_index"] = request_index
                result = self._infer_remote(request_payload)
                if self.stop_event.is_set() or not rclpy.ok(context=self.context):
                    return False
                if self.policy_protocol == "sharpa_v3":
                    with self.lock:
                        if not self._sharpa_v3_snapshot_context_is_current_locked(
                            snapshot.sharpa_v3
                        ):
                            return False
                response_received_ns = now_ns()
                payload = self._policy_response_payload(
                    result,
                    window[-1],
                    pred_seq,
                    stamp_ns,
                    request_info,
                    request_started_ns,
                    response_received_ns,
                    request,
                )
            except Exception as exc:  # noqa: BLE001 - keep ROS callback alive.
                with self.lock:
                    self.obs_dropped += 1
                    self.policy_failures += 1
                    self.last_error = f"policy call failed: {exc}"
                return False

        if self.stop_event.is_set() or not rclpy.ok(context=self.context):
            return False

        pred = PolicyPred()
        set_header(pred, "policy_client", self.get_clock())
        pred.seq = pred_seq
        pred.provider = self.provider
        pred.payload_json = json_dumps(_jsonable(payload))
        with self.lock:
            self.last_published_pred_seq = pred_seq
            self.last_pred_request_id = int(request.request_id)
            self.last_pred_msg = pred
            execution = payload.get("_ws_sharpa_v4")
            self.last_action_id = (
                str(execution.get("action_id"))
                if isinstance(execution, dict) and execution.get("action_id")
                else None
            )
        self.pred_pub.publish(pred)

        with self.lock:
            self.pred_published += 1
            self.last_pred_time = time.monotonic()
            self.last_response_received_ns = response_received_ns
            self.last_request_info = request_info
            self.last_error = ""
        return True

    def _dry_run_payload(
        self,
        obs: ObsSample,
        pred_seq: int,
        stamp_ns: int,
    ) -> dict[str, Any]:
        provider = self.provider.strip().lower()
        unified = provider in UNIFIED_SHARPA62_PROVIDERS
        baseline = provider in BASELINE_PROVIDERS
        return {
            "schema": (
                UNIFIED_ACTION_SCHEMA
                if unified
                else "dreamzero_sharpa62_action.v1"
            ),
            "seq": pred_seq,
            "stamp_ns": stamp_ns,
            "provider": self.provider,
            "server_url": self.server_url,
            "dry_run": True,
            "obs": {
                "seq": int(obs.seq),
                "provider": obs.provider,
                "payload": json_or_raw(obs.payload_json),
                "image_rgb_len": len(obs.image_rgb),
            },
            "shape": [self.action_horizon, self.dry_run_action_dim],
            "action_horizon": self.action_horizon,
            "action_hz": self.actor_send_hz,
            "action_hand_pose_62d": [
                [0.0] * self.dry_run_action_dim
                for _ in range(self.action_horizon)
            ],
            "action_hand_pose_62d_relative_eef": [
                [0.0] * self.dry_run_action_dim
                for _ in range(self.action_horizon)
            ],
            "action_space": UNIFIED_ACTION_SPACE,
            "wrist_frame": (
                BASELINE_WRIST_FRAME
                if baseline
                else UNIFIED_WRIST_FRAME
                if unified
                else "hip"
            ),
            "layout": (
                BASELINE_ACTION_LAYOUT
                if baseline
                else UNIFIED_ACTION_LAYOUT
                if unified
                else "left_wrist_9d,right_wrist_9d,sharpa_q44"
            ),
        }

    def _build_policy_request(
        self,
        snapshot: PolicyInputSnapshot,
        *,
        inference_request: InferenceRequest | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        window = snapshot.window
        obs = window[-1]
        if self.policy_protocol == "sharpa_v3":
            if inference_request is None:
                raise ValueError("v3 request builder requires InferenceRequest")
            return self._build_sharpa_v3_request(snapshot, inference_request)
        provider = self.provider.strip().lower()
        if provider in BASELINE_PROVIDERS:
            return self._build_baseline_request(snapshot)
        if provider in GCC_WIRE_PROVIDERS:
            return self._build_gcc_request(snapshot)
        if provider in SHARPA62_PROVIDERS:
            return self._build_sharpa62_request(list(window))

        parsed = json_or_raw(obs.payload_json)
        image_rgb, image_info = self._image_from_obs(obs, parsed.get("json"))
        request = {
            "schema": "ws.policy_observation.v1",
            "endpoint": "infer",
            "provider": self.provider,
            "session_id": self.session_id,
            "observation": parsed,
            "image_rgb": image_rgb,
            "timestamp_unix_s": time.time(),
        }
        return request, {
            "schema": "ws.policy_request_info.v1",
            "provider": self.provider,
            "image": image_info,
            "observation_schema": (
                parsed["json"].get("schema")
                if parsed.get("valid") and isinstance(parsed.get("json"), dict)
                else None
            ),
        }

    def _build_baseline_request(
        self,
        snapshot: PolicyInputSnapshot,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = snapshot.latest
        parsed = json_or_raw(obs.payload_json)
        wrapper = parsed.get("json") if parsed.get("valid") else None
        if not isinstance(wrapper, dict):
            raise ValueError("baseline PolicyObs payload is not valid JSON")
        hand_pose, hand_info = self._hand_pose_from_obs(wrapper)
        model_image = wrapper.get("model_image")
        if (
            not isinstance(model_image, dict)
            or model_image.get("valid") is not True
        ):
            raise ValueError("baseline observation has no valid model image")
        try:
            model_image_age_ms = float(model_image.get("age_ms"))
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline model image age_ms is not numeric") from exc
        if (
            not np.isfinite(model_image_age_ms)
            or model_image_age_ms < 0.0
            or model_image_age_ms > self.baseline_image_max_age_ms
        ):
            raise ValueError(
                "baseline model image is stale: "
                f"age_ms={model_image_age_ms:g}, "
                f"limit_ms={self.baseline_image_max_age_ms:g}"
            )
        image, image_info = self._image_from_obs(obs, wrapper)
        if image_info.get("fallback"):
            raise ValueError("baseline observation has missing or bad image bytes")
        image_jpeg = self._encode_jpeg(image)
        provider = self.provider.strip().lower()
        policy_family = "trex" if provider in TREX_PROVIDERS else "vitacformer"
        request, baseline_info = build_baseline_wire_request(
            snapshot.baseline_history,
            provider=policy_family,
            current_observation=wrapper,
            current_tactile_data=obs.tactile_data,
            current_hand_pose_62d=hand_pose,
            current_ego_view_jpeg=image_jpeg,
            current_obs_seq=obs.seq,
            current_timestamp_unix_s=obs.timestamp_unix_s,
            session_id=self.session_id,
            prompt=self.prompt,
            deformation_max_age_ms=self.baseline_deformation_max_age_ms,
        )
        image_info = dict(image_info)
        image_info.update(
            {
                "transport_encoding": "jpeg",
                "transport_quality": 90,
                "transport_bytes": len(image_jpeg),
            }
        )
        history = snapshot.baseline_history
        return request, {
            "schema": "ws.policy_request_info.v1",
            "provider": self.provider,
            "mode": f"{policy_family}_sharpa62_from_ws_obs",
            "hand_pose": hand_info,
            "image": image_info,
            "baseline": baseline_info,
            "window": {
                "obs_seqs": [int(frame.obs_seq) for frame in history],
                "frame_count": len(history),
                "stride_obs": 1,
                "obs_rate_hz": self.obs_rate_hz,
                "effective_window_hz": self.obs_rate_hz,
                "span_s": (len(history) - 1) / self.obs_rate_hz,
            },
            "observation_schema": wrapper.get("schema"),
        }

    def _build_sharpa_v3_request(
        self,
        snapshot: PolicyInputSnapshot,
        inference_request: InferenceRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        history = snapshot.sharpa_v3
        if history is None:
            raise ValueError("v3 request has no ready history snapshot")
        parsed = json_or_raw(inference_request.payload_json)
        request_payload = parsed.get("json") if parsed.get("valid") else None
        feedback = (
            request_payload.get("execution_feedback")
            if isinstance(request_payload, dict)
            else None
        )
        if self.sharpa_v3_force_initial_feedback:
            feedback = None
        prompt = self.sharpa_v3_effective_prompt or self.prompt
        request, info = build_sharpa_v3_observation(
            history.frames,
            metadata_format=history.metadata_format,
            session_id=self.session_id,
            request_id=int(inference_request.request_id),
            prompt=prompt,
            execution_feedback=feedback,
        )
        info["provider"] = self.provider
        info["request_reason"] = inference_request.reason
        info["request_trigger_action_seq"] = int(
            inference_request.trigger_action_seq
        )
        info["history_generation"] = history.generation
        info["format_revision"] = history.format_revision
        info["stream_lengths"] = history.stream_lengths
        info["stream_capacities"] = history.stream_capacities
        info["stream_required_lengths"] = history.stream_required_lengths
        return request, info

    def _build_gcc_request(
        self,
        snapshot: PolicyInputSnapshot,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = snapshot.latest
        parsed = json_or_raw(obs.payload_json)
        wrapper = parsed.get("json") if parsed.get("valid") else None
        if not isinstance(wrapper, dict):
            raise ValueError("GCC-compatible PolicyObs payload is not valid JSON")
        hand_pose, hand_info = self._hand_pose_from_obs(wrapper)
        image, image_info = self._image_from_obs(obs, wrapper)
        image_jpeg = self._encode_jpeg(image)
        request, gcc_info = build_gcc_wire_request(
            snapshot.gcc_history,
            current_observation=wrapper,
            current_tactile_data=obs.tactile_data,
            current_hand_pose_62d=hand_pose,
            current_ego_view_jpeg=image_jpeg,
            current_obs_seq=obs.seq,
            current_timestamp_unix_s=obs.timestamp_unix_s,
            session_id=self.session_id,
            prompt=self.prompt,
            require_full_real_history=True,
            deformation_max_age_ms=self.gcc_deformation_max_age_ms,
        )
        image_info = dict(image_info)
        image_info.update(
            {
                "transport_encoding": "jpeg",
                "transport_quality": 90,
                "transport_bytes": len(image_jpeg),
            }
        )
        return request, {
            "schema": "ws.policy_request_info.v1",
            "provider": self.provider,
            "mode": (
                "pace_n17_sharpa62_from_ws_obs"
                if self.provider.strip().lower() in PACE_PROVIDERS
                else "gcc_n17_sharpa62_from_ws_obs"
            ),
            "hand_pose": hand_info,
            "image": image_info,
            "gcc": gcc_info,
            "window": {
                "obs_seqs": [
                    int(frame.obs_seq) for frame in snapshot.gcc_history
                ],
                "frame_count": len(snapshot.gcc_history),
                "stride_obs": 1,
                "obs_rate_hz": self.obs_rate_hz,
                "effective_window_hz": self.obs_rate_hz,
                "span_s": (
                    (len(snapshot.gcc_history) - 1) / self.obs_rate_hz
                ),
            },
            "observation_schema": wrapper.get("schema"),
        }

    def _build_sharpa62_request(
        self,
        window: list[ObsSample],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = window[-1]
        parsed = json_or_raw(obs.payload_json)
        wrapper = parsed.get("json") if parsed.get("valid") else {}
        if (
            len(window) == 1
            and isinstance(wrapper, dict)
            and wrapper.get("schema") == "dreamzero_sharpa62_observation.v1"
        ):
            request = dict(wrapper)
            request.setdefault("endpoint", "infer")
            request.setdefault("session_id", self.session_id)
            info = {
                "schema": "ws.policy_request_info.v1",
                "provider": self.provider,
                "mode": "passthrough",
                "observation_schema": wrapper.get("schema"),
            }
            return request, info

        provider = self.provider.strip().lower()
        request_window = [window[-1]] if provider in GROOT_PROVIDERS else window
        hand_poses = []
        images = []
        timestamps = []
        hand_info: dict[str, Any] | None = None
        image_info: dict[str, Any] | None = None
        for sample in request_window:
            sample_parsed = json_or_raw(sample.payload_json)
            sample_wrapper = (
                sample_parsed.get("json") if sample_parsed.get("valid") else {}
            )
            hand_pose, hand_info = self._hand_pose_from_obs(sample_wrapper)
            image_rgb, image_info = self._image_from_obs(sample, sample_wrapper)
            hand_poses.append(hand_pose)
            images.append(image_rgb)
            timestamps.append(sample.timestamp_unix_s)

        if len(request_window) == 1:
            ego_view: np.ndarray = images[0]
            hand_pose_payload: np.ndarray = hand_poses[0]
            timestamp_payload: float | np.ndarray = float(timestamps[0])
        else:
            ego_view = np.stack(images, axis=0).astype(np.uint8)
            hand_pose_payload = np.stack(hand_poses, axis=0).astype(np.float32)
            timestamp_payload = np.asarray(timestamps, dtype=np.float64)

        request: dict[str, Any] = {
            "schema": "dreamzero_sharpa62_observation.v1",
            "endpoint": "infer",
            "session_id": self.session_id,
            "observation/hand_pose_62d": hand_pose_payload,
            "observation/timestamp_unix_s": timestamp_payload,
        }
        if provider in GROOT_PROVIDERS:
            image_jpeg = self._encode_jpeg(ego_view)
            request["observation/ego_view_jpeg"] = image_jpeg
            image_info = dict(image_info or {})
            image_info.update(
                {
                    "transport_encoding": "jpeg",
                    "transport_quality": 90,
                    "transport_bytes": len(image_jpeg),
                }
            )
        else:
            request["observation/ego_view"] = ego_view
        if self.prompt:
            request["prompt"] = self.prompt
        tactile_info: dict[str, Any] = {}
        if provider in GROOT_MOT_PROVIDERS:
            tactile_fields, tactile_info = build_mot_tactile_request(
                wrapper,
                window[-1].tactile_data,
            )
            request.update(tactile_fields)
        return request, {
            "schema": "ws.policy_request_info.v1",
            "provider": self.provider,
            "mode": "sharpa62_from_ws_obs",
            "hand_pose": hand_info or {},
            "image": image_info or {},
            "tactile": tactile_info,
            "window": {
                "obs_seqs": [int(sample.seq) for sample in request_window],
                "frame_count": len(request_window),
                "source_window_frames": len(window),
                "target_frames_after_first": self.policy_window_frames,
                "stride_obs": self.policy_window_stride,
                "obs_rate_hz": self.obs_rate_hz,
                "actor_send_hz": self.actor_send_hz,
                "effective_window_hz": self.obs_rate_hz
                / float(self.policy_window_stride),
                "span_s": (
                    (len(request_window) - 1)
                    * self.policy_window_stride
                    / float(self.obs_rate_hz)
                    if len(request_window) > 1
                    else 0.0
                ),
            },
            "observation_schema": (
                wrapper.get("schema") if isinstance(wrapper, dict) else None
            ),
        }

    def _hand_pose_from_obs(self, wrapper: Any) -> tuple[np.ndarray, dict[str, Any]]:
        hand = self._find_hand_pose_62d(wrapper)
        if hand is not None:
            return hand, {
                "source": "ws_obs_payload_fk_62d",
                "fallback": False,
                "layout": "left_wrist_9d,right_wrist_9d,sharpa_q44",
            }
        if not self.allow_zero_wrist_fallback:
            if not isinstance(wrapper, dict):
                raise ValueError("policy observation is not a JSON object")
            policy_input = wrapper.get("policy_input")
            reason = None
            if isinstance(policy_input, dict):
                reason = policy_input.get("reason")
            detail = f": {reason}" if reason else ""
            raise ValueError(
                "policy observation has no valid FK hand_pose_62d"
                f"{detail}"
            )

        state = self._robot_state_json(wrapper)
        sharpa_q44 = self._sharpa_q44_from_state(state)
        if sharpa_q44 is None:
            hand_pose = np.zeros(62, dtype=np.float32)
            return hand_pose, {
                "source": "zero",
                "fallback": True,
                "reason": "missing_sharpa_q44",
            }
        hand_pose = np.zeros(62, dtype=np.float32)
        hand_pose[18:62] = sharpa_q44
        return hand_pose, {
            "source": "robot_state_sharpa_q44_zero_wrist18",
            "fallback": True,
            "reason": "explicit_zero_wrist_fallback_enabled",
        }

    def _find_hand_pose_62d(self, wrapper: Any) -> np.ndarray | None:
        if not isinstance(wrapper, dict):
            return None
        candidates = [
            wrapper.get("observation/hand_pose_62d"),
            wrapper.get("hand_pose_62d"),
        ]
        for key in ("policy_input", "dreamzero", "converted_state"):
            value = wrapper.get(key)
            if isinstance(value, dict):
                if key == "policy_input" and value.get("valid") is False:
                    continue
                candidates.append(value.get("hand_pose_62d"))
                candidates.append(value.get("observation/hand_pose_62d"))
        for value in candidates:
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape == (62,):
                if not np.all(np.isfinite(array)):
                    raise ValueError("hand_pose_62d contains NaN or Inf")
                return array
            if array.shape == (1, 62):
                if not np.all(np.isfinite(array)):
                    raise ValueError("hand_pose_62d contains NaN or Inf")
                return array[0]
        return None

    def _robot_state_json(self, wrapper: dict[str, Any]) -> dict[str, Any] | None:
        robot_state = wrapper.get("robot_state")
        if not isinstance(robot_state, dict):
            return None
        payload = robot_state.get("payload")
        if not isinstance(payload, dict):
            return None
        state = payload.get("json")
        return state if isinstance(state, dict) else None

    def _sharpa_q44_from_state(
        self,
        state: dict[str, Any] | None,
    ) -> np.ndarray | None:
        if not isinstance(state, dict):
            return None
        sharpa = state.get("sharpa")
        if not isinstance(sharpa, dict):
            return None
        q = sharpa.get("q")
        if not isinstance(q, list) or len(q) != 44:
            return None
        out = np.asarray(q, dtype=np.float32)
        if out.shape != (44,) or not np.all(np.isfinite(out)):
            return None
        return out

    def _image_from_obs(
        self,
        obs: ObsSample,
        wrapper: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        data = bytes(obs.image_rgb)
        width = 0
        height = 0
        encoding = ""
        if isinstance(wrapper, dict):
            model_image = wrapper.get("model_image")
            if isinstance(model_image, dict):
                width = int(model_image.get("width") or 0)
                height = int(model_image.get("height") or 0)
                encoding = str(model_image.get("encoding") or "")
        if width > 0 and height > 0 and len(data) == width * height * 3:
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()
            if encoding.lower() == "bgr8":
                image = image[:, :, ::-1].copy()
            image = self._resize_model_image(image)
            return image, {
                "source": "PolicyObs.image_rgb",
                "fallback": False,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "encoding": "rgb8",
            }
        return np.zeros((160, 320, 3), dtype=np.uint8), {
            "source": "black",
            "fallback": True,
            "reason": "missing_or_bad_model_image",
            "input_bytes": len(data),
            "input_width": width,
            "input_height": height,
            "input_encoding": encoding,
        }

    @staticmethod
    def _resize_model_image(image: np.ndarray) -> np.ndarray:
        if image.shape == (160, 320, 3):
            return image.astype(np.uint8, copy=False)
        try:
            from PIL import Image

            pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
            pil = pil.resize((320, 160), Image.Resampling.BILINEAR)
            return np.asarray(pil, dtype=np.uint8).copy()
        except Exception:
            return np.zeros((160, 320, 3), dtype=np.uint8)

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> bytes:
        from PIL import Image

        output = io.BytesIO()
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(
            output,
            format="JPEG",
            quality=90,
            subsampling=0,
        )
        return output.getvalue()

    def _set_sharpa_v3_format(self, value: Any) -> dict[str, Any]:
        active_format = _validate_sharpa_v3_format_capacity(value)
        with self.lock:
            self.sharpa_v3_history.configure(active_format)
            self.sharpa_v3_active_format = active_format
        return active_format

    def _ensure_sharpa_v3_client(self) -> None:
        if not self.server_url:
            raise ValueError("server_url is empty")
        if self.stop_event.is_set():
            raise ConnectionError("policy client is shutting down")
        with self.client_lock:
            client = self.client
        if client is None:
            candidate = SharpaV3PolicyClient(
                self.server_url,
                timeout_s=self.request_timeout_s,
                ssh_host=self.ssh_host,
                ssh_remote_host=self.ssh_remote_host,
                ssh_remote_port=self.ssh_remote_port,
            )
            try:
                metadata = validate_sharpa_v3_server_metadata(
                    candidate.metadata,
                    configured_prompt=self.prompt,
                    expected_policy_family=(
                        SHARPA_V3_POLICY_FAMILY_BY_PROVIDER.get(
                            self.provider.strip().lower()
                        )
                    ),
                )
                active_format = _validate_sharpa_v3_format_capacity(
                    metadata["metadata_format"]
                )
                effective_prompt = self.prompt or str(
                    metadata.get("prompt") or ""
                )
                if not effective_prompt:
                    raise ValueError(
                        "v3 server and workstation both have an empty task prompt"
                    )
            except Exception:
                candidate.close()
                raise
            installed = False
            with self.client_lock:
                if not self.stop_event.is_set() and self.client is None:
                    self.client = candidate
                    client = candidate
                    installed = True
                else:
                    client = self.client
            if not installed:
                candidate.close()
                if self.stop_event.is_set():
                    raise ConnectionError("policy client is shutting down")
            else:
                with self.lock:
                    self.sharpa_v3_history.configure(active_format)
                    self.sharpa_v3_active_format = active_format
                    self.sharpa_v3_metadata = metadata
                    self.sharpa_v3_effective_prompt = effective_prompt
                    self.sharpa_v3_force_initial_feedback = True
        if not isinstance(client, SharpaV3PolicyClient):
            raise ValueError("v3 client was not initialized")
        with self.lock:
            reset_pending = self.sharpa_v3_reset_pending
            reset_request_id = self.last_request_id
            reset_generation = self.sharpa_v3_history.generation
        if reset_pending:
            reset_result = client.reset(
                self.session_id,
                request_id=reset_request_id,
            )
            if self.stop_event.is_set():
                raise ConnectionError("policy client is shutting down")
            with self.client_lock:
                if self.client is not client:
                    raise ConnectionError("policy client connection was closed")
            if reset_result.get("schema") != "sharpa_policy_reset.v1":
                raise ValueError("v3 reset response schema is invalid")
            if reset_result.get("session_id") != self.session_id:
                raise ValueError("v3 reset response session_id is invalid")
            active_format = _validate_sharpa_v3_format_capacity(
                reset_result.get("metadata_format")
            )
            with self.lock:
                if self.sharpa_v3_history.generation == reset_generation:
                    self.sharpa_v3_history.configure(active_format)
                    self.sharpa_v3_active_format = active_format
                    self.sharpa_v3_reset_pending = False
                    self.sharpa_v3_force_initial_feedback = True

    def _install_legacy_client(
        self,
    ) -> MsgpackPolicyWsClient | SharpaV3PolicyClient:
        if self.stop_event.is_set():
            raise ConnectionError("policy client is shutting down")
        with self.client_lock:
            client = self.client
        if client is not None:
            return client
        candidate = MsgpackPolicyWsClient(
            self.server_url,
            timeout_s=self.request_timeout_s,
            expect_initial_message=self.expect_initial_message,
            ssh_host=self.ssh_host,
            ssh_remote_host=self.ssh_remote_host,
            ssh_remote_port=self.ssh_remote_port,
        )
        try:
            provider = self.provider.strip().lower()
            if provider in BASELINE_PROVIDERS:
                validate_baseline_server_metadata(
                    candidate.metadata,
                    self.provider,
                )
            if provider in PACE_PROVIDERS:
                validate_pace_server_metadata(
                    candidate.metadata,
                    self.provider,
                )
        except Exception:
            candidate.close()
            if self.provider.strip().lower() in BASELINE_PROVIDERS:
                with self.lock:
                    self._clear_baseline_history_locked(
                        "server_metadata_mismatch"
                    )
            if self.provider.strip().lower() in PACE_PROVIDERS:
                with self.lock:
                    self._clear_gcc_history_locked("server_metadata_mismatch")
            raise
        installed = False
        with self.client_lock:
            if not self.stop_event.is_set() and self.client is None:
                self.client = candidate
                client = candidate
                installed = True
            else:
                client = self.client
        if not installed:
            candidate.close()
            if self.stop_event.is_set():
                raise ConnectionError("policy client is shutting down")
        if client is None:
            raise ConnectionError("policy client connection was closed")
        return client

    def _infer_remote(self, request: dict[str, Any]) -> PolicyRpcResult:
        if not self.server_url:
            raise ValueError("server_url is empty")
        if self.stop_event.is_set():
            raise ConnectionError("policy client is shutting down")
        if self.policy_protocol == "sharpa_v3":
            with self.client_lock:
                client = self.client
            if not isinstance(client, SharpaV3PolicyClient):
                raise ValueError("v3 client was not initialized")
            try:
                result = client.infer(request)
            except Exception:
                with self.client_lock:
                    detached = self.client is client
                    if detached:
                        self.client = None
                client.close()
                if detached:
                    with self.lock:
                        self.sharpa_v3_metadata = None
                        self.sharpa_v3_active_format = None
                        self.sharpa_v3_effective_prompt = ""
                        self.sharpa_v3_force_initial_feedback = True
                raise
            with self.lock:
                self.policy_calls += 1
                self.last_latency_s = result.latency_s
            return result
        client = self._install_legacy_client()
        try:
            result = client.infer(request)
        except Exception:
            with self.client_lock:
                detached = self.client is client
                if detached:
                    self.client = None
            client.close()
            if detached:
                provider = self.provider.strip().lower()
                if provider in BASELINE_PROVIDERS:
                    with self.lock:
                        self._clear_baseline_history_locked(
                            "server_connection_reset"
                        )
                if provider in GCC_WIRE_PROVIDERS:
                    with self.lock:
                        self._clear_gcc_history_locked(
                            "server_connection_reset"
                        )
            raise
        with self.lock:
            self.policy_calls += 1
            self.last_latency_s = result.latency_s
        return result

    def _materialize_prediction_video(
        self,
        remote_path: str,
        request_index: int,
    ) -> str | None:
        source = Path(remote_path).expanduser()
        if not self.ssh_host:
            if source.is_file() and source.stat().st_size > 0:
                return str(source.resolve())
            return None

        self.pred_video_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.pred_video_dir / f"request_{int(request_index):06d}.mp4"
        if local_path.is_file() and local_path.stat().st_size > 0:
            return str(local_path)
        temporary_path = local_path.with_suffix(".mp4.tmp")
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            self.ssh_host,
            f"cat -- {shlex.quote(remote_path)}",
        ]
        ssh_environment = os.environ.copy()
        for variable in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "OPENSSL_CONF",
            "OPENSSL_MODULES",
        ):
            ssh_environment.pop(variable, None)
        try:
            if not self.ssh_remote_host:
                raise ValueError("ssh_remote_host is empty")
            with temporary_path.open("wb") as output:
                result = subprocess.run(
                    command,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    env=ssh_environment,
                    timeout=min(30.0, max(5.0, self.request_timeout_s)),
                    check=False,
                )
            if result.returncode != 0:
                return None
            if temporary_path.stat().st_size <= 0:
                return None
            temporary_path.replace(local_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        videos = sorted(
            self.pred_video_dir.glob("request_*.mp4"),
            key=lambda path: path.stat().st_mtime,
        )
        for stale in videos[:-12]:
            stale.unlink(missing_ok=True)
        return str(local_path)

    def _attach_dashboard_video_path(
        self,
        response: dict[str, Any],
        request_info: dict[str, Any],
    ) -> dict[str, Any]:
        video_path = _prediction_video_path(response)
        if not video_path:
            return response
        request_index = int(request_info.get("request_index", 0))
        try:
            local_path = self._materialize_prediction_video(
                video_path,
                request_index,
            )
        except Exception:
            local_path = None
        if local_path:
            response = dict(response)
            response["dashboard_video_path"] = local_path
        return response

    def _policy_response_payload(
        self,
        result: PolicyRpcResult,
        obs: ObsSample,
        pred_seq: int,
        stamp_ns: int,
        request_info: dict[str, Any],
        request_started_ns: int,
        response_received_ns: int,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        response = result.payload
        provider = self.provider.strip().lower()
        if self.policy_protocol == "sharpa_v3":
            response, next_format = action_to_policy_payload(
                response,
                expected_session_id=self.session_id,
                expected_request_id=int(request.request_id),
            )
            if next_format is not None:
                self._set_sharpa_v3_format(next_format)
            self.sharpa_v3_force_initial_feedback = False
        elif provider in BASELINE_PROVIDERS:
            response = validate_baseline_sharpa62_response(response)
        elif provider in UNIFIED_SHARPA62_PROVIDERS:
            response = validate_unified_sharpa62_response(
                response,
                expected_policy_family=(
                    PACE_POLICY_FAMILY
                    if provider in PACE_PROVIDERS
                    else None
                ),
            )
        elif provider in SHARPA62_PROVIDERS:
            action = np.asarray(
                response.get("action_hand_pose_62d"),
                dtype=np.float32,
            )
            expected_shape = (self.action_horizon, 62)
            if action.shape != expected_shape:
                raise ValueError(
                    f"action_hand_pose_62d shape is {action.shape}, "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError("action_hand_pose_62d contains NaN or Inf")
            response = dict(response)
            response["action_hand_pose_62d"] = action
            response.setdefault("schema", "dreamzero_sharpa62_action.v1")
            response.setdefault("action_space", "sharpa_dexretarget_position_62d")
            server_horizon = self._optional_number(response.get("action_horizon"))
            if (
                server_horizon is not None
                and int(server_horizon) != self.action_horizon
            ):
                raise ValueError(
                    f"server action_horizon={server_horizon}, "
                    f"expected {self.action_horizon}"
                )
            server_hz = self._optional_number(response.get("action_hz"))
            if server_hz is not None and not np.isclose(
                server_hz,
                self.actor_send_hz,
            ):
                raise ValueError(
                    f"server action_hz={server_hz}, expected {self.actor_send_hz}"
                )
            response["action_horizon"] = self.action_horizon
            response["action_hz"] = self.actor_send_hz
        response = self._attach_dashboard_video_path(response, request_info)
        response["_ws_policy_client"] = self._result_metadata(
            obs,
            pred_seq,
            stamp_ns,
            result.latency_s,
            request_info,
            request_started_ns,
            response_received_ns,
            request,
        )
        return response

    def _result_metadata(
        self,
        obs: ObsSample,
        pred_seq: int,
        stamp_ns: int,
        latency_s: float,
        request_info: dict[str, Any],
        request_started_ns: int,
        response_received_ns: int,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        return {
            "schema": "ws.policy_client.result_meta.v1",
            "seq": pred_seq,
            "stamp_ns": stamp_ns,
            "provider": self.provider,
            "server_url": self.server_url,
            "latency_s": latency_s,
            "obs_seq": int(obs.seq),
            "request_obs_stamp_ns": int(obs.stamp_ns),
            "request_started_ns": int(request_started_ns),
            "response_received_ns": int(response_received_ns),
            "request_id": int(request.request_id),
            "request_stamp_ns": int(request.request_stamp_ns),
            "request_trigger_action_seq": int(request.trigger_action_seq),
            "request_reason": request.reason,
            "request_payload": json_or_raw(request.payload_json),
            "robot_state_anchor": self._robot_state_wrapper(obs),
            "request": request_info,
        }

    @staticmethod
    def _robot_state_wrapper(obs: ObsSample) -> dict[str, Any] | None:
        parsed = json_or_raw(obs.payload_json)
        wrapper = parsed.get("json") if parsed.get("valid") else None
        if not isinstance(wrapper, dict):
            return None
        robot_state = wrapper.get("robot_state")
        return robot_state if isinstance(robot_state, dict) else None

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        if value is None:
            return None
        array = np.asarray(value)
        if array.size != 1:
            return None
        try:
            number = float(array.reshape(-1)[0])
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self.lock:
            v3_lengths = self.sharpa_v3_history.stream_lengths()
            v3_capacities = self.sharpa_v3_history.stream_capacities()
            v3_required = self.sharpa_v3_history.stream_required_lengths()
            payload = {
                "schema": "ws.policy_client.status.v1",
                "provider": self.provider,
                "policy_protocol": self.policy_protocol,
                "server_url": self.server_url,
                "transport": {
                    "kind": "ssh_command_stream" if self.ssh_host else "direct",
                    "ssh_host": self.ssh_host,
                    "remote_host": self.ssh_remote_host if self.ssh_host else "",
                    "remote_port": self.ssh_remote_port if self.ssh_host else 0,
                },
                "session_id": self.session_id,
                "sharpa_v3": {
                    "connected": isinstance(self.client, SharpaV3PolicyClient),
                    "policy_family": (
                        self.sharpa_v3_metadata.get("policy_family")
                        if isinstance(self.sharpa_v3_metadata, dict)
                        else None
                    ),
                    "metadata_format_id": (
                        self.sharpa_v3_active_format.get("format_id")
                        if isinstance(self.sharpa_v3_active_format, dict)
                        else None
                    ),
                    "effective_prompt": self.sharpa_v3_effective_prompt,
                    "history_ready": self.sharpa_v3_history.ready,
                    "history_generation": self.sharpa_v3_history.generation,
                    "format_revision": self.sharpa_v3_history.format_revision,
                    "history_retained": max(v3_lengths.values(), default=0),
                    "history_capacity": max(v3_capacities.values(), default=0),
                    "history_required": max(v3_required.values(), default=0),
                    "streams": {
                        name: {
                            "length": v3_lengths[name],
                            "capacity": v3_capacities[name],
                            "required": v3_required[name],
                            "ready": (
                                v3_required[name] == 0
                                or v3_lengths[name] >= v3_required[name]
                            ),
                        }
                        for name in v3_capacities
                    },
                    "history_resets": self.sharpa_v3_history_resets,
                    "history_last_reset_reason": (
                        self.sharpa_v3_history_last_reset_reason
                    ),
                },
                "dry_run": self.dry_run,
                "request_timeout_s": self.request_timeout_s,
                "schedule": {
                    "mode": "policy_client_synchronous",
                    "execution_done_topic": getattr(
                        self, "execution_done_topic", "/ws/execution_done"
                    ),
                    "action_horizon": self.action_horizon,
                    "actor_send_hz": self.actor_send_hz,
                    "obs_rate_hz": self.obs_rate_hz,
                    "policy_window_frames": self.policy_window_frames,
                    "policy_window_stride": self.policy_window_stride,
                    "initial_window_frames": self.initial_window_frames,
                    "baseline_history_frames": BASELINE_HISTORY_FRAMES,
                    "baseline_history_max_gap_s": (
                        self.baseline_history_max_gap_s
                    ),
                    "baseline_image_max_age_ms": (
                        self.baseline_image_max_age_ms
                    ),
                    "baseline_wrench_max_age_ms": (
                        self.baseline_wrench_max_age_ms
                    ),
                    "baseline_deformation_max_age_ms": (
                        self.baseline_deformation_max_age_ms
                    ),
                    "gcc_history_frames": self.gcc_history_frames,
                    "gcc_history_max_gap_s": self.gcc_history_max_gap_s,
                    "gcc_joint_max_age_ms": self.gcc_joint_max_age_ms,
                    "gcc_wrench_max_age_ms": self.gcc_wrench_max_age_ms,
                    "gcc_deformation_max_age_ms": (
                        self.gcc_deformation_max_age_ms
                    ),
                },
                "counts": {
                    "obs_received": self.obs_received,
                    "pred_published": self.pred_published,
                    "obs_dropped": self.obs_dropped,
                    "policy_calls": self.policy_calls,
                    "policy_failures": self.policy_failures,
                    "execution_done_received": getattr(
                        self, "execution_done_received", 0
                    ),
                },
                "request": {
                    "index_next": self.request_index,
                    "inflight": self.request_inflight,
                    "inflight_age_ms": age_ms(self.last_request_start_time, now),
                    "obs_buffer_len": len(self.obs_buffer),
                    "baseline_history_len": len(self.baseline_history),
                    "baseline_history_real_count": len(
                        self.baseline_history
                    ),
                    "baseline_history_resets": self.baseline_history_resets,
                    "baseline_history_last_reset_reason": (
                        self.baseline_history_last_reset_reason
                    ),
                    "gcc_history_len": len(self.gcc_history),
                    "gcc_history_real_count": len(self.gcc_history),
                    "gcc_history_resets": self.gcc_history_resets,
                    "gcc_history_last_reset_reason": (
                        self.gcc_history_last_reset_reason
                    ),
                    "last_request_id": self.last_request_id,
                    "last_action_id": getattr(self, "last_action_id", None),
                    "last_published_pred_seq": self.last_published_pred_seq,
                    "last_trigger_action_seq": self.last_trigger_action_seq,
                },
                "last": {
                    "obs_seq": self.last_obs_seq,
                    "obs_age_ms": age_ms(self.last_obs_time, now),
                    "pred_age_ms": age_ms(self.last_pred_time, now),
                    "latency_s": self.last_latency_s,
                    "request_started_ns": self.last_request_started_ns,
                    "response_received_ns": self.last_response_received_ns,
                    "request": self.last_request_info,
                },
                "last_error": self.last_error,
            }
            ok = (self.dry_run or bool(self.server_url)) and not self.last_error
        self.status_pub.publish(
            make_status(self.get_clock(), "policy_client", ok, payload)
        )

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self.request_event.set()
        with self.client_lock:
            client = self.client
            self.client = None
        if client is not None:
            client.close()
        worker = getattr(self, "worker", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyClient()
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
