from __future__ import annotations

from collections import deque
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from ws_core.gcc_wire import (
    FINGER_ORDER,
    GCC_HISTORY_FRAMES,
    GCC_REQUEST_SCHEMA,
)
from ws_core.policy_client import (
    GCC_WIRE_PROVIDERS,
    PACE_POLICY_FAMILY,
    PACE_PROVIDERS,
    PACE_REQUEST_SCHEMA,
    PolicyClient,
    UNIFIED_ACTION_HZ,
    UNIFIED_ACTION_HORIZON,
    UNIFIED_ACTION_LAYOUT,
    UNIFIED_ACTION_SCHEMA,
    UNIFIED_ACTION_SPACE,
    UNIFIED_WRIST_FRAME,
    validate_pace_server_metadata,
    validate_unified_sharpa62_response,
)


JOINT_ORDER = [f"joint_{index}" for index in range(44)]


def wrapper(seq: int) -> dict:
    return {
        "schema": "ws.policy_obs.v1",
        "stamp_ns": 1_000_000 + seq,
        "policy_input": {
            "valid": True,
            "hand_pose_62d": [0.0] * 18 + [float(seq)] * 44,
        },
        "robot_state": {
            "age_ms": 5.0,
            "payload": {
                "valid": True,
                "json": {
                    "sharpa": {
                        "age_ms": 10.0,
                        "q_exe": [float(seq)] * 44,
                        "q_exe_valid": [True] * 44,
                        "q_cmd": [float(seq)] * 44,
                        "q_cmd_valid": [True] * 44,
                        "tau": [float(seq)] * 44,
                        "tau_valid": [True] * 44,
                        # All frames intentionally share this source stamp:
                        # history is driven by /ws/obs, not SharpA callbacks.
                        "feedback_stamp_ns": 777,
                        "joint_order": JOINT_ORDER,
                        "joint_layout": "sharpa_joint_order.v1",
                    },
                    "tactile": {
                        "force_age_ms": 10.0,
                        "order": list(FINGER_ORDER),
                        "tactile_layout": (
                            "sharpa_tactile_right_then_left_pinky_to_thumb.v1"
                        ),
                        "wrench": [[float(seq)] * 6] * 10,
                        "wrench_valid": [True] * 10,
                    },
                },
            }
        },
        "model_image": {"width": 320, "height": 160, "encoding": "rgb8"},
        "robot_tactile": {
            "age_ms": 10.0,
            "metadata": {"valid": True, "json": {"entries": []}}
        },
    }


def message(seq: int) -> SimpleNamespace:
    return SimpleNamespace(
        seq=seq,
        provider="obs_sync",
        payload_json=json.dumps(wrapper(seq)),
        image_rgb=b"",
        tactile_data=b"",
    )


def make_callback_client(provider: str) -> PolicyClient:
    client = object.__new__(PolicyClient)
    client.provider = provider
    client.lock = threading.Lock()
    client.obs_buffer = deque(maxlen=128)
    client.latest_obs = None
    client.gcc_history = deque(maxlen=GCC_HISTORY_FRAMES)
    client.gcc_history_generation = 0
    client.gcc_history_max_gap_s = 0.25
    client.gcc_joint_max_age_ms = 150.0
    client.gcc_wrench_max_age_ms = 150.0
    client.gcc_deformation_max_age_ms = 150.0
    client.gcc_history_resets = 0
    client.gcc_history_last_reset_reason = ""
    client.last_gcc_obs_seq = None
    client.last_gcc_obs_time = None
    client.obs_received = 0
    client.last_obs_seq = None
    client.last_obs_time = None
    client.last_error = ""
    client.request_index = 0
    client.initial_window_frames = 1
    client.policy_window_frames = 4
    client.policy_window_stride = 2
    client.allow_zero_wrist_fallback = False
    client.obs_rate_hz = 30.0
    client.session_id = "pace-test"
    client.prompt = "Unscrew the bottle cap."
    return client


@pytest.mark.parametrize("provider", ["gcc", "gcc_n17_sharpa62"])
def test_gcc_history_uses_ws_obs_ticks_without_sharpa_deduplication(
    provider: str,
) -> None:
    client = make_callback_client(provider)
    for seq in range(1, 10):
        client._on_obs(message(seq))

    history_seqs = [frame.obs_seq for frame in client.gcc_history]
    assert history_seqs == list(range(1, 10))
    assert len(client.obs_buffer) == 1
    snapshot = client._select_obs_window()
    assert snapshot is not None
    assert snapshot.latest.seq == 9
    snapshot_seqs = [frame.obs_seq for frame in snapshot.gcc_history]
    assert snapshot_seqs == list(range(1, 10))


@pytest.mark.parametrize("provider", sorted(PACE_PROVIDERS))
def test_pace_uses_gcc_compatible_nine_frame_request(
    provider: str,
) -> None:
    assert provider in GCC_WIRE_PROVIDERS
    client = make_callback_client(provider)
    for seq in range(1, 10):
        client._on_obs(message(seq))

    snapshot = client._select_obs_window()
    assert snapshot is not None
    assert [frame.obs_seq for frame in snapshot.gcc_history] == list(range(1, 10))
    request, info = client._build_policy_request(snapshot)

    assert request["schema"] == GCC_REQUEST_SCHEMA
    assert request["session_id"] == "pace-test"
    assert request["prompt"] == "Unscrew the bottle cap."
    assert request["observation/q_exe_history_9x44"].shape == (9, 44)
    assert request["observation/q_cmd_history_9x44"].shape == (9, 44)
    assert request["observation/tau_history_9x44"].shape == (9, 44)
    assert request["observation/tactile_wrench_history_9x10x6"].shape == (
        9,
        10,
        6,
    )
    assert request["observation/tactile_deformation_10x64x64"].shape == (
        10,
        64,
        64,
    )
    assert int(request["history_real_count"]) == 9
    assert info["provider"] == provider
    assert info["mode"] == "pace_n17_sharpa62_from_ws_obs"


def test_pace_snapshot_generation_changes_when_history_resets() -> None:
    client = make_callback_client("pace")
    for seq in range(1, 10):
        client._on_obs(message(seq))
    snapshot = client._select_obs_window()
    assert snapshot is not None

    with client.lock:
        client._clear_gcc_history_locked("pipeline_reset")

    assert snapshot.gcc_history_generation != client.gcc_history_generation
    assert client.gcc_history_last_reset_reason == "pipeline_reset"
    assert client._select_obs_window() is None


def test_gcc_history_clears_on_obs_seq_rollback_and_long_gap() -> None:
    client = make_callback_client("gcc_n17_sharpa62")
    for seq in range(1, 5):
        client._on_obs(message(seq))
    client._on_obs(message(1))
    assert [frame.obs_seq for frame in client.gcc_history] == [1]
    assert client.gcc_history_last_reset_reason == "obs_seq_reset"

    client.last_gcc_obs_time -= 1.0
    client._on_obs(message(2))
    assert [frame.obs_seq for frame in client.gcc_history] == [2]
    assert client.gcc_history_last_reset_reason == "observation_gap"


def test_gcc_does_not_request_with_stale_current_joint_state() -> None:
    client = make_callback_client("gcc_n17_sharpa62")
    for seq in range(1, 10):
        payload = wrapper(seq)
        if seq == 9:
            payload["robot_state"]["payload"]["json"]["sharpa"][
                "age_ms"
            ] = 151.0
        msg = message(seq)
        msg.payload_json = json.dumps(payload)
        client._on_obs(msg)

    assert len(client.gcc_history) == 9
    assert not client.gcc_history[-1].q_exe_valid.any()
    assert client._select_obs_window() is None

    fresh_inner_stale_transport = wrapper(10)
    fresh_inner_stale_transport["robot_state"]["age_ms"] = 151.0
    msg = message(10)
    msg.payload_json = json.dumps(fresh_inner_stale_transport)
    client._on_obs(msg)
    assert not client.gcc_history[-1].q_exe_valid.any()
    assert not client.gcc_history[-1].tactile_wrench_valid.any()
    assert client._select_obs_window() is None


@pytest.mark.parametrize("provider", ["groot", "groot_n17_sharpa62"])
def test_groot_retains_only_current_observation_and_no_gcc_history(
    provider: str,
) -> None:
    client = make_callback_client(provider)
    for seq in range(1, 10):
        client._on_obs(message(seq))

    assert len(client.obs_buffer) == 1
    assert client.obs_buffer[-1].seq == 9
    assert not client.gcc_history
    snapshot = client._select_obs_window()
    assert snapshot is not None
    assert [sample.seq for sample in snapshot.window] == [9]


def valid_response() -> dict:
    return {
        "schema": UNIFIED_ACTION_SCHEMA,
        "action_hand_pose_62d": np.zeros(
            (UNIFIED_ACTION_HORIZON, 62),
            dtype=np.float32,
        ),
        "action_horizon": UNIFIED_ACTION_HORIZON,
        "action_hz": UNIFIED_ACTION_HZ,
        "action_space": UNIFIED_ACTION_SPACE,
        "wrist_frame": UNIFIED_WRIST_FRAME,
        "layout": UNIFIED_ACTION_LAYOUT,
    }


def valid_pace_server_metadata() -> dict:
    return {
        "schema": "sharpa62_policy_server.v1",
        "request_schema": PACE_REQUEST_SCHEMA,
        "accepted_request_schemas": [PACE_REQUEST_SCHEMA, GCC_REQUEST_SCHEMA],
        "response_schema": UNIFIED_ACTION_SCHEMA,
        "policy_family": PACE_POLICY_FAMILY,
        "execute_joint_source": "reconstructed_q_cmd",
        "state_dim": 62,
        "action_dim": 62,
        "action_horizon": UNIFIED_ACTION_HORIZON,
        "action_hz": UNIFIED_ACTION_HZ,
        "history_length": GCC_HISTORY_FRAMES,
        "action_space": UNIFIED_ACTION_SPACE,
        "output_wrist_frame": UNIFIED_WRIST_FRAME,
        "layout": UNIFIED_ACTION_LAYOUT,
    }


@pytest.mark.parametrize("provider", sorted(PACE_PROVIDERS))
def test_pace_metadata_rejects_compatible_wrong_policy(provider: str) -> None:
    metadata = valid_pace_server_metadata()
    assert validate_pace_server_metadata(metadata, provider) == metadata

    metadata["policy_family"] = "gcc_n17"
    with pytest.raises(ValueError, match="policy_family"):
        validate_pace_server_metadata(metadata, provider)


def test_pace_response_requires_native_float32_and_matching_family() -> None:
    response = valid_response()
    response["metadata"] = {
        "policy_family": PACE_POLICY_FAMILY,
        "execute_joint_source": "predicted_q_exe",
    }
    validated = validate_unified_sharpa62_response(
        response,
        expected_policy_family=PACE_POLICY_FAMILY,
    )
    assert validated["action_hand_pose_62d"].dtype == np.float32

    response["metadata"]["policy_family"] = "gcc_n17"
    with pytest.raises(ValueError, match="policy_family"):
        validate_unified_sharpa62_response(
            response,
            expected_policy_family=PACE_POLICY_FAMILY,
        )

    response = valid_response()
    response["action_hand_pose_62d"] = np.zeros((40, 62), dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        validate_unified_sharpa62_response(
            response,
            expected_policy_family=PACE_POLICY_FAMILY,
        )


def test_unified_response_validation_accepts_only_exact_contract() -> None:
    validated = validate_unified_sharpa62_response(valid_response())
    assert validated["action_hand_pose_62d"].shape == (40, 62)

    mutations = [
        ("schema", "groot_n17_sharpa62_action.v1"),
        ("action_horizon", 39),
        ("action_hz", 15.0),
        ("action_space", "other"),
        ("wrist_frame", "hip"),
        ("layout", "left_wrist_9d,right_wrist_9d,sharpa_q44"),
    ]
    for field, value in mutations:
        response = valid_response()
        response[field] = value
        with pytest.raises(ValueError):
            validate_unified_sharpa62_response(response)

    response = valid_response()
    del response["schema"]
    with pytest.raises(ValueError):
        validate_unified_sharpa62_response(response)

    response = valid_response()
    response["action_hand_pose_62d"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_unified_sharpa62_response(response)
