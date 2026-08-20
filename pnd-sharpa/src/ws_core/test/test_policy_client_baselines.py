from __future__ import annotations

from collections import deque
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from ws_core.baseline_wire import BASELINE_HISTORY_FRAMES, FINGER_ORDER
from ws_core.policy_client import (
    BASELINE_ACTION_HORIZON,
    BASELINE_ACTION_HZ,
    BASELINE_ACTION_LAYOUT,
    BASELINE_ACTION_SCHEMA,
    BASELINE_ACTION_SPACE,
    BASELINE_PROVIDERS,
    BASELINE_REQUEST_SCHEMA,
    BASELINE_SERVER_SCHEMA,
    BASELINE_WRIST_FRAME,
    PolicyClient,
    TREX_PROVIDERS,
    VITACFORMER_PROVIDERS,
    validate_baseline_sharpa62_response,
    validate_baseline_server_metadata,
)


TACTILE_LAYOUT = "sharpa_tactile_right_then_left_pinky_to_thumb.v1"


def wrapper(seq: int) -> dict:
    entries = []
    for index, name in enumerate(FINGER_ORDER):
        side, finger = name.split("_", 1)
        entries.append(
            {
                "side": side,
                "finger": finger,
                "valid": True,
                "raw_offset": index * 16,
                "raw_length": 16,
                "height": 4,
                "width": 4,
            }
        )
    return {
        "schema": "ws.policy_obs.v1",
        "stamp_ns": 1_000_000 + seq,
        "policy_input": {
            "valid": True,
            "hand_pose_62d": [float(seq)] * 62,
        },
        "robot_state": {
            "age_ms": 5.0,
            "payload": {
                "valid": True,
                "json": {
                    "tactile": {
                        "force_age_ms": 10.0,
                        "order": list(FINGER_ORDER),
                        "tactile_layout": TACTILE_LAYOUT,
                        "wrench": [[float(seq)] * 6] * 10,
                        "wrench_valid": [True] * 10,
                    }
                },
            },
        },
        "model_image": {
            "valid": True,
            "age_ms": 5.0,
            "width": 320,
            "height": 160,
            "encoding": "rgb8",
        },
        "robot_tactile": {
            "age_ms": 10.0,
            "metadata": {"valid": True, "json": {"entries": entries}},
        },
    }


def message(seq: int) -> SimpleNamespace:
    tactile_data = b"".join(bytes([index]) * 16 for index in range(10))
    return SimpleNamespace(
        seq=seq,
        provider="obs_sync",
        payload_json=json.dumps(wrapper(seq)),
        image_rgb=np.zeros((160, 320, 3), dtype=np.uint8).tobytes(),
        tactile_data=tactile_data,
    )


def make_callback_client(provider: str) -> PolicyClient:
    client = object.__new__(PolicyClient)
    client.provider = provider
    client.session_id = "test-session"
    client.prompt = "Place the egg and apple."
    client.allow_zero_wrist_fallback = False
    client.lock = threading.Lock()
    client.request_event = threading.Event()
    client.obs_buffer = deque(maxlen=128)
    client.latest_obs = None
    client.baseline_history = deque(maxlen=BASELINE_HISTORY_FRAMES)
    client.baseline_history_generation = 0
    client.baseline_history_max_gap_s = 0.25
    client.baseline_image_max_age_ms = 150.0
    client.baseline_wrench_max_age_ms = 150.0
    client.baseline_deformation_max_age_ms = 150.0
    client.baseline_history_resets = 0
    client.baseline_history_last_reset_reason = ""
    client.last_baseline_obs_seq = None
    client.last_baseline_obs_time = None
    client.gcc_history = deque(maxlen=9)
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
    client.obs_rate_hz = 30.0
    client.actor_send_hz = 15.0
    client.last_request_id = 0
    client.last_pred_request_id = None
    client.last_pred_msg = None
    client.pending_request = None
    client.request_inflight = False
    client.last_trigger_action_seq = None
    return client


@pytest.mark.parametrize("provider", sorted(TREX_PROVIDERS))
def test_trex_waits_for_sixteen_real_observation_ticks(provider: str) -> None:
    client = make_callback_client(provider)
    for seq in range(1, 16):
        client._on_obs(message(seq))

    assert client._select_obs_window() is None
    client._on_obs(message(16))

    snapshot = client._select_obs_window()
    assert snapshot is not None
    assert [frame.obs_seq for frame in snapshot.baseline_history] == list(
        range(1, 17)
    )
    request, info = client._build_policy_request(snapshot)
    assert request["observation/tactile_wrench_history_16x10x6"].shape == (
        16,
        10,
        6,
    )
    assert request["observation/hand_pose_62d"].shape == (62,)
    assert request["observation/tactile_deformation_10x64x64"].shape == (
        10,
        64,
        64,
    )
    assert info["mode"] == "trex_sharpa62_from_ws_obs"


@pytest.mark.parametrize("provider", sorted(VITACFORMER_PROVIDERS))
def test_vitacformer_waits_for_eighteen_ticks_and_uses_last_sixteen_states(
    provider: str,
) -> None:
    client = make_callback_client(provider)
    for seq in range(1, 18):
        client._on_obs(message(seq))

    assert client._select_obs_window() is None
    client._on_obs(message(18))

    snapshot = client._select_obs_window()
    assert snapshot is not None
    assert [frame.obs_seq for frame in snapshot.baseline_history] == list(
        range(1, 19)
    )
    request, info = client._build_policy_request(snapshot)
    wrench = request["observation/tactile_wrench_history_18x10x6"]
    states = request["observation/hand_pose_history_16x62"]
    assert wrench.shape == (18, 10, 6)
    assert wrench[0, 0, 0] == 1.0
    assert wrench[-1, 0, 0] == 18.0
    assert states.shape == (16, 62)
    assert states[0, 0] == 3.0
    assert states[-1, 0] == 18.0
    assert info["mode"] == "vitacformer_sharpa62_from_ws_obs"


def test_baseline_history_clears_on_seq_rollback_and_long_gap() -> None:
    client = make_callback_client("vitacformer")
    for seq in range(1, 5):
        client._on_obs(message(seq))

    client._on_obs(message(1))
    assert [frame.obs_seq for frame in client.baseline_history] == [1]
    assert client.baseline_history_last_reset_reason == "obs_seq_reset"

    client.last_baseline_obs_time -= 1.0
    client._on_obs(message(2))
    assert [frame.obs_seq for frame in client.baseline_history] == [2]
    assert client.baseline_history_last_reset_reason == "observation_gap"


def test_pipeline_reset_clears_baseline_history() -> None:
    client = make_callback_client("trex")
    for seq in range(1, 5):
        client._on_obs(message(seq))
    request = SimpleNamespace(
        request_id=1,
        trigger_action_seq=0,
        reason="pipeline_reset",
        payload_json="{}",
    )

    client._on_inference_request(request)

    assert not client.baseline_history
    assert client.baseline_history_last_reset_reason == "pipeline_reset"
    assert client.pending_request is request
    assert client.request_event.is_set()


@pytest.mark.parametrize("provider", sorted(BASELINE_PROVIDERS))
def test_baseline_provider_routes_through_baseline_request_builder(
    provider: str,
) -> None:
    client = make_callback_client(provider)
    required = 16 if provider in TREX_PROVIDERS else 18
    for seq in range(1, required + 1):
        client._on_obs(message(seq))
    snapshot = client._select_obs_window()
    assert snapshot is not None

    request, _ = client._build_policy_request(snapshot)

    assert request["schema"] == "dreamzero_sharpa62_observation.v1"
    assert isinstance(request["observation/ego_view_jpeg"], bytes)
    assert request["session_id"] == "test-session"
    assert request["prompt"] == "Place the egg and apple."
    if provider in TREX_PROVIDERS:
        assert "observation/tactile_wrench_history_16x10x6" in request
        assert "observation/tactile_deformation_10x64x64" in request
    else:
        assert "observation/tactile_wrench_history_18x10x6" in request
        assert "observation/hand_pose_history_16x62" in request


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["model_image"].update(valid=False), "valid"),
        (lambda payload: payload["model_image"].update(age_ms=151.0), "stale"),
        (lambda payload: payload["model_image"].update(width=321), "image bytes"),
    ],
)
def test_baseline_request_rejects_invalid_or_stale_image(
    mutation,
    match: str,
) -> None:
    client = make_callback_client("trex")
    for seq in range(1, 17):
        client._on_obs(message(seq))
    snapshot = client._select_obs_window()
    assert snapshot is not None
    payload = json.loads(snapshot.latest.payload_json)
    mutation(payload)
    bad_sample = snapshot.latest.__class__(
        **{
            **snapshot.latest.__dict__,
            "payload_json": json.dumps(payload),
        }
    )
    bad_snapshot = snapshot.__class__(
        window=(bad_sample,),
        baseline_history=snapshot.baseline_history,
        baseline_history_generation=snapshot.baseline_history_generation,
    )

    with pytest.raises(ValueError, match=match):
        client._build_policy_request(bad_snapshot)


def valid_baseline_response() -> dict:
    return {
        "schema": BASELINE_ACTION_SCHEMA,
        "action_hand_pose_62d": np.zeros(
            (BASELINE_ACTION_HORIZON, 62),
            dtype=np.float32,
        ),
        "action_horizon": BASELINE_ACTION_HORIZON,
        "action_hz": BASELINE_ACTION_HZ,
        "action_space": BASELINE_ACTION_SPACE,
        "wrist_frame": BASELINE_WRIST_FRAME,
        "layout": BASELINE_ACTION_LAYOUT,
    }


def valid_server_metadata(provider: str) -> dict:
    is_trex = provider in TREX_PROVIDERS
    return {
        "schema": BASELINE_SERVER_SCHEMA,
        "policy_family": "trex" if is_trex else "vitacformer",
        "request_schema": BASELINE_REQUEST_SCHEMA,
        "response_schema": BASELINE_ACTION_SCHEMA,
        "action_horizon": BASELINE_ACTION_HORIZON,
        "action_hz": BASELINE_ACTION_HZ,
        "action_space": BASELINE_ACTION_SPACE,
        "output_wrist_frame": BASELINE_WRIST_FRAME,
        "layout": BASELINE_ACTION_LAYOUT,
        "required_state_history": 1 if is_trex else 16,
        "required_wrench_history": 16 if is_trex else 18,
        "required_tactile_deformation": is_trex,
    }


@pytest.mark.parametrize("provider", sorted(BASELINE_PROVIDERS))
def test_baseline_server_metadata_rejects_wrong_policy_on_shared_port(
    provider: str,
) -> None:
    metadata = valid_server_metadata(provider)
    assert validate_baseline_server_metadata(metadata, provider) == metadata

    metadata["policy_family"] = (
        "vitacformer" if provider in TREX_PROVIDERS else "trex"
    )
    with pytest.raises(ValueError, match="policy_family"):
        validate_baseline_server_metadata(metadata, provider)


def test_baseline_response_validation_accepts_only_exact_contract() -> None:
    validated = validate_baseline_sharpa62_response(valid_baseline_response())
    assert validated["action_hand_pose_62d"].shape == (24, 62)
    assert validated["action_hand_pose_62d"].dtype == np.float32

    mutations = (
        ("schema", "sharpa62_policy_action.v1"),
        ("action_horizon", 23),
        ("action_hz", 30.0),
        ("action_space", "other"),
        ("wrist_frame", "hip"),
        ("layout", "left_wrist_9d,right_wrist_9d,sharpa_q44"),
    )
    for field, value in mutations:
        response = valid_baseline_response()
        response[field] = value
        with pytest.raises(ValueError):
            validate_baseline_sharpa62_response(response)

    for field in (
        "schema",
        "action_horizon",
        "action_hz",
        "action_space",
        "wrist_frame",
        "layout",
    ):
        response = valid_baseline_response()
        del response[field]
        with pytest.raises(ValueError):
            validate_baseline_sharpa62_response(response)

    response = valid_baseline_response()
    response["action_hand_pose_62d"] = np.zeros((23, 62), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_baseline_sharpa62_response(response)

    response = valid_baseline_response()
    response["action_hand_pose_62d"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_baseline_sharpa62_response(response)

    response = valid_baseline_response()
    response["action_hand_pose_62d"] = response[
        "action_hand_pose_62d"
    ].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        validate_baseline_sharpa62_response(response)
