from __future__ import annotations

import json

import numpy as np

from ws_core.policy_client import ObsSample, PolicyClient
from ws_core.tactile_wire import DEFORMATION_KEY, FINGER_ORDER, FORCE_KEY


def make_client(provider: str) -> PolicyClient:
    client = object.__new__(PolicyClient)
    client.provider = provider
    client.session_id = "test-session"
    client.prompt = "test prompt"
    client.allow_zero_wrist_fallback = False
    client.policy_window_frames = 4
    client.policy_window_stride = 2
    client.obs_rate_hz = 30.0
    client.actor_send_hz = 30.0
    return client


def make_sample() -> ObsSample:
    raw = b"".join(bytes([index]) * 16 for index in range(10))
    entries = [
        {
            "side": name.split("_", 1)[0],
            "finger": name.split("_", 1)[1],
            "valid": True,
            "offset": index * 16,
            "length": 16,
            "height": 4,
            "width": 4,
        }
        for index, name in enumerate(FINGER_ORDER)
    ]
    wrapper = {
        "policy_input": {
            "valid": True,
            "hand_pose_62d": [0.0] * 62,
        },
        "model_image": {"width": 320, "height": 160, "encoding": "rgb8"},
        "robot_state": {
            "payload": {
                "valid": True,
                "json": {
                    "tactile": {
                        "order": list(FINGER_ORDER),
                        "force": [[1.0, 2.0, 3.0]] * 10,
                        "torque": [[4.0, 5.0, 6.0]] * 10,
                        "force_valid": [True] * 10,
                    }
                },
            }
        },
        "robot_tactile": {
            "metadata": {"valid": True, "json": {"entries": entries}}
        },
    }
    return ObsSample(
        seq=1,
        provider="obs_sync",
        payload_json=json.dumps(wrapper),
        image_rgb=np.zeros((160, 320, 3), dtype=np.uint8).tobytes(),
        tactile_data=raw,
        recv_time=0.0,
        timestamp_unix_s=1.0,
        stamp_ns=1,
    )


def test_original_groot_request_has_no_mot_fields() -> None:
    request, info = make_client("groot_n17_sharpa62")._build_sharpa62_request(
        [make_sample()]
    )
    assert FORCE_KEY not in request
    assert DEFORMATION_KEY not in request
    assert info["tactile"] == {}


def test_mot_request_has_force_and_raw_deformation() -> None:
    request, info = make_client(
        "groot_n17_mot_sharpa62"
    )._build_sharpa62_request([make_sample()])
    assert request[FORCE_KEY].shape == (10, 6)
    assert request[DEFORMATION_KEY].shape == (10, 64, 64)
    assert info["tactile"]["force_valid"] == 10
    assert info["tactile"]["deformation_valid"] == 10
