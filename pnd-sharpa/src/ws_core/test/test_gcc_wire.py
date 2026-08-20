from __future__ import annotations

import copy

import numpy as np
import pytest

from ws_core.gcc_wire import (
    FINGER_ORDER,
    GCC_REQUEST_SCHEMA,
    build_gcc_request,
    extract_gcc_history_frame,
)


JOINT_ORDER = tuple(f"joint_{index}" for index in range(44))


def observation(seq: int) -> dict:
    entries = []
    offset = 0
    for index, name in enumerate(FINGER_ORDER):
        side, finger = name.split("_", 1)
        entries.append(
            {
                "side": side,
                "finger": finger,
                "valid": True,
                "raw_offset": offset,
                "raw_length": 16,
                "height": 4,
                "width": 4,
            }
        )
        offset += 16
    return {
        "schema": "ws.policy_obs.v1",
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
                        "q_cmd": [float(seq) + 0.25] * 44,
                        "q_cmd_valid": [True] * 44,
                        "tau": [float(seq) + 0.5] * 44,
                        "tau_valid": [True] * 44,
                        "feedback_stamp_ns": 123,
                        "joint_order": list(JOINT_ORDER),
                        "joint_layout": "sharpa_joint_order.v1",
                    },
                    "tactile": {
                        "force_age_ms": 10.0,
                        "order": list(FINGER_ORDER),
                        "tactile_layout": (
                            "sharpa_tactile_right_then_left_pinky_to_thumb.v1"
                        ),
                        "wrench": [
                            [float(seq + finger)] * 6 for finger in range(10)
                        ],
                        "wrench_valid": [True] * 10,
                    },
                },
            }
        },
        "robot_tactile": {
            "age_ms": 10.0,
            "metadata": {
                "valid": True,
                "json": {"entries": entries},
            }
        },
    }


def history() -> list:
    return [
        extract_gcc_history_frame(
            observation(seq),
            obs_seq=seq,
            obs_stamp_ns=1_000_000_000 + seq,
            timestamp_unix_s=float(seq),
        )
        for seq in range(1, 10)
    ]


def test_gcc_request_stacks_selected_history_and_current_deformation() -> None:
    raw = b"".join(bytes([index]) * 16 for index in range(10))
    request, info = build_gcc_request(
        history(),
        current_observation=observation(9),
        current_tactile_data=raw,
        current_hand_pose_62d=np.asarray(
            [0.0] * 18 + [9.0] * 44,
            dtype=np.float32,
        ),
        current_ego_view_jpeg=b"jpeg",
        current_obs_seq=9,
        current_timestamp_unix_s=9.0,
        session_id="test",
    )

    assert request["schema"] == GCC_REQUEST_SCHEMA
    assert request["observation/ego_view_jpeg"] == b"jpeg"
    assert request["observation/hand_pose_62d"].shape == (62,)
    assert request["observation/q_exe_history_9x44"].shape == (9, 44)
    assert request["observation/q_exe_history_9x44"][0, 0] == 1.0
    assert request["observation/q_exe_history_9x44"][-1, 0] == 9.0
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
    assert request["observation/tactile_deformation_valid_10"].all()
    assert "observation/ego_view_history_9" not in request
    assert "observation/hand_pose_history_9x62" not in request
    assert "observation/tactile_deformation_history_9x10x64x64" not in request
    assert request["history_is_real_9"].all()
    assert int(request["history_real_count"]) == 9
    assert request["joint_layout"] == "sharpa_joint_order.v1"
    assert request["joint_order"] == list(JOINT_ORDER)
    assert request["tactile_order"] == list(FINGER_ORDER)
    assert info["history_obs_seqs"] == list(range(1, 10))


def test_padding_mask_forces_all_history_valid_masks_false() -> None:
    real = np.ones(9, dtype=bool)
    real[:2] = False
    request, _ = build_gcc_request(
        history(),
        current_observation=observation(9),
        current_tactile_data=b"",
        current_hand_pose_62d=np.zeros(62, dtype=np.float32),
        current_ego_view_jpeg=b"jpeg",
        current_obs_seq=9,
        current_timestamp_unix_s=9.0,
        session_id="test",
        history_is_real=real,
        require_full_real_history=False,
    )

    assert not request["observation/q_exe_valid_history_9x44"][:2].any()
    assert not request["observation/q_cmd_valid_history_9x44"][:2].any()
    assert not request["observation/tau_valid_history_9x44"][:2].any()
    assert not request[
        "observation/tactile_wrench_valid_history_9x10"
    ][:2].any()
    assert int(request["history_real_count"]) == 7


def test_gcc_request_rejects_non_monotonic_or_short_history() -> None:
    frames = history()
    with pytest.raises(ValueError, match="expected 9"):
        build_gcc_request(
            frames[:-1],
            current_observation=observation(9),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=9,
            current_timestamp_unix_s=9.0,
            session_id="test",
        )

    repeated = list(frames)
    repeated[-1] = copy.copy(repeated[-2])
    with pytest.raises(ValueError, match="not strictly increasing"):
        build_gcc_request(
            repeated,
            current_observation=observation(8),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=8,
            current_timestamp_unix_s=8.0,
            session_id="test",
        )


def test_freshness_gates_clear_stale_sensor_masks() -> None:
    stale = observation(1)
    state = stale["robot_state"]["payload"]["json"]
    state["sharpa"]["age_ms"] = 151.0
    state["tactile"]["force_age_ms"] = 151.0
    frame = extract_gcc_history_frame(
        stale,
        obs_seq=1,
        obs_stamp_ns=1,
        timestamp_unix_s=1.0,
        joint_max_age_ms=150.0,
        wrench_max_age_ms=150.0,
    )
    assert not frame.q_exe_valid.any()
    assert not frame.q_cmd_valid.any()
    assert not frame.tau_valid.any()
    assert not frame.tactile_wrench_valid.any()

    stale_transport = observation(2)
    stale_transport["robot_state"]["age_ms"] = 151.0
    frame = extract_gcc_history_frame(
        stale_transport,
        obs_seq=2,
        obs_stamp_ns=2,
        timestamp_unix_s=2.0,
        joint_max_age_ms=150.0,
        wrench_max_age_ms=150.0,
    )
    assert not frame.q_exe_valid.any()
    assert not frame.tactile_wrench_valid.any()

    stale["robot_tactile"]["age_ms"] = 151.0
    raw = b"".join(bytes([index]) * 16 for index in range(10))
    request, _ = build_gcc_request(
        history(),
        current_observation=stale,
        current_tactile_data=raw,
        current_hand_pose_62d=np.asarray(
            [0.0] * 18 + [9.0] * 44,
            dtype=np.float32,
        ),
        current_ego_view_jpeg=b"jpeg",
        current_obs_seq=9,
        current_timestamp_unix_s=9.0,
        session_id="test",
        deformation_max_age_ms=150.0,
    )
    assert not request["observation/tactile_deformation_valid_10"].any()
