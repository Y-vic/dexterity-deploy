from __future__ import annotations

import copy

import numpy as np
import pytest

from ws_core.baseline_wire import (
    BASELINE_REQUEST_SCHEMA,
    FINGER_ORDER,
    build_baseline_request,
    extract_baseline_history_frame,
)


TACTILE_LAYOUT = "sharpa_tactile_right_then_left_pinky_to_thumb.v1"


def observation(seq: int, *, reversed_order: bool = False) -> dict:
    order = list(FINGER_ORDER)
    if reversed_order:
        order.reverse()
    entries = []
    offset = 0
    for name in FINGER_ORDER:
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
    wrench = [
        [float(seq), float(FINGER_ORDER.index(name)), 2.0, 3.0, 4.0, 5.0]
        for name in order
    ]
    return {
        "schema": "ws.policy_obs.v1",
        "robot_state": {
            "age_ms": 5.0,
            "payload": {
                "valid": True,
                "json": {
                    "tactile": {
                        "force_age_ms": 10.0,
                        "order": order,
                        "tactile_layout": TACTILE_LAYOUT,
                        "wrench": wrench,
                        "wrench_valid": [True] * 10,
                    }
                },
            },
        },
        "robot_tactile": {
            "age_ms": 10.0,
            "metadata": {"valid": True, "json": {"entries": entries}},
        },
    }


def history(count: int = 18) -> list:
    frames = []
    for seq in range(1, count + 1):
        frames.append(
            extract_baseline_history_frame(
                observation(seq, reversed_order=seq == 1),
                hand_pose_62d=np.full(62, float(seq), dtype=np.float32),
                obs_seq=seq,
                obs_stamp_ns=1_000_000 + seq,
                timestamp_unix_s=float(seq),
                wrench_max_age_ms=150.0,
            )
        )
    return frames


def raw_deformation() -> bytes:
    return b"".join(bytes([index]) * 16 for index in range(10))


def test_extract_reorders_wrench_and_clears_stale_values() -> None:
    wrapper = observation(1, reversed_order=True)
    frame = extract_baseline_history_frame(
        wrapper,
        hand_pose_62d=np.arange(62, dtype=np.float32),
        obs_seq=1,
        obs_stamp_ns=1,
        timestamp_unix_s=1.0,
        wrench_max_age_ms=150.0,
    )
    assert frame.tactile_order == tuple(FINGER_ORDER)
    np.testing.assert_array_equal(frame.tactile_wrench[:, 1], np.arange(10))

    wrapper["robot_state"]["age_ms"] = 141.0
    stale = extract_baseline_history_frame(
        wrapper,
        hand_pose_62d=np.zeros(62, dtype=np.float32),
        obs_seq=2,
        obs_stamp_ns=2,
        timestamp_unix_s=2.0,
        wrench_max_age_ms=150.0,
    )
    assert not stale.tactile_wrench_valid.any()
    assert not stale.tactile_wrench.any()


def test_trex_request_uses_latest_sixteen_wrenches_and_current_sensors() -> None:
    frames = history()
    request, info = build_baseline_request(
        frames,
        provider="trex",
        current_observation=observation(18),
        current_tactile_data=raw_deformation(),
        current_hand_pose_62d=np.full(62, 99.0, dtype=np.float32),
        current_ego_view_jpeg=b"jpeg",
        current_obs_seq=18,
        current_timestamp_unix_s=18.5,
        session_id="test",
        prompt="Place the egg.",
        deformation_max_age_ms=150.0,
    )

    assert request["schema"] == BASELINE_REQUEST_SCHEMA
    assert request["observation/ego_view_jpeg"] == b"jpeg"
    assert request["prompt"] == "Place the egg."
    np.testing.assert_array_equal(
        request["observation/hand_pose_62d"],
        np.full(62, 99.0, dtype=np.float32),
    )
    wrench = request["observation/tactile_wrench_history_16x10x6"]
    assert wrench.shape == (16, 10, 6)
    assert wrench[0, 0, 0] == 3.0
    assert wrench[-1, 0, 0] == 18.0
    assert request["observation/tactile_wrench_valid_history_16x10"].all()
    assert request["observation/tactile_deformation_10x64x64"].shape == (
        10,
        64,
        64,
    )
    assert request["observation/tactile_deformation_valid_10"].all()
    assert "observation/hand_pose_history_16x62" not in request
    assert info["history_obs_seqs"] == list(range(3, 19))
    assert info["history_frame_count"] == 16


@pytest.mark.parametrize("provider", ["vitacformer", "vitac"])
def test_vitacformer_request_uses_eighteen_wrenches_and_sixteen_states(
    provider: str,
) -> None:
    request, info = build_baseline_request(
        history(),
        provider=provider,
        current_observation=observation(18),
        current_tactile_data=b"",
        current_hand_pose_62d=np.full(62, 99.0, dtype=np.float32),
        current_ego_view_jpeg=b"jpeg",
        current_obs_seq=18,
        current_timestamp_unix_s=18.0,
        session_id="test",
    )

    wrench = request["observation/tactile_wrench_history_18x10x6"]
    states = request["observation/hand_pose_history_16x62"]
    assert wrench.shape == (18, 10, 6)
    assert wrench[0, 0, 0] == 1.0
    assert wrench[-1, 0, 0] == 18.0
    assert states.shape == (16, 62)
    assert states[0, 0] == 3.0
    assert states[-1, 0] == 18.0
    assert "observation/hand_pose_62d" not in request
    assert "observation/tactile_deformation_10x64x64" not in request
    assert "prompt" not in request
    assert info["history_obs_seqs"] == list(range(1, 19))
    assert info["state_history_frame_count"] == 16


def test_request_rejects_short_non_monotonic_or_mismatched_history() -> None:
    frames = history()
    with pytest.raises(ValueError, match="expected at least 16"):
        build_baseline_request(
            frames[:15],
            provider="trex",
            current_observation=observation(15),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=15,
            current_timestamp_unix_s=15.0,
            session_id="test",
        )

    repeated = list(frames)
    repeated[-1] = copy.copy(repeated[-2])
    with pytest.raises(ValueError, match="not strictly increasing"):
        build_baseline_request(
            repeated,
            provider="vitacformer",
            current_observation=observation(17),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=17,
            current_timestamp_unix_s=17.0,
            session_id="test",
        )

    with pytest.raises(ValueError, match="does not match"):
        build_baseline_request(
            frames,
            provider="vitacformer",
            current_observation=observation(18),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=99,
            current_timestamp_unix_s=18.0,
            session_id="test",
        )


def test_request_rejects_unknown_provider_and_invalid_history_layout() -> None:
    frames = history()
    with pytest.raises(ValueError, match="unsupported baseline provider"):
        build_baseline_request(
            frames,
            provider="gcc",
            current_observation=observation(18),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=18,
            current_timestamp_unix_s=18.0,
            session_id="test",
        )

    changed = list(frames)
    changed[-1] = copy.copy(changed[-1])
    object.__setattr__(changed[-1], "tactile_layout", "changed")
    with pytest.raises(ValueError, match="layout changed"):
        build_baseline_request(
            changed,
            provider="vitacformer",
            current_observation=observation(18),
            current_tactile_data=b"",
            current_hand_pose_62d=np.zeros(62, dtype=np.float32),
            current_ego_view_jpeg=b"jpeg",
            current_obs_seq=18,
            current_timestamp_unix_s=18.0,
            session_id="test",
        )
