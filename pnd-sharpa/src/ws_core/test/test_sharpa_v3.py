from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from ws_core.sharpa_v3 import (
    ACTION_SCHEMA,
    ERROR_SCHEMA,
    HAND_JOINT_ORDER,
    HISTORY_STREAM_MAX_CAPACITIES,
    METADATA_FORMAT_SCHEMA,
    OBSERVATION_SCHEMA,
    TACTILE_ORDER,
    SharpaV3Frame,
    SharpaV3History,
    SharpaV3ProtocolError,
    SharpaV3ServerError,
    action_to_policy_payload,
    build_observation,
    extract_frame,
    required_stream_lengths,
)


def metadata_format(
    format_id: str = "gcc_default_v1",
    *,
    ego_current: bool = True,
    state_current: bool = True,
    tau_history: int = 8,
    tau_current: bool = True,
    wrench_history: int = 8,
    wrench_current: bool = True,
    deformation_history: int = 0,
    deformation_current: bool = True,
) -> dict:
    return {
        "schema": METADATA_FORMAT_SCHEMA,
        "format_id": format_id,
        "image": {
            "ego_cam": {"history_len": 0, "current": ego_current},
            "left_wrist_cam": {"history_len": 0, "current": False},
            "right_wrist_cam": {"history_len": 0, "current": False},
        },
        "state": {
            "history_len": 0,
            "current": state_current,
            "left_wrist": {"joint": False, "eef": True},
            "right_wrist": {"joint": False, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": {"history_len": tau_history, "current": tau_current},
            "wrench": {
                "history_len": wrench_history,
                "current": wrench_current,
            },
            "deformation": {
                "history_len": deformation_history,
                "current": deformation_current,
            },
        },
    }


def frame(index: int) -> SharpaV3Frame:
    value = float(index)
    return SharpaV3Frame(
        obs_seq=index,
        timestamp_ns=1_000_000 + index,
        image_jpeg=b"jpeg" + bytes([index]),
        image_valid=True,
        left_eef=np.full(9, value, dtype=np.float32),
        right_eef=np.full(9, value + 10, dtype=np.float32),
        left_hand=np.full(22, value + 20, dtype=np.float32),
        right_hand=np.full(22, value + 30, dtype=np.float32),
        state_valid=True,
        left_tau=np.full(22, value + 40, dtype=np.float32),
        right_tau=np.full(22, value + 50, dtype=np.float32),
        left_tau_valid=np.ones(22, dtype=bool),
        right_tau_valid=np.ones(22, dtype=bool),
        left_wrench=np.full((5, 6), value + 60, dtype=np.float32),
        right_wrench=np.full((5, 6), value + 70, dtype=np.float32),
        left_wrench_valid=np.ones(5, dtype=bool),
        right_wrench_valid=np.ones(5, dtype=bool),
        left_deformation=np.full((5, 240, 240), index, dtype=np.uint8),
        right_deformation=np.full((5, 240, 240), index + 10, dtype=np.uint8),
        left_deformation_valid=np.ones(5, dtype=bool),
        right_deformation_valid=np.ones(5, dtype=bool),
    )


def test_history_has_fixed_capacities_and_tracks_format_requirements() -> None:
    history = SharpaV3History()
    assert HISTORY_STREAM_MAX_CAPACITIES == {
        "ego_cam": 3,
        "left_wrist_cam": 3,
        "right_wrist_cam": 3,
        "state": 19,
        "tau": 19,
        "wrench": 19,
        "deformation": 3,
    }
    assert history.stream_capacities() == HISTORY_STREAM_MAX_CAPACITIES
    assert history.stream_required_lengths() == {
        name: 0 for name in HISTORY_STREAM_MAX_CAPACITIES
    }

    requirement = metadata_format()
    expected_required_lengths = {
        "ego_cam": 1,
        "left_wrist_cam": 0,
        "right_wrist_cam": 0,
        "state": 1,
        "tau": 9,
        "wrench": 9,
        "deformation": 1,
    }
    assert required_stream_lengths(requirement) == expected_required_lengths
    assert history.configure(requirement) == expected_required_lengths
    assert history.stream_required_lengths() == expected_required_lengths
    assert history.stream_capacities() == HISTORY_STREAM_MAX_CAPACITIES

    history_only = metadata_format(
        "history_only_v1",
        ego_current=False,
        state_current=False,
        tau_history=2,
        tau_current=False,
        wrench_history=0,
        wrench_current=False,
        deformation_history=0,
        deformation_current=False,
    )
    history_only["image"]["left_wrist_cam"] = {
        "history_len": 2,
        "current": False,
    }
    history_only["state"]["history_len"] = 1
    assert required_stream_lengths(history_only) == {
        "ego_cam": 0,
        "left_wrist_cam": 3,
        "right_wrist_cam": 0,
        "state": 2,
        "tau": 3,
        "wrench": 0,
        "deformation": 0,
    }


def test_history_caches_before_metadata_and_switches_format_immediately() -> None:
    history = SharpaV3History()
    for index in range(1, 20):
        history.append(frame(index))

    assert not history.configured
    assert not history.ready
    assert history.snapshot() is None
    assert history.generation == 0
    assert history.format_revision == 0
    assert history.stream_lengths() == HISTORY_STREAM_MAX_CAPACITIES

    fast = metadata_format(
        "trex_fast_v1",
        ego_current=False,
        state_current=False,
        tau_history=0,
        tau_current=False,
        wrench_history=15,
        wrench_current=True,
        deformation_current=True,
    )
    history.configure(fast)
    fast_snapshot = history.snapshot()
    assert fast_snapshot is not None
    assert history.ready
    assert history.format_revision == 1
    assert [item.obs_seq for item in fast_snapshot.frames] == list(range(4, 20))
    assert fast_snapshot.stream_capacities == HISTORY_STREAM_MAX_CAPACITIES
    assert fast_snapshot.stream_required_lengths == {
        "ego_cam": 0,
        "left_wrist_cam": 0,
        "right_wrist_cam": 0,
        "state": 0,
        "tau": 0,
        "wrench": 16,
        "deformation": 1,
    }

    slow = metadata_format(
        "trex_slow_v1",
        ego_current=True,
        state_current=True,
        tau_history=0,
        tau_current=False,
        wrench_history=15,
        wrench_current=True,
        deformation_current=True,
    )
    history.configure(slow)
    slow_snapshot = history.snapshot()
    assert slow_snapshot is not None
    assert history.ready
    assert history.format_revision == 2
    assert slow_snapshot.anchor_obs_seq == fast_snapshot.anchor_obs_seq == 19
    assert [item.obs_seq for item in slow_snapshot.frames] == list(range(4, 20))
    assert slow_snapshot.frames[-1].image_valid
    assert slow_snapshot.frames[-1].state_valid
    assert history.is_current(slow_snapshot)
    assert not history.is_current(fast_snapshot)


def test_history_accepts_each_stream_at_its_fixed_limit() -> None:
    requirement = metadata_format(
        "all_stream_limits_v1",
        tau_history=18,
        wrench_history=18,
        deformation_history=2,
    )
    requirement["image"]["ego_cam"]["history_len"] = 2
    requirement["image"]["left_wrist_cam"] = {
        "history_len": 2,
        "current": True,
    }
    requirement["image"]["right_wrist_cam"] = {
        "history_len": 2,
        "current": True,
    }
    requirement["state"]["history_len"] = 18

    history = SharpaV3History()
    assert history.configure(requirement) == HISTORY_STREAM_MAX_CAPACITIES
    for index in range(1, 20):
        history.append(frame(index))

    snapshot = history.snapshot()
    assert snapshot is not None
    assert snapshot.stream_required_lengths == HISTORY_STREAM_MAX_CAPACITIES
    assert [item.obs_seq for item in snapshot.frames] == list(range(1, 20))


@pytest.mark.parametrize(
    ("stream_name", "history_len", "capacity"),
    (
        ("ego_cam", 3, 3),
        ("left_wrist_cam", 3, 3),
        ("right_wrist_cam", 3, 3),
        ("state", 19, 19),
        ("tau", 19, 19),
        ("wrench", 19, 19),
        ("deformation", 3, 3),
    ),
)
def test_history_rejects_format_above_each_stream_limit(
    stream_name: str,
    history_len: int,
    capacity: int,
) -> None:
    requirement = metadata_format(
        "over_capacity_v1",
        ego_current=False,
        state_current=False,
        tau_history=0,
        tau_current=False,
        wrench_history=0,
        wrench_current=False,
        deformation_history=0,
        deformation_current=False,
    )
    if stream_name in requirement["image"]:
        requirement["image"][stream_name] = {
            "history_len": history_len,
            "current": False,
        }
    elif stream_name == "state":
        requirement["state"]["history_len"] = history_len
    else:
        requirement["sensor"][stream_name]["history_len"] = history_len

    history = SharpaV3History()
    history.append(frame(1))
    lengths_before = history.stream_lengths()
    with pytest.raises(
        SharpaV3ProtocolError,
        match=rf"{stream_name}.*capacity is {capacity}",
    ):
        history.configure(requirement)

    assert not history.configured
    assert history.format_revision == 0
    assert history.stream_lengths() == lengths_before
    assert history.stream_capacities() == HISTORY_STREAM_MAX_CAPACITIES


def test_history_only_payload_excludes_the_current_anchor_tick() -> None:
    requirement = metadata_format(
        "tau_history_only_v1",
        ego_current=False,
        state_current=False,
        tau_history=2,
        tau_current=False,
        wrench_history=0,
        wrench_current=False,
        deformation_history=0,
        deformation_current=False,
    )
    history = SharpaV3History()
    history.configure(requirement)
    for index in range(1, 4):
        history.append(frame(index))

    snapshot = history.snapshot()
    assert snapshot is not None
    assert [item.obs_seq for item in snapshot.frames] == [1, 2, 3]
    missing = snapshot.frames[0]
    assert not missing.image_valid
    assert not missing.state_valid
    assert missing.left_eef.shape == (9,)
    assert missing.left_eef.dtype == np.float32
    assert missing.left_deformation.shape == (5, 240, 240)
    assert missing.left_deformation.dtype == np.uint8
    assert not missing.left_deformation_valid.any()

    observation, _ = build_observation(
        snapshot.frames,
        metadata_format=snapshot.metadata_format,
        session_id="history-only",
        request_id=1,
        prompt="fixed task",
        execution_feedback=None,
    )
    tau = observation["sensor"]["tau"]
    np.testing.assert_array_equal(tau["history"]["left"][:, 0], [41.0, 42.0])
    assert tau["current"] is None


def test_history_streams_do_not_retain_evicted_heavy_modalities() -> None:
    history = SharpaV3History()
    first = frame(1)
    old_wrench = weakref.ref(first.left_wrench)
    old_deformation = weakref.ref(first.left_deformation)
    history.append(first)
    del first
    history.append(frame(2))
    history.append(frame(3))
    gc.collect()

    assert old_wrench() is not None
    assert old_deformation() is not None

    history.append(frame(4))
    gc.collect()
    assert old_wrench() is not None
    assert old_deformation() is None

    history.clear()
    gc.collect()
    assert old_wrench() is None


def test_history_generation_and_anchor_reject_stale_snapshots() -> None:
    history = SharpaV3History()
    history.configure(metadata_format())
    for index in range(1, 10):
        history.append(frame(index))
    snapshot = history.snapshot()
    assert snapshot is not None

    history.append(frame(10))
    assert not history.is_current(snapshot)
    current = history.snapshot()
    assert current is not None
    generation = history.generation
    revision = history.format_revision

    history.clear()
    assert history.generation == generation + 1
    assert history.format_revision == revision
    assert not history.ready
    assert history.snapshot() is None
    assert not history.is_current(current)


def test_builds_exact_metadata_driven_gcc_observation() -> None:
    frames = [frame(index) for index in range(1, 10)]
    observation, info = build_observation(
        frames,
        metadata_format=metadata_format(),
        session_id="episode-1",
        request_id=3,
        prompt="fixed task",
        execution_feedback={
            "last_action_id": "episode-1:request:2",
            "executed_steps": 40,
            "success": True,
        },
    )

    assert set(observation) == {
        "schema",
        "metadata_format_id",
        "session_id",
        "request_id",
        "timestamp_ns",
        "prompt",
        "image",
        "state",
        "sensor",
        "execution_feedback",
    }
    assert observation["schema"] == OBSERVATION_SCHEMA
    assert observation["image"]["ego_cam"]["history"] == []
    assert observation["image"]["ego_cam"]["current"]["data"].startswith(
        b"jpeg"
    )
    assert observation["image"]["left_wrist_cam"] == {
        "history": [],
        "current": None,
    }
    assert observation["state"]["history"] is None
    state = observation["state"]["current"]
    assert state["left_wrist"]["eef"].shape == (9,)
    assert state["left_wrist"]["eef_def"] == "absolute"
    assert state["left_wrist"]["joint"] is None
    assert state["hand_joint"]["left"].shape == (22,)

    tau = observation["sensor"]["tau"]
    assert tau["history"]["left"].shape == (8, 22)
    assert tau["history"]["left"].dtype == np.float32
    np.testing.assert_array_equal(tau["history"]["left"][:, 0], np.arange(41, 49))
    assert tau["current"]["left"][0] == 49
    wrench = observation["sensor"]["wrench"]
    assert wrench["history"]["left"].shape == (8, 5, 6)
    assert wrench["current"]["right"].shape == (5, 6)
    deformation = observation["sensor"]["deformation"]
    assert deformation["history"] is None
    assert deformation["current"]["left"].shape == (5, 240, 240)
    assert observation["execution_feedback"]["executed_steps"] == 40
    assert info["metadata_format_id"] == "gcc_default_v1"


def test_trex_fast_format_omits_image_and_state_but_keeps_sensors() -> None:
    frames = [frame(index) for index in range(1, 17)]
    observation, _ = build_observation(
        frames,
        metadata_format=metadata_format(
            "trex_fast_v1",
            ego_current=False,
            state_current=False,
            tau_history=0,
            tau_current=False,
            wrench_history=15,
            wrench_current=True,
            deformation_current=True,
        ),
        session_id="episode-trex",
        request_id=2,
        prompt="fixed task",
        execution_feedback=None,
    )

    assert observation["image"]["ego_cam"]["current"] is None
    assert observation["state"] == {"history": None, "current": None}
    assert observation["sensor"]["tau"] == {
        "history": None,
        "current": None,
    }
    assert observation["sensor"]["wrench"]["history"]["left"].shape == (
        15,
        5,
        6,
    )
    assert observation["execution_feedback"] == {
        "last_action_id": None,
        "executed_steps": 0,
        "success": True,
    }


def valid_action(*, next_format: dict | None = None) -> dict:
    left = np.arange(16 * 9, dtype=np.float32).reshape(16, 9)
    right = left + 1_000
    hand = np.arange(16 * 44, dtype=np.float32).reshape(16, 44) + 2_000
    return {
        "schema": ACTION_SCHEMA,
        "session_id": "episode-trex",
        "request_id": 7,
        "action_id": "episode-trex:chunk:11",
        "revision": 4,
        "timestamp_ns": 123,
        "execution": {
            "frequency_hz": 15.0,
            "action_length": 16,
            "execute_start": 4,
            "execute_length": 4,
        },
        "action": {
            "left_wrist": {"joint": None, "eef": left, "eef_def": "absolute"},
            "right_wrist": {"joint": None, "eef": right, "eef_def": "absolute"},
            "hand_joint": {"left": hand[:, :22], "right": hand[:, 22:]},
        },
        "auxiliary": {
            "video": {"ego": None, "left_wrist": None, "right_wrist": None},
            "tactile": {"deformation": None, "wrench": None, "hand_tau": None},
        },
        "diagnostics": {
            "policy_family": "trex",
            "checkpoint_id": "ckpt",
            "checkpoint_path": "/checkpoint/ckpt",
            "inference_latency_ms": 3.5,
        },
        "next_metadata_format": next_format,
    }


def test_action_is_strictly_validated_concatenated_and_sliced() -> None:
    next_format = metadata_format(
        "trex_fast_v1",
        ego_current=False,
        state_current=False,
        tau_history=0,
        tau_current=False,
        wrench_history=15,
        wrench_current=True,
    )
    payload, active_format = action_to_policy_payload(
        valid_action(next_format=next_format),
        expected_session_id="episode-trex",
        expected_request_id=7,
    )

    assert payload["action_hand_pose_62d"].shape == (4, 62)
    np.testing.assert_array_equal(
        payload["action_hand_pose_62d"][0, :9],
        valid_action()["action"]["left_wrist"]["eef"][4],
    )
    assert payload["action_hz"] == 15.0
    assert payload["eef_def"] == "absolute"
    assert payload["_ws_sharpa_v4"] == {
        "action_id": "episode-trex:chunk:11",
        "revision": 4,
        "frequency_hz": 15.0,
        "action_length": 16,
        "execute_start": 4,
        "execute_length": 4,
        "server_driven_execution": True,
    }
    assert active_format is not None
    assert active_format["format_id"] == "trex_fast_v1"


def test_action_preserves_prediction_video_path_from_debug() -> None:
    response = valid_action()
    response["debug"] = {
        "server_video_pred_path": "/tmp/video_pred/request_000007.mp4",
        "request_debug": np.asarray(3, dtype=np.int64),
    }

    payload, _ = action_to_policy_payload(
        response,
        expected_session_id="episode-trex",
        expected_request_id=7,
    )

    assert payload["debug"]["server_video_pred_path"] == (
        "/tmp/video_pred/request_000007.mp4"
    )
    assert payload["debug"]["request_debug"] == np.asarray(3, dtype=np.int64)
    assert payload["diagnostics"]["server_video_pred_path"] == (
        "/tmp/video_pred/request_000007.mp4"
    )


def test_action_accepts_prediction_video_path_from_diagnostics() -> None:
    response = valid_action()
    response["diagnostics"]["server_video_pred_path"] = (
        "/tmp/video_pred/request_000008.mp4"
    )

    payload, _ = action_to_policy_payload(
        response,
        expected_session_id="episode-trex",
        expected_request_id=7,
    )

    assert payload["debug"]["server_video_pred_path"] == (
        "/tmp/video_pred/request_000008.mp4"
    )
    assert payload["diagnostics"]["server_video_pred_path"] == (
        "/tmp/video_pred/request_000008.mp4"
    )


def test_action_accepts_prediction_video_path_at_top_level() -> None:
    response = valid_action()
    response["server_video_pred_path"] = "/tmp/video_pred/request_000009.mp4"

    payload, _ = action_to_policy_payload(
        response,
        expected_session_id="episode-trex",
        expected_request_id=7,
    )

    assert payload["server_video_pred_path"] == (
        "/tmp/video_pred/request_000009.mp4"
    )
    assert payload["debug"]["server_video_pred_path"] == (
        "/tmp/video_pred/request_000009.mp4"
    )
    assert payload["diagnostics"]["server_video_pred_path"] == (
        "/tmp/video_pred/request_000009.mp4"
    )


def test_action_rejects_wrong_dtype_and_structured_error() -> None:
    response = valid_action()
    response["action"]["left_wrist"]["eef"] = response["action"][
        "left_wrist"
    ]["eef"].astype(np.float64)
    with pytest.raises(SharpaV3ProtocolError, match="dtype"):
        action_to_policy_payload(
            response,
            expected_session_id="episode-trex",
            expected_request_id=7,
        )

    with pytest.raises(SharpaV3ServerError) as error:
        action_to_policy_payload(
            {
                "schema": ERROR_SCHEMA,
                "request_id": 7,
                "error": {
                    "code": "INVALID_POLICY_MESSAGE",
                    "message": "bad metadata format",
                    "retryable": False,
                },
            },
            expected_session_id="episode-trex",
            expected_request_id=7,
        )
    assert error.value.code == "INVALID_POLICY_MESSAGE"
    assert not error.value.retryable


def test_observation_requires_real_history_and_current_camera() -> None:
    with pytest.raises(SharpaV3ProtocolError, match="requires 9"):
        build_observation(
            [frame(index) for index in range(1, 9)],
            metadata_format=metadata_format(),
            session_id="episode-1",
            request_id=1,
            prompt="fixed task",
            execution_feedback=None,
        )

    invalid = frame(9)
    object.__setattr__(invalid, "image_valid", False)
    with pytest.raises(SharpaV3ProtocolError, match="ego camera"):
        build_observation(
            [frame(index) for index in range(1, 9)] + [invalid],
            metadata_format=metadata_format(),
            session_id="episode-1",
            request_id=1,
            prompt="fixed task",
            execution_feedback=None,
        )


def test_extract_frame_requires_fresh_state_and_explicit_joint_order() -> None:
    joint_order = list(reversed(HAND_JOINT_ORDER))
    tactile_order = list(reversed(TACTILE_ORDER))
    observation = {
        "policy_input": {
            "valid": True,
            "hand_pose_62d": np.arange(62, dtype=np.float32),
        },
        "robot_state": {
            "valid": True,
            "age_ms": 5.0,
            "payload": {
                "valid": True,
                "json": {
                    "sharpa": {
                        "age_ms": 5.0,
                        "tau": np.arange(44, dtype=np.float32),
                        "tau_valid": np.ones(44, dtype=bool),
                        "joint_order": joint_order,
                    },
                    "tactile": {
                        "force_age_ms": 5.0,
                        "order": tactile_order,
                        "wrench": np.arange(60, dtype=np.float32).reshape(10, 6),
                        "wrench_valid": np.ones(10, dtype=bool),
                    },
                },
            },
        },
    }

    extracted = extract_frame(
        observation,
        image_jpeg=b"jpeg",
        tactile_data=b"",
        obs_seq=1,
        timestamp_ns=1,
        image_valid=True,
        joint_max_age_ms=150.0,
        wrench_max_age_ms=150.0,
    )

    assert extracted.state_valid
    source_joint_index = {
        name: index for index, name in enumerate(joint_order)
    }
    assert extracted.left_tau[0] == source_joint_index[HAND_JOINT_ORDER[0]]
    source_tactile_index = {
        name: index for index, name in enumerate(tactile_order)
    }
    assert extracted.left_wrench[0, 0] == 6 * source_tactile_index[TACTILE_ORDER[0]]

    observation["robot_state"]["age_ms"] = 151.0
    del observation["robot_state"]["payload"]["json"]["sharpa"]["joint_order"]
    stale = extract_frame(
        observation,
        image_jpeg=b"jpeg",
        tactile_data=b"",
        obs_seq=2,
        timestamp_ns=2,
        image_valid=True,
        joint_max_age_ms=150.0,
        wrench_max_age_ms=150.0,
    )
    assert not stale.state_valid
    assert not stale.left_tau_valid.any()
    assert not stale.right_tau_valid.any()
