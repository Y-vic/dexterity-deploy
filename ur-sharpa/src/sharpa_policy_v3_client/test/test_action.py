from copy import deepcopy

import numpy as np
import pytest

from sharpa_policy_v3_client.action import (
    ACTION_SCHEMA,
    ActionValidationError,
    parse_policy_action,
)


def _pose(length: int, offset: float = 0.0) -> np.ndarray:
    value = np.zeros((length, 9), dtype=np.float32)
    value[:, :3] = offset
    value[:, 3] = 1.0
    value[:, 7] = 1.0
    return value


def action_result() -> dict:
    length = 4
    hand = np.arange(length * 44, dtype=np.float32).reshape(length, 44)
    return {
        "schema": ACTION_SCHEMA,
        "session_id": "episode-001",
        "request_id": 0,
        "action_id": "action-001",
        "revision": 0,
        "timestamp_ns": 123,
        "execution": {
            "frequency_hz": 20.0,
            "action_length": length,
            "execute_start": 1,
            "execute_length": 2,
        },
        "action": {
            "left_wrist": {
                "joint": None,
                "eef": _pose(length),
                "eef_def": "absolute",
            },
            "right_wrist": {
                "joint": None,
                "eef": _pose(length, 0.1),
                "eef_def": None,
            },
            "hand_joint": {"left": hand[:, :22], "right": hand[:, 22:]},
        },
        "auxiliary": {
            "video": {"ego": None, "left_wrist": None, "right_wrist": None},
            "tactile": {"deformation": None, "wrench": None, "hand_tau": None},
        },
        "diagnostics": {
            "policy_family": "mock",
            "checkpoint_id": "ckpt",
            "checkpoint_path": "/ckpt",
            "inference_latency_ms": 1.5,
        },
        "next_metadata_format": None,
    }


def test_parses_v4_absolute_action_and_joins_hands() -> None:
    value = action_result()
    result = parse_policy_action(
        value,
        expected_session_id="episode-001",
        expected_request_id=0,
    )

    assert result.schema == "sharpa_policy_action.v4"
    assert result.left_wrist_action_type == "eef"
    assert result.right_wrist_action_type == "eef"
    assert result.left_wrist.shape == (4, 9)
    assert result.hand_joint.shape == (4, 44)
    np.testing.assert_array_equal(
        result.hand_joint[:, :22], value["action"]["hand_joint"]["left"]
    )
    assert result.execution_slice == slice(1, 3)
    assert not result.left_wrist.flags.writeable


def test_interface_dict_allows_extra_fields() -> None:
    value = action_result()
    value["model_private"] = {"anything": True}
    value["action"]["left_wrist"]["confidence"] = 0.9
    parse_policy_action(value)


@pytest.mark.parametrize(
    "path",
    [
        ("auxiliary", "video", "ego"),
        ("action", "left_wrist", "eef_def"),
        ("action", "hand_joint", "right"),
    ],
)
def test_rejects_missing_minimum_key(path: tuple[str, ...]) -> None:
    value = action_result()
    target = value
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(ActionValidationError, match="missing fields"):
        parse_policy_action(value)


def test_rejects_relative_public_eef() -> None:
    value = action_result()
    value["action"]["left_wrist"]["eef_def"] = "relative"
    with pytest.raises(ActionValidationError, match="must be absolute"):
        parse_policy_action(value)


def test_rejects_wrong_hand_width_and_nonfinite_eef() -> None:
    value = action_result()
    value["action"]["hand_joint"]["left"] = np.zeros((4, 21), dtype=np.float32)
    with pytest.raises(ActionValidationError, match="dimension 22"):
        parse_policy_action(value)

    value = action_result()
    value["action"]["right_wrist"]["eef"][0, 0] = np.nan
    with pytest.raises(ActionValidationError, match="finite"):
        parse_policy_action(value)


def test_copies_next_metadata_format() -> None:
    value = action_result()
    value["next_metadata_format"] = {"format_id": "fast", "private": [1]}
    result = parse_policy_action(value)
    expected = deepcopy(value["next_metadata_format"])
    value["next_metadata_format"]["private"].append(2)
    assert result.next_metadata_format == expected
