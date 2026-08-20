import numpy as np
import pytest

from sharpa_interface.server.action import (
    ACTION_SCHEMA,
    ActionValidationError,
    empty_auxiliary,
    parse_policy_action,
)


def valid_action() -> dict:
    length = 16
    return {
        "schema": ACTION_SCHEMA,
        "session_id": "episode-trex",
        "request_id": 7,
        "action_id": "episode-trex:chunk:11",
        "revision": 4,
        "timestamp_ns": 123,
        "execution": {
            "frequency_hz": 30.0,
            "action_length": length,
            "execute_start": 4,
            "execute_length": 4,
        },
        "action": {
            "left_wrist": {
                "joint": None,
                "eef": np.zeros((length, 9), dtype=np.float32),
                "eef_def": "absolute",
            },
            "right_wrist": {
                "joint": None,
                "eef": np.ones((length, 9), dtype=np.float32),
                "eef_def": None,
            },
            "hand_joint": {
                "left": np.zeros((length, 22), dtype=np.float32),
                "right": np.ones((length, 22), dtype=np.float32),
            },
        },
        "auxiliary": empty_auxiliary(),
        "diagnostics": {"policy_family": "trex"},
        "next_metadata_format": None,
        "model_extra": "allowed",
    }


def test_action_is_extensible_and_returns_execution_slice() -> None:
    parsed = parse_policy_action(
        valid_action(),
        expected_session_id="episode-trex",
        expected_request_id=7,
    )

    assert parsed.execution_slice == slice(4, 8)
    assert parsed.right_wrist.eef_def == "absolute"
    executable = parsed.executable_action()
    assert executable["left_wrist"]["eef"].shape == (4, 9)
    assert executable["hand_joint"]["right"].shape == (4, 22)


def test_action_requires_every_minimum_key() -> None:
    response = valid_action()
    del response["auxiliary"]["video"]["left_wrist"]

    with pytest.raises(ActionValidationError, match="missing fields: left_wrist"):
        parse_policy_action(
            response,
            expected_session_id="episode-trex",
            expected_request_id=7,
        )


def test_relative_eef_is_rejected_at_public_interface() -> None:
    response = valid_action()
    response["action"]["left_wrist"]["eef_def"] = "relative"

    with pytest.raises(ActionValidationError, match="interface boundary"):
        parse_policy_action(
            response,
            expected_session_id="episode-trex",
            expected_request_id=7,
        )


def test_action_rejects_out_of_bounds_execution_slice() -> None:
    response = valid_action()
    response["execution"]["execute_start"] = 14

    with pytest.raises(ActionValidationError, match="slice exceeds"):
        parse_policy_action(
            response,
            expected_session_id="episode-trex",
            expected_request_id=7,
        )
