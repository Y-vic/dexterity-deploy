import numpy as np
import pytest

from sharpa_interface.server.buffers import PolicyInputBuffers
from sharpa_interface.server.observation import (
    ObservationValidationError,
    validate_policy_observation,
)


def metadata_format() -> dict:
    return {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": "test_v1",
        "image": {
            name: {"history_len": 0, "current": name == "ego_cam"}
            for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
        },
        "state": {
            "history_len": 0,
            "current": True,
            "left_wrist": {"joint": False, "eef": True},
            "right_wrist": {"joint": False, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": {"history_len": 1, "current": True},
            "wrench": {"history_len": 0, "current": False},
            "deformation": {"history_len": 0, "current": False},
        },
        "server_extension": True,
    }


def state_frame(stamp: int) -> dict:
    wrist = {
        "joint": None,
        "eef": np.zeros(9, dtype=np.float32),
        "eef_def": "absolute",
    }
    return {
        "timestamp_ns": stamp,
        "left_wrist": wrist,
        "right_wrist": wrist,
        "hand_joint": {
            "left": np.zeros(22, dtype=np.float32),
            "right": np.zeros(22, dtype=np.float32),
        },
        "valid": True,
    }


def sensor_frame(stamp: int, shape: tuple[int, ...], dtype: type) -> dict:
    valid_shape = shape[:-1] if shape == (5, 6) else shape
    if shape == (5, 240, 240):
        valid_shape = (5,)
    return {
        "timestamp_ns": stamp,
        "left": np.zeros(shape, dtype=dtype),
        "right": np.zeros(shape, dtype=dtype),
        "valid": {
            "left": np.ones(valid_shape, dtype=np.bool_),
            "right": np.ones(valid_shape, dtype=np.bool_),
        },
    }


def test_independent_buffers_build_metadata_selected_input() -> None:
    buffers = PolicyInputBuffers()
    buffers.push(
        "ego_cam",
        {"encoding": "jpeg", "data": b"jpeg", "timestamp_ns": 3, "valid": True},
    )
    buffers.push("state", state_frame(3))
    buffers.push("tau", sensor_frame(1, (22,), np.float32))
    buffers.push("tau", sensor_frame(2, (22,), np.float32))

    observation = buffers.build_observation(
        metadata_format(),
        session_id="episode",
        request_id=0,
        timestamp_ns=4,
    )

    assert observation["image"]["left_wrist_cam"] == {"history": [], "current": None}
    assert observation["sensor"]["tau"]["history"]["left"].shape == (1, 22)
    assert observation["sensor"]["tau"]["current"]["timestamp_ns"] == 2
    assert observation["state"]["current"]["left_wrist"]["eef_def"] == "absolute"


def test_input_allows_extra_keys_but_not_missing_required_keys() -> None:
    buffers = PolicyInputBuffers()
    buffers.push(
        "ego_cam",
        {"encoding": "jpeg", "data": b"jpeg", "timestamp_ns": 3, "valid": True},
    )
    buffers.push("state", state_frame(3))
    buffers.push("tau", sensor_frame(1, (22,), np.float32))
    buffers.push("tau", sensor_frame(2, (22,), np.float32))
    observation = buffers.build_observation(
        metadata_format(), session_id="episode", request_id=0, timestamp_ns=4
    )
    observation["extra"] = None
    validate_policy_observation(observation, metadata_format())

    del observation["image"]["right_wrist_cam"]
    with pytest.raises(ObservationValidationError, match="missing fields"):
        validate_policy_observation(observation, metadata_format())


def test_metadata_extensions_are_preserved() -> None:
    metadata = metadata_format()
    metadata["extension"] = {"model": "trex"}
    metadata["image"]["ego_cam"]["stride"] = 2

    from sharpa_interface.server.metadata import validate_metadata_format

    normalized = validate_metadata_format(metadata)

    assert normalized["extension"] == {"model": "trex"}
    assert normalized["image"]["ego_cam"]["stride"] == 2
