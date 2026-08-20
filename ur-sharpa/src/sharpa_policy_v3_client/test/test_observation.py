from copy import deepcopy

import numpy as np
import pytest

from sharpa_policy_v3_client.buffers import (
    DEFORMATION_FRAME_BYTE_SIZE,
    TAU_FRAME_BYTE_SIZE,
    WRENCH_FRAME_BYTE_SIZE,
    CameraFrame,
    DeformationFrame,
    ObservationBuffers,
    StateFrame,
    TauFrame,
    WrenchFrame,
)
from sharpa_policy_v3_client.observation import (
    OBSERVATION_SCHEMA,
    ObservationBuilder,
    ObservationCapacityError,
    ObservationNotReady,
    ObservationValidationError,
    build_observation,
    validate_format_capacity,
)


def disabled_metadata_format() -> dict:
    return {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": "format-001",
        "image": {
            "ego_cam": {"history_len": 0, "current": False},
            "left_wrist_cam": {"history_len": 0, "current": False},
            "right_wrist_cam": {"history_len": 0, "current": False},
        },
        "state": {
            "history_len": 0,
            "current": False,
            "left_wrist": {"joint": False, "eef": False},
            "right_wrist": {"joint": False, "eef": False},
            "hand_joint": {"left": False, "right": False},
        },
        "sensor": {
            "tau": {"history_len": 0, "current": False},
            "wrench": {"history_len": 0, "current": False},
            "deformation": {"history_len": 0, "current": False},
        },
    }


def full_metadata_format(history_len: int = 2) -> dict:
    value = disabled_metadata_format()
    for selector in value["image"].values():
        selector.update(history_len=history_len, current=True)
    value["state"].update(history_len=history_len, current=True)
    value["state"]["left_wrist"].update(joint=True, eef=True)
    value["state"]["right_wrist"].update(joint=True, eef=True)
    value["state"]["hand_joint"].update(left=True, right=True)
    for selector in value["sensor"].values():
        selector.update(history_len=history_len, current=True)
    return value


def camera_frame(timestamp_ns: int, camera_index: int) -> CameraFrame:
    return CameraFrame(
        timestamp_ns=timestamp_ns,
        encoding="jpeg",
        data=f"jpeg-{camera_index}-{timestamp_ns}".encode(),
        valid=True,
    )


def eef_pose(position: float = 0.0) -> np.ndarray:
    value = np.zeros((9,), dtype=np.float32)
    value[0:3] = position
    value[3] = 1.0
    value[7] = 1.0
    return value


def state_frame(timestamp_ns: int, *, left_width: int = 7) -> StateFrame:
    return StateFrame(
        timestamp_ns=timestamp_ns,
        left_joint=np.arange(left_width, dtype=np.float32) + timestamp_ns,
        left_eef=eef_pose(float(timestamp_ns)),
        left_eef_frame="robot_base",
        right_joint=np.arange(6, dtype=np.float32) - timestamp_ns,
        right_eef=eef_pose(float(-timestamp_ns)),
        right_eef_frame="robot_base",
        left_hand_joint=np.arange(22, dtype=np.float32) + timestamp_ns,
        right_hand_joint=np.arange(22, dtype=np.float32) - timestamp_ns,
        valid=timestamp_ns != 200,
    )


def tau_frame(timestamp_ns: int) -> TauFrame:
    return TauFrame(
        timestamp_ns=timestamp_ns,
        left=np.arange(22, dtype=np.float32) + timestamp_ns,
        right=np.arange(22, dtype=np.float32) - timestamp_ns,
        left_valid=np.ones((22,), dtype=np.bool_),
        right_valid=np.zeros((22,), dtype=np.bool_),
    )


def wrench_frame(timestamp_ns: int) -> WrenchFrame:
    base = np.arange(30, dtype=np.float32).reshape(5, 6)
    return WrenchFrame(
        timestamp_ns=timestamp_ns,
        left=base + timestamp_ns,
        right=base - timestamp_ns,
        left_valid=np.ones((5,), dtype=np.bool_),
        right_valid=np.zeros((5,), dtype=np.bool_),
    )


def deformation_frame(timestamp_ns: int) -> DeformationFrame:
    return DeformationFrame(
        timestamp_ns=timestamp_ns,
        left=np.full((5, 240, 240), timestamp_ns % 256, dtype=np.uint8),
        right=np.full(
            (5, 240, 240),
            (timestamp_ns + 1) % 256,
            dtype=np.uint8,
        ),
        left_valid=np.ones((5,), dtype=np.bool_),
        right_valid=np.zeros((5,), dtype=np.bool_),
    )


def populate_full_buffers(
    buffers: ObservationBuffers,
    timestamps: tuple[int, ...] = (100, 200, 300),
) -> None:
    for timestamp_ns in timestamps:
        for camera_index, camera_name in enumerate(buffers.cameras):
            buffers.push_camera(
                camera_name,
                camera_frame(timestamp_ns, camera_index),
            )
        buffers.push_state(state_frame(timestamp_ns))
        buffers.push_tau(tau_frame(timestamp_ns))
        buffers.push_wrench(wrench_frame(timestamp_ns))
        buffers.push_deformation(deformation_frame(timestamp_ns))


def assert_array_contract(
    value: np.ndarray,
    shape: tuple[int, ...],
    dtype: type[np.generic],
) -> None:
    assert isinstance(value, np.ndarray)
    assert value.shape == shape
    assert value.dtype == np.dtype(dtype)


def build_disabled(
    buffers: ObservationBuffers | None = None,
    **overrides,
) -> dict:
    arguments = {
        "session_id": "episode-001",
        "request_id": 0,
        "timestamp_ns": 999,
        "prompt": "",
    }
    arguments.update(overrides)
    return build_observation(
        buffers or ObservationBuffers(),
        disabled_metadata_format(),
        **arguments,
    )


def test_builds_complete_observation_with_exact_shapes_dtypes_and_timestamps():
    buffers = ObservationBuffers()
    populate_full_buffers(buffers)
    metadata_format = full_metadata_format()
    original_metadata_format = deepcopy(metadata_format)

    result = build_observation(
        buffers,
        metadata_format,
        session_id="episode-001",
        request_id=7,
        timestamp_ns=999,
        prompt="pick up the object",
    )

    assert metadata_format == original_metadata_format
    assert set(result) == {
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
    assert result["schema"] == OBSERVATION_SCHEMA
    assert result["metadata_format_id"] == "format-001"
    assert result["session_id"] == "episode-001"
    assert result["request_id"] == 7
    assert result["timestamp_ns"] == 999
    assert result["prompt"] == "pick up the object"
    assert result["execution_feedback"] == {
        "last_action_id": None,
        "executed_steps": 0,
        "success": True,
    }

    for camera_name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
        camera = result["image"][camera_name]
        assert [frame["timestamp_ns"] for frame in camera["history"]] == [
            100,
            200,
        ]
        assert camera["current"]["timestamp_ns"] == 300
        assert all(frame["encoding"] == "jpeg" for frame in camera["history"])
        assert all(type(frame["data"]) is bytes for frame in camera["history"])
        assert all(type(frame["valid"]) is bool for frame in camera["history"])

    history = result["state"]["history"]
    current = result["state"]["current"]
    assert history is not None
    assert current is not None
    np.testing.assert_array_equal(
        history["timestamp_ns"],
        np.asarray([100, 200], dtype=np.int64),
    )
    assert_array_contract(history["timestamp_ns"], (2,), np.int64)
    assert_array_contract(history["left_wrist"]["joint"], (2, 7), np.float32)
    assert_array_contract(history["left_wrist"]["eef"], (2, 9), np.float32)
    assert_array_contract(history["right_wrist"]["joint"], (2, 6), np.float32)
    assert_array_contract(history["right_wrist"]["eef"], (2, 9), np.float32)
    assert_array_contract(history["hand_joint"]["left"], (2, 22), np.float32)
    assert_array_contract(history["hand_joint"]["right"], (2, 22), np.float32)
    assert_array_contract(history["valid"], (2,), np.bool_)
    assert history["left_wrist"]["eef_def"] == "absolute"
    assert history["right_wrist"]["eef_def"] == "absolute"
    np.testing.assert_array_equal(
        history["hand_joint"]["left"][0],
        np.arange(22, dtype=np.float32) + 100,
    )
    assert current["timestamp_ns"] == 300
    assert_array_contract(current["left_wrist"]["joint"], (7,), np.float32)
    assert_array_contract(current["left_wrist"]["eef"], (9,), np.float32)
    assert_array_contract(current["right_wrist"]["joint"], (6,), np.float32)
    assert_array_contract(current["right_wrist"]["eef"], (9,), np.float32)
    assert_array_contract(current["hand_joint"]["left"], (22,), np.float32)
    assert_array_contract(current["hand_joint"]["right"], (22,), np.float32)
    assert type(current["valid"]) is bool

    tau_history = result["sensor"]["tau"]["history"]
    tau_current = result["sensor"]["tau"]["current"]
    assert_array_contract(tau_history["left"], (2, 22), np.float32)
    assert_array_contract(tau_history["right"], (2, 22), np.float32)
    assert_array_contract(tau_history["timestamp_ns"], (2,), np.int64)
    assert_array_contract(tau_history["valid"]["left"], (2, 22), np.bool_)
    assert_array_contract(tau_history["valid"]["right"], (2, 22), np.bool_)
    assert tau_current["timestamp_ns"] == 300
    assert_array_contract(tau_current["left"], (22,), np.float32)
    assert_array_contract(tau_current["right"], (22,), np.float32)
    assert_array_contract(tau_current["valid"]["left"], (22,), np.bool_)
    assert_array_contract(tau_current["valid"]["right"], (22,), np.bool_)

    wrench_history = result["sensor"]["wrench"]["history"]
    wrench_current = result["sensor"]["wrench"]["current"]
    assert_array_contract(wrench_history["left"], (2, 5, 6), np.float32)
    assert_array_contract(wrench_history["right"], (2, 5, 6), np.float32)
    assert_array_contract(wrench_history["timestamp_ns"], (2,), np.int64)
    assert_array_contract(wrench_history["valid"]["left"], (2, 5), np.bool_)
    assert_array_contract(wrench_history["valid"]["right"], (2, 5), np.bool_)
    assert wrench_current["timestamp_ns"] == 300
    assert_array_contract(wrench_current["left"], (5, 6), np.float32)
    assert_array_contract(wrench_current["right"], (5, 6), np.float32)
    assert_array_contract(wrench_current["valid"]["left"], (5,), np.bool_)
    assert_array_contract(wrench_current["valid"]["right"], (5,), np.bool_)
    wrench_marker = np.arange(30, dtype=np.float32).reshape(5, 6)
    np.testing.assert_array_equal(wrench_history["left"][0], wrench_marker + 100)
    np.testing.assert_array_equal(wrench_history["right"][0], wrench_marker - 100)

    deformation_history = result["sensor"]["deformation"]["history"]
    deformation_current = result["sensor"]["deformation"]["current"]
    assert_array_contract(
        deformation_history["left"],
        (2, 5, 240, 240),
        np.uint8,
    )
    assert_array_contract(
        deformation_history["right"],
        (2, 5, 240, 240),
        np.uint8,
    )
    assert_array_contract(deformation_history["timestamp_ns"], (2,), np.int64)
    assert_array_contract(
        deformation_history["valid"]["left"],
        (2, 5),
        np.bool_,
    )
    assert_array_contract(
        deformation_history["valid"]["right"],
        (2, 5),
        np.bool_,
    )
    assert deformation_current["timestamp_ns"] == 300
    assert_array_contract(
        deformation_current["left"],
        (5, 240, 240),
        np.uint8,
    )
    assert_array_contract(
        deformation_current["right"],
        (5, 240, 240),
        np.uint8,
    )
    assert_array_contract(deformation_current["valid"]["left"], (5,), np.bool_)
    assert_array_contract(deformation_current["valid"]["right"], (5,), np.bool_)


def test_zero_history_and_disabled_current_need_no_buffered_frames():
    result = build_disabled()

    for camera in result["image"].values():
        assert camera == {"history": [], "current": None}
    assert result["state"] == {"history": None, "current": None}
    for sensor in result["sensor"].values():
        assert sensor == {"history": None, "current": None}


def test_history_without_current_uses_latest_history_len_frames():
    metadata_format = disabled_metadata_format()
    metadata_format["image"]["ego_cam"] = {
        "history_len": 2,
        "current": False,
    }
    buffers = ObservationBuffers()
    for timestamp_ns in (10, 20, 30):
        buffers.push_camera("ego_cam", camera_frame(timestamp_ns, 0))

    result = build_observation(
        buffers,
        metadata_format,
        session_id="episode-001",
        request_id=0,
        timestamp_ns=30,
    )

    assert [
        frame["timestamp_ns"]
        for frame in result["image"]["ego_cam"]["history"]
    ] == [20, 30]
    assert result["image"]["ego_cam"]["current"] is None


def test_unrequested_state_components_are_explicit_none():
    metadata_format = disabled_metadata_format()
    metadata_format["state"].update(history_len=1, current=True)
    metadata_format["state"]["left_wrist"]["joint"] = True
    buffers = ObservationBuffers()
    buffers.push_state(
        StateFrame(
            timestamp_ns=10,
            left_joint=np.arange(7, dtype=np.float32),
        )
    )
    buffers.push_state(
        StateFrame(
            timestamp_ns=20,
            left_joint=np.arange(7, dtype=np.float32) + 1,
        )
    )

    result = build_observation(
        buffers,
        metadata_format,
        session_id="episode-001",
        request_id=0,
        timestamp_ns=20,
    )

    history = result["state"]["history"]
    current = result["state"]["current"]
    assert history["left_wrist"]["joint"].shape == (1, 7)
    assert current["left_wrist"]["joint"].shape == (7,)
    for state in (history, current):
        assert state["left_wrist"]["eef"] is None
        assert state["left_wrist"]["eef_def"] is None
        assert state["right_wrist"] == {
            "joint": None,
            "eef": None,
            "eef_def": None,
        }
        assert state["hand_joint"] == {"left": None, "right": None}


def test_reports_precise_path_when_history_is_not_ready():
    metadata_format = disabled_metadata_format()
    metadata_format["image"]["ego_cam"] = {
        "history_len": 2,
        "current": False,
    }
    buffers = ObservationBuffers()
    buffers.push_camera("ego_cam", camera_frame(10, 0))

    with pytest.raises(ObservationNotReady) as error:
        build_observation(
            buffers,
            metadata_format,
            session_id="episode-001",
            request_id=0,
            timestamp_ns=20,
        )

    assert error.value.path == "image.ego_cam"
    assert error.value.required == 2
    assert error.value.available == 1


def test_rejects_requested_state_component_missing_from_selected_frame():
    metadata_format = disabled_metadata_format()
    metadata_format["state"]["current"] = True
    metadata_format["state"]["left_wrist"]["joint"] = True
    buffers = ObservationBuffers()
    buffers.push_state(StateFrame(timestamp_ns=10))

    with pytest.raises(
        ObservationValidationError,
        match=r"state\.current\.left_wrist\.joint is requested but missing",
    ):
        build_observation(
            buffers,
            metadata_format,
            session_id="episode-001",
            request_id=0,
            timestamp_ns=10,
        )


def test_rejects_joint_width_change_between_history_and_current():
    metadata_format = disabled_metadata_format()
    metadata_format["state"].update(history_len=1, current=True)
    metadata_format["state"]["left_wrist"]["joint"] = True
    buffers = ObservationBuffers()
    buffers.push_state(state_frame(10, left_width=7))
    buffers.push_state(state_frame(20, left_width=8))

    with pytest.raises(ObservationValidationError, match="shape changed"):
        build_observation(
            buffers,
            metadata_format,
            session_id="episode-001",
            request_id=0,
            timestamp_ns=20,
        )


def test_preserves_explicit_execution_feedback():
    feedback = {
        "last_action_id": "action-007",
        "executed_steps": 3,
        "success": False,
    }

    result = build_disabled(execution_feedback=feedback)

    assert result["execution_feedback"] == feedback
    assert result["execution_feedback"] is not feedback


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"session_id": ""}, "session_id"),
        ({"session_id": b"episode"}, "session_id"),
        ({"request_id": True}, "request_id"),
        ({"request_id": -1}, "request_id"),
        ({"timestamp_ns": -1}, "timestamp_ns"),
        ({"prompt": b"prompt"}, "prompt"),
    ],
    ids=[
        "empty-session",
        "nonstring-session",
        "boolean-request",
        "negative-request",
        "negative-timestamp",
        "nonstring-prompt",
    ],
)
def test_rejects_invalid_top_level_fields(overrides, match):
    with pytest.raises(ObservationValidationError, match=match):
        build_disabled(**overrides)


@pytest.mark.parametrize(
    "feedback",
    [
        "not-a-mapping",
        {"last_action_id": None, "executed_steps": 0},
        {
            "last_action_id": None,
            "executed_steps": 0,
            "success": True,
            "extra": False,
        },
        {"last_action_id": "", "executed_steps": 0, "success": True},
        {"last_action_id": None, "executed_steps": True, "success": True},
        {"last_action_id": None, "executed_steps": 0, "success": 1},
    ],
    ids=[
        "nonmapping",
        "missing-field",
        "unknown-field",
        "empty-action-id",
        "boolean-steps",
        "nonboolean-success",
    ],
)
def test_rejects_invalid_execution_feedback(feedback):
    with pytest.raises(ObservationValidationError, match="execution_feedback"):
        build_disabled(execution_feedback=feedback)


def test_observation_builder_delegates_to_shared_buffer_set():
    buffers = ObservationBuffers()
    builder = ObservationBuilder(buffers)

    result = builder.build(
        disabled_metadata_format(),
        session_id="episode-002",
        request_id=4,
        timestamp_ns=123,
    )

    assert result["session_id"] == "episode-002"
    assert result["request_id"] == 4
    with pytest.raises(TypeError, match="ObservationBuffers"):
        ObservationBuilder(object())


def test_format_capacity_uses_history_plus_requested_current():
    buffers = ObservationBuffers(camera_frame_capacity=2)
    metadata_format = disabled_metadata_format()
    metadata_format["image"]["ego_cam"] = {
        "history_len": 2,
        "current": False,
    }

    normalized = validate_format_capacity(
        buffers,
        metadata_format,
        max_message_size=64 * 1024 * 1024,
    )
    assert normalized == metadata_format
    assert normalized is not metadata_format

    metadata_format["image"]["ego_cam"]["current"] = True
    with pytest.raises(ObservationCapacityError) as error:
        validate_format_capacity(
            buffers,
            metadata_format,
            max_message_size=64 * 1024 * 1024,
        )
    assert error.value.path == "image.ego_cam"
    assert error.value.resource == "frames"
    assert error.value.required == 3
    assert error.value.available == 2


@pytest.mark.parametrize(
    ("sensor_name", "capacity_parameter", "required_bytes"),
    [
        ("tau", "tau_byte_capacity", TAU_FRAME_BYTE_SIZE),
        ("wrench", "wrench_byte_capacity", WRENCH_FRAME_BYTE_SIZE),
        (
            "deformation",
            "deformation_byte_capacity",
            DEFORMATION_FRAME_BYTE_SIZE,
        ),
    ],
)
def test_format_capacity_rejects_fixed_sensor_buffer_byte_overflow(
    sensor_name,
    capacity_parameter,
    required_bytes,
):
    buffers = ObservationBuffers(**{capacity_parameter: required_bytes - 1})
    metadata_format = disabled_metadata_format()
    metadata_format["sensor"][sensor_name] = {
        "history_len": 0,
        "current": True,
    }

    with pytest.raises(ObservationCapacityError) as error:
        validate_format_capacity(
            buffers,
            metadata_format,
            max_message_size=64 * 1024 * 1024,
        )

    assert error.value.path == f"sensor.{sensor_name}"
    assert error.value.resource == "buffer bytes"
    assert error.value.required == required_bytes
    assert error.value.available == required_bytes - 1


def test_format_capacity_rejects_fixed_state_buffer_byte_overflow():
    buffers = ObservationBuffers(state_byte_capacity=247)
    metadata_format = disabled_metadata_format()
    metadata_format["state"]["current"] = True
    metadata_format["state"]["left_wrist"]["eef"] = True
    metadata_format["state"]["right_wrist"]["eef"] = True
    metadata_format["state"]["hand_joint"] = {"left": True, "right": True}

    with pytest.raises(ObservationCapacityError) as error:
        validate_format_capacity(
            buffers,
            metadata_format,
            max_message_size=64 * 1024 * 1024,
        )

    assert error.value.path == "state"
    assert error.value.resource == "buffer bytes"
    assert error.value.required == 248
    assert error.value.available == 247


def test_format_capacity_rejects_fixed_wire_payload_over_message_limit():
    buffers = ObservationBuffers()
    metadata_format = disabled_metadata_format()
    metadata_format["sensor"]["deformation"] = {
        "history_len": 1,
        "current": True,
    }

    with pytest.raises(ObservationCapacityError) as error:
        validate_format_capacity(
            buffers,
            metadata_format,
            max_message_size=1_000_000,
        )

    assert error.value.path == "observation"
    assert error.value.resource == "message bytes"
    assert error.value.required > 1_000_000
    assert error.value.available == 1_000_000


@pytest.mark.parametrize("max_message_size", [0, -1])
def test_format_capacity_rejects_nonpositive_message_limit(max_message_size):
    with pytest.raises(ValueError, match="positive"):
        validate_format_capacity(
            ObservationBuffers(),
            disabled_metadata_format(),
            max_message_size=max_message_size,
        )


def test_format_capacity_rejects_noninteger_message_limit():
    with pytest.raises(TypeError, match="integer"):
        validate_format_capacity(
            ObservationBuffers(),
            disabled_metadata_format(),
            max_message_size=True,
        )


def test_build_rejects_selected_variable_payload_before_wire_encoding():
    buffers = ObservationBuffers()
    buffers.push_camera(
        "ego_cam",
        CameraFrame(
            timestamp_ns=1,
            encoding="jpeg",
            data=b"x" * 600,
            valid=True,
        ),
    )
    buffers.push_camera(
        "ego_cam",
        CameraFrame(
            timestamp_ns=2,
            encoding="jpeg",
            data=b"y" * 600,
            valid=True,
        ),
    )
    metadata_format = disabled_metadata_format()
    metadata_format["image"]["ego_cam"] = {
        "history_len": 2,
        "current": False,
    }

    with pytest.raises(ObservationCapacityError) as error:
        build_observation(
            buffers,
            metadata_format,
            session_id="episode-001",
            request_id=0,
            timestamp_ns=2,
            max_message_size=1_000,
        )

    assert error.value.resource == "selected raw bytes"
    assert error.value.required == 1_201
    assert error.value.available == 1_000
