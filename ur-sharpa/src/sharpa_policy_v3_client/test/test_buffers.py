from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sharpa_policy_v3_client.buffers import (
    DEFORMATION_FRAME_BYTE_SIZE,
    TAU_FRAME_BYTE_SIZE,
    WRENCH_FRAME_BYTE_SIZE,
    BufferCapacityError,
    BufferUnderflowError,
    CameraBuffer,
    CameraFrame,
    DeformationFrame,
    FrameValidationError,
    ObservationBuffers,
    StateFrame,
    TauFrame,
    WrenchFrame,
)


def camera_frame(
    timestamp_ns: int,
    *,
    data: bytes | None = None,
    valid: bool = True,
) -> CameraFrame:
    if data is None:
        data = f"jpeg-{timestamp_ns}".encode()
    return CameraFrame(
        timestamp_ns=timestamp_ns,
        encoding="jpeg",
        data=data,
        valid=valid,
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
        left_joint=np.full((left_width,), timestamp_ns, dtype=np.float32),
        left_eef=eef_pose(float(timestamp_ns)),
        left_eef_frame="robot_base",
        right_joint=np.full((6,), timestamp_ns, dtype=np.float32),
        right_eef=eef_pose(float(-timestamp_ns)),
        right_eef_frame="robot_base",
        left_hand_joint=np.arange(22, dtype=np.float32) + timestamp_ns,
        right_hand_joint=np.arange(22, dtype=np.float32) - timestamp_ns,
        valid=True,
    )


def tau_frame(timestamp_ns: int) -> TauFrame:
    return TauFrame(
        timestamp_ns=timestamp_ns,
        left=np.full((22,), timestamp_ns, dtype=np.float32),
        right=np.full((22,), -timestamp_ns, dtype=np.float32),
        left_valid=np.ones((22,), dtype=np.bool_),
        right_valid=np.zeros((22,), dtype=np.bool_),
    )


def wrench_frame(timestamp_ns: int) -> WrenchFrame:
    return WrenchFrame(
        timestamp_ns=timestamp_ns,
        left=np.full((5, 6), timestamp_ns, dtype=np.float32),
        right=np.full((5, 6), -timestamp_ns, dtype=np.float32),
        left_valid=np.ones((5,), dtype=np.bool_),
        right_valid=np.zeros((5,), dtype=np.bool_),
    )


def deformation_frame(timestamp_ns: int) -> DeformationFrame:
    return DeformationFrame(
        timestamp_ns=timestamp_ns,
        left=np.full((5, 240, 240), timestamp_ns % 256, dtype=np.uint8),
        right=np.full((5, 240, 240), (timestamp_ns + 1) % 256, dtype=np.uint8),
        left_valid=np.ones((5,), dtype=np.bool_),
        right_valid=np.zeros((5,), dtype=np.bool_),
    )


def test_buffer_sorts_out_of_order_frames_and_replaces_duplicate_timestamp():
    buffer = CameraBuffer(frame_capacity=8, byte_capacity=1024)

    buffer.append(camera_frame(30, data=b"thirty"))
    buffer.append(camera_frame(10, data=b"ten"))
    buffer.append(camera_frame(20, data=b"old"))
    buffer.append(camera_frame(20, data=b"replacement"))

    assert buffer.timestamps == (10, 20, 30)
    assert [frame.data for frame in buffer.frames()] == [
        b"ten",
        b"replacement",
        b"thirty",
    ]
    assert buffer.total_bytes == len(b"tenreplacementthirty")


def test_frame_capacity_evicts_oldest_timestamp_even_for_late_arrival():
    buffer = CameraBuffer(frame_capacity=2, byte_capacity=1024)

    assert buffer.append(camera_frame(10))
    assert buffer.append(camera_frame(20))
    assert buffer.append(camera_frame(30))
    assert buffer.timestamps == (20, 30)

    assert not buffer.append(camera_frame(5))
    assert buffer.timestamps == (20, 30)


def test_byte_capacity_evicts_oldest_and_rejects_oversized_frame():
    buffer = CameraBuffer(frame_capacity=10, byte_capacity=5)

    buffer.append(camera_frame(1, data=b"aa"))
    buffer.append(camera_frame(2, data=b"bb"))
    buffer.append(camera_frame(3, data=b"cc"))

    assert buffer.timestamps == (2, 3)
    assert buffer.total_bytes == 4
    with pytest.raises(BufferCapacityError, match="buffer limit"):
        buffer.append(camera_frame(4, data=b"123456"))


def test_selection_excludes_physical_current_from_history():
    buffer = CameraBuffer(frame_capacity=8, byte_capacity=1024)
    for timestamp_ns in (1, 2, 3, 4):
        buffer.append(camera_frame(timestamp_ns))

    history, current = buffer.select(history_len=2, current=True)
    assert [frame.timestamp_ns for frame in history] == [2, 3]
    assert current is not None
    assert current.timestamp_ns == 4

    history, current = buffer.select(history_len=2, current=False)
    assert [frame.timestamp_ns for frame in history] == [3, 4]
    assert current is None

    history, current = buffer.select(history_len=0, current=False)
    assert history == ()
    assert current is None

    history, current = buffer.select(history_len=0, current=True)
    assert history == ()
    assert current is not None
    assert current.timestamp_ns == 4


def test_history_without_current_requires_exactly_history_len_frames():
    buffer = CameraBuffer(frame_capacity=8, byte_capacity=1024)
    for timestamp_ns in (1, 2):
        buffer.append(camera_frame(timestamp_ns))

    history, current = buffer.select(history_len=2, current=False)
    assert [frame.timestamp_ns for frame in history] == [1, 2]
    assert current is None

    with pytest.raises(BufferUnderflowError) as error:
        buffer.select(history_len=3, current=False)

    assert error.value.required == 3
    assert error.value.available == 2


def test_clear_removes_every_modality_and_resets_byte_counts():
    buffers = ObservationBuffers()
    for name in buffers.cameras:
        buffers.push_camera(name, camera_frame(1))
    buffers.push_state(state_frame(1))
    buffers.push_tau(tau_frame(1))
    buffers.push_wrench(wrench_frame(1))
    buffers.push_deformation(deformation_frame(1))

    buffers.clear()

    assert all(len(buffer) == 0 for buffer in buffers.cameras.values())
    assert all(buffer.total_bytes == 0 for buffer in buffers.cameras.values())
    assert len(buffers.state) == 0
    assert len(buffers.tau) == 0
    assert len(buffers.wrench) == 0
    assert len(buffers.deformation) == 0
    assert buffers.state.total_bytes == 0
    assert buffers.tau.total_bytes == 0
    assert buffers.wrench.total_bytes == 0
    assert buffers.deformation.total_bytes == 0


def test_clear_source_only_removes_the_named_source():
    buffers = ObservationBuffers()
    buffers.push_camera("ego_cam", camera_frame(1))
    buffers.push_camera("left_wrist_cam", camera_frame(1))
    buffers.push_state(state_frame(1))
    buffers.push_tau(tau_frame(1))
    buffers.push_wrench(wrench_frame(1))
    buffers.push_deformation(deformation_frame(1))

    buffers.clear_source("ego_cam")

    assert len(buffers.camera("ego_cam")) == 0
    assert len(buffers.camera("left_wrist_cam")) == 1
    assert len(buffers.state) == 1

    buffers.clear_source("state")
    assert len(buffers.state) == 0
    buffers.clear_source("tau")
    buffers.clear_source("wrench")
    buffers.clear_source("deformation")
    assert len(buffers.tau) == 0
    assert len(buffers.wrench) == 0
    assert len(buffers.deformation) == 0
    with pytest.raises(ValueError, match="unknown observation source"):
        buffers.clear_source("overhead_cam")


def test_fixed_frame_byte_constants_match_frame_storage():
    assert tau_frame(1).byte_size == TAU_FRAME_BYTE_SIZE
    assert wrench_frame(1).byte_size == WRENCH_FRAME_BYTE_SIZE
    assert deformation_frame(1).byte_size == DEFORMATION_FRAME_BYTE_SIZE


def test_frames_copy_source_arrays_and_make_them_read_only():
    source = np.arange(7, dtype=np.float32)
    frame = StateFrame(timestamp_ns=1, left_joint=source)
    source[0] = 99.0

    assert frame.left_joint is not None
    assert frame.left_joint[0] == 0.0
    assert frame.left_joint.flags.c_contiguous
    assert not frame.left_joint.flags.writeable
    with pytest.raises(ValueError):
        frame.left_joint[0] = 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestamp_ns": -1, "encoding": "jpeg", "data": b"x", "valid": True},
        {"timestamp_ns": True, "encoding": "jpeg", "data": b"x", "valid": True},
        {"timestamp_ns": 1, "encoding": "png", "data": b"x", "valid": True},
        {"timestamp_ns": 1, "encoding": "jpeg", "data": bytearray(b"x"), "valid": True},
        {"timestamp_ns": 1, "encoding": "jpeg", "data": b"", "valid": True},
        {"timestamp_ns": 1, "encoding": "jpeg", "data": b"x", "valid": False},
        {"timestamp_ns": 1, "encoding": "jpeg", "data": b"x", "valid": 1},
    ],
    ids=[
        "negative-timestamp",
        "boolean-timestamp",
        "encoding",
        "non-bytes",
        "valid-empty",
        "invalid-nonempty",
        "nonboolean-valid",
    ],
)
def test_camera_frame_rejects_invalid_wire_facts(kwargs):
    with pytest.raises(FrameValidationError):
        CameraFrame(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_joint": np.zeros((7,), dtype=np.float64)},
        {"left_joint": np.empty((0,), dtype=np.float32)},
        {"left_joint": np.full((7,), np.nan, dtype=np.float32)},
        {"left_eef": np.zeros((8,), dtype=np.float32), "left_eef_frame": "robot_base"},
        {"left_eef": np.full((9,), np.inf, dtype=np.float32), "left_eef_frame": "robot_base"},
        {"left_eef": eef_pose()},
        {"left_eef": eef_pose(), "left_eef_frame": "world"},
        {"left_eef_frame": "robot_base"},
        {"left_hand_joint": np.zeros((21,), dtype=np.float32)},
        {"right_hand_joint": np.zeros((22,), dtype=np.float64)},
        {"valid": 1},
    ],
    ids=[
        "joint-dtype",
        "empty-joint",
        "joint-nan",
        "eef-shape",
        "eef-inf",
        "eef-frame-missing",
        "eef-frame-wrong",
        "frame-without-eef",
        "hand-shape",
        "hand-dtype",
        "validity-type",
    ],
)
def test_state_frame_rejects_invalid_shape_dtype_finite_and_frame(kwargs):
    with pytest.raises(FrameValidationError):
        StateFrame(timestamp_ns=1, **kwargs)


@pytest.mark.parametrize("case", ["zero", "collinear", "near-collinear"])
def test_state_frame_rejects_degenerate_rot6d(case):
    value = eef_pose()
    if case == "zero":
        value[3:6] = 0.0
    elif case == "collinear":
        value[6:9] = value[3:6] * 2.0
    else:
        value[6:9] = np.asarray([1.0, 1.0e-6, 0.0], dtype=np.float32)

    with pytest.raises(FrameValidationError, match="Rot6D"):
        StateFrame(
            timestamp_ns=1,
            left_eef=value,
            left_eef_frame="robot_base",
        )


def test_tau_frame_rejects_invalid_shape_dtype_finite_and_validity():
    valid = np.ones((22,), dtype=np.bool_)

    with pytest.raises(FrameValidationError, match="dtype"):
        TauFrame(
            timestamp_ns=1,
            left=np.zeros((22,), dtype=np.float64),
            right=np.zeros((22,), dtype=np.float32),
            left_valid=valid,
            right_valid=valid,
        )
    with pytest.raises(FrameValidationError, match="shape"):
        TauFrame(
            timestamp_ns=1,
            left=np.zeros((21,), dtype=np.float32),
            right=np.zeros((22,), dtype=np.float32),
            left_valid=valid,
            right_valid=valid,
        )
    with pytest.raises(FrameValidationError, match="NaN or Inf"):
        values = np.zeros((22,), dtype=np.float32)
        values[0] = np.inf
        TauFrame(
            timestamp_ns=1,
            left=values,
            right=np.zeros((22,), dtype=np.float32),
            left_valid=valid,
            right_valid=valid,
        )
    with pytest.raises(FrameValidationError, match="dtype"):
        TauFrame(
            timestamp_ns=1,
            left=np.zeros((22,), dtype=np.float32),
            right=np.zeros((22,), dtype=np.float32),
            left_valid=np.ones((22,), dtype=np.uint8),
            right_valid=valid,
        )


def test_wrench_and_deformation_frames_reject_invalid_contracts():
    with pytest.raises(FrameValidationError, match="shape"):
        WrenchFrame(
            timestamp_ns=1,
            left=np.zeros((5, 5), dtype=np.float32),
            right=np.zeros((5, 6), dtype=np.float32),
            left_valid=np.ones((5,), dtype=np.bool_),
            right_valid=np.ones((5,), dtype=np.bool_),
        )
    with pytest.raises(FrameValidationError, match="dtype"):
        DeformationFrame(
            timestamp_ns=1,
            left=np.zeros((5, 240, 240), dtype=np.float32),
            right=np.zeros((5, 240, 240), dtype=np.uint8),
            left_valid=np.ones((5,), dtype=np.bool_),
            right_valid=np.ones((5,), dtype=np.bool_),
        )
    with pytest.raises(FrameValidationError, match="shape"):
        DeformationFrame(
            timestamp_ns=1,
            left=np.zeros((5, 240, 240), dtype=np.uint8),
            right=np.zeros((5, 240, 240), dtype=np.uint8),
            left_valid=np.ones((4,), dtype=np.bool_),
            right_valid=np.ones((5,), dtype=np.bool_),
        )


def test_observation_buffers_reject_unknown_camera_and_wrong_frame_type():
    buffers = ObservationBuffers()

    with pytest.raises(ValueError, match="unknown camera"):
        buffers.camera("overhead_cam")
    with pytest.raises(TypeError, match="expected CameraFrame"):
        buffers.push_camera("ego_cam", state_frame(1))


def test_fixed_source_capacities_retain_only_the_latest_frames():
    buffers = ObservationBuffers()

    for timestamp_ns in range(1, 23):
        buffers.push_camera("ego_cam", camera_frame(timestamp_ns))
        buffers.push_state(state_frame(timestamp_ns))
        buffers.push_tau(tau_frame(timestamp_ns))
        buffers.push_wrench(wrench_frame(timestamp_ns))
        buffers.push_deformation(deformation_frame(timestamp_ns))

    assert buffers.camera("ego_cam").timestamps == (20, 21, 22)
    assert buffers.state.timestamps == tuple(range(4, 23))
    assert buffers.tau.timestamps == tuple(range(4, 23))
    assert buffers.wrench.timestamps == tuple(range(4, 23))
    assert buffers.deformation.timestamps == (20, 21, 22)


def test_concurrent_push_and_select_preserve_sorted_unique_frames():
    buffers = ObservationBuffers(camera_frame_capacity=512)
    buffers.push_camera("ego_cam", camera_frame(0))

    def push_range(start: int) -> None:
        for timestamp_ns in range(start, start + 100):
            buffers.push_camera("ego_cam", camera_frame(timestamp_ns))

    def read_current() -> None:
        for _ in range(500):
            with buffers.locked():
                history, current = buffers.camera("ego_cam").select(
                    history_len=0,
                    current=True,
                )
            assert history == ()
            assert current is not None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(push_range, 1),
            executor.submit(push_range, 101),
            executor.submit(push_range, 201),
            executor.submit(read_current),
        ]
        for future in futures:
            future.result()

    timestamps = buffers.camera("ego_cam").timestamps
    assert timestamps == tuple(range(301))
    assert len(timestamps) == len(set(timestamps))
