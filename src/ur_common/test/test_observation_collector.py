import numpy as np

from ur_common.observation_collector import ObservationCollector, buffer_status


def eef_pose(position: float = 0.0) -> np.ndarray:
    value = np.zeros(9, dtype=np.float32)
    value[0:3] = position
    value[3] = 1.0
    value[7] = 1.0
    return value


def state_metadata_format() -> dict:
    return {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": "state-current",
        "image": {
            name: {"history_len": 0, "current": False}
            for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
        },
        "state": {
            "history_len": 0,
            "current": True,
            "left_wrist": {"joint": True, "eef": True},
            "right_wrist": {"joint": True, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            name: {"history_len": 0, "current": False}
            for name in ("tau", "wrench", "deformation")
        },
    }


def test_joins_latest_ur_and_hand_without_recomputing_eef():
    collector = ObservationCollector()
    assert not collector.push_ur(
        timestamp_ns=10,
        left_joint=np.zeros(6),
        right_joint=np.ones(6),
        left_eef=eef_pose(1.0),
        right_eef=eef_pose(2.0),
        eef_frame="canonical",
        valid=True,
    )
    assert collector.push_hand(
        timestamp_ns=11,
        joints=np.arange(44),
        left_valid=True,
        right_valid=True,
    )

    payload = collector.build(
        state_metadata_format(),
        session_id="session",
        request_id=4,
        timestamp_ns=12,
        prompt="move",
    )

    current = payload["state"]["current"]
    assert current["timestamp_ns"] == 11
    np.testing.assert_array_equal(current["left_wrist"]["eef"], eef_pose(1.0))
    np.testing.assert_array_equal(
        current["hand_joint"]["right"], np.arange(22, 44, dtype=np.float32)
    )


def test_rejects_noncanonical_or_invalid_ur_from_state_buffer():
    collector = ObservationCollector()
    collector.push_hand(
        timestamp_ns=1,
        joints=np.zeros(44),
        left_valid=True,
        right_valid=True,
    )

    for frame, valid in (("legacy", True), ("canonical", False)):
        assert not collector.push_ur(
            timestamp_ns=2,
            left_joint=np.zeros(6),
            right_joint=np.zeros(6),
            left_eef=eef_pose(),
            right_eef=eef_pose(),
            eef_frame=frame,
            valid=valid,
        )

    assert len(collector.buffers.state) == 0


def test_sensor_ingestion_and_status_keep_existing_buffer_contract():
    collector = ObservationCollector()
    collector.push_camera(
        "ego_cam", timestamp_ns=10, data=b"jpeg", valid=True
    )
    collector.push_tau(
        timestamp_ns=11,
        left=np.zeros(22),
        right=np.ones(22),
        left_valid=True,
        right_valid=False,
    )
    collector.push_wrench(
        timestamp_ns=12,
        left=np.zeros(30),
        right=np.ones(30),
        left_valid=np.ones(5, dtype=bool),
        right_valid=np.zeros(5, dtype=bool),
    )
    collector.push_deformation(
        timestamp_ns=13,
        left=np.zeros((5, 240, 240), dtype=np.uint8),
        right=np.ones((5, 240, 240), dtype=np.uint8),
        left_valid=np.ones(5, dtype=bool),
        right_valid=np.zeros(5, dtype=bool),
    )

    status = collector.status()
    assert status["ego_cam"]["frames"] == 1
    assert status["tau"]["frames"] == 1
    assert status["wrench"]["frames"] == 1
    assert status["deformation"]["frames"] == 1


def test_buffer_status_splits_history_from_current_frame():
    class FakeBuffer:
        timestamps = (10, 20, 30)
        frame_capacity = 8
        total_bytes = 128

        def __len__(self):
            return len(self.timestamps)

    status = buffer_status(FakeBuffer())

    assert status["frames"] == 3
    assert status["history_frames"] == 2
    assert status["current_frames"] == 1
    assert status["oldest_timestamp_ns"] == 10
    assert status["newest_timestamp_ns"] == 30
