import numpy as np
import pytest

from robot_states_node.robot_states_node import (
    ObservationNotReady,
    RobotStateAggregator,
    buffer_status,
)


class FakeKinematics:
    def arm_fk_eef9(self, joints, side):
        out = np.zeros(9, dtype=np.float32)
        out[:3] = np.asarray(joints[:3])
        out[3] = 1.0; out[7] = 1.0
        return out


def test_snapshot_requires_both_hardware_sources():
    agg = RobotStateAggregator(FakeKinematics())
    with pytest.raises(ObservationNotReady):
        agg.snapshot()
    agg.update_ur(1, np.zeros(6), np.zeros(6))
    with pytest.raises(ObservationNotReady):
        agg.snapshot()


def test_snapshot_fk_and_policy_shape():
    agg = RobotStateAggregator(FakeKinematics())
    agg.update_ur(10, np.zeros(6), np.ones(6))
    agg.update_hand(11, np.arange(44, dtype=np.float32))
    payload = agg.policy_obs(session_id="s", request_id=4)
    assert payload["state"]["current"]["left_wrist"]["eef"].shape == (9,)
    assert payload["state"]["current"]["hand_joint"]["right"].shape == (22,)
    assert payload["request_id"] == 4
    assert agg.snapshot().eef_frame == "canonical"


def test_buffer_status_splits_history_from_current_frame():
    class FakeBuffer:
        timestamps = [10, 20, 30]
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
