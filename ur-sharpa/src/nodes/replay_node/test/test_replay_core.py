from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replay_node.replay_node import (
    ReplayConfig,
    ReplaySample,
    ReplaySequencer,
    execution_identity,
)


REAL_SAMPLE = Path(
    "/media/yns/T9/0824_yns_unscrew_cap/success/"
    "2026-08-24_16-01-44.683076"
)


def test_requested_sample_loads_and_repairs_invalid_hand_rows():
    if not REAL_SAMPLE.is_dir():
        pytest.skip("hardware sample is not mounted")
    sample = ReplaySample.load(REAL_SAMPLE)
    assert sample.row_count == 1472
    assert sample.recorded_rate_hz == pytest.approx(30.0)
    assert sample.arm_joint.shape == (1472, 12)
    assert sample.hand_joint.shape == (1472, 44)
    assert len(sample.repaired_hand_rows) == 71
    assert np.isfinite(sample.hand_joint).all()


def test_sequencer_slices_all_rows_without_overlap():
    sample = ReplaySample(
        path=Path("/tmp/sample"),
        name="sample",
        elapsed_ns=np.arange(5, dtype=np.int64),
        arm_joint=np.arange(60, dtype=np.float32).reshape(5, 12) / 100.0,
        hand_joint=np.arange(220, dtype=np.float32).reshape(5, 44) / 100.0,
        recorded_rate_hz=30.0,
        repaired_hand_rows=(),
        joint_source="measured",
    )
    sequencer = ReplaySequencer(sample, ReplayConfig(execution_steps=2))
    chunks = [sequencer.next_segment(), sequencer.next_segment(), sequencer.next_segment()]
    assert [(row["execute_start"], row["execute_length"]) for row in chunks] == [
        (0, 2), (2, 2), (4, 1)
    ]
    assert sequencer.next_segment() is None
    assert chunks[1]["left_arm_joint"] == sample.arm_joint[2:4, :6].reshape(-1).tolist()
    assert chunks[2]["hand_joint"] == sample.hand_joint[4:5].reshape(-1).tolist()


def test_preview_does_not_consume_execution_progress():
    sample = ReplaySample(
        path=Path("/tmp/sample"),
        name="sample",
        elapsed_ns=np.arange(5, dtype=np.int64),
        arm_joint=np.zeros((5, 12), dtype=np.float32),
        hand_joint=np.zeros((5, 44), dtype=np.float32),
        recorded_rate_hz=30.0,
        repaired_hand_rows=(),
        joint_source="measured",
    )
    sequencer = ReplaySequencer(sample, ReplayConfig(execution_steps=2))

    first = sequencer.preview_segment(0)
    second = sequencer.preview_segment(2)

    assert (first["execute_start"], first["execute_length"]) == (0, 2)
    assert (second["execute_start"], second["execute_length"]) == (2, 2)
    assert sequencer.next_row == 0
    assert sequencer.next_segment()["execute_start"] == 0


def test_loop_boundary_requires_explicit_new_cycle_and_keeps_unique_identity():
    sample = ReplaySample(
        path=Path("/tmp/sample"),
        name="sample",
        elapsed_ns=np.arange(2, dtype=np.int64),
        arm_joint=np.zeros((2, 12), dtype=np.float32),
        hand_joint=np.zeros((2, 44), dtype=np.float32),
        recorded_rate_hz=30.0,
        repaired_hand_rows=(),
        joint_source="measured",
    )
    sequencer = ReplaySequencer(
        sample, ReplayConfig(execution_steps=2, loop=True)
    )
    first = sequencer.next_segment()
    assert sequencer.next_segment() is None

    sequencer.begin_next_cycle()
    second = sequencer.next_segment()
    assert second["execute_start"] == 0
    assert second["action_id"] != first["action_id"]
    assert second["request_id"] > first["request_id"]

    old_session = second["session_id"]
    sequencer.reset()
    restarted = sequencer.next_segment()
    assert restarted["session_id"] != old_session
    assert restarted["action_id"] != second["action_id"]


def test_execution_identity_includes_slice_and_revision_fields():
    base = {
        "session_id": "session",
        "request_id": 7,
        "action_id": "action",
        "revision": 2,
        "execute_start": 40,
        "execute_length": 20,
    }
    expected = execution_identity(SimpleNamespace(**base))
    for key in base:
        changed = dict(base)
        changed[key] = f"other-{key}" if key in {"session_id", "action_id"} else 99
        assert execution_identity(SimpleNamespace(**changed)) != expected


def test_config_rejects_unsafe_values():
    with pytest.raises(ValueError, match="execution_steps"):
        ReplayConfig(execution_steps=0)
    with pytest.raises(ValueError, match="frequency_hz"):
        ReplayConfig(frequency_hz=2.0)
    with pytest.raises(ValueError, match="execution_steps"):
        ReplayConfig(execution_steps=1.5)
    with pytest.raises(ValueError, match="boolean"):
        ReplayConfig(loop="false")
