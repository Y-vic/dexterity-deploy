from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from replay_node.sample import _resample_source_rows

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
CURRENT_SAMPLE = Path(
    "/media/yns/T9/0825_yns_flip_book/success/"
    "2026-08-25_13-29-23.210490"
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


@pytest.mark.parametrize("joint_source", ("measured", "commanded"))
def test_current_sample_resamples_stale_hand_source(joint_source):
    if not CURRENT_SAMPLE.is_dir():
        pytest.skip("hardware sample is not mounted")

    sample = ReplaySample.load(CURRENT_SAMPLE, joint_source)

    assert sample.row_count == 613
    assert len(sample.repaired_hand_rows) == 57
    assert np.max(np.abs(np.diff(sample.hand_joint, axis=0))) < 0.4


def test_source_timestamps_interpolate_stale_recovery():
    values = np.asarray(
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.9, -0.6)),
        dtype=np.float32,
    )
    timestamps = np.asarray((100, 100, 100, 400), dtype=np.int64)
    target = np.asarray((100, 200, 300, 400), dtype=np.int64)

    resampled, repaired = _resample_source_rows(
        values,
        np.ones(4, dtype=bool),
        timestamps,
        target,
        "hand trajectory",
    )

    assert repaired == []
    assert np.allclose(
        resampled,
        ((0.0, 0.0), (0.3, -0.2), (0.6, -0.4), (0.9, -0.6)),
    )
    assert np.max(np.abs(np.diff(resampled, axis=0))) == pytest.approx(0.3)


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


def test_dry_run_uses_the_same_finite_sequencer_progress():
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
    sequencer = ReplaySequencer(
        sample, ReplayConfig(execution_steps=2, dry_run=True)
    )

    first = sequencer.next_segment()
    second = sequencer.next_segment()
    third = sequencer.next_segment()

    assert (first["execute_start"], first["execute_length"]) == (0, 2)
    assert (second["execute_start"], second["execute_length"]) == (2, 2)
    assert (third["execute_start"], third["execute_length"]) == (4, 1)
    assert sequencer.next_row == 5
    assert sequencer.next_segment() is None


def test_loop_replay_is_rejected():
    with pytest.raises(ValueError, match="loop replay is disabled"):
        ReplayConfig(execution_steps=2, loop=True)


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


def test_replay_stop_keeps_sample_and_resets_sequencer_cursor():
    source = (
        Path(__file__).parents[1] / "replay_node" / "replay_node.py"
    ).read_text(encoding="utf-8")
    finish_stop = source.split("def _finish_stop(self) -> None:", 1)[1].split(
        "def _control(", 1
    )[0]

    assert "if self._unload_requested:" in finish_stop
    assert "self._sequencer.reset()" in finish_stop
    assert 'self._state = "ready" if self._sequencer is not None else "idle"' in finish_stop


def test_config_rejects_unsafe_values():
    with pytest.raises(ValueError, match="execution_steps"):
        ReplayConfig(execution_steps=1001)
    with pytest.raises(ValueError, match="execution_steps"):
        ReplayConfig(execution_steps=0)
    with pytest.raises(ValueError, match="frequency_hz"):
        ReplayConfig(frequency_hz=2.0)
    with pytest.raises(ValueError, match="execution_steps"):
        ReplayConfig(execution_steps=1.5)
    with pytest.raises(ValueError, match="boolean"):
        ReplayConfig(loop="false")
    with pytest.raises(ValueError, match="execution_scope"):
        ReplayConfig(execution_scope="arms_only")


def test_empty_sample_path_does_not_resolve_to_working_directory():
    from replay_node.replay_node import DEFAULT_SAMPLE_DIR
    assert DEFAULT_SAMPLE_DIR == ""
    with pytest.raises(ValueError, match="select a Replay sample"):
        ReplaySample.load("")
