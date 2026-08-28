import json

import numpy as np

from actor_ik_node.hand_tracking import (
    CommandMatcher,
    HandTrackingWriter,
    TRACKING_SCHEMA,
    tracking_record,
)


def test_command_matcher_preserves_actor_identity_for_dispatched_target():
    matcher = CommandMatcher(capacity=4)
    matcher.add(
        action_id="action-7",
        revision=2,
        step_index=11,
        dry_run=False,
        requested_q_cmd=np.full(44, 0.1),
        received_at_ns=123,
    )
    matcher.add(
        action_id="action-7",
        revision=2,
        step_index=12,
        dry_run=False,
        requested_q_cmd=np.full(44, 0.2),
        received_at_ns=456,
    )

    command, error = matcher.nearest(np.full(44, 0.2))

    assert command is not None
    assert command.action_id == "action-7"
    assert command.step_index == 12
    assert command.received_at_ns == 456
    assert error == 0.0


def test_tracking_record_contains_same_snapshot_command_and_measured_error():
    matcher = CommandMatcher()
    requested = np.linspace(-0.2, 0.2, 44)
    matcher.add(
        action_id="replay-segment",
        revision=1,
        step_index=9,
        dry_run=False,
        requested_q_cmd=requested,
        received_at_ns=900,
    )
    command, match_error = matcher.nearest(requested)
    measured = requested.copy()
    measured[22] += 0.03

    record = tracking_record(
        hand_timestamp_ns=2_000_000_000,
        measured_q=measured,
        left_valid=True,
        right_valid=True,
        dispatched_q_cmd=requested,
        q_cmd_valid=True,
        q_cmd_timestamp_ns=1_990_000_000,
        q_cmd_source=3,
        control_mode="position",
        position_control_ready=True,
        control_error="",
        command=command,
        command_match_max_abs_rad=match_error,
        recorded_at_ns=3,
        recorded_monotonic_ns=4,
    )

    assert record["schema"] == TRACKING_SCHEMA
    assert record["action_id"] == "replay-segment"
    assert record["step_index"] == 9
    assert record["tracking_valid"] is True
    assert record["q_cmd_age_ms"] == 10.0
    assert record["left_max_abs_error_rad"] == 0.0
    assert np.isclose(record["right_max_abs_error_rad"], 0.03)
    assert np.isclose(record["tracking_error"][22], 0.03)


def test_invalid_command_cache_keeps_state_but_omits_tracking_error():
    record = tracking_record(
        hand_timestamp_ns=10,
        measured_q=np.zeros(44),
        left_valid=True,
        right_valid=True,
        dispatched_q_cmd=np.zeros(44),
        q_cmd_valid=False,
        q_cmd_timestamp_ns=0,
        q_cmd_source=0,
        control_mode="position",
        position_control_ready=True,
        control_error="",
    )

    assert record["tracking_valid"] is False
    assert record["tracking_error"] is None
    assert record["left_max_abs_error_rad"] is None
    assert record["right_max_abs_error_rad"] is None


def test_writer_appends_jsonl_to_selected_actor_trace(tmp_path):
    writer = HandTrackingWriter()
    assert writer.select_trace(tmp_path)
    writer.append({"schema": TRACKING_SCHEMA, "step_index": 3})
    writer.close()

    rows = [json.loads(line) for line in (tmp_path / "hand_tracking.jsonl").read_text().splitlines()]
    assert rows == [{"schema": TRACKING_SCHEMA, "step_index": 3}]
