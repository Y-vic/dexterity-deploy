import json
import numpy as np
import pytest
import threading
from pathlib import Path
from types import SimpleNamespace

from actor_ik_node.actor_ik_node import (
    ActorIkCore,
    ActorIkError,
    ChunkLatencyTracker,
    IkPlan,
    PlannedAction,
    SequentialPlanBuffer,
    execution_edges,
    limit_joint_target,
    required_execution_devices,
    next_step_due,
)
from actor_ik_node.trajectory import (
    ExecutionTraceWriter,
    PreviewTraceWriter,
    build_trajectory_payload,
)


def test_hands_only_scope_excludes_ur_from_publish_and_ack_devices():
    assert required_execution_devices("hands_only") == {"sharpa"}
    assert required_execution_devices("full") == {"ur", "sharpa"}
    with pytest.raises(ActorIkError, match="execution scope"):
        required_execution_devices("arms_only")


class FakeKinematics:
    def arm_ik_eef9(self, eef, near, side, ref_joint=None):
        assert side in {"left", "right"}
        return np.asarray(near) + (0.01 if side == "left" else 0.02)


def _eef(n):
    value = np.zeros((n, 9), dtype=np.float32)
    value[:, 3] = 1.0
    value[:, 7] = 1.0
    return value


def test_plan_preserves_step_count_and_advances_seed():
    plan = ActorIkCore(FakeKinematics()).make_plan(
        left_eef=_eef(2), right_eef=_eef(2), hand_joint=np.zeros((2, 44), np.float32),
        execute_length=2, left_seed=np.zeros(6), right_seed=np.zeros(6),
    )
    assert plan.left_joint.shape == (2, 6)
    assert plan.right_joint.shape == (2, 6)
    assert np.allclose(plan.left_joint[-1], 0.02)
    assert np.allclose(plan.right_joint[-1], 0.04)


def test_plan_rejects_wrong_flattened_shape():
    with pytest.raises(ActorIkError, match="left_eef"):
        ActorIkCore(FakeKinematics()).make_plan(
            left_eef=np.zeros(9, np.float32), right_eef=_eef(1),
            hand_joint=np.zeros((1, 44), np.float32), execute_length=2,
            left_seed=np.zeros(6), right_seed=np.zeros(6),
        )


def test_joint_target_is_limited_against_measured_state():
    current = np.asarray((0.0, 1.0, -1.0), dtype=np.float32)
    target = np.asarray((0.5, 0.8, -2.0), dtype=np.float32)
    limited = limit_joint_target(current, target, 0.25)
    assert np.allclose(limited, (0.25, 0.8, -1.25))
    assert np.max(np.abs(limited - current)) <= 0.25


def test_next_step_due_enforces_requested_period():
    assert next_step_due(10.0, 20.0) == pytest.approx(10.05)
    deadlines = [10.0]
    for _ in range(300):
        deadlines.append(next_step_due(deadlines[-1], 30.0))
    assert deadlines[-1] == pytest.approx(20.0)
    with pytest.raises(ActorIkError, match="frequency_hz"):
        next_step_due(10.0, 0.0)


def test_execution_edges_unify_sync_and_rtc_without_mode_branch():
    assert execution_edges(40, 0, 39) == (False, False)
    assert execution_edges(40, 0, 40) == (True, True)
    assert execution_edges(40, 10, 29) == (False, False)
    assert execution_edges(40, 10, 30) == (True, False)
    assert execution_edges(40, 10, 40) == (False, True)


def _planned(action_id, *, session_id="session", end=0.0):
    action = SimpleNamespace(
        session_id=session_id,
        request_id=int(action_id),
        action_id=f"chunk-{action_id}",
        revision=0,
    )
    plan = IkPlan(
        left_joint=np.full((2, 6), end, dtype=np.float32),
        right_joint=np.full((2, 6), end + 1.0, dtype=np.float32),
        hand_joint=np.zeros((2, 44), dtype=np.float32),
    )
    return PlannedAction(action=action, plan=plan, provider="policy")


def test_sequential_plan_buffer_accepts_one_prefetched_chunk_without_promoting_it():
    plans = SequentialPlanBuffer()
    first = _planned(0, end=2.0)
    second = _planned(1, end=4.0)

    assert plans.admit(first) == "current"
    assert plans.admit(second) == "next"
    assert plans.current is first
    assert plans.next is second

    left_seed, right_seed = plans.planning_seeds(np.zeros(6), np.zeros(6))
    assert np.allclose(left_seed, 2.0)
    assert np.allclose(right_seed, 3.0)

    finished, promoted = plans.finish_current()
    assert finished is first
    assert promoted is second
    assert plans.current is second
    assert plans.next is None


def test_sequential_plan_buffer_rejects_a_third_chunk_and_cross_session_queue():
    plans = SequentialPlanBuffer()
    plans.admit(_planned(0))

    with pytest.raises(ActorIkError, match="same session"):
        plans.admit(_planned(1, session_id="other"))

    plans.admit(_planned(1))
    with pytest.raises(ActorIkError, match="already buffered"):
        plans.admit(_planned(2))


def test_actor_trajectory_contains_ik_plan_and_diagnostics():
    action = SimpleNamespace(
        session_id="session",
        request_id=4,
        action_id="action",
        revision=2,
        timestamp_ns=10,
        frequency_hz=30.0,
        action_length=2,
        execute_start=0,
        execute_length=2,
        eef_frame="canonical",
        left_eef=_eef(2).reshape(-1).tolist(),
        right_eef=_eef(2).reshape(-1).tolist(),
    )
    plan = SimpleNamespace(
        left_joint=np.zeros((2, 6), dtype=np.float32),
        right_joint=np.ones((2, 6), dtype=np.float32),
        hand_joint=np.zeros((2, 44), dtype=np.float32),
    )

    payload = build_trajectory_payload(
        action, plan, provider="policy", dry_run=True, trace_dir="/tmp/trace"
    )

    assert payload["dry_run"] is True
    assert payload["ik_applied"] is True
    assert len(payload["left_arm_joint"]) == 12
    assert payload["diagnostics"]["schema"] == "sharpa.actor_ik.trajectory.v1"


def test_preview_trace_writer_flushes_trajectory_and_summary(tmp_path):
    writer = PreviewTraceWriter(tmp_path)
    path = writer.start()
    writer.trajectory({"action_id": "action", "dry_run": True})
    writer.event("command_ack", device="ur", success=True)
    finished = writer.stop()

    assert finished == path
    run_dir = next(tmp_path.iterdir())
    assert '"action_id":"action"' in (run_dir / "trajectories.jsonl").read_text()
    assert '"trajectory_count":1' in (run_dir / "summary.json").read_text()


def test_execution_trace_covers_live_mode_and_external_events(tmp_path):
    writer = ExecutionTraceWriter(tmp_path)
    path = writer.start(mode="live", provider="replay")
    writer.record(
        {
            "node": "sharpa_node",
            "event": "hardware_sdk_command_started",
            "action_id": "action",
        }
    )
    writer.stop("execution_stopped", final_state={"missing_acks": ["sharpa"]})

    run_dir = tmp_path / path.split("/")[-1]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    final_state = json.loads((run_dir / "final_state.json").read_text())
    assert manifest["mode"] == "live"
    assert manifest["provider"] == "replay"
    assert manifest["state"] == "complete"
    assert summary["event_count"] == 3
    assert final_state["missing_acks"] == ["sharpa"]


@pytest.mark.parametrize('dry_run', [False, True])
def test_execution_trace_serializes_real_ros_float32_commands(tmp_path, dry_run):
    from ur_msgs.msg import ArmCommand, HandCommand

    target = np.linspace(-0.1, 0.1, 56, dtype=np.float32)
    arm, hand = ArmCommand(), HandCommand()
    arm.left_joint = target[:6].tolist()
    arm.right_joint = target[6:12].tolist()
    hand.q_cmd = target[12:].tolist()
    writer = ExecutionTraceWriter(tmp_path)
    path = Path(writer.start(mode='preview' if dry_run else 'live', provider='policy'))
    writer.event('command_published', action_id='policy-0', dry_run=dry_run,
                 left_arm_joint=list(arm.left_joint), right_arm_joint=list(arm.right_joint),
                 hand_joint=list(hand.q_cmd), source=hand.source)
    writer.stop()

    records = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    command = next(row for row in records if row['event'] == 'command_published')
    assert command['left_arm_joint'] + command['right_arm_joint'] + command['hand_joint'] == target.tolist()
    assert command['action_id'] == 'policy-0' and command['dry_run'] is dry_run


def test_execution_trace_preserves_numpy_numbers_in_nested_records(tmp_path):
    writer = ExecutionTraceWriter(tmp_path)
    path = Path(writer.start())
    timestamp = 1792252799280343695
    writer.event('diagnostic', nested={'delta': np.float32(0.125), 'success': np.bool_(True)},
                 timestamp_ns=np.int64(timestamp), joints=np.array([[0.25, 0.5]], dtype=np.float32))
    writer.stop(final_state={'count': np.int64(2), 'joints': np.array([0.25], dtype=np.float32)})

    records = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    record = next(row for row in records if row['event'] == 'diagnostic')
    assert record['nested'] == {'delta': 0.125, 'success': True}
    assert record['timestamp_ns'] == timestamp
    assert record['joints'] == [[0.25, 0.5]]
    assert json.loads((path / 'final_state.json').read_text()) == {'count': 2, 'joints': [0.25]}


def test_failed_event_serialization_leaves_journal_intact(tmp_path):
    writer = ExecutionTraceWriter(tmp_path)
    path = Path(writer.start())
    before = (path / 'events.jsonl').read_bytes()
    with pytest.raises(TypeError, match='not JSON serializable'):
        writer.event('invalid', nested={'unsupported': object()})
    assert (path / 'events.jsonl').read_bytes() == before
    writer.event('local_policy_rejected', reason='test failure')
    writer.stop()
    records = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    assert [row['event'] for row in records] == ['execution_started', 'local_policy_rejected', 'execution_stopped']
    assert json.loads((path / 'summary.json').read_text())['event_count'] == 3


def test_execution_trace_summarizes_nonblocking_ack_warnings(tmp_path):
    writer = ExecutionTraceWriter(tmp_path)
    writer.start(mode="live", provider="policy")
    writer.event(
        "hardware_command_result_received",
        device="sharpa",
        ack_delivery_latency_ms=128.5,
        success=True,
    )
    writer.event(
        "hardware_command_ack_late_warning",
        missing_acks=["sharpa"],
        requested_period_ms=1000.0 / 30.0,
        execution_continued=True,
    )
    writer.event(
        "late_hardware_command_ack",
        device="sharpa",
        ack_delivery_latency_ms=1602.0,
        success=True,
    )
    path = writer.stop()

    summary = json.loads((tmp_path / path.split("/")[-1] / "summary.json").read_text())
    assert summary["ack_diagnostics"] == {
        "late_warning_count": 1,
        "late_ack_count": 1,
        "max_delivery_latency_ms": 1602.0,
    }


def test_external_trace_queue_is_bounded_and_separate_from_control_log(
    tmp_path, monkeypatch
):
    writer = ExecutionTraceWriter(tmp_path, external_event_queue_capacity=2)
    path = writer.start(mode="live", provider="policy")
    write_started = threading.Event()
    allow_write = threading.Event()
    original_append = writer._append_jsonl_at

    def blocked_append(path, payload):
        if path.name == "node_events.jsonl":
            write_started.set()
            assert allow_write.wait(timeout=1.0)
        original_append(path, payload)

    monkeypatch.setattr(writer, "_append_jsonl_at", blocked_append)
    assert writer.record_external({"event": "node-0"})
    assert write_started.wait(timeout=1.0)
    assert writer.record_external({"event": "node-1"})
    assert writer.record_external({"event": "node-2"})
    assert not writer.record_external({"event": "node-3"})
    writer.event(
        "hardware_command_ack_late_warning",
        missing_acks=["sharpa"],
        execution_continued=True,
    )
    allow_write.set()
    finished = writer.stop()

    assert finished == path
    run_dir = tmp_path / path.split("/")[-1]
    assert "hardware_command_ack_late_warning" in (
        run_dir / "events.jsonl"
    ).read_text()
    assert "node-0" in (run_dir / "node_events.jsonl").read_text()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["external_trace_diagnostics"]["queue_capacity"] == 2
    assert summary["external_trace_diagnostics"]["attempted_count"] == 4
    assert summary["external_trace_diagnostics"]["pending_at_stop"] == 2
    assert summary["external_trace_diagnostics"]["dropped_count"] == 3


def test_execution_trace_retention_removes_oldest_completed_run(tmp_path):
    writer = ExecutionTraceWriter(tmp_path, max_runs=2)
    paths = []
    for _ in range(3):
        paths.append(writer.start(mode="preview", provider="policy"))
        writer.stop()

    remaining = sorted(
        str(path) for path in tmp_path.iterdir() if (path / "manifest.json").is_file()
    )
    assert len(remaining) == 2
    assert paths[0] not in remaining
    assert paths[1:] == remaining


def test_execution_trace_retention_includes_migrated_legacy_preview(tmp_path):
    legacy = tmp_path / "legacy_previews" / "old-preview"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "sharpa.actor_ik.preview_trace.v1",
                "created_at_ns": 1,
            }
        )
    )
    unrelated = tmp_path / "do-not-delete"
    unrelated.mkdir()

    ExecutionTraceWriter(tmp_path, retention_days=1)

    assert not legacy.exists()
    assert unrelated.exists()


def test_chunk_latency_tracks_current_and_session_average():
    tracker = ChunkLatencyTracker()
    assert tracker.ensure_context(session_id="session", provider="policy")
    tracker.mark_chunk_complete(
        session_id="session",
        provider="policy",
        request_id=0,
        action_id="chunk-0",
        revision=0,
        completed_at_ns=1_000_000_000,
    )

    first = tracker.mark_next_command_published(
        session_id="session",
        provider="policy",
        request_id=1,
        action_id="chunk-1",
        revision=0,
        published_at_ns=1_125_000_000,
    )

    assert first is not None
    assert first["current_latency_ms"] == pytest.approx(125.0)
    assert first["average_latency_ms"] == pytest.approx(125.0)
    assert first["sample_count"] == 1
    assert first["previous_request_id"] == 0
    assert first["current_request_id"] == 1

    tracker.mark_chunk_complete(
        session_id="session",
        provider="policy",
        request_id=1,
        action_id="chunk-1",
        revision=0,
        completed_at_ns=2_000_000_000,
    )
    second = tracker.mark_next_command_published(
        session_id="session",
        provider="policy",
        request_id=2,
        action_id="chunk-2",
        revision=0,
        published_at_ns=2_275_000_000,
    )

    assert second is not None
    assert second["current_latency_ms"] == pytest.approx(275.0)
    assert second["average_latency_ms"] == pytest.approx(200.0)
    assert second["sample_count"] == 2


def test_chunk_latency_excludes_the_normal_command_period():
    tracker = ChunkLatencyTracker()
    tracker.ensure_context(session_id="session", provider="policy")
    final_command_deadline = 10.0
    next_command_deadline = next_step_due(final_command_deadline, 30.0)
    tracker.mark_chunk_complete(
        session_id="session",
        provider="policy",
        request_id=0,
        action_id="chunk-0",
        revision=0,
        completed_at_ns=int(next_command_deadline * 1_000_000_000),
    )

    sample = tracker.mark_next_command_published(
        session_id="session",
        provider="policy",
        request_id=1,
        action_id="chunk-1",
        revision=0,
        published_at_ns=int((next_command_deadline + 0.0008) * 1_000_000_000),
    )

    assert sample is not None
    assert sample["current_latency_ms"] == pytest.approx(0.8, abs=1e-5)


def test_chunk_latency_requires_next_identity_and_resets_context():
    tracker = ChunkLatencyTracker()
    tracker.mark_chunk_complete(
        session_id="first",
        provider="policy",
        request_id=3,
        action_id="chunk",
        revision=1,
        completed_at_ns=10,
    )

    assert tracker.mark_next_command_published(
        session_id="first",
        provider="policy",
        request_id=3,
        action_id="chunk",
        revision=1,
        published_at_ns=20,
    ) is None
    assert tracker.ensure_context(session_id="second", provider="policy")
    assert tracker.mark_next_command_published(
        session_id="second",
        provider="policy",
        request_id=4,
        action_id="next",
        revision=0,
        published_at_ns=30,
    ) is None
    assert tracker.snapshot()["sample_count"] == 0
    assert tracker.snapshot()["current_latency_ms"] is None


def test_chunk_latency_excludes_pause_and_stop_clears_only_actor_owned_state():
    tracker = ChunkLatencyTracker()
    tracker.ensure_context(session_id="session", provider="policy")
    tracker.mark_chunk_complete(
        session_id="session",
        provider="policy",
        request_id=0,
        action_id="chunk-0",
        revision=0,
        completed_at_ns=1_000_000_000,
    )
    assert tracker.pause(1_050_000_000)
    assert tracker.resume(2_050_000_000)

    sample = tracker.mark_next_command_published(
        session_id="session",
        provider="policy",
        request_id=1,
        action_id="chunk-1",
        revision=0,
        published_at_ns=2_125_000_000,
    )
    assert sample["current_latency_ms"] == pytest.approx(125.0)

    # Monitor owns the last published value, so resetting Actor's tracker must
    # not mutate that already-published snapshot.
    monitor_snapshot = tracker.snapshot()
    tracker.reset_statistics(provider="policy")
    assert monitor_snapshot["current_latency_ms"] == pytest.approx(125.0)
    assert monitor_snapshot["average_latency_ms"] == pytest.approx(125.0)
    assert monitor_snapshot["sample_count"] == 1
    cleared = tracker.snapshot()
    assert cleared["current_latency_ms"] is None
    assert cleared["average_latency_ms"] is None
    assert cleared["sample_count"] == 0


def test_chunk_completed_while_paused_starts_gap_when_resumed():
    tracker = ChunkLatencyTracker()
    tracker.ensure_context(session_id="session", provider="policy")
    tracker.pause(1_000_000_000)
    tracker.mark_chunk_complete(
        session_id="session",
        provider="policy",
        request_id=4,
        action_id="chunk-4",
        revision=0,
        completed_at_ns=1_500_000_000,
    )
    tracker.resume(3_000_000_000)

    sample = tracker.mark_next_command_published(
        session_id="session",
        provider="policy",
        request_id=5,
        action_id="chunk-5",
        revision=0,
        published_at_ns=3_080_000_000,
    )

    assert sample["current_latency_ms"] == pytest.approx(80.0)
