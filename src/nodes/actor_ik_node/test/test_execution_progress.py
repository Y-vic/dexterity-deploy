"""Command publication counts and non-gating hardware ACK diagnostics."""

from types import SimpleNamespace

from actor_ik_node.execution_progress import ActorExecutionProgress


def action(**fields):
    return SimpleNamespace(
        **dict(session_id="session", request_id=7, action_id="chunk", revision=2,
               action_length=30, execute_start=8, execute_length=3) | fields
    )


def tracker_for(chunk, *, scope="full", dry_run=False):
    tracker = ActorExecutionProgress()
    tracker.start(chunk, provider="policy", execution_scope=scope,
                  dry_run=dry_run,
                  required_devices=frozenset({"sharpa"} if scope == "hands_only" else {"ur", "sharpa"}))
    return tracker


def ack(tracker, chunk, step, device, **fields):
    values = dict(device=device, source=3, dry_run=False, success=True, timestamp_ns=200)
    return tracker.acknowledged(chunk, step, **(values | fields))


def test_nonzero_slice_reports_original_length_and_cumulative_dispatch_count():
    chunk = action()
    tracker = tracker_for(chunk)
    assert tracker.dispatched(chunk, 0, 100)
    assert not tracker.dispatched(chunk, 0, 100)
    assert not tracker.dispatched(chunk, 2, 100)
    progress = tracker.snapshot(paused=True)
    assert progress["active"] and progress["paused"]
    assert progress["action_length"] == 30
    assert progress["execute_start"] == 8
    assert progress["execute_length"] == 3
    assert progress["dispatched_steps"] == 1
    assert progress["executed_steps"] == 9
    assert progress["acknowledged_steps"] == 0
    assert tracker.dispatched(chunk, 1, 110)
    assert tracker.dispatched(chunk, 2, 120)
    assert tracker.finish(chunk, success=True)
    progress = tracker.snapshot(paused=True)
    assert not progress["active"] and not progress["paused"]
    assert progress["success"]
    assert progress["dispatched_steps"] == 3
    assert progress["executed_steps"] == 11


def test_required_device_ack_prefix_handles_out_of_order_duplicate_and_failed_results():
    chunk = action()
    tracker = tracker_for(chunk)
    for step in range(3):
        tracker.dispatched(chunk, step, 100)
    assert not ack(tracker, chunk, 1, "ur")
    assert not ack(tracker, chunk, 1, "sharpa")
    assert not ack(tracker, chunk, 0, "ur")
    assert not ack(tracker, chunk, 0, "ur")
    assert ack(tracker, chunk, 0, "sharpa")
    assert tracker.snapshot()["acknowledged_steps"] == 2
    assert not ack(tracker, chunk, 2, "ur", success=False)
    assert not ack(tracker, chunk, 2, "sharpa")
    assert tracker.snapshot()["acknowledged_steps"] == 2
    tracker.finish(chunk, success=True)
    assert tracker.snapshot()["success"]  # ACKs never gate scheduling completion.


def test_stale_identity_mode_timestamp_and_unpublished_steps_do_not_advance_ack_progress():
    chunk = action()
    tracker = tracker_for(chunk, scope="hands_only")
    tracker.dispatched(chunk, 0, 100)
    for stale in [action(session_id="old"), action(request_id=6), action(revision=1), action(action_id="old")]:
        assert not ack(tracker, stale, 0, "sharpa")
        assert not tracker.finish(stale, success=False, error="old failure")
    assert not ack(tracker, chunk, 0, "ur")
    assert not ack(tracker, chunk, 0, "sharpa", source=2)
    assert not ack(tracker, chunk, 0, "sharpa", dry_run=True)
    assert not ack(tracker, chunk, 0, "sharpa", timestamp_ns=99)
    assert not ack(tracker, chunk, 1, "sharpa")
    tracker.finish(chunk, success=False, error="stopped")
    assert ack(tracker, chunk, 0, "sharpa")  # Late diagnostics for latest terminal action.
    assert tracker.snapshot()["error"] == "stopped"
    assert tracker.snapshot()["acknowledged_steps"] == 1
    replacement = action(session_id="next", request_id=0)
    tracker.start(replacement, provider="policy", execution_scope="hands_only",
                  dry_run=False, required_devices=frozenset({"sharpa"}))
    assert not ack(tracker, chunk, 0, "sharpa")
    assert tracker.snapshot()["dispatched_steps"] == 0
    assert tracker.snapshot()["acknowledged_steps"] == 0
