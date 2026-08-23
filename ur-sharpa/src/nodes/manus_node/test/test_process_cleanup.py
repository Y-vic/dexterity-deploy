import signal
import subprocess

from manus_node.manus_node import ManusHealthTracker, RestartLimiter, terminate_process_group


class FakeProcess:
    pid = 123

    def __init__(self, timeouts=0):
        self.timeouts = timeouts
        self.wait_calls = 0

    def poll(self):
        return None

    def wait(self, timeout):
        self.wait_calls += 1
        if self.wait_calls <= self.timeouts:
            raise subprocess.TimeoutExpired("fake", timeout)


def test_terminate_process_group_sends_term(monkeypatch):
    signals = []
    monkeypatch.setattr("manus_node.manus_node._process_tree_groups", lambda process_id: {456})
    monkeypatch.setattr("os.killpg", lambda group, sig: None if sig == 0 else signals.append((group, sig)))

    terminate_process_group(FakeProcess())

    assert set(signals[:2]) == {(123, signal.SIGTERM), (456, signal.SIGTERM)}
    assert set(signals[2:]) == {(123, signal.SIGKILL), (456, signal.SIGKILL)}


def test_terminate_process_group_escalates_to_kill(monkeypatch):
    signals = []
    monkeypatch.setattr("manus_node.manus_node._process_tree_groups", lambda process_id: {456, 789})
    monkeypatch.setattr("os.killpg", lambda group, sig: None if sig == 0 else signals.append((group, sig)))

    terminate_process_group(FakeProcess(timeouts=1))

    assert set(signals[:3]) == {
        (123, signal.SIGTERM), (456, signal.SIGTERM), (789, signal.SIGTERM)
    }
    assert set(signals[3:]) == {
        (123, signal.SIGKILL), (456, signal.SIGKILL), (789, signal.SIGKILL)
    }


def test_health_tracker_keeps_left_available_until_missing_side_timeout():
    health = ManusHealthTracker(
        startup_grace_s=1.0, raw_timeout_s=2.0,
        incomplete_timeout_s=0.5, action_timeout_s=2.0,
    )
    health.reset(10.0)
    health.observe_raw(25, 0, 11.0)
    health.observe_action(11.0)

    assert not health.raw_complete(11.1)
    assert health.side_complete("left", 11.1)
    assert not health.side_complete("right", 11.1)
    assert health.fault(True, True, 11.4) == ""
    health.observe_raw(25, 0, 11.5)
    health.observe_action(11.5)
    assert "right glove" in health.fault(True, True, 11.6)


def test_health_tracker_accepts_asynchronous_left_and_right_raw_frames():
    health = ManusHealthTracker(
        startup_grace_s=0.0, raw_timeout_s=2.0,
        incomplete_timeout_s=1.0, action_timeout_s=2.0,
    )
    health.reset(10.0)
    health.observe_raw(25, 0, 10.1)
    health.observe_raw(0, 25, 10.2)
    health.observe_action(10.2)

    assert health.raw_complete(10.3)
    assert health.side_complete("left", 10.3)
    assert health.side_complete("right", 10.3)
    assert health.incomplete_sides(10.3) == []
    assert health.fault(True, True, 10.3) == ""

    health.observe_raw(25, 0, 10.8)
    health.observe_action(10.8)
    assert health.fault(True, True, 11.1) == ""
    assert "right glove" in health.fault(True, True, 11.21)


def test_incomplete_glove_timeout_is_not_delayed_by_startup_grace():
    health = ManusHealthTracker(
        startup_grace_s=8.0, raw_timeout_s=2.0,
        incomplete_timeout_s=1.0, action_timeout_s=2.0,
    )
    health.reset(10.0)
    health.observe_raw(25, 0, 10.1)
    health.observe_action(10.1)

    assert health.fault(True, True, 11.0) == ""
    health.observe_raw(25, 0, 11.0)
    health.observe_action(11.0)
    assert "right glove" in health.fault(True, True, 11.11)


def test_health_tracker_cancels_side_restart_when_both_gloves_recover():
    health = ManusHealthTracker(
        startup_grace_s=0.0, raw_timeout_s=2.0,
        incomplete_timeout_s=0.5, action_timeout_s=2.0,
    )
    health.reset(10.0)
    health.observe_raw(25, 0, 10.1)
    health.observe_action(10.1)
    assert health.fault(True, True, 10.4) == ""

    health.observe_raw(25, 25, 10.5)
    health.observe_action(10.5)
    assert health.incomplete_since == 0.0
    assert health.fault(True, True, 11.0) == ""


def test_health_tracker_requires_fresh_raw_and_action_streams():
    health = ManusHealthTracker(startup_grace_s=0.0, raw_timeout_s=1.0, action_timeout_s=1.0)
    health.reset(10.0)
    health.observe_raw(25, 25, 10.1)
    health.observe_action(10.1)

    assert health.raw_complete(10.2)
    assert health.fault(True, True, 10.2) == ""
    assert "2044" in health.fault(True, True, 11.2)


def test_restart_limiter_stops_after_maximum_until_cleared():
    limiter = RestartLimiter(maximum=3, window_s=10.0)
    assert limiter.reserve(1.0)[0]
    assert limiter.reserve(2.0)[0]
    assert limiter.reserve(3.0)[0]
    allowed, retry_after = limiter.reserve(20.0)
    assert not allowed
    assert retry_after == 0.0
    assert limiter.attempts == 3
    limiter.clear()
    assert limiter.reserve(21.0)[0]
