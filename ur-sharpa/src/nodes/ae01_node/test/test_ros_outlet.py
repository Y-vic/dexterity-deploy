"""Exercise the ROS shell with messages and an SDK child double, without a ROS graph."""
import io
import json
import signal
from types import SimpleNamespace as NS

import pytest
import rclpy
import rclpy.node
from ur_msgs.msg import ControlStatus, HandCommand
from ur_msgs.msg import WrenchFrame
from std_msgs.msg import Bool

from ae01_node.ae01_node import main
from ae01_node.sdk_worker import PACKET_PREFIX
from ae01_node.feedback import FeedbackInput


@pytest.mark.parametrize('feedback_enabled', [True, False])
def test_outlet_gating_single_hand_source_identity_and_child_cleanup(monkeypatch, feedback_enabled):
    published, events, arguments = {}, [], {}
    now = [10.0]
    packet = dict(monotonic_s=10., sides={"left": dict(online=True, ready=True,
        q=[.1]*22, raw_seen_s=10., robot_seen_s=10., robot_timestamp_ns=100)},
        feedback=dict(input_fresh=True, left_active=True, right_active=False, rate_hz=30.))
    worker = NS(stdout=io.StringIO(PACKET_PREFIX + json.dumps(packet) + "\n"), stdin=io.StringIO(),
        poll=lambda: None, terminate=lambda: events.append("terminate"), wait=lambda **_: events.append("wait"))
    feedback = []

    class Writer:
        def __init__(self, stream): assert stream is worker.stdin
        def send(self, value):
            feedback.append(value)
            return True

    def popen(command, **kwargs):
        arguments.update(command=command, **kwargs)
        return worker

    class ImmediateThread:
        def __init__(self, target, **_): self.target = target
        def start(self): self.target()
        def join(self, **_): events.append("join")

    class FakeNode:
        def __init__(self, *_): self.parameters = {}
        def declare_parameter(self, name, value):
            self.parameters[name] = feedback_enabled if name == 'feedback_enabled' else value
        def get_parameter(self, name): return NS(value=self.parameters[name])
        def create_publisher(self, _, topic, __):
            return NS(publish=lambda msg: published.setdefault(topic, []).append(msg))
        def create_subscription(self, *_): pass
        def create_timer(self, *_): pass
        def get_clock(self):
            from builtin_interfaces.msg import Time
            return NS(now=lambda: NS(to_msg=lambda: Time(sec=10)))
        def destroy_node(self): events.append("destroy")

    running = [True]
    def spin(node):
        node._tick()
        assert "/sharpa/command" not in published  # No authoritative gate yet.
        node._on_control(ControlStatus(active_source=2, sharpa_state=1))
        node._on_wrench(WrenchFrame(timestamp_ns=10_000_000_000,
            left=[float(i) for i in range(30)], right=[1.]*30,
            left_valid=[True]*5, right_valid=[True]*5))
        node._publish_feedback()
        if feedback_enabled:
            assert feedback[-1]['sides']['left']['enabled']
            assert not feedback[-1]['sides']['right']['enabled']
            assert feedback[-1]['sides']['left']['forces'][0] == [24., 25., 26.]
            node._on_stop(Bool(data=True))
            assert not feedback[-1]['sides']['left']['enabled']
            node._on_stop(Bool(data=False))
            assert feedback[-1]['sides']['left']['enabled']
        node._tick()
        assert len(published["/sharpa/command"]) == 1
        command = published["/sharpa/command"][0]
        assert command.source == HandCommand.SOURCE_VIVE
        assert command.header.frame_id == "ae01/robot"
        assert command.left_enable and not command.right_enable
        assert list(command.q_cmd[:22]) == pytest.approx([.1]*22)
        assert list(command.q_cmd[22:]) == [0.]*22
        assert not command.dry_run and not command.is_hil and not command.strict_targets
        node._tick()
        assert len(published["/sharpa/command"]) == 1  # No replay of the same cached frame.
        now[0] = 10.3
        node._tick()
        assert len(published["/sharpa/command"]) == 1
        node._publish_status()
        status = json.loads(published["/teleop/glove_status"][-1].data)
        assert status["provider"] == "ae01" and not status["left_valid"]
        assert status['feedback']['enabled'] == feedback_enabled
        assert not status['feedback']['input_fresh'] and not status['feedback']['left_active']
        node._publish_feedback()
        if feedback_enabled:
            assert not feedback[-1]['sides']['left']['enabled']

    monkeypatch.setattr(rclpy.node, "Node", FakeNode)
    monkeypatch.setattr(rclpy, "init", lambda **_: None)
    monkeypatch.setattr(rclpy, "ok", lambda: running[0])
    monkeypatch.setattr(rclpy, "shutdown", lambda: running.__setitem__(0, False))
    monkeypatch.setattr(rclpy, "spin", spin)
    monkeypatch.setattr("ae01_node.ae01_node.subprocess.Popen", popen)
    monkeypatch.setattr("ae01_node.ae01_node.threading.Thread", ImmediateThread)
    monkeypatch.setattr("ae01_node.ae01_node.time.monotonic", lambda: now[0])
    monkeypatch.setattr('ae01_node.ae01_node.FeedbackWriter', Writer)
    monkeypatch.setattr('ae01_node.ae01_node.FeedbackInput', lambda age, future: FeedbackInput(
        age, future, clock=lambda: now[0], wall_ns=lambda: int(now[0]*1e9)))
    # Default-bound clock must be replaced too; no real SDK or live ROS nodes participate.
    from ae01_node.core import AE01Core
    monkeypatch.setattr("ae01_node.ae01_node.AE01Core", lambda timeout: AE01Core(timeout, clock=lambda: now[0]))
    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        main()
    finally:
        signal.signal(signal.SIGTERM, original_handler)
    assert "--parent-pid" in arguments["command"]
    assert "LD_LIBRARY_PATH" not in arguments["env"]
    assert ('--feedback' in arguments['command']) == feedback_enabled
    if not feedback_enabled:
        assert not feedback
    else:
        assert not feedback[-1]['sides']['left']['enabled']
    assert events == ["terminate", "wait", "join", "destroy"]
    assert worker.stdout.closed
    assert worker.stdin.closed
