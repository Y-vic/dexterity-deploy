"""Run the actual policy wrapper with manual futures and fake ROS transports."""

from concurrent.futures import Future
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from policy_client_node import policy_client_node as policy
from sharpa_interface.server.action import parse_policy_action


class FakeNode:
    def __init__(self, name):
        self.parameters, self.messages = {}, {}
        self._test_publisher_qos, self._test_subscriptions = {}, {}

    def declare_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def create_publisher(self, cls, topic, depth):
        self._test_publisher_qos[topic] = depth
        return SimpleNamespace(publish=self.messages.setdefault(topic, []).append)

    def create_subscription(self, cls, topic, callback, qos, *args, **kwargs):
        self._test_subscriptions[topic] = (cls, callback, qos)

    def create_timer(self, *args, **kwargs):
        pass

    def get_clock(self):
        from builtin_interfaces.msg import Time
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns(), to_msg=Time))

    def get_logger(self):
        return SimpleNamespace(warning=lambda value: None)

    def destroy_node(self):
        pass


@pytest.fixture
def run_policy(monkeypatch):
    import rclpy
    import rclpy.executors
    import rclpy.node
    import ur_common.camera_transport

    monkeypatch.setattr(rclpy, "init", lambda **kwargs: None)
    monkeypatch.setattr(rclpy.node, "Node", FakeNode)
    monkeypatch.setattr(ur_common.camera_transport, "SharedCameraReader",
                        lambda **kwargs: SimpleNamespace(close=lambda: None))

    def run(check):
        class FakeExecutor:
            def __init__(self, **kwargs):
                pass

            def add_node(self, node):
                self.node = node

            def spin(self):
                check(self.node)

            def shutdown(self):
                pass

        monkeypatch.setattr(rclpy.executors, "MultiThreadedExecutor", FakeExecutor)
        policy.main()

    return run


def start_request(node):
    node._metadata = {"metadata_format": {"format_id": "test", "image": {}, "state": {}, "sensor": {}}}
    node._observations = SimpleNamespace(build=lambda *args, **kwargs: {}, status=lambda: {})
    node._machine_control_ready = node._control_ready = True
    future = Future()
    node._inference_pool.shutdown(wait=False)
    node._inference_pool = SimpleNamespace(submit=lambda *args, **kwargs: future,
                                           shutdown=lambda **kwargs: None)
    node._request_observation(node._core.begin())
    return future


def result_for(node):
    eef = np.zeros((30, 9), np.float32)
    eef[:, 3] = eef[:, 7] = 1
    payload = {
        "schema": "sharpa_policy_action.v5", "session_id": node._core.session_id,
        "request_id": node._pending.request_id, "action_id": "chunk", "revision": 0,
        "timestamp_ns": 1,
        "auxiliary": {"video": {"ego": None, "left_wrist": None, "right_wrist": None},
                      "tactile": {"deformation": None, "wrench": None, "hand_tau": None}},
        "diagnostics": None, "next_metadata_format": None,
        "execution": {"frequency_hz": 30.0, "action_length": 30,
                      "execute_start": 8, "execute_length": 2, "rtc": 0},
        "action": {"left_wrist": {"eef": eef, "joint": None, "eef_def": "absolute"},
                   "right_wrist": {"eef": eef, "joint": None, "eef_def": "absolute"},
                   "hand_joint": {"left": np.zeros((30, 22), np.float32),
                                  "right": np.zeros((30, 22), np.float32)}},
    }
    return parse_policy_action(payload, expected_session_id=node._core.session_id,
                               expected_request_id=node._pending.request_id), .125


def latest(node):
    return json.loads(node.messages["/policy/status"][-1].data)


def test_recording_subscription_matches_status_publisher_and_receives_periodic_switch(run_policy, monkeypatch):
    from rclpy.qos import DurabilityPolicy, QoSCompatibility, qos_check_compatible
    from status_node.status_node import StatusNode

    # Exercise the real status constructor with the same offline ROS transports.
    base = StatusNode.__mro__[1]
    for name in ("__init__", "declare_parameter", "get_parameter", "create_publisher",
                 "create_subscription", "create_timer", "get_clock"):
        monkeypatch.setattr(base, name, getattr(FakeNode, name))
    status = StatusNode()
    offered = status._test_publisher_qos["/control/policy_recording"]

    def check(node):
        _, callback, requested = node._test_subscriptions["/control/policy_recording"]
        assert requested.depth == 1
        assert requested.durability == DurabilityPolicy.VOLATILE
        assert offered.durability == DurabilityPolicy.VOLATILE
        compatibility, reason = qos_check_compatible(offered, requested)
        # Humble reports WARNING for unresolved system-default liveliness;
        # an incompatible durability pairing is an ERROR.
        assert compatibility != QoSCompatibility.ERROR, reason
        initial_epoch = node._epoch
        status.machine.policy_recording = True
        status._publish()
        callback(status.messages["/control/policy_recording"][-1])
        assert node._policy_recording
        assert node._epoch == initial_epoch + 1
        assert not node._recording_ready()
        status._publish()
        callback(status.messages["/control/policy_recording"][-1])
        assert node._epoch == initial_epoch + 1  # Repeated heartbeats are idempotent.
        status.machine.policy_recording = False
        status._publish()
        callback(status.messages["/control/policy_recording"][-1])
        assert not node._policy_recording
        assert node._epoch == initial_epoch + 2
        assert node._recording_ready()

    run_policy(check)


def test_request_start_complete_and_rollout_reset_preserve_model_history(run_policy):
    def check(node):
        future = start_request(node)
        assert latest(node)["inference_pending"]
        assert latest(node)["pending_request_id"] == 0
        future.set_result(result_for(node))
        node._tick()
        completed = latest(node)
        assert not completed["inference_pending"]
        assert completed["inference_count"] == 1
        assert completed["latency_s"] == .125
        assert completed["average_latency_s"] == .125
        command = node.messages["/policy/executable_action"][-1]
        assert command.action_length == 30 and command.execute_start == 8
        assert command.execute_length == 2 and len(command.left_eef) == 18
        reset = Future()
        reset.set_result({"metadata_format": node._metadata["metadata_format"]})
        node._service_context = ("reset", node._client, node._epoch, "new-rollout")
        node._service_future = reset
        node._finish_service()
        assert latest(node)["session_id"] == "new-rollout"
        assert latest(node)["inference_count"] == 1
        assert not latest(node)["inference_pending"]
    run_policy(check)


def test_stopped_request_result_is_discarded_and_idle_status_retains_one_hz(run_policy, monkeypatch):
    import time as clock
    now = [100.0]
    monkeypatch.setattr(clock, "monotonic", lambda: now[0])

    def check(node):
        future = start_request(node)
        result = result_for(node)
        before = len(node.messages["/policy/status"])
        now[0] += .1
        node._publish_timing_status()
        assert len(node.messages["/policy/status"]) == before + 1
        assert latest(node)["pending_elapsed_s"] == pytest.approx(.1)
        node._invalidate_request("stopped")
        assert not latest(node)["inference_pending"]
        future.set_result(result)
        node._tick()
        assert node._request_metrics.inference_count == 0
        assert not node.messages["/policy/executable_action"]
        before = len(node.messages["/policy/status"])
        now[0] += .2
        node._publish_timing_status()
        assert len(node.messages["/policy/status"]) == before
        now[0] += 1.0
        node._publish_timing_status()
        assert len(node.messages["/policy/status"]) == before + 1
    run_policy(check)
