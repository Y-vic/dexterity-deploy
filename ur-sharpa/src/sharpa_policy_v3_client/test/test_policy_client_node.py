from __future__ import annotations

from concurrent.futures import Future
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest


pytest.importorskip("rclpy")
pytest.importorskip("sharpa_policy_v3_interfaces")

from sharpa_policy_v3_client.action import parse_policy_action
from sharpa_policy_v3_client.policy_client_node import PolicyClientNode
from sharpa_policy_v3_client.session import SessionSnapshot


def parsed_action():
    horizon = 3
    left = np.arange(horizon * 7, dtype=np.float32).reshape(horizon, 7)
    right = np.zeros((horizon, 9), dtype=np.float32)
    right[:, 3] = 1.0
    right[:, 7] = 1.0
    return parse_policy_action(
        {
            "schema": "sharpa_policy_action.v4",
            "session_id": "episode-001",
            "request_id": 4,
            "action_id": "action-004",
            "revision": 2,
            "timestamp_ns": 123,
            "execution": {
                "frequency_hz": 20.0,
                "action_length": horizon,
                "execute_start": 1,
                "execute_length": 2,
            },
            "action": {
                "left_wrist": {
                    "joint": left,
                    "eef": None,
                    "eef_def": None,
                },
                "right_wrist": {
                    "joint": None,
                    "eef": right,
                    "eef_def": "absolute",
                },
                "hand_joint": {
                    "left": np.zeros((horizon, 22), dtype=np.float32),
                    "right": np.zeros((horizon, 22), dtype=np.float32),
                },
            },
            "auxiliary": {
                "video": {
                    "ego": None,
                    "left_wrist": None,
                    "right_wrist": None,
                },
                "tactile": {
                    "deformation": None,
                    "wrench": None,
                    "hand_tau": None,
                },
            },
            "diagnostics": {
                "policy_family": "mock",
                "checkpoint_id": "checkpoint",
                "checkpoint_path": "/checkpoint",
                "inference_latency_ms": 1.5,
            },
            "next_metadata_format": None,
        }
    )


def test_action_message_preserves_types_shapes_and_slice():
    action = parsed_action()
    node = SimpleNamespace(
        runtime=SimpleNamespace(active_metadata_format={"format_id": "unused"}),
        _require_uint=PolicyClientNode._require_uint,
    )

    message = PolicyClientNode._action_message(node, action)

    assert message.session_id == "episode-001"
    assert message.request_id == 4
    assert message.left_wrist_action_type == "joint"
    assert message.left_wrist_dimension == 7
    assert len(message.left_wrist) == 3 * 7
    assert message.right_wrist_action_type == "eef"
    assert message.right_wrist_dimension == 9
    assert len(message.right_wrist) == 3 * 9
    assert message.hand_joint_dimension == 44
    assert len(message.hand_joint) == 3 * 44
    assert message.execute_start == 1
    assert message.execute_length == 2


def test_fault_message_keeps_explicit_action_identity():
    action = parsed_action()
    node = SimpleNamespace(
        runtime=SimpleNamespace(session_id="episode-001"),
        _state_lock=threading.RLock(),
        _pending_action=None,
    )

    fault = PolicyClientNode._fault_message(
        node,
        code="publish_failed",
        message="failed",
        retryable=False,
        safe_stop=True,
        clear_plan=True,
        correlated_action=action,
    )

    assert fault.has_request_id
    assert fault.request_id == 4
    assert fault.has_action_id
    assert fault.action_id == "action-004"
    assert fault.has_revision
    assert fault.revision == 2


class _InspectingPublisher:
    def __init__(self, owner):
        self.owner = owner
        self.messages = []
        self.state_at_publish = None

    def publish(self, message):
        self.messages.append(message)
        self.state_at_publish = (
            self.owner._pending_action,
            self.owner._awaiting_feedback,
            self.owner._inflight,
        )


class _InferenceHarness:
    def __init__(self, future):
        self._state_lock = threading.RLock()
        self._infer_future = future
        self._closing = False
        self._epoch = 0
        self._connected = True
        self._inflight = True
        self._pending_action = None
        self._awaiting_feedback = False
        self._wire_feedback = None
        self._phase = "inflight"
        self.require_execution_feedback = True
        self.action_pub = _InspectingPublisher(self)

    @staticmethod
    def _action_message(action):
        return action.action_id

    def _clear_inference_locked(self, future):
        PolicyClientNode._clear_inference_locked(self, future)

    def _handle_async_failure(self, *args, **kwargs):
        raise AssertionError((args, kwargs))


def test_action_state_is_committed_before_publish_and_inflight_clears_after():
    action = parsed_action()
    future = Future()
    future.set_result(action)
    node = _InferenceHarness(future)

    PolicyClientNode._on_inference_done(node, future, request_epoch=0)

    pending, awaiting, inflight = node.action_pub.state_at_publish
    assert pending is action
    assert awaiting is True
    assert inflight is True
    assert node._pending_action is action
    assert node._awaiting_feedback is True
    assert node._inflight is False


def test_stale_epoch_drops_action_without_publish():
    future = Future()
    future.set_result(parsed_action())
    node = _InferenceHarness(future)
    node._epoch = 2

    PolicyClientNode._on_inference_done(node, future, request_epoch=1)

    assert node.action_pub.messages == []
    assert node._pending_action is None
    assert node._inflight is False


class _ObservationClient:
    def __init__(self):
        self.requests = []
        self.future = Future()

    @staticmethod
    def service_is_ready():
        return True

    def call_async(self, request):
        self.requests.append(request)
        return self.future


def test_request_uses_active_metadata_format_for_state_service():
    metadata_format = {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": "format-from-previous-action",
    }
    client = _ObservationClient()
    node = SimpleNamespace(
        _state_lock=threading.RLock(),
        _closing=False,
        _connected=True,
        _inflight=False,
        require_execution_feedback=True,
        _awaiting_feedback=False,
        observation_client=client,
        runtime=SimpleNamespace(
            active_metadata_format=metadata_format,
            snapshot=lambda: SessionSnapshot(
                session_id="episode-001",
                request_id=7,
                format_id="format-from-previous-action",
                connected=True,
                last_action_id="action-006",
            ),
        ),
        transport=SimpleNamespace(effective_message_size=1024),
        prompt="pick up the object",
        _wire_feedback={
            "last_action_id": "action-006",
            "executed_steps": 2,
            "success": True,
        },
        _epoch=3,
        _infer_future=None,
        _phase="ready",
    )

    PolicyClientNode._request_tick(node)

    assert node._phase == "building_observation"
    assert node._inflight is True
    assert node._infer_future is client.future
    request = client.requests[0]
    assert json.loads(request.metadata_format_json) == metadata_format
    assert request.session_id == "episode-001"
    assert request.request_id == 7
    assert request.prompt == "pick up the object"
    assert json.loads(request.execution_feedback_json)["last_action_id"] == (
        "action-006"
    )
