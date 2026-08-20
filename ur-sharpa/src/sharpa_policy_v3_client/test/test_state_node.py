from __future__ import annotations

import json
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest


pytest.importorskip("rclpy")
pytest.importorskip("sharpa_policy_v3_interfaces")

from sharpa_policy_v3_client.buffers import DeformationFrame, ObservationBuffers
from sharpa_policy_v3_client.mock_server import empty_metadata_format
from sharpa_policy_v3_client.observation import ObservationBuilder
from sharpa_policy_v3_client.serialization import unpackb
from sharpa_policy_v3_client.state_node import StateNode
from sharpa_policy_v3_interfaces.msg import HandStateFrame, UrStateFrame
from sharpa_policy_v3_interfaces.srv import BuildObservation


def _deformation(timestamp_ns: int) -> DeformationFrame:
    return DeformationFrame(
        timestamp_ns=timestamp_ns,
        left=np.full((5, 240, 240), timestamp_ns, dtype=np.uint8),
        right=np.full((5, 240, 240), timestamp_ns + 10, dtype=np.uint8),
        left_valid=np.ones((5,), dtype=np.bool_),
        right_valid=np.ones((5,), dtype=np.bool_),
    )


def _service_harness(
    buffers: ObservationBuffers,
    *,
    aggregate_hardware_state: bool = False,
    has_merged_state: bool = False,
):
    return SimpleNamespace(
        aggregate_hardware_state=aggregate_hardware_state,
        _has_merged_state=has_merged_state,
        max_message_size=64 * 1024 * 1024,
        buffers=buffers,
        observation_builder=ObservationBuilder(buffers),
        _observation_error=StateNode._observation_error,
    )


def _request(metadata_format: dict) -> BuildObservation.Request:
    request = BuildObservation.Request()
    request.metadata_format_json = json.dumps(metadata_format)
    request.session_id = "episode-001"
    request.request_id = 0
    request.timestamp_ns = 123456789
    request.prompt = "test"
    request.execution_feedback_json = ""
    request.max_message_size = 64 * 1024 * 1024
    return request


def test_optional_source_values_reject_stale_dimensions():
    assert StateNode._optional_float_vector(
        False,
        0,
        [],
        "left_wrist_joint",
    ) is None

    with pytest.raises(ValueError, match="dimension must be 0"):
        StateNode._optional_float_vector(
            False,
            7,
            [],
            "left_wrist_joint",
        )

    assert StateNode._optional_float_vector(
        False,
        22,
        [],
        "left_hand_joint",
        expected_dimension=22,
        shared_dimension=True,
    ) is None


def test_build_observation_waits_for_first_merged_hardware_state():
    node = _service_harness(
        ObservationBuffers(),
        aggregate_hardware_state=True,
        has_merged_state=False,
    )

    response = StateNode._on_build_observation(
        node,
        _request(empty_metadata_format()),
        BuildObservation.Response(),
    )

    assert response.success is False
    assert response.retryable is True
    assert response.error_code == "state_not_ready"


def test_build_observation_selects_two_retained_deformation_frames():
    buffers = ObservationBuffers(deformation_frame_capacity=2)
    for timestamp_ns in (1, 2, 3):
        buffers.push_deformation(_deformation(timestamp_ns))
    metadata_format = empty_metadata_format()
    metadata_format["sensor"]["deformation"] = {
        "history_len": 1,
        "current": True,
    }
    node = _service_harness(buffers)

    response = StateNode._on_build_observation(
        node,
        _request(metadata_format),
        BuildObservation.Response(),
    )

    assert response.success is True
    observation = unpackb(bytes(response.observation_msgpack))
    deformation = observation["sensor"]["deformation"]
    assert deformation["history"]["timestamp_ns"].tolist() == [2]
    assert deformation["current"]["timestamp_ns"] == 3


def test_build_observation_rejects_metadata_beyond_local_capacity():
    buffers = ObservationBuffers(deformation_frame_capacity=2)
    metadata_format = empty_metadata_format()
    metadata_format["sensor"]["deformation"] = {
        "history_len": 2,
        "current": True,
    }
    node = _service_harness(buffers)

    response = StateNode._on_build_observation(
        node,
        _request(metadata_format),
        BuildObservation.Response(),
    )

    assert response.success is False
    assert response.retryable is False
    assert response.error_code == "metadata_capacity_exceeded"


def test_hardware_state_merge_preserves_device_fields_and_timestamps():
    now_ns = time.time_ns()
    ur = UrStateFrame()
    ur.timestamp_ns = now_ns
    ur.joint_dimension = 6
    ur.left_joint = np.arange(6, dtype=np.float32).tolist()
    ur.right_joint = (np.arange(6, dtype=np.float32) + 10).tolist()
    ur.eef_dimension = 9
    ur.left_eef = [0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ur.right_eef = [0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ur.eef_frame = "robot_base"
    ur.normal_mode = True
    ur.valid = True
    hand = HandStateFrame()
    hand.timestamp_ns = now_ns + 1_000_000
    hand.joint_dimension = 22
    hand.left_joint = np.arange(22, dtype=np.float32).tolist()
    hand.right_joint = (np.arange(22, dtype=np.float32) + 30).tolist()
    hand.left_valid = True
    hand.right_valid = True
    published = []
    buffered = []

    def buffer_state(message):
        buffered.append(message)
        return True

    node = SimpleNamespace(
        _source_lock=threading.Lock(),
        _latest_ur_state=ur,
        _latest_hand_state=hand,
        _last_merged_state_key=None,
        _has_merged_state=False,
        max_state_age_ns=250_000_000,
        max_state_skew_ns=50_000_000,
        state_pub=SimpleNamespace(publish=published.append),
        _on_state=buffer_state,
    )

    StateNode._merge_hardware_state(node)

    assert len(published) == 1
    assert buffered == published
    assert node._has_merged_state is True
    message = published[0]
    assert message.timestamp_ns == hand.timestamp_ns
    assert message.left_wrist_joint == ur.left_joint
    assert message.right_wrist_eef == ur.right_eef
    assert message.left_hand_joint == hand.left_joint
    assert message.right_hand_joint == hand.right_joint
