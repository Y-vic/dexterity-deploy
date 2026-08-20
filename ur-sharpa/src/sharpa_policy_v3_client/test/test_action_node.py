from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("rclpy")
pytest.importorskip("sharpa_policy_v3_interfaces")

from sharpa_policy_v3_client.action_node import ActionNode
from sharpa_policy_v3_interfaces.msg import PolicyActionV3, UrStateFrame


def action_message() -> PolicyActionV3:
    horizon = 4
    pose = np.asarray(
        [0.25, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    message = PolicyActionV3()
    message.action_id = "action-1"
    message.frequency_hz = 30.0
    message.action_length = horizon
    message.execute_start = 1
    message.execute_length = 2
    message.left_wrist_action_type = "eef"
    message.left_wrist_dimension = 9
    message.left_wrist = np.repeat(pose[None], horizon, axis=0).reshape(-1).tolist()
    message.right_wrist_action_type = "eef"
    message.right_wrist_dimension = 9
    message.right_wrist = np.repeat(pose[None], horizon, axis=0).reshape(-1).tolist()
    message.hand_joint_dimension = 44
    message.hand_joint = np.arange(horizon * 44, dtype=np.float32).tolist()
    return message


def ur_state() -> UrStateFrame:
    pose = [0.25, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    message = UrStateFrame()
    message.eef_dimension = 9
    message.left_eef = pose
    message.right_eef = pose
    message.normal_mode = True
    message.valid = True
    return message


def test_action_node_prepares_only_negotiated_slice():
    prepared = ActionNode._prepare(action_message(), ur_state())

    assert prepared.execute_start == 1
    assert prepared.execute_length == 2
    assert prepared.left_wrist.shape == (2, 9)
    assert prepared.right_wrist.shape == (2, 9)
    assert prepared.hand_joint.shape == (2, 44)
    np.testing.assert_array_equal(
        prepared.hand_joint[0],
        np.arange(44, 88, dtype=np.float32),
    )


def test_action_node_requires_normal_ur_state():
    state = ur_state()
    state.normal_mode = False

    with pytest.raises(Exception, match="normal-mode"):
        ActionNode._prepare(action_message(), state)
