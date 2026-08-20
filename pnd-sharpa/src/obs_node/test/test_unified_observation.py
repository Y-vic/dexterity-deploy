from __future__ import annotations

import math
import threading

from sharpa_node.common import SHARPA_JOINT_NAMES
from teleop_interfaces.msg import (
    SharpaJointState,
    TactileDeformImage,
    TactileDeformImageArray,
    TactileForce6D,
    TactileForce6DArray,
)

from obs_node.obs_node import (
    SHARPA_JOINT_LAYOUT,
    TACTILE_LAYOUT,
    TACTILE_ORDER,
    ObsNode,
)


def _bare_node() -> ObsNode:
    # Payload builders do not need a ROS context or sockets.
    return object.__new__(ObsNode)


def _sharpa_message() -> SharpaJointState:
    msg = SharpaJointState()
    msg.joint_state.header.stamp.sec = 123
    msg.joint_state.header.stamp.nanosec = 456
    msg.joint_state.name = list(SHARPA_JOINT_NAMES)
    msg.joint_state.position = [float(index) for index in range(44)]
    msg.joint_state.velocity = [float(index) / 10.0 for index in range(43)]
    msg.joint_state.effort = [float(index) / 100.0 for index in range(44)]
    msg.q_cmd = [float(index) + 100.0 for index in range(44)]
    msg.q_cmd_valid = True
    return msg


def test_on_sharpa_keeps_wrapper_and_q_cmd() -> None:
    node = _bare_node()
    node.lock = threading.Lock()
    node.sharpa_msg = None
    node.sharpa_time = None
    node.sharpa_received = 0
    msg = _sharpa_message()

    node._on_sharpa(msg)

    assert node.sharpa_msg is msg
    assert list(node.sharpa_msg.q_cmd) == list(msg.q_cmd)
    assert node.sharpa_received == 1
    assert node.sharpa_time is not None


def test_sharpa_payload_has_element_masks_source_stamp_and_layout() -> None:
    node = _bare_node()
    msg = _sharpa_message()
    msg.joint_state.position[2] = math.nan
    msg.joint_state.effort[3] = math.inf
    msg.q_cmd[4] = math.nan

    payload = node._sharpa_joint_payload(msg, recv_time=10.0, now=10.25)

    assert payload["feedback_stamp_ns"] == 123_000_000_456
    assert payload["stamp_ns"] == payload["feedback_stamp_ns"]
    assert payload["joint_layout"] == SHARPA_JOINT_LAYOUT
    assert payload["joint_order"] == list(SHARPA_JOINT_NAMES)
    assert len(payload["q_exe"]) == 44
    assert len(payload["q_exe_valid"]) == 44
    assert payload["q_exe"][2] == 0.0
    assert payload["q_exe_valid"][2] is False
    assert payload["joint_velocity_valid"][-1] is False
    assert payload["tau"][3] == 0.0
    assert payload["tau_valid"][3] is False
    assert payload["q_cmd"][0] == 100.0
    assert payload["q_cmd"][4] == 0.0
    assert payload["q_cmd_valid"][4] is False
    assert payload["q_cmd_message_valid"] is True
    # Legacy names remain exact aliases of the explicit facts.
    assert payload["q"] == payload["q_exe"]
    assert payload["dq"] == payload["joint_velocity"]
    assert payload["age_ms"] == 250.0


def test_invalid_q_cmd_message_is_zero_filled_and_masked() -> None:
    node = _bare_node()
    msg = _sharpa_message()
    msg.q_cmd_valid = False

    payload = node._sharpa_joint_payload(msg, recv_time=None, now=20.0)

    assert payload["q_cmd_message_valid"] is False
    assert payload["q_cmd"] == [0.0] * 44
    assert payload["q_cmd_valid"] == [False] * 44


def test_wrench_is_fixed_order_with_per_finger_metadata() -> None:
    node = _bare_node()
    msg = TactileForce6DArray()
    msg.header.stamp.sec = 20
    msg.header.stamp.nanosec = 30
    entry = TactileForce6D()
    entry.side = "left"
    entry.finger = "thumb"
    entry.channel = 9
    entry.frame_id = 7
    entry.sensor_time = 42.5
    entry.force = [1.0, 2.0, 3.0]
    entry.torque = [4.0, 5.0, 6.0]
    msg.forces = [entry]

    payload = node._tactile_state_payload(
        msg,
        force_time=9.5,
        contact_msg=None,
        contact_time=None,
        now=10.0,
    )

    index = TACTILE_ORDER.index("left_thumb")
    assert payload["tactile_layout"] == TACTILE_LAYOUT
    assert payload["force_stamp_ns"] == 20_000_000_030
    assert len(payload["wrench"]) == 10
    assert payload["wrench"][index] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert payload["wrench_valid"][index] is True
    assert payload["wrench_frame_id"][index] == 7
    assert payload["wrench_sensor_timestamp"][index] == 42.5


def test_deformation_bulk_has_fixed_layout_and_raw_offsets() -> None:
    node = _bare_node()
    msg = TactileDeformImageArray()
    msg.header.stamp.sec = 30
    msg.header.stamp.nanosec = 40
    image = TactileDeformImage()
    image.side = "left"
    image.finger = "thumb"
    image.channel = 9
    image.frame_id = 8
    image.sensor_time = 43.5
    image.height = 2
    image.width = 2
    image.data = [1, 2, 3, 4]
    msg.images = [image]

    metadata, raw = node._tactile_bulk_metadata(
        msg,
        seq=11,
        stamp_ns=12,
        nearest_obs_seq=10,
    )

    index = TACTILE_ORDER.index("left_thumb")
    entry = metadata["entries"][index]
    assert metadata["tactile_layout"] == TACTILE_LAYOUT
    assert metadata["order"] == list(TACTILE_ORDER)
    assert metadata["image_count"] == 10
    assert metadata["ros_stamp_ns"] == 30_000_000_040
    assert metadata["valid"][index] is True
    assert metadata["frame_id"][index] == 8
    assert metadata["sensor_timestamp"][index] == 43.5
    assert entry["raw_offset"] == 0
    assert entry["raw_length"] == 4
    assert raw == b"\x01\x02\x03\x04"
