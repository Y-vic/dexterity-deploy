import math

import pytest

from quest_node.webvr_protocol import (
    Quaternion,
    Vector3,
    WebVRProtocolError,
    HandExecutionGate,
    calibration_from_sample,
    flatten_joy_buttons,
    hand_execution_is_ready,
    hand_positions_are_usable,
    parse_webvr_message,
    position_distance,
    validate_zero_pose_sample,
    vr_pose_to_ros,
)


def webvr_payload():
    pose = {
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0},
    }
    return {
        "timestamp": 1234,
        "Head": pose,
        "LeftHand": {
            **pose,
            "position": {"x": -0.4, "y": 1.2, "z": -0.53},
        },
        "RightHand": {
            **pose,
            "position": {"x": 0.4, "y": 1.2, "z": -0.53},
        },
        "Joy": {
            "axes": [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, 0.6, 0.7],
            "buttons": [
                [0, False],
                [1, True],
                [0, True],
                [1, False],
                [1, True],
                [0, False],
            ],
        },
    }


def test_parse_preserves_pnd_joy_layout_and_boolean_touches():
    sample = parse_webvr_message(webvr_payload())

    assert sample.joy_axes == (0.1, 0.2, 0.3, 0.4, -0.1, -0.2, 0.6, 0.7)
    assert flatten_joy_buttons(sample.joy_buttons) == [
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
    ]
    assert sample.poses["Head"].quaternion.w == pytest.approx(1.0)
    assert not hand_positions_are_usable(sample)


def test_missing_tracking_metadata_fails_closed():
    sample = parse_webvr_message(webvr_payload())

    assert not sample.poses["LeftHand"].tracking.connected
    assert sample.poses["LeftHand"].tracking.position == "Unavailable"
    assert not sample.poses["RightHand"].tracking.position_is_known


def test_position_distance_measures_single_hand_jump():
    previous_payload = calibration_payload()
    current_payload = calibration_payload()
    current_payload["RightHand"]["position"]["x"] += 0.25

    previous = parse_webvr_message(previous_payload)
    current = parse_webvr_message(current_payload)

    assert position_distance(
        previous.poses["RightHand"],
        current.poses["RightHand"],
    ) == pytest.approx(0.25)


def test_calibration_command_is_parsed_separately_from_joy():
    payload = calibration_payload()
    payload["Calibration"] = {"pressed": True}
    payload["Joy"]["buttons"][4] = [0, 0]

    sample = parse_webvr_message(payload)

    assert sample.calibration_pressed


def test_calibration_command_falls_back_to_joy_button():
    payload = calibration_payload()
    payload["Joy"]["buttons"][4] = [1, 1]

    assert parse_webvr_message(payload).calibration_pressed


def test_parse_preserves_optional_source_timing_metadata():
    payload = calibration_payload()
    payload["sequence"] = 42
    payload["monotonicTimestampMs"] = 123.5

    sample = parse_webvr_message(payload)

    assert sample.source_sequence == 42
    assert sample.source_monotonic_ms == pytest.approx(123.5)


@pytest.mark.parametrize("sequence", [-1, 1.5, "invalid"])
def test_parse_rejects_invalid_source_sequence(sequence):
    payload = calibration_payload()
    payload["sequence"] = sequence

    with pytest.raises(WebVRProtocolError, match="sequence"):
        parse_webvr_message(payload)


def test_position_distance_is_independent_of_tracking_state():
    previous = parse_webvr_message(calibration_payload())
    current_payload = calibration_payload()
    current_payload["RightHand"]["tracking"]["position"] = "Inferred"
    current_payload["RightHand"]["position"]["x"] += 0.25
    current = parse_webvr_message(current_payload)

    assert position_distance(
        previous.poses["RightHand"],
        current.poses["RightHand"],
    ) == pytest.approx(0.25)


def test_hand_gate_holds_only_jumping_hand():
    left_gate = HandExecutionGate("LeftHand")
    right_gate = HandExecutionGate("RightHand")
    sample = parse_webvr_message(calibration_payload())
    jumped_payload = calibration_payload()
    jumped_payload["RightHand"]["position"]["x"] += 0.2
    jumped = parse_webvr_message(jumped_payload)

    left_gate.observe(
        sample.poses["LeftHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    right_gate.observe(
        sample.poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    held = right_gate.observe(
        jumped.poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )

    assert left_gate.state == "Normal"
    assert right_gate.state == "Suspect"
    assert held == sample.poses["RightHand"]


def test_hand_gate_recovers_near_pose_without_calibration():
    gate = HandExecutionGate("RightHand")
    sample = parse_webvr_message(calibration_payload())
    gate.observe(
        sample.poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    lost_payload = calibration_payload()
    lost_payload["RightHand"]["tracking"]["position"] = "Lost"
    gate.observe(
        parse_webvr_message(lost_payload).poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    recovered_payload = calibration_payload()
    recovered_payload["RightHand"]["position"]["x"] += 0.05

    recovered = gate.observe(
        parse_webvr_message(recovered_payload).poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )

    assert gate.state == "Normal"
    assert recovered == parse_webvr_message(recovered_payload).poses["RightHand"]


def test_hand_gate_accepts_large_recovery_jump_with_a():
    gate = HandExecutionGate("RightHand")
    sample = parse_webvr_message(calibration_payload())
    gate.observe(
        sample.poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    lost_payload = calibration_payload()
    lost_payload["RightHand"]["tracking"]["position"] = "Lost"
    gate.observe(
        parse_webvr_message(lost_payload).poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    recovered_payload = calibration_payload()
    recovered_payload["RightHand"]["position"]["x"] += 0.3
    recovered = parse_webvr_message(recovered_payload).poses["RightHand"]

    assert gate.observe(
        recovered,
        calibration_command=True,
        jump_threshold=0.1,
    ) == recovered
    assert gate.state == "Normal"


def test_hand_gate_reanchors_far_stable_recovery_without_position_jump():
    gate = HandExecutionGate("RightHand")
    initial = parse_webvr_message(calibration_payload()).poses["RightHand"]
    assert gate.observe(
        initial,
        calibration_command=False,
        jump_threshold=0.1,
    ) == initial

    lost_payload = calibration_payload()
    lost_payload["RightHand"]["tracking"]["position"] = "Lost"
    held = gate.observe(
        parse_webvr_message(lost_payload).poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    assert gate.state == "Lost"
    assert held.position == initial.position

    recovered = None
    for delta in (0.30, 0.31, 0.32):
        payload = calibration_payload()
        payload["RightHand"]["position"]["x"] += delta
        payload["RightHand"]["quaternion"] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0,
        }
        recovered = gate.observe(
            parse_webvr_message(payload).poses["RightHand"],
            calibration_command=False,
            jump_threshold=0.1,
        )

    assert gate.state == "Normal"
    assert gate.ready
    assert recovered.position == initial.position
    assert recovered.quaternion == pytest.approx(initial.quaternion)

    moved_payload = calibration_payload()
    moved_payload["RightHand"]["position"]["x"] += 0.34
    moved_payload["RightHand"]["quaternion"] = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "w": 1.0,
    }
    moved = gate.observe(
        parse_webvr_message(moved_payload).poses["RightHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )
    assert moved.position.x - recovered.position.x == pytest.approx(0.02)


def test_hand_gate_requires_consecutive_reasonable_recovery_frames():
    gate = HandExecutionGate("RightHand")
    initial = parse_webvr_message(calibration_payload()).poses["RightHand"]
    gate.observe(initial, calibration_command=False, jump_threshold=0.1)

    for delta in (0.30, 0.60, 0.90):
        payload = calibration_payload()
        payload["RightHand"]["position"]["x"] += delta
        held = gate.observe(
            parse_webvr_message(payload).poses["RightHand"],
            calibration_command=False,
            jump_threshold=0.1,
        )
        assert held.position == initial.position

    assert gate.state == "Recovering"
    assert gate.recovery_frames == 1


def test_hand_execution_is_not_ready_while_either_gate_holds():
    sample = parse_webvr_message(calibration_payload())
    gates = {
        "LeftHand": HandExecutionGate("LeftHand"),
        "RightHand": HandExecutionGate("RightHand"),
    }
    execution = {
        name: gate.observe(
            sample.poses[name],
            calibration_command=False,
            jump_threshold=0.1,
        )
        for name, gate in gates.items()
    }
    assert hand_execution_is_ready(gates, execution)

    lost_payload = calibration_payload()
    lost_payload["LeftHand"]["tracking"]["position"] = "Lost"
    execution["LeftHand"] = gates["LeftHand"].observe(
        parse_webvr_message(lost_payload).poses["LeftHand"],
        calibration_command=False,
        jump_threshold=0.1,
    )

    assert not hand_execution_is_ready(gates, execution)


def test_zero_pose_calibration_accepts_inferred_hand_position():
    payload = calibration_payload()
    payload["LeftHand"]["tracking"] = {
        "connected": True,
        "position": "Inferred",
        "rotation": "Known",
    }

    sample = parse_webvr_message(payload)

    assert hand_positions_are_usable(sample)
    validate_zero_pose_sample(sample)


def test_arms_forward_calibration_accepts_inferred_hand_position():
    payload = calibration_payload()
    payload["RightHand"]["tracking"] = {
        "connected": True,
        "position": "Inferred",
        "rotation": "Known",
    }

    sample = parse_webvr_message(payload)

    assert hand_positions_are_usable(sample)
    calibration_from_sample(sample)


@pytest.mark.parametrize("state", ["Lost", "Cached", "Unavailable"])
def test_calibration_rejects_unusable_hand_position(state):
    payload = calibration_payload()
    payload["LeftHand"]["tracking"]["position"] = state
    sample = parse_webvr_message(payload)

    assert not hand_positions_are_usable(sample)
    with pytest.raises(WebVRProtocolError, match="LeftHand position tracking"):
        validate_zero_pose_sample(sample)


def test_vr_to_ros_applies_coordinate_conversion_and_calibrated_offset():
    sample = parse_webvr_message(webvr_payload())
    converted = vr_pose_to_ros(
        sample.poses["Head"],
        scale=2.0,
        position_offset=Vector3(0.0, 0.0, 1.0),
    )

    assert converted.position.x == pytest.approx(-6.0)
    assert converted.position.y == pytest.approx(-2.0)
    assert converted.position.z == pytest.approx(5.0)
    assert converted.quaternion.x == pytest.approx(0.0)
    assert converted.quaternion.y == pytest.approx(0.0)
    assert converted.quaternion.z == pytest.approx(0.0)
    assert converted.quaternion.w == pytest.approx(1.0)


def calibration_payload():
    payload = webvr_payload()
    payload["Head"]["position"] = {"x": 0.1, "y": 1.65, "z": 0.0}
    payload["Head"]["quaternion"] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    payload["LeftHand"]["position"] = {"x": -0.35, "y": 1.35, "z": -0.7}
    payload["LeftHand"]["quaternion"] = {
        "x": 0.0,
        "y": math.sqrt(0.5),
        "z": 0.0,
        "w": math.sqrt(0.5),
    }
    payload["LeftHand"]["tracking"] = {
        "connected": True,
        "position": "Known",
        "rotation": "Known",
    }
    payload["RightHand"]["position"] = {"x": 0.45, "y": 1.4, "z": -0.7}
    payload["RightHand"]["quaternion"] = {
        "x": math.sqrt(0.5),
        "y": 0.0,
        "z": 0.0,
        "w": math.sqrt(0.5),
    }
    payload["RightHand"]["tracking"] = {
        "connected": True,
        "position": "Known",
        "rotation": "Known",
    }
    return payload


def test_calibration_maps_sample_to_adam_pro_reference_pose():
    sample = parse_webvr_message(calibration_payload())

    calibration = calibration_from_sample(sample)
    calibrated_poses = {
        name: vr_pose_to_ros(
            sample.poses[name],
            scale=calibration.scale,
            position_rotation=calibration.position_rotation,
            position_offset=calibration.position_offsets[name],
            quaternion_offset=calibration.quaternion_offsets[name],
        )
        for name in sample.poses
    }

    assert calibration.scale == pytest.approx(0.53 / 0.7)
    expected_positions = {
        "Head": Vector3(0.0186, 0.0204, 1.5715),
        "LeftHand": Vector3(0.54, 0.2, 1.4),
        "RightHand": Vector3(0.54, -0.2, 1.4),
    }
    hand_quaternions = {
        "LeftHand": Quaternion(0.5, -0.5, -0.5, 0.5),
        "RightHand": Quaternion(0.5, 0.5, -0.5, -0.5),
    }
    for name, expected_position in expected_positions.items():
        assert calibrated_poses[name].position == pytest.approx(expected_position)
        expected_quaternion = (
            Quaternion(0.0, 0.0, 0.0, 1.0) if name == "Head" else hand_quaternions[name]
        )
        assert calibrated_poses[name].quaternion == pytest.approx(expected_quaternion)


def test_second_calibration_sample_replaces_full_reference():
    first_sample = parse_webvr_message(calibration_payload())
    second_payload = calibration_payload()
    second_payload["Head"]["position"]["z"] = 0.2
    second_payload["LeftHand"]["position"]["z"] = -0.6
    second_payload["RightHand"]["position"]["z"] = -0.6
    second_sample = parse_webvr_message(second_payload)
    first = calibration_from_sample(first_sample)
    second = calibration_from_sample(second_sample)

    using_old_reference = vr_pose_to_ros(
        second_sample.poses["LeftHand"],
        scale=first.scale,
        position_rotation=first.position_rotation,
        position_offset=first.position_offsets["LeftHand"],
        quaternion_offset=first.quaternion_offsets["LeftHand"],
    )
    using_new_reference = vr_pose_to_ros(
        second_sample.poses["LeftHand"],
        scale=second.scale,
        position_rotation=second.position_rotation,
        position_offset=second.position_offsets["LeftHand"],
        quaternion_offset=second.quaternion_offsets["LeftHand"],
    )

    assert using_old_reference.position.x != pytest.approx(0.5)
    assert using_new_reference.position == pytest.approx(Vector3(0.54, 0.2, 1.4))


def test_zero_pose_calibration_accepts_current_non_forward_pose():
    payload = calibration_payload()
    payload["LeftHand"]["position"] = {"x": -0.2, "y": 0.8, "z": 0.1}
    payload["RightHand"]["position"] = {"x": 0.2, "y": 0.7, "z": 0.0}

    validate_zero_pose_sample(parse_webvr_message(payload))


def test_zero_pose_calibration_rejects_hands_too_close():
    payload = calibration_payload()
    payload["LeftHand"]["position"] = {"x": 0.0, "y": 1.0, "z": -0.5}
    payload["RightHand"]["position"] = {"x": 0.01, "y": 1.0, "z": -0.5}

    with pytest.raises(WebVRProtocolError, match="must be separated"):
        validate_zero_pose_sample(parse_webvr_message(payload))


def test_calibration_rejects_hands_without_forward_extension():
    payload = calibration_payload()
    payload["LeftHand"]["position"]["z"] = 0.0
    payload["RightHand"]["position"]["z"] = 0.0
    sample = parse_webvr_message(payload)

    with pytest.raises(WebVRProtocolError, match="extended horizontally forward"):
        calibration_from_sample(sample)


def test_calibration_rejects_hands_at_different_heights():
    payload = calibration_payload()
    payload["RightHand"]["position"]["y"] -= 0.3

    with pytest.raises(WebVRProtocolError, match="same height"):
        calibration_from_sample(parse_webvr_message(payload))


def test_calibration_preserves_relative_hand_motion_and_orientation():
    calibration_sample = parse_webvr_message(calibration_payload())
    calibration = calibration_from_sample(calibration_sample)

    moved_payload = calibration_payload()
    moved_payload["LeftHand"]["position"]["x"] -= 0.1
    moved_payload["LeftHand"]["quaternion"] = {
        "x": math.sqrt(0.5),
        "y": 0.0,
        "z": 0.0,
        "w": math.sqrt(0.5),
    }
    moved_sample = parse_webvr_message(moved_payload)
    moved_left = vr_pose_to_ros(
        moved_sample.poses["LeftHand"],
        scale=calibration.scale,
        position_rotation=calibration.position_rotation,
        position_offset=calibration.position_offsets["LeftHand"],
        quaternion_offset=calibration.quaternion_offsets["LeftHand"],
    )

    calibrated_left = vr_pose_to_ros(
        calibration_sample.poses["LeftHand"],
        scale=calibration.scale,
        position_rotation=calibration.position_rotation,
        position_offset=calibration.position_offsets["LeftHand"],
        quaternion_offset=calibration.quaternion_offsets["LeftHand"],
    )
    assert moved_left.position.y - calibrated_left.position.y == pytest.approx(
        0.1 * calibration.scale
    )
    assert moved_left.quaternion != calibrated_left.quaternion


def test_calibration_rejects_head_not_facing_horizontally():
    payload = calibration_payload()
    payload["Head"]["quaternion"] = {
        "x": math.sqrt(0.5),
        "y": 0.0,
        "z": 0.0,
        "w": math.sqrt(0.5),
    }

    with pytest.raises(WebVRProtocolError, match="head must face horizontally"):
        calibration_from_sample(parse_webvr_message(payload))


def test_calibration_rejects_head_pitch_over_twenty_degrees():
    payload = calibration_payload()
    pitch = math.radians(25.0)
    payload["Head"]["quaternion"] = {
        "x": math.sin(0.5 * pitch),
        "y": 0.0,
        "z": 0.0,
        "w": math.cos(0.5 * pitch),
    }

    with pytest.raises(WebVRProtocolError, match="within 20 degrees"):
        calibration_from_sample(parse_webvr_message(payload))


def test_calibration_rejects_head_roll_over_twenty_degrees():
    payload = calibration_payload()
    roll = math.radians(25.0)
    payload["Head"]["quaternion"] = {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(0.5 * roll),
        "w": math.cos(0.5 * roll),
    }

    with pytest.raises(WebVRProtocolError, match="within 20 degrees"):
        calibration_from_sample(parse_webvr_message(payload))


def test_rejects_non_finite_pose_values():
    payload = webvr_payload()
    payload["Head"]["position"]["x"] = math.inf

    with pytest.raises(WebVRProtocolError, match="finite number"):
        parse_webvr_message(payload)


def test_rejects_wrong_joy_lengths():
    payload = webvr_payload()
    payload["Joy"]["axes"] = [0.0] * 7

    with pytest.raises(WebVRProtocolError, match="exactly 8"):
        parse_webvr_message(payload)
