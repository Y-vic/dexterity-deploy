import json

import numpy as np
import pytest

from vive_node.tracker_source import (
    AdaptivePoseFilter,
    LEFT_VIVE_TO_WRIST,
    TrackerLatch,
    apply_relative_pose_delta,
    interpolate_pose,
    limit_pose_delta,
    limit_ur_tcp_target,
    parse_tracker_message,
    retarget_vive_frame,
    retarget_vive_wrist,
)
from ur_common.geometry import UrSharpAWireGeometry, matrix_to_rotvec, rotvec_to_matrix
from vive_node.vive_node import (
    solve_ik_with_cartesian_backtracking,
    validate_ik_joint_step,
    vive_side_validity,
)


def message(topic, position, orientation=None):
    payload = {
        "position": dict(zip(("x", "y", "z"), position)),
        "orientation": dict(
            zip(("x", "y", "z", "w"), orientation or (0.0, 0.0, 0.0, 1.0))
        ),
    }
    return f"{topic} {json.dumps(payload)}"


def test_parse_fqx_topic_json_pose():
    topic, transform = parse_tracker_message(
        message("left_elbow", (1.0, 2.0, 3.0))
    )

    assert topic == "left_elbow"
    assert np.allclose(transform[:3, :3], np.eye(3))
    assert np.allclose(transform[:3, 3], (1.0, 2.0, 3.0))


def test_parse_rejects_wrong_topic_and_degenerate_orientation():
    with pytest.raises(ValueError, match="unexpected Vive topic"):
        parse_tracker_message(message("manus", (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="degenerate"):
        parse_tracker_message(
            message("chest", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
        )


def test_complete_frame_requires_all_three_fresh_topics():
    latch = TrackerLatch()
    identity = np.eye(4)
    latch.put("left_elbow", identity, received=10.0)
    latch.put("right_elbow", identity, received=10.0)
    assert latch.peek() is None
    assert latch.missing_topics() == ("chest",)
    latch.put("chest", identity, received=10.0)

    frame = latch.peek()
    assert frame is not None
    assert frame.stale_topics(0.25, now=10.2) == ()
    assert set(frame.stale_topics(0.25, now=10.3)) == {
        "left_elbow",
        "right_elbow",
        "chest",
    }


def test_retarget_matches_fqx_basis_and_head_translation():
    latch = TrackerLatch()
    poses = {}
    for topic, xyz in (
        ("left_elbow", (1.0, 2.0, 4.0)),
        ("right_elbow", (-1.0, 2.0, 4.0)),
        ("chest", (0.0, 1.0, 3.0)),
    ):
        _, poses[topic] = parse_tracker_message(message(topic, xyz))
        latch.put(topic, poses[topic], received=10.0)

    targets, origin = retarget_vive_frame(latch.peek())

    assert np.allclose(origin, (3.0, 0.0, 1.0))
    assert np.allclose(targets["left"][:3, 3], (1.05, 1.0, 1.40))
    assert np.allclose(targets["right"][:3, 3], (1.05, -1.0, 1.40))
    assert np.allclose(
        targets["left"][:3, :3],
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        @ LEFT_VIVE_TO_WRIST.T,
    )


def test_single_wrist_retarget_does_not_require_other_arm_sample():
    latch = TrackerLatch()
    _, left = parse_tracker_message(message("left_elbow", (1.0, 2.0, 4.0)))
    _, chest = parse_tracker_message(message("chest", (0.0, 1.0, 3.0)))
    latch.put("left_elbow", left, received=10.0)
    latch.put("chest", chest, received=10.0)

    frame = latch.peek()
    assert frame is not None
    assert frame.right is None
    target, origin = retarget_vive_wrist(frame, "left")

    np.testing.assert_allclose(origin, (3.0, 0.0, 1.0))
    np.testing.assert_allclose(target[:3, 3], (1.05, 1.0, 1.40))


def test_side_validity_keeps_fresh_left_when_right_is_missing():
    latch = TrackerLatch()
    _, left = parse_tracker_message(message("left_elbow", (1.0, 2.0, 4.0)))
    _, chest = parse_tracker_message(message("chest", (0.0, 1.0, 3.0)))
    latch.put("left_elbow", left, received=10.0)
    latch.put("chest", chest, received=10.0)

    frame = latch.peek()

    assert vive_side_validity(frame, frame.stale_topics(0.25, now=10.1)) == {
        "left": True,
        "right": False,
    }


def test_zero_tracker_delta_returns_exact_robot_activation_pose():
    tracker_origin = np.eye(4)
    tracker_origin[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    tracker_origin[:3, 3] = (0.3, -0.4, 1.2)
    robot_origin = np.eye(4)
    robot_origin[:3, :3] = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    robot_origin[:3, 3] = (0.55, 0.21, 0.48)

    target = apply_relative_pose_delta(
        tracker_origin.copy(), tracker_origin, robot_origin
    )

    np.testing.assert_array_equal(target, robot_origin)


def test_tracker_translation_delta_maps_one_to_one():
    tracker_origin = np.eye(4)
    tracker_origin[:3, 3] = (1.0, 2.0, 3.0)
    tracker_now = tracker_origin.copy()
    tracker_now[:3, 3] += (0.12, -0.08, 0.03)
    robot_origin = np.eye(4)
    robot_origin[:3, 3] = (0.4, -0.2, 0.7)

    target = apply_relative_pose_delta(
        tracker_now, tracker_origin, robot_origin, translation_scale=1.0
    )

    np.testing.assert_allclose(target[:3, 3], (0.52, -0.28, 0.73))


def test_tracker_rotation_is_relative_not_absolute():
    tracker_origin = np.eye(4)
    tracker_origin[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    rotation_delta = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    tracker_now = tracker_origin.copy()
    tracker_now[:3, :3] = rotation_delta @ tracker_origin[:3, :3]
    robot_origin = np.eye(4)
    robot_origin[:3, :3] = np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    )

    target = apply_relative_pose_delta(tracker_now, tracker_origin, robot_origin)

    np.testing.assert_allclose(
        target[:3, :3], rotation_delta @ robot_origin[:3, :3], atol=1.0e-12
    )
    assert not np.allclose(target[:3, :3], tracker_now[:3, :3])


def test_adaptive_pose_filter_holds_jitter_inside_deadband():
    pose_filter = AdaptivePoseFilter()
    origin = np.eye(4)
    pose_filter.reset(origin, timestamp=10.0)
    jitter = origin.copy()
    jitter[:3, 3] = (0.001, -0.0005, 0.0002)
    jitter[:3, :3] = rotvec_to_matrix((0.0, 0.0, np.deg2rad(0.4)))

    filtered = pose_filter.update(jitter, timestamp=10.0 + 1.0 / 30.0)

    np.testing.assert_array_equal(filtered, origin)


def test_adaptive_pose_filter_preserves_sustained_motion():
    pose_filter = AdaptivePoseFilter()
    origin = np.eye(4)
    pose_filter.reset(origin, timestamp=10.0)
    target = origin.copy()
    target[:3, 3] = (0.10, 0.0, 0.0)

    filtered = origin
    for index in range(1, 121):
        filtered = pose_filter.update(target, timestamp=10.0 + index / 30.0)

    assert 0.098 <= filtered[0, 3] <= 0.10
    np.testing.assert_allclose(filtered[1:3, 3], 0.0)


def test_adaptive_pose_filter_accumulates_sub_deadband_motion():
    pose_filter = AdaptivePoseFilter()
    origin = np.eye(4)
    pose_filter.reset(origin, timestamp=10.0)

    filtered = origin
    for index in range(1, 101):
        target = origin.copy()
        target[0, 3] = index * 0.0002
        filtered = pose_filter.update(target, timestamp=10.0 + index / 30.0)

    assert filtered[0, 3] > 0.015
    assert filtered[0, 3] <= 0.020


def test_adaptive_cutoff_follows_fast_motion_more_closely():
    fixed_filter = AdaptivePoseFilter(position_beta=0.0)
    adaptive_filter = AdaptivePoseFilter(position_beta=5.0)
    origin = np.eye(4)
    fixed_filter.reset(origin, timestamp=10.0)
    adaptive_filter.reset(origin, timestamp=10.0)
    target = origin.copy()
    target[0, 3] = 0.03

    fixed = fixed_filter.update(target, timestamp=10.0 + 1.0 / 30.0)
    adaptive = adaptive_filter.update(target, timestamp=10.0 + 1.0 / 30.0)

    assert 0.0 < fixed[0, 3] < adaptive[0, 3] < target[0, 3]


def test_adaptive_pose_filter_filters_rotation_on_so3():
    pose_filter = AdaptivePoseFilter(rotation_deadband_rad=0.0)
    origin = np.eye(4)
    pose_filter.reset(origin, timestamp=10.0)
    target = origin.copy()
    target[:3, :3] = rotvec_to_matrix((0.2, -0.1, 0.3))

    filtered = pose_filter.update(target, timestamp=10.0 + 1.0 / 30.0)

    filtered_angle = np.linalg.norm(matrix_to_rotvec(filtered[:3, :3]))
    target_angle = np.linalg.norm(matrix_to_rotvec(target[:3, :3]))
    assert 0.0 < filtered_angle < target_angle
    np.testing.assert_allclose(
        filtered[:3, :3].T @ filtered[:3, :3], np.eye(3), atol=1.0e-12
    )
    assert np.linalg.det(filtered[:3, :3]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("frequency_hz", 0.0),
        ("position_deadband_m", -0.001),
        ("rotation_min_cutoff_hz", np.inf),
        ("rotation_beta", -1.0),
    ),
)
def test_adaptive_pose_filter_rejects_invalid_configuration(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        AdaptivePoseFilter(**{parameter: value})


def test_pose_delta_below_limits_passes_through_exactly():
    previous = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = (0.01, -0.02, 0.005)
    target[:3, :3] = rotvec_to_matrix((0.03, -0.02, 0.01))

    limited = limit_pose_delta(previous, target, 0.03, 0.12)

    np.testing.assert_array_equal(limited, target)


def test_pose_delta_limits_translation_norm_and_rotation_angle():
    previous = np.eye(4)
    previous[:3, 3] = (0.4, -0.2, 0.7)
    previous[:3, :3] = rotvec_to_matrix((0.1, -0.2, 0.3))
    target = np.eye(4)
    target[:3, 3] = previous[:3, 3] + (0.06, 0.08, 0.0)
    target[:3, :3] = previous[:3, :3] @ rotvec_to_matrix((0.0, 0.0, 0.4))

    limited = limit_pose_delta(previous, target, 0.03, 0.12)

    translation_delta = limited[:3, 3] - previous[:3, 3]
    rotation_delta = previous[:3, :3].T @ limited[:3, :3]
    assert np.linalg.norm(translation_delta) == pytest.approx(0.03)
    np.testing.assert_allclose(translation_delta / 0.03, (0.6, 0.8, 0.0))
    assert np.linalg.norm(matrix_to_rotvec(rotation_delta)) == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("translation_limit", "rotation_limit", "message_text"),
    ((0.0, 0.12, "max_translation_delta_m"), (0.03, np.inf, "max_rotation_delta_rad")),
)
def test_pose_delta_rejects_invalid_limits(
    translation_limit, rotation_limit, message_text
):
    with pytest.raises(ValueError, match=message_text):
        limit_pose_delta(
            np.eye(4), np.eye(4), translation_limit, rotation_limit
        )


def test_ur_tcp_delta_is_limited_after_capture_frame_conversion():
    class FakeKinematics:
        geometry = UrSharpAWireGeometry()

        @staticmethod
        def arm_fk(_joint):
            return np.eye(4)

    kinematics = FakeKinematics()
    measured_ur_tcp = np.eye(4)
    measured_capture_tcp = kinematics.geometry.ur_base_tcp_to_capture_tcp(
        measured_ur_tcp, "left"
    )
    desired_capture_tcp = measured_capture_tcp.copy()
    desired_capture_tcp[:3, 3] += (0.10, -0.04, 0.02)
    desired_capture_tcp[:3, :3] = (
        desired_capture_tcp[:3, :3]
        @ rotvec_to_matrix((0.0, 0.0, 0.6))
    )

    limited = limit_ur_tcp_target(
        kinematics, "left", np.zeros(6), desired_capture_tcp, 0.03, 0.12
    )

    assert np.linalg.norm(limited[:3, 3] - measured_ur_tcp[:3, 3]) == pytest.approx(0.03)
    relative_rotation = measured_ur_tcp[:3, :3].T @ limited[:3, :3]
    assert np.linalg.norm(matrix_to_rotvec(relative_rotation)) == pytest.approx(0.12)


def test_pose_interpolation_uses_translation_and_shortest_rotation_fraction():
    target = np.eye(4)
    target[:3, 3] = (0.12, -0.04, 0.08)
    target[:3, :3] = rotvec_to_matrix((0.0, 0.0, 0.8))

    interpolated = interpolate_pose(np.eye(4), target, 0.25)

    assert np.allclose(interpolated[:3, 3], (0.03, -0.01, 0.02))
    assert np.allclose(matrix_to_rotvec(interpolated[:3, :3]), (0.0, 0.0, 0.2))


@pytest.mark.parametrize("fraction", (-0.01, 1.01, np.nan))
def test_pose_interpolation_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        interpolate_pose(np.eye(4), np.eye(4), fraction)


def test_vive_ik_backtracks_cartesian_target_without_crossing_flip_gate():
    class FakeSolver:
        def __init__(self):
            self.last_target_joint = np.zeros(6)
            self.last_rejection = None

    class FakeKinematics:
        def __init__(self):
            self.fake_solver = FakeSolver()
            self.requested_x = []

        def solver(self, _side):
            return self.fake_solver

        @staticmethod
        def arm_fk(_joint):
            return np.eye(4)

        def arm_ik(self, target, _near, side, ref_joint):
            assert side == "left"
            assert np.allclose(ref_joint, np.zeros(6))
            x_value = float(target[0, 3])
            self.requested_x.append(x_value)
            if x_value > 0.05:
                self.fake_solver.last_rejection = {
                    "stage": "WRIST_FLIP_HOLD",
                    "minimum_max_delta_rad": 2.55,
                }
                return self.fake_solver.last_target_joint.copy()
            self.fake_solver.last_rejection = None
            candidate = np.full(6, x_value)
            self.fake_solver.last_target_joint = candidate.copy()
            return candidate

    kinematics = FakeKinematics()
    target = np.eye(4)
    target[0, 3] = 0.10

    solution, rejection, fraction, direct_rejection = (
        solve_ik_with_cartesian_backtracking(
            kinematics,
            "left",
            np.zeros(6),
            np.zeros(6),
            target,
            0.35,
        )
    )

    assert rejection is None
    assert direct_rejection["stage"] == "WRIST_FLIP_HOLD"
    assert fraction == pytest.approx(0.5)
    assert np.allclose(solution, 0.05)
    assert kinematics.requested_x == pytest.approx([0.10, 0.075, 0.05])


def test_vive_ik_backtrack_never_bypasses_joint_step_gate():
    class FakeSolver:
        def __init__(self):
            self.last_target_joint = np.full(6, 0.1)
            self.last_rejection = None

    class FakeKinematics:
        def __init__(self):
            self.fake_solver = FakeSolver()

        def solver(self, _side):
            return self.fake_solver

        @staticmethod
        def arm_fk(_joint):
            return np.eye(4)

        def arm_ik(self, target, _near, side, ref_joint):
            del side, ref_joint
            if target[0, 3] >= 0.10:
                self.fake_solver.last_rejection = {"stage": "WRIST_FLIP_HOLD"}
                return self.fake_solver.last_target_joint.copy()
            self.fake_solver.last_rejection = None
            candidate = np.full(6, 0.5)
            self.fake_solver.last_target_joint = candidate.copy()
            return candidate

    kinematics = FakeKinematics()
    target = np.eye(4)
    target[0, 3] = 0.10

    solution, rejection, fraction, _ = solve_ik_with_cartesian_backtracking(
        kinematics,
        "left",
        np.zeros(6),
        np.zeros(6),
        target,
        0.35,
    )

    assert rejection["stage"] == "WRIST_FLIP_HOLD"
    assert fraction == 0.0
    assert np.allclose(solution, 0.1)
    assert np.allclose(kinematics.fake_solver.last_target_joint, 0.1)


def test_vive_rejects_ik_branch_jump_against_measured_joints():
    with pytest.raises(RuntimeError, match=r"stage=INITIAL_STEP_LIMIT.*axes=\[4\]"):
        validate_ik_joint_step(
            np.zeros(6),
            [0.0, 0.0, 0.0, 0.0, 0.36, 0.0],
            0.35,
            "left",
            stage="INITIAL_STEP_LIMIT",
        )


def test_vive_joint_step_treats_full_turn_as_equivalent():
    delta = validate_ik_joint_step(
        np.zeros(6), [2.0 * np.pi + 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 0.35, "left"
    )

    assert delta[0] == pytest.approx(0.1)
