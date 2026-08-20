import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from quest_node.command_gate import ADAM_COMMAND_JOINTS_19
from quest_node.quest_retarget import (
    Pose3,
    QuestRetargetNode,
    RetargetCalibration,
    compute_alignment_rotation,
    targets_from_calibration,
)
from quest_node.adam_bimanual_ik import (
    ELBOW_FRAMES,
    AdamBimanualIkSolver,
    two_bone_elbow_target,
)


MODEL_PATH = (
    "/opt/pnd/pnd_teleop/install/adam_description/share/adam_description/"
    "urdf/adam_pro/adam_pro.urdf"
)
BIAS_POSITIONS = {name: 0.0 for name in ADAM_COMMAND_JOINTS_19}
BIAS_POSITIONS.update(
    {
        "dof_pos/neckPitch": math.radians(46.0),
        "dof_pos/shoulderPitch_Left": math.radians(-10.0),
        "dof_pos/elbow_Left": math.radians(-90.0),
        "dof_pos/wristYaw_Left": math.radians(-90.0),
        "dof_pos/shoulderPitch_Right": math.radians(-10.0),
        "dof_pos/elbow_Right": math.radians(-90.0),
        "dof_pos/wristYaw_Right": math.radians(90.0),
    }
)


def pose(position, rotation=None):
    return Pose3(
        position=np.asarray(position, dtype=np.float64),
        rotation=np.eye(3) if rotation is None else np.asarray(rotation),
    )


def test_alignment_uses_forward_left_up_robot_axes():
    alignment = compute_alignment_rotation(
        np.array([0.5, 0.3, 1.2]),
        np.array([0.5, -0.3, 1.2]),
    )

    assert alignment == pytest.approx(np.eye(3))


def test_relative_wrist_translation_is_one_to_one_from_a_zero():
    initial_tracker = {
        "Head": pose([0.0, 0.0, 1.6]),
        "Left": pose([0.5, 0.3, 1.3]),
        "Right": pose([0.5, -0.3, 1.3]),
    }
    initial_robot = {
        "Head": pose([0.0, 0.0, 1.55]),
        "Left": pose([0.32, 0.22, 1.16]),
        "Right": pose([0.32, -0.22, 1.16]),
    }
    calibration = RetargetCalibration(
        alignment_rotation=np.eye(3),
        tracker_initial=initial_tracker,
        robot_initial=initial_robot,
    )
    current = dict(initial_tracker)
    current["Left"] = pose([0.6, 0.25, 1.5])

    targets = targets_from_calibration(current, calibration)

    assert targets["Left"].position - initial_robot["Left"].position == pytest.approx(
        [0.1, -0.05, 0.2]
    )
    assert targets["Right"].position == pytest.approx(initial_robot["Right"].position)


def test_relative_wrist_rotation_is_applied_from_robot_bias_orientation():
    tracker_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    robot_rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    initial_tracker = {
        name: pose([0.0, 0.0, 0.0]) for name in ("Head", "Left", "Right")
    }
    initial_robot = {
        "Head": pose([0.0, 0.0, 0.0]),
        "Left": pose([0.0, 0.0, 0.0], robot_rotation),
        "Right": pose([0.0, 0.0, 0.0]),
    }
    current = dict(initial_tracker)
    current["Left"] = pose([0.0, 0.0, 0.0], tracker_rotation)
    calibration = RetargetCalibration(
        alignment_rotation=np.eye(3),
        tracker_initial=initial_tracker,
        robot_initial=initial_robot,
    )

    targets = targets_from_calibration(current, calibration)

    assert targets["Left"].rotation == pytest.approx(tracker_rotation @ robot_rotation)


def test_temporary_tracking_hold_preserves_retarget_calibration():
    node = object.__new__(QuestRetargetNode)
    node.tracking_ready = True
    node.last_tracking_time = 1.0
    node.last_source_sequence = 10
    calibration = object()
    node.calibration = calibration
    node.calibration_pending = False

    node._status_callback(
        String(
            data=(
                '{"event":"frame","connected":true,"calibrated":true,'
                '"tracking_fresh":false,"source_sequence":11}'
            )
        )
    )

    assert node.calibration is calibration
    assert not node.tracking_ready

    node._status_callback(
        String(
            data=(
                '{"event":"tracking_stale","connected":true,'
                '"calibrated":false,"tracking_fresh":false}'
            )
        )
    )

    assert node.calibration is None


def test_two_bone_elbow_target_selects_requested_outer_branch():
    shoulder = np.array([0.0, 0.2, 0.4])
    wrist = np.array([0.4, 0.2, 0.3])

    left_elbow = two_bone_elbow_target(
        shoulder,
        wrist,
        np.array([0.0, 1.0, 0.0]),
        0.25,
        0.28,
    )
    inner_elbow = two_bone_elbow_target(
        shoulder,
        wrist,
        np.array([0.0, -1.0, 0.0]),
        0.25,
        0.28,
    )

    assert left_elbow[1] > shoulder[1]
    assert inner_elbow[1] < shoulder[1]
    assert np.linalg.norm(left_elbow - shoulder) == pytest.approx(0.25)
    assert np.linalg.norm(left_elbow - wrist) == pytest.approx(0.28)


@pytest.fixture(scope="module")
def solver():
    return AdamBimanualIkSolver(MODEL_PATH, retarget_method="local_qp")


def test_solver_calibration_uses_exact_bias_joint_pose(solver):
    reference = solver.set_reference(BIAS_POSITIONS)

    assert solver.model.nq == 14
    assert tuple(solver.model.names[1:]) == tuple(
        name.removeprefix("dof_pos/") for name in ADAM_COMMAND_JOINTS_19[5:]
    )
    assert solver.positions_19() == pytest.approx(
        [BIAS_POSITIONS[name] for name in ADAM_COMMAND_JOINTS_19]
    )
    assert reference["Left"].position == pytest.approx(
        [0.31938506, 0.22269691, 0.2122904],
        abs=1.0e-6,
    )
    assert reference["Right"].position == pytest.approx(
        [0.31938506, -0.22169691, 0.2122904],
        abs=1.0e-6,
    )


def test_zero_tracker_delta_keeps_bias_pose(solver):
    reference = solver.set_reference(BIAS_POSITIONS)
    solver.set_targets(reference)
    solver.solve(iterations=5, solve_dt=0.05)

    assert solver.positions_19() == pytest.approx(
        [BIAS_POSITIONS[name] for name in ADAM_COMMAND_JOINTS_19],
        abs=1.0e-7,
    )
    assert max(solver.wrist_errors(reference).values()) < 0.01


def test_reachable_motion_is_not_clipped_by_an_output_speed_limit(solver):
    reference = solver.set_reference(BIAS_POSITIONS)
    targets = {
        name: Pose3(value.position.copy(), value.rotation.copy())
        for name, value in reference.items()
    }
    targets["Left"] = Pose3(
        targets["Left"].position + np.array([0.1, 0.0, 0.1]),
        targets["Left"].rotation,
    )
    targets["Right"] = Pose3(
        targets["Right"].position + np.array([0.1, 0.0, 0.1]),
        targets["Right"].rotation,
    )
    previous = np.asarray(solver.positions_19())
    solver.set_targets(targets)
    solver.solve(iterations=5, solve_dt=0.05)
    current = np.asarray(solver.positions_19())

    assert np.max(np.abs(current - previous)) > 0.06
    first_frame_error = max(solver.wrist_errors(targets).values())
    assert first_frame_error < 50.0

    for _ in range(4):
        solver.solve(iterations=5, solve_dt=0.05)

    assert max(solver.wrist_errors(targets).values()) < 1.0


def test_fixed_wrist_position_rotation_stays_out_of_shoulder_and_elbow(solver):
    elbow_solver = AdamBimanualIkSolver(MODEL_PATH, retarget_method="elbow_pole")
    reference = elbow_solver.set_reference(BIAS_POSITIONS)
    targets = {
        name: Pose3(value.position.copy(), value.rotation.copy())
        for name, value in reference.items()
    }
    targets["Left"] = Pose3(
        targets["Left"].position,
        Rotation.from_rotvec([math.radians(45.0), 0.0, 0.0]).as_matrix()
        @ targets["Left"].rotation,
    )
    initial = np.asarray(elbow_solver.positions_19())[5:12]

    for _ in range(12):
        elbow_solver.set_targets(targets)
        elbow_solver.solve(iterations=5, solve_dt=0.05)

    delta_degrees = np.degrees(
        np.asarray(elbow_solver.positions_19())[5:12] - initial
    )
    assert np.max(np.abs(delta_degrees[:4])) < 1.0
    assert np.max(np.abs(delta_degrees[4:])) > 30.0
    assert elbow_solver.wrist_errors(targets)["Left"] < 0.1


@pytest.mark.parametrize(
    "translation",
    [
        [0.35, 0.0, 0.0],
        [0.35, 0.15, 0.0],
        [0.35, 0.0, 0.15],
        [0.45, -0.15, -0.15],
    ],
)
def test_unreachable_wrist_target_converges_without_joint_cycle(translation):
    local_solver = AdamBimanualIkSolver(MODEL_PATH, retarget_method="local_qp")
    reference = local_solver.set_reference(BIAS_POSITIONS)
    targets = {
        name: Pose3(value.position.copy(), value.rotation.copy())
        for name, value in reference.items()
    }
    targets["Left"] = Pose3(
        targets["Left"].position + np.asarray(translation),
        targets["Left"].rotation,
    )
    history = []

    for _ in range(80):
        local_solver.set_targets(targets)
        local_solver.solve(iterations=5, solve_dt=0.05)
        history.append(np.asarray(local_solver.positions_19())[5:12])

    tail = np.asarray(history[-20:])
    max_step = np.max(np.abs(np.diff(tail, axis=0)))

    assert np.degrees(max_step) < 0.1
    assert np.all(np.isfinite(tail))


def test_head_motion_changes_only_neck_and_keeps_waist_at_bias(solver):
    reference = solver.set_reference(BIAS_POSITIONS)
    targets = {
        name: Pose3(value.position.copy(), value.rotation.copy())
        for name, value in reference.items()
    }
    targets["Head"] = Pose3(
        targets["Head"].position,
        Rotation.from_euler("z", 0.2).as_matrix() @ targets["Head"].rotation,
    )
    solver.set_targets(targets)
    solver.solve(iterations=5, solve_dt=0.05)
    values = dict(zip(ADAM_COMMAND_JOINTS_19, solver.positions_19(), strict=True))

    assert [values[name] for name in ADAM_COMMAND_JOINTS_19[:3]] == pytest.approx(
        [BIAS_POSITIONS[name] for name in ADAM_COMMAND_JOINTS_19[:3]]
    )
    assert values["dof_pos/neckYaw"] != pytest.approx(BIAS_POSITIONS["dof_pos/neckYaw"])


def test_disabled_neck_tracking_keeps_bias_neck_pose(solver):
    reference = solver.set_reference(BIAS_POSITIONS)
    targets = {
        name: Pose3(value.position.copy(), value.rotation.copy())
        for name, value in reference.items()
    }
    targets["Head"] = Pose3(
        targets["Head"].position,
        Rotation.from_euler("zy", [0.2, -0.15]).as_matrix()
        @ targets["Head"].rotation,
    )

    solver.set_targets(targets, track_neck=False)
    solver.solve(iterations=5, solve_dt=0.05)
    values = dict(zip(ADAM_COMMAND_JOINTS_19, solver.positions_19(), strict=True))

    assert values["dof_pos/neckYaw"] == pytest.approx(
        BIAS_POSITIONS["dof_pos/neckYaw"]
    )
    assert values["dof_pos/neckPitch"] == pytest.approx(
        BIAS_POSITIONS["dof_pos/neckPitch"]
    )
