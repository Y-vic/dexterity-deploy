import math
import numpy as np
import pinocchio as pin
import pytest

from quest_node.adam_bimanual_ik import (
    EE_FRAMES,
    ELBOW_FRAMES,
    AdamBimanualIkSolver,
    Pose3,
)
from quest_node.command_gate import ADAM_COMMAND_JOINTS_19

casadi = pytest.importorskip("casadi")


MODEL_PATH = (
    "/opt/pnd/pnd_teleop/install/adam_description/share/adam_description/"
    "urdf/adam_pro/adam_pro.urdf"
)
BIAS_POSITIONS = {
    name: value
    for name, value in zip(
        ADAM_COMMAND_JOINTS_19,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            math.radians(25.0),
            0.0,
            math.radians(-90.0),
            0.0,
            0.0,
            0.0,
            0.0,
            math.radians(-25.0),
            0.0,
            math.radians(-90.0),
            0.0,
            0.0,
            0.0,
        ],
        strict=True,
    )
}


@pytest.fixture(scope="module")
def nonlinear_solver():
    solver = AdamBimanualIkSolver(
        MODEL_PATH,
        retarget_method="nonlinear_ik",
        nonlinear_filter_enabled=False,
    )
    solver.set_reference(BIAS_POSITIONS)
    return solver


def test_symbolic_forward_kinematics_matches_pinocchio(nonlinear_solver):
    random = np.random.default_rng(7)
    forward_kinematics = nonlinear_solver.nonlinear_problem["forward_kinematics"]
    for _ in range(10):
        qpos = random.uniform(
            nonlinear_solver.model.lowerPositionLimit,
            nonlinear_solver.model.upperPositionLimit,
        )
        symbolic_transforms = forward_kinematics(qpos)
        pin.forwardKinematics(nonlinear_solver.model, nonlinear_solver.data, qpos)
        pin.updateFramePlacements(nonlinear_solver.model, nonlinear_solver.data)
        for side, symbolic_transform in zip(
            ("Left", "Right"), symbolic_transforms[:2], strict=True
        ):
            numeric_transform = nonlinear_solver.data.oMf[
                nonlinear_solver.model.getFrameId(EE_FRAMES[side])
            ].homogeneous
            assert np.asarray(symbolic_transform) == pytest.approx(
                numeric_transform,
                abs=1.0e-10,
            )
        for side, symbolic_position in zip(
            ("Left", "Right"), symbolic_transforms[2:], strict=True
        ):
            numeric_position = nonlinear_solver.data.oMf[
                nonlinear_solver.model.getFrameId(ELBOW_FRAMES[side])
            ].translation
            assert np.asarray(symbolic_position).reshape(3) == pytest.approx(
                numeric_position,
                abs=1.0e-10,
            )


def test_nonlinear_warmup_leaves_clean_bias_state(nonlinear_solver):
    assert nonlinear_solver.last_output_qpos == pytest.approx(
        nonlinear_solver.default_qpos
    )
    assert nonlinear_solver.targets is None
    assert not nonlinear_solver.nonlinear_history


def test_nonlinear_solver_tracks_reachable_targets_within_joint_limits(
    nonlinear_solver,
):
    reference = nonlinear_solver.set_reference(BIAS_POSITIONS)
    targets = {
        name: Pose3(pose.position.copy(), pose.rotation.copy())
        for name, pose in reference.items()
    }
    for side, lateral_delta in (("Left", 0.02), ("Right", -0.02)):
        position = targets[side].position.copy()
        position[1] += lateral_delta
        targets[side] = Pose3(position, targets[side].rotation)

    nonlinear_solver.set_targets(targets)
    nonlinear_solver.solve(iterations=5, solve_dt=0.05)

    assert max(nonlinear_solver.wrist_errors(targets).values()) < 5.0
    assert np.all(
        nonlinear_solver.last_output_qpos
        >= nonlinear_solver.model.lowerPositionLimit - 1.0e-12
    )
    assert np.all(
        nonlinear_solver.last_output_qpos
        <= nonlinear_solver.model.upperPositionLimit + 1.0e-12
    )
    assert min(nonlinear_solver.elbow_outer_distances().values()) >= 0.0


def test_nonlinear_inner_elbow_protection_is_independent_per_arm(
    nonlinear_solver,
):
    nonlinear_solver.set_reference(BIAS_POSITIONS)
    previous = nonlinear_solver.last_output_qpos.copy()
    random = np.random.default_rng(19)
    candidate = None
    violating_side = None
    previous_distances = nonlinear_solver.elbow_outer_distances(previous)
    for _ in range(5000):
        trial = random.uniform(
            nonlinear_solver.model.lowerPositionLimit,
            nonlinear_solver.model.upperPositionLimit,
        )
        trial_distances = nonlinear_solver.elbow_outer_distances(trial)
        sides = [
            side
            for side in ("Left", "Right")
            if trial_distances[side] < min(0.0, previous_distances[side])
        ]
        if len(sides) == 1:
            candidate = trial
            violating_side = sides[0]
            break
    assert candidate is not None

    protected = nonlinear_solver._protect_outward_elbows(candidate, previous)
    other_side = "Right" if violating_side == "Left" else "Left"
    assert protected[nonlinear_solver.arm_side_indices[violating_side]] == pytest.approx(
        previous[nonlinear_solver.arm_side_indices[violating_side]]
    )
    assert protected[nonlinear_solver.arm_side_indices[other_side]] == pytest.approx(
        candidate[nonlinear_solver.arm_side_indices[other_side]]
    )


def test_nonlinear_rotation_residual_stays_accurate_near_half_turn():
    angle = math.pi - 1.0e-6
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )

    residual = AdamBimanualIkSolver._casadi_rotation_residual(
        casadi,
        casadi.DM(rotation),
    )

    assert np.linalg.norm(np.asarray(residual)) == pytest.approx(angle, abs=1.0e-6)


def test_nonlinear_filter_uses_newest_first_weights(nonlinear_solver):
    nonlinear_solver.nonlinear_history.clear()
    for value in (1.0, 2.0, 3.0):
        filtered = nonlinear_solver._filter_nonlinear_qpos(np.full(14, value))

    expected = (0.4 * 3.0 + 0.3 * 2.0 + 0.2 * 1.0) / 0.9
    assert filtered == pytest.approx(np.full(14, expected))
