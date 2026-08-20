import math

import numpy as np
import pytest
from pink.tasks import FrameTask

from quest_node.adam_bimanual_ik import (
    EE_FRAMES,
    SHOULDER_PRIOR_FRAMES,
    AdamBimanualIkSolver,
    Pose3,
)
from quest_node.command_gate import ADAM_COMMAND_JOINTS_19


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


def calibrated_shoulder_prior_solver():
    solver = AdamBimanualIkSolver(MODEL_PATH, retarget_method="shoulder_prior")
    reference = solver.set_reference(BIAS_POSITIONS)
    solver.set_targets(reference)
    return solver, reference


def test_shoulder_prior_tasks_use_configured_weights_and_frames():
    solver, _ = calibrated_shoulder_prior_solver()

    tasks = solver._pink_tasks()

    assert len(tasks) == 4
    assert all(isinstance(task, FrameTask) for task in tasks)
    assert [task.frame for task in tasks] == [
        EE_FRAMES["Left"],
        EE_FRAMES["Right"],
        SHOULDER_PRIOR_FRAMES["Left"],
        SHOULDER_PRIOR_FRAMES["Right"],
    ]
    assert tasks[0].cost == pytest.approx([20.0, 20.0, 20.0, 18.0, 18.0, 18.0])
    assert tasks[1].cost == pytest.approx([20.0, 20.0, 20.0, 18.0, 18.0, 18.0])
    assert tasks[2].cost == pytest.approx([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert tasks[3].cost == pytest.approx([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert [task.gain for task in tasks] == pytest.approx([1.0] * 4)
    assert [task.lm_damping for task in tasks] == pytest.approx([1.0] * 4)


def test_shoulder_targets_are_fixed_to_a_bias_orientation():
    solver, reference = calibrated_shoulder_prior_solver()
    initial_tasks = solver._pink_tasks()
    initial_shoulder_rotations = [
        task.transform_target_to_world.rotation.copy()
        for task in initial_tasks[2:]
    ]
    moved_targets = {
        name: Pose3(value.position.copy(), value.rotation.copy())
        for name, value in reference.items()
    }
    moved_targets["Left"] = Pose3(
        moved_targets["Left"].position + np.array([0.05, 0.03, 0.02]),
        moved_targets["Left"].rotation,
    )
    moved_targets["Right"] = Pose3(
        moved_targets["Right"].position + np.array([0.04, -0.02, -0.01]),
        moved_targets["Right"].rotation,
    )

    solver.set_targets(moved_targets)
    moved_tasks = solver._pink_tasks()

    for side, task, initial_rotation in zip(
        ("Left", "Right"), moved_tasks[2:], initial_shoulder_rotations, strict=True
    ):
        assert task.transform_target_to_world.rotation == pytest.approx(
            initial_rotation
        )
        assert task.transform_target_to_world.rotation == pytest.approx(
            solver.shoulder_prior_orientations[side]
        )


def test_shoulder_prior_zero_tracker_delta_keeps_exact_bias_pose():
    solver, reference = calibrated_shoulder_prior_solver()

    solver.solve(iterations=5, solve_dt=0.05)

    assert solver.positions_19() == pytest.approx(
        [BIAS_POSITIONS[name] for name in ADAM_COMMAND_JOINTS_19],
        abs=1.0e-7,
    )
    assert max(solver.wrist_errors(reference).values()) < 0.01
