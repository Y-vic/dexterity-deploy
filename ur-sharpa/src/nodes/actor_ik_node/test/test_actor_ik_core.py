import numpy as np
import pytest

from actor_ik_node.actor_ik_node import (
    ActorIkCore,
    ActorIkError,
    limit_joint_target,
    next_step_due,
)


class FakeKinematics:
    def arm_ik_eef9(self, eef, near, side, ref_joint=None):
        assert side in {"left", "right"}
        return np.asarray(near) + (0.01 if side == "left" else 0.02)


def _eef(n):
    value = np.zeros((n, 9), dtype=np.float32)
    value[:, 3] = 1.0
    value[:, 7] = 1.0
    return value


def test_plan_preserves_step_count_and_advances_seed():
    plan = ActorIkCore(FakeKinematics()).make_plan(
        left_eef=_eef(2), right_eef=_eef(2), hand_joint=np.zeros((2, 44), np.float32),
        execute_length=2, left_seed=np.zeros(6), right_seed=np.zeros(6),
    )
    assert plan.left_joint.shape == (2, 6)
    assert plan.right_joint.shape == (2, 6)
    assert np.allclose(plan.left_joint[-1], 0.02)
    assert np.allclose(plan.right_joint[-1], 0.04)


def test_plan_rejects_wrong_flattened_shape():
    with pytest.raises(ActorIkError, match="left_eef"):
        ActorIkCore(FakeKinematics()).make_plan(
            left_eef=np.zeros(9, np.float32), right_eef=_eef(1),
            hand_joint=np.zeros((1, 44), np.float32), execute_length=2,
            left_seed=np.zeros(6), right_seed=np.zeros(6),
        )


def test_joint_target_is_limited_against_measured_state():
    current = np.asarray((0.0, 1.0, -1.0), dtype=np.float32)
    target = np.asarray((0.5, 0.8, -2.0), dtype=np.float32)
    limited = limit_joint_target(current, target, 0.25)
    assert np.allclose(limited, (0.25, 0.8, -1.25))
    assert np.max(np.abs(limited - current)) <= 0.25


def test_next_step_due_enforces_requested_period():
    assert next_step_due(10.0, 20.0) == pytest.approx(10.05)
    with pytest.raises(ActorIkError, match="frequency_hz"):
        next_step_due(10.0, 0.0)
