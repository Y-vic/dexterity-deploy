import numpy as np
import pytest

from ur_common.command_guards import (
    ExecutionContractError, execution_profile, hand_bounds, project_hand_plan,
)


def test_policy_projection_preserves_exact_firmware_boundaries_and_input():
    lower, upper = hand_bounds()
    raw = np.stack((lower, upper))
    original = raw.copy()
    projected = project_hand_plan(raw)
    np.testing.assert_array_equal(projected, original)
    np.testing.assert_array_equal(raw, original)
    assert projected.dtype == np.float32
    assert not np.shares_memory(projected, raw)


def test_policy_projection_bounds_small_corrections_in_asymmetric_wire_order():
    lower, upper = hand_bounds()
    raw = np.stack((lower, upper))
    raw[0, 27] -= .031
    raw[1, 35] += .032
    original = raw.copy()
    projected = project_hand_plan(raw)
    np.testing.assert_array_equal(projected, np.stack((lower, upper)))
    np.testing.assert_array_equal(raw, original)


def test_policy_projection_rejects_excessive_later_row_with_joint_context():
    lower, upper = hand_bounds()
    ceiling = execution_profile()['max_projection_rad'][35]
    raw = np.tile((lower + upper) / 2, (3, 1))
    raw[2, 35] = upper[35] + ceiling + .01
    with pytest.raises(ExecutionContractError, match='hand_projection_too_large') as caught:
        project_hand_plan(raw)
    row = caught.value.violations[0]
    assert row['chunk_step'] == 2
    assert row['group'] == 'right_hand' and row['joint_number'] == 14
    assert row['limit_rad'] == ceiling
    assert row['upper_rad'] == upper[35]
    assert row['delta_rad'] > ceiling


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_policy_projection_rejects_nonfinite_target(invalid):
    raw = np.zeros((2, 44), dtype=np.float32)
    raw[1, 43] = invalid
    with pytest.raises(ValueError, match='finite'):
        project_hand_plan(raw)


@pytest.mark.parametrize('shape', [(0, 44), (44,), (2, 43), (1, 2, 44)])
def test_policy_projection_rejects_invalid_plan_shape(shape):
    with pytest.raises(ValueError, match='plan'):
        project_hand_plan(np.zeros(shape, dtype=np.float32))
