import numpy as np

from bias_node.trajectory import (
    SHARPA_HOME_STEPS,
    SHARPA_HOME_TOLERANCE_RAD,
    check_hand_arrival,
)


def test_sharpa_home_uses_deployed_v3_ramp_length():
    assert SHARPA_HOME_STEPS == 180


def test_sharpa_arrival_is_relative_to_saved_bias_not_numeric_zero():
    target = np.linspace(-0.2, 0.6, 44)
    measured = target.copy()
    measured[22] += 0.07

    result = check_hand_arrival(measured, target)

    assert SHARPA_HOME_TOLERANCE_RAD == 0.08
    assert result.reached
    assert result.per_side_error["left"] == 0.0
    assert np.isclose(result.per_side_error["right"], 0.07)


def test_materially_closed_hand_still_fails_home_check():
    target = np.linspace(-0.2, 0.6, 44)
    measured = target.copy()
    measured[22] += SHARPA_HOME_TOLERANCE_RAD + 0.01

    result = check_hand_arrival(measured, target)

    assert not result.reached
    assert "right SharpA" in result.reason
