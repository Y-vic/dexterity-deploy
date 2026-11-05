import numpy as np
import pytest
from ur_common.hil import HumanControl
from ur_common.command_guards import hand_bounds

def test_hil_takeover_anchor_and_per_hand_offset_are_continuous():
    now = [1.0]
    hil = HumanControl(clock=lambda: now[0])
    anchor = np.linspace(0, .1, 56).astype(np.float32)
    hil.begin(anchor)
    np.testing.assert_array_equal(hil.target(), anchor)
    for side, offset in [('left',0), ('right',6)]:
        hil.arm(side, anchor[offset:offset+6], 'old-token')
        assert side+'_arm' not in hil.samples
        hil.arm(side, anchor[offset:offset+6], hil.token)
        hil.hand(side, np.full(22, 1.5))
    np.testing.assert_allclose(hil.target(), anchor, atol=2e-7)
    hil.hand('left', np.full(22, 1.55))
    np.testing.assert_allclose(hil.target()[12:34], anchor[12:34]+.05, atol=2e-7)
    np.testing.assert_array_equal(hil.target()[34:], anchor[34:])
    now[0] += .3
    with pytest.raises(RuntimeError, match='stale'):
        hil.target()
    hil.end()
    hil.begin(anchor + .1)
    assert not hil.hand_origin and not hil.samples

@pytest.mark.parametrize('delta', [-.01, -.5])
def test_hil_saturates_recorded_takeover_underflows_before_strict_validation(delta):
    from ur_common.command_guards import hand_bounds, validate_hand_range
    lo, hi = hand_bounds()
    anchor = np.r_[np.zeros(12), np.clip(np.zeros(44), lo, hi)].astype(np.float32)
    anchor[16] = lo[4]
    hil = HumanControl(clock=lambda: 1.)
    hil.begin(anchor)
    for side, start in (('left', 0), ('right', 6)):
        hil.arm(side, anchor[start:start+6], hil.token)
        hil.hand(side, np.full(22, .5, np.float32))
    glove = np.full(22, .5, np.float32)
    glove[4] += delta  # Observed left joint 5 is already at its firmware lower bound.
    glove[0] += .1
    hil.hand('left', glove)
    result = hil.target()
    validate_hand_range(result[12:])
    assert result[16] == lo[4]
    assert result[12] == pytest.approx(.1)
    np.testing.assert_array_equal(result[:12], anchor[:12])
    np.testing.assert_array_equal(result[34:], anchor[34:])
    glove[4] = .55
    hil.hand('left', glove)
    assert hil.target()[16] == pytest.approx(lo[4]+.05, abs=1e-6)

def test_hil_saturates_upper_bounds_and_rejects_nonfinite_inputs():
    from ur_common.command_guards import hand_bounds
    lo, hi = hand_bounds()
    hil = HumanControl(clock=lambda: 1.)
    hil.begin(np.r_[np.zeros(12), hi])
    for side in ('left', 'right'):
        hil.arm(side, np.zeros(6), hil.token)
        hil.hand(side, np.zeros(22))
        hil.hand(side, np.ones(22))
    np.testing.assert_array_equal(hil.target()[12:], hi)
    with pytest.raises(ValueError, match='finite'):
        hil.hand('left', np.full(22, np.nan))
