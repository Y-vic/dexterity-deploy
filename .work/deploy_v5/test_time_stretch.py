import numpy as np
import pytest

import repr_bridge as rb
from time_stretch import read_time_scale, stretch_chunk, stretched_length


def hands():
    return {
        "left": np.linspace(-0.2, 0.3, rb.HAND_DIM, dtype=np.float32),
        "right": np.linspace(0.4, -0.1, rb.HAND_DIM, dtype=np.float32),
    }


def chunk():
    out = np.zeros((rb.ACTION_HORIZON, rb.ACTION_DIM), dtype=np.float32)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        t = np.arange(1, rb.ACTION_HORIZON + 1, dtype=np.float32)[:, None]
        out[:, rb.SLICES[f"{side}_wrist_trans"]] = sign * t * 0.004
        out[:, rb.SLICES[f"{side}_wrist_rotvec"]] = sign * t * 0.01
        start = hands()[side]
        out[:, rb.SLICES[f"{side}_hand_joints"]] = start + sign * t * 0.02
    return out


def test_point_three_expands_ten_samples_to_thirty_four():
    assert stretched_length(10, 0.3) == 34


def test_time_stretch_preserves_full_wrist_and_hand_endpoint():
    original = chunk()
    slowed = stretch_chunk(original, hands(), 0.3)
    assert slowed.shape == (34, rb.ACTION_DIM)
    np.testing.assert_array_equal(slowed[-1], original[-1])


def test_time_stretch_reduces_each_increment_without_spatial_shrink():
    original = chunk()
    slowed = stretch_chunk(original, hands(), 0.3)
    anchor = np.zeros(rb.ACTION_DIM, dtype=np.float32)
    for side in rb.SIDES:
        anchor[rb.SLICES[f"{side}_hand_joints"]] = hands()[side]
    original_steps = np.diff(np.vstack([anchor, original]), axis=0)
    slowed_steps = np.diff(np.vstack([anchor, slowed]), axis=0)
    limit = 0.3 * np.max(np.abs(original_steps))
    assert np.max(np.abs(slowed_steps)) <= limit + 1e-6


def test_native_scale_is_an_exact_copy():
    original = chunk()
    native = stretch_chunk(original, hands(), 1.0)
    np.testing.assert_array_equal(native, original)
    assert native is not original


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "bad"])
def test_invalid_time_scale_is_rejected(value):
    with pytest.raises(ValueError):
        read_time_scale({"SHARPA_DEPLOY_TIME_SCALE": value})
