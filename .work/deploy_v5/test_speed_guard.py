"""The ceiling must be INERT inside the corpus envelope; the scale must be exact.

A ceiling that quietly rescaled normal task motion would reintroduce the
underprediction this deployment is already fighting, so "does nothing at corpus
speed" is the property under test, not an afterthought.
"""

import numpy as np
import pytest

import repr_bridge as rb
import speed_guard as sg


def chunk_at(wrist_step_mm: float, hand_step_rad: float = 0.0) -> np.ndarray:
    """A chunk moving both wrists at a constant speed along +x."""
    chunk = np.zeros((rb.ACTION_HORIZON, rb.ACTION_DIM), dtype=np.float32)
    for k in range(rb.ACTION_HORIZON):
        for side in rb.SIDES:
            chunk[k, rb.SLICES[f"{side}_wrist_trans"].start] = (
                (k + 1) * wrist_step_mm / 1000.0
            )
            chunk[k, rb.SLICES[f"{side}_hand_joints"]] = k * hand_step_rad
    return chunk


def test_inert_at_fastest_corpus_phase_mean():
    """6.4 mm/step is the fastest phase mean in the corpus."""
    chunk = chunk_at(6.4)
    out, report = sg.govern(chunk)
    assert report["capped"] == []
    np.testing.assert_array_equal(out, chunk)


def test_inert_at_corpus_worst_single_step():
    """18.5 mm is the widest single step the corpus contains."""
    chunk = chunk_at(18.5)
    out, report = sg.govern(chunk)
    assert report["capped"] == []
    np.testing.assert_array_equal(out, chunk)


def test_ceiling_engages_past_the_cap():
    out, report = sg.govern(chunk_at(40.0))
    assert [c["channel"] for c in report["capped"]] == ["left_wrist", "right_wrist"]
    for side in rb.SIDES:
        assert sg.wrist_steps_mm(out, side).max() == pytest.approx(
            sg.MAX_WRIST_STEP_MM, rel=1e-5
        )


def test_ceiling_preserves_direction():
    """Rescaling, not clipping: the path shape must survive."""
    chunk = chunk_at(40.0)
    out, _ = sg.govern(chunk)
    before = np.asarray(chunk[:, rb.SLICES["left_wrist_trans"]], np.float64)[-1]
    after = np.asarray(out[:, rb.SLICES["left_wrist_trans"]], np.float64)[-1]
    np.testing.assert_allclose(
        before / np.linalg.norm(before), after / np.linalg.norm(after), atol=1e-6
    )


def test_hand_start_posture_is_preserved():
    chunk = chunk_at(1.0, hand_step_rad=1.0)
    out, report = sg.govern(chunk)
    assert any(c["channel"].endswith("hand") for c in report["capped"])
    for side in rb.SIDES:
        sl = rb.SLICES[f"{side}_hand_joints"]
        np.testing.assert_allclose(out[0, sl], chunk[0, sl], atol=1e-6)
        assert sg.hand_steps_rad(out, side).max() == pytest.approx(
            sg.MAX_HAND_STEP_RAD, rel=1e-5
        )


def test_scale_is_exact_and_applied_after_the_ceiling():
    chunk = chunk_at(10.0)
    out, report = sg.govern(chunk, scale=0.5)
    assert report["capped"] == []
    assert report["scale"] == 0.5
    for side in rb.SIDES:
        assert sg.wrist_steps_mm(out, side).max() == pytest.approx(5.0, rel=1e-5)


def test_ceiling_reported_even_when_scaled_under_it():
    """A slowed excursion must still be reported, or the log hides the runaway."""
    _, report = sg.govern(chunk_at(60.0), scale=0.25)
    assert [c["channel"] for c in report["capped"]] == ["left_wrist", "right_wrist"]
    assert report["max_wrist_step_mm"] == pytest.approx(60.0, rel=1e-3)


def test_scale_composes_with_the_ceiling():
    out, _ = sg.govern(chunk_at(40.0), scale=0.5)
    for side in rb.SIDES:
        assert sg.wrist_steps_mm(out, side).max() == pytest.approx(
            sg.MAX_WRIST_STEP_MM * 0.5, rel=1e-5
        )


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "1.5", "nan"])
def test_read_scale_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        sg.read_scale({sg.SCALE_ENV: bad})


@pytest.mark.parametrize("value,want", [("", 1.0), ("1.0", 1.0), ("0.3", 0.3)])
def test_read_scale_accepts_valid_values(value, want):
    assert sg.read_scale({sg.SCALE_ENV: value}) == pytest.approx(want)


def test_govern_rejects_wrong_shape_and_scale():
    with pytest.raises(ValueError):
        sg.govern(np.zeros((rb.ACTION_HORIZON, 12), dtype=np.float32))
    with pytest.raises(ValueError):
        sg.govern(chunk_at(1.0), scale=0.0)
