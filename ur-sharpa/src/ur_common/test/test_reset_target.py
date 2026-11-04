import json

import numpy as np
import pytest

from ur_common.reset_target import episode_initial_target, validate_target


def sample(tmp_path, *, arm_valid=True, hand_valid=True):
    root = tmp_path / "dataset"
    episode = root / "success" / "one"
    (episode / "ur").mkdir(parents=True)
    (episode / "sharpa").mkdir()
    (episode / "COMPLETE").touch()
    (episode / "manifest.json").write_text(json.dumps({
        "schema": "ur_sharpa_sample.v1", "state": "complete", "quality": {"ok": True},
    }))
    arm = np.arange(24, dtype=np.float32).reshape(2, 12) * 0.1
    hand = np.arange(88, dtype=np.float32).reshape(2, 44) * 0.01
    np.savez(episode / "ur" / "physical_12.npz", q=arm, valid=[arm_valid, True])
    np.savez(episode / "sharpa" / "joints_44.npz", q=hand, q_valid=[hand_valid, True])
    return root, arm, hand


def test_episode_reset_uses_first_measured_row_for_both_arms_and_hands(tmp_path):
    root, arm, hand = sample(tmp_path)
    target = episode_initial_target(str(root), "success/one")
    assert target["source"] == "episode"
    np.testing.assert_array_equal(target["left_joint"], arm[0, :6])
    np.testing.assert_array_equal(target["right_joint"], arm[0, 6:])
    np.testing.assert_array_equal(target["left_hand_joint"], hand[0, :22])
    np.testing.assert_array_equal(target["right_hand_joint"], hand[0, 22:])


def test_episode_reset_rejects_invalid_first_row_and_path_escape(tmp_path):
    root, _, _ = sample(tmp_path, hand_valid=False)
    with pytest.raises(ValueError, match="initial row"):
        episode_initial_target(str(root), "success/one")
    with pytest.raises(ValueError, match="invalid episode path"):
        episode_initial_target(str(root), "../elsewhere")


def test_motion_boundary_rejects_out_of_range_episode_target(tmp_path):
    root, _, _ = sample(tmp_path)
    target = episode_initial_target(str(root), "success/one")
    target["left_joint"][0] = 7.0
    with pytest.raises(ValueError, match="left_joint"):
        validate_target(target)
