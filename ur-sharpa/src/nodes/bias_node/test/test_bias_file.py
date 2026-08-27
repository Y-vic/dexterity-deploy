import json
from types import SimpleNamespace

import numpy as np
import pytest

from bias_node.bias_node import BiasNode


def _node_with_path(path):
    node = object.__new__(BiasNode)
    node.bias_file = path
    return node


def test_loads_saved_hand_bias_as_44d_reset_target(tmp_path):
    path = tmp_path / "bias.json"
    left = np.linspace(-0.1, 0.4, 22)
    right = np.linspace(0.2, 0.7, 22)
    path.write_text(
        json.dumps({
            "left_hand_joint": left.tolist(),
            "right_hand_joint": right.tolist(),
        }),
        encoding="utf-8",
    )
    node = _node_with_path(path)

    target = node._load_hand_bias_target_locked()

    np.testing.assert_array_equal(target, np.concatenate((left, right)))


def test_missing_hand_bias_fails_instead_of_assuming_zero(tmp_path):
    path = tmp_path / "bias.json"
    path.write_text(json.dumps({"left_joint": [0.0] * 6}), encoding="utf-8")
    node = _node_with_path(path)

    with pytest.raises(RuntimeError, match="left_hand_joint and right_hand_joint"):
        node._load_hand_bias_target_locked()


def test_save_current_bias_includes_both_hands(tmp_path):
    path = tmp_path / "bias.json"
    node = _node_with_path(path)
    left_arm = np.linspace(-0.3, 0.2, 6)
    right_arm = np.linspace(0.4, 0.9, 6)
    hand = np.linspace(-0.5, 0.8, 44)
    node._measured_arm_joints = lambda: {
        "left": left_arm,
        "right": right_arm,
    }
    node._measured_hand_joints = lambda: hand
    node._ur_state = SimpleNamespace(
        left_eef=np.arange(9, dtype=float),
        right_eef=np.arange(9, dtype=float) + 10,
        eef_frame="canonical",
    )

    node._save_current_bias_locked()

    payload = json.loads(path.read_text(encoding="utf-8"))
    np.testing.assert_array_equal(payload["left_hand_joint"], hand[:22])
    np.testing.assert_array_equal(payload["right_hand_joint"], hand[22:])
