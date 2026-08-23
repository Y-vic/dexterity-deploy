"""SharpA hand fingertip FK: shape contract and model-cache reuse.

Needs `pinocchio`. When it is absent the module skips with a clear reason
instead of stubbing FK, since a faked fingertip position proves nothing.

The joint-order test is pure string comparison and runs everywhere -- it is the
one that catches an interface/URDF ordering mismatch, which would silently
scramble every fingertip.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur_common import sharpa_fk
from ur_common.kinematics import hand_fk_available

NUM_TIPS = 5


def _load_interface_contract():
    """Load interface/sharpa_interface/sharpa.py directly, by file path.

    `import sharpa_interface.sharpa` would run the package __init__, which pulls
    in the server subpackage and therefore msgpack. sharpa.py is deliberately
    dependency-free frozen constants, so loading the file alone lets the order
    check run without the server's transport dependencies installed.
    """
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(
        here, "..", "..", "..", "..", "interface", "sharpa_interface", "sharpa.py"))
    if not os.path.isfile(path):
        return None
    import sys

    name = "_frozen_sharpa_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # sharpa.py uses `from __future__ import annotations` plus a dataclass, and
    # dataclasses resolves those string annotations through sys.modules. Register
    # before exec or the decorator raises AttributeError.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


# ---------------------------------------------------------------------------
# Order contract (no pinocchio needed)
# ---------------------------------------------------------------------------
def test_joint_order_matches_the_frozen_interface_order() -> None:
    """URDF order must equal the frozen interface order, prefix aside.

    `sharpa_fk` relies on this to map the 44D interface vector straight through.
    If the interface joint order ever changes, this test fails first.
    """
    contract = _load_interface_contract()
    if contract is None:
        pytest.skip("interface/sharpa_interface/sharpa.py not found next to this package")

    assert sharpa_fk.NUM_JOINTS_PER_HAND == contract.NUM_JOINTS_PER_HAND == 22
    assert sharpa_fk.NUM_JOINTS_TOTAL == contract.NUM_JOINTS_TOTAL == 44
    assert sharpa_fk.LEFT_JOINT_ORDER == [
        f"left_{n}" for n in contract.JOINT_NAMES_PER_HAND
    ]
    assert sharpa_fk.RIGHT_JOINT_ORDER == [
        f"right_{n}" for n in contract.JOINT_NAMES_PER_HAND
    ]


def test_joint_and_tip_counts() -> None:
    assert len(sharpa_fk.LEFT_JOINT_ORDER) == 22
    assert len(sharpa_fk.RIGHT_JOINT_ORDER) == 22
    assert len(sharpa_fk.LEFT_TIP_LINKS) == NUM_TIPS
    assert len(sharpa_fk.RIGHT_TIP_LINKS) == NUM_TIPS
    # thumb -> pinky, matching the interface finger order
    assert sharpa_fk.LEFT_TIP_LINKS[0] == "left_thumb_fingertip"
    assert sharpa_fk.LEFT_TIP_LINKS[-1] == "left_pinky_fingertip"
    assert len(sharpa_fk.HAND_KEYPOINT_LINKS["left"]) == 21
    assert sharpa_fk.HAND_KEYPOINT_LINKS["left"][0] == "left_hand_C_MC"
    assert sharpa_fk.HAND_KEYPOINT_LINKS["right"][-1] == "right_pinky_fingertip"


def test_urdf_assets_are_bundled() -> None:
    """The two URDFs must ship with the package (meshes deliberately do not)."""
    import os

    assert os.path.isfile(sharpa_fk.LEFT_URDF_PATH), sharpa_fk.LEFT_URDF_PATH
    assert os.path.isfile(sharpa_fk.RIGHT_URDF_PATH), sharpa_fk.RIGHT_URDF_PATH


def test_split_interface_hand_joints() -> None:
    q = np.arange(44, dtype=np.float64)
    left, right = sharpa_fk.split_interface_hand_joints(q)
    assert left.shape == (22,)
    assert right.shape == (22,)
    assert np.allclose(left, np.arange(22))       # left first
    assert np.allclose(right, np.arange(22, 44))  # then right
    with pytest.raises(ValueError):
        sharpa_fk.split_interface_hand_joints(np.zeros(43))


# ---------------------------------------------------------------------------
# FK (needs pinocchio)
# ---------------------------------------------------------------------------
pinocchio_required = pytest.mark.skipif(
    not hand_fk_available(),
    reason="pinocchio not installed; hand fingertip FK cannot run",
)


@pytest.fixture()
def sample_joints() -> tuple[np.ndarray, np.ndarray]:
    left_deg = np.array([20, 5, 30, -8, 35, 25, 5, 40, 20, 30, 0, 35, 15,
                         28, -6, 32, 12, 12, 22, 6, 26, 14], dtype=float)
    right_deg = np.array([18, -4, 28, 7, 30, 24, -3, 36, 18, 32, 2, 34, 16,
                          26, 5, 30, 10, 10, 20, -4, 24, 12], dtype=float)
    return np.deg2rad(left_deg), np.deg2rad(right_deg)


@pinocchio_required
def test_fingertip_poses_shape(sample_joints) -> None:
    left, right = sample_joints
    poses = sharpa_fk.hand_joints_to_fingertip_poses(left, right)
    assert len(poses) == 2 * NUM_TIPS  # 5 left + 5 right, v1 signature
    for p in poses:
        assert np.asarray(p).shape == (3,)
        assert np.isfinite(p).all()


@pinocchio_required
def test_fingertip_transforms_shape(sample_joints) -> None:
    left, right = sample_joints
    out = sharpa_fk.hand_joints_to_fingertip_transforms(left, right)
    assert set(out) == {"left", "right"}
    for side in ("left", "right"):
        assert out[side].shape == (NUM_TIPS, 4, 4)
        for t in out[side]:
            assert np.allclose(t[3], (0, 0, 0, 1))
            assert np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1.0e-9)


@pinocchio_required
def test_hand_keypoints_match_dashboard_skeleton(sample_joints) -> None:
    left, right = sample_joints
    keypoints = sharpa_fk.hand_joints_to_keypoints(left, right)

    assert set(keypoints) == {"left", "right"}
    assert keypoints["left"].shape == (21, 3)
    assert keypoints["right"].shape == (21, 3)
    assert np.allclose(keypoints["left"][0], np.zeros(3))
    assert np.allclose(keypoints["right"][0], np.zeros(3))
    assert np.isfinite(keypoints["left"]).all()
    assert np.isfinite(keypoints["right"]).all()


@pinocchio_required
def test_facade_returns_per_hand_arrays(sample_joints) -> None:
    from ur_common.kinematics import SafetyConfig, UrCommonKinematics

    left, right = sample_joints
    kin = UrCommonKinematics(SafetyConfig())
    out = kin.hand_fingertip_positions(np.concatenate((left, right)))
    assert set(out) == {"left", "right"}
    assert out["left"].shape == (NUM_TIPS, 3)
    assert out["right"].shape == (NUM_TIPS, 3)
    # Must agree with the underlying port.
    flat = sharpa_fk.hand_joints_to_fingertip_poses(left, right)
    assert np.allclose(out["left"], np.asarray(flat[:NUM_TIPS]))
    assert np.allclose(out["right"], np.asarray(flat[NUM_TIPS:]))


@pinocchio_required
def test_model_cache_is_reused_across_calls(sample_joints) -> None:
    """The Pinocchio models must be built once, not per frame.

    Rebuilding from URDF every frame was the original performance problem.
    """
    left, right = sample_joints
    sharpa_fk.clear_hand_model_cache()
    assert sharpa_fk.hand_model_cache_info()["left_builds"] == 0

    sharpa_fk.hand_joints_to_fingertip_poses(left, right)
    first = sharpa_fk.hand_model_cache_info()
    assert first["left_builds"] == 1
    assert first["right_builds"] == 1

    for _ in range(20):
        sharpa_fk.hand_joints_to_fingertip_poses(left, right)
    after = sharpa_fk.hand_model_cache_info()
    assert after["left_builds"] == 1, "left hand model was rebuilt; cache broken"
    assert after["right_builds"] == 1, "right hand model was rebuilt; cache broken"


@pinocchio_required
def test_fk_is_deterministic(sample_joints) -> None:
    left, right = sample_joints
    a = np.asarray(sharpa_fk.hand_joints_to_fingertip_poses(left, right))
    b = np.asarray(sharpa_fk.hand_joints_to_fingertip_poses(left, right))
    assert np.allclose(a, b, atol=0.0)


@pinocchio_required
def test_different_joints_move_the_fingertips(sample_joints) -> None:
    left, right = sample_joints
    base = np.asarray(sharpa_fk.hand_joints_to_fingertip_poses(left, right))
    bent = left.copy()
    bent[5] += 0.5  # index_MCP_FE
    moved = np.asarray(sharpa_fk.hand_joints_to_fingertip_poses(bent, right))
    assert not np.allclose(base[1], moved[1])  # left index tip moved
    assert np.allclose(base[5:], moved[5:])    # right hand untouched


@pinocchio_required
def test_wrong_joint_count_raises(sample_joints) -> None:
    left, right = sample_joints
    with pytest.raises(ValueError):
        sharpa_fk.hand_joints_to_fingertip_poses(left[:10], right)
