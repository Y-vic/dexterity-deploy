"""Frame transform round-trips: UR base <-> capture root <-> canonical frame.

Every conversion in UrSharpAWireGeometry must be exactly invertible. If one of
these fails, the observation path and the execution path disagree about where
the hand is, which is the calibration drift this package exists to prevent.

Pure NumPy, no hardware, no compiled bindings.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur_common import geometry as g
from ur_common.geometry import UrSharpAWireGeometry

from test_geometry import random_transform  # noqa: E402

POSE9_TOL = 1.0e-5   # any leg crossing float32 pose9
F64_TOL = 1.0e-9     # pure float64 legs

SIDES = ("left", "right")


@pytest.fixture(scope="module")
def geom() -> UrSharpAWireGeometry:
    return UrSharpAWireGeometry()


@pytest.mark.parametrize("side", SIDES)
def test_ur_base_capture_root_roundtrip(geom: UrSharpAWireGeometry, side: str) -> None:
    rng = np.random.default_rng(100)
    for _ in range(200):
        ur_base = random_transform(rng)
        capture = geom.ur_base_tcp_to_capture_tcp(ur_base, side)
        back = geom.capture_tcp_to_ur_base_tcp(capture, side)
        assert np.allclose(back, ur_base, atol=F64_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_capture_root_model_roundtrip(geom: UrSharpAWireGeometry, side: str) -> None:
    rng = np.random.default_rng(101)
    for _ in range(200):
        capture = random_transform(rng)
        model = geom.capture_tcp_to_model_pose(capture, side)
        assert model.shape == (9,)
        back = geom.model_pose_to_capture_tcp(model, side)
        assert np.allclose(back, capture, atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_capture_root_canonical_roundtrip(
    geom: UrSharpAWireGeometry, side: str
) -> None:
    rng = np.random.default_rng(106)
    for _ in range(200):
        capture = random_transform(rng)
        canonical = geom.capture_tcp_to_canonical_pose(capture, side)
        assert canonical.shape == (9,)
        back = geom.canonical_pose_to_capture_tcp(canonical, side)
        assert np.allclose(back, capture, atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_canonical_pose_uses_pnd_column_rot6d(
    geom: UrSharpAWireGeometry, side: str
) -> None:
    capture = np.eye(4, dtype=np.float64)
    canonical = geom.capture_tcp_to_canonical_pose(capture, side)
    canonical_transform = g.column_pose9_to_transform(canonical)
    expected = np.eye(4, dtype=np.float64)
    capture_from_wrist = capture @ g.UR_TCP_FROM_SHARPA_HAND[side]
    expected[:3, :3] = g.UR_ROOT_TO_PRETRAIN @ capture_from_wrist[:3, :3]
    expected[:3, 3] = g.UR_ROOT_TO_PRETRAIN @ capture_from_wrist[:3, 3]
    assert np.allclose(canonical_transform, expected, atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_full_chain_ur_base_to_model_and_back(geom: UrSharpAWireGeometry, side: str) -> None:
    """UR base -> capture root -> model -> capture root -> UR base is identity."""
    rng = np.random.default_rng(102)
    for _ in range(200):
        ur_base = random_transform(rng)
        model = geom.capture_tcp_to_model_pose(
            geom.ur_base_tcp_to_capture_tcp(ur_base, side), side
        )
        back = geom.capture_tcp_to_ur_base_tcp(
            geom.model_pose_to_capture_tcp(model, side), side
        )
        assert np.allclose(back, ur_base, atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_composed_transform_is_identity(geom: UrSharpAWireGeometry, side: str) -> None:
    """The composed forward*inverse matrix itself must be the identity."""
    root = geom.capture_root_from_ur_base[side]
    wrist = geom.capture_tcp_to_ur_tcp[side]
    assert np.allclose(root @ np.linalg.inv(root), np.eye(4), atol=F64_TOL)
    assert np.allclose(wrist @ np.linalg.inv(wrist), np.eye(4), atol=F64_TOL)
    hand = g.UR_TCP_FROM_SHARPA_HAND[side]
    assert np.allclose(hand @ np.linalg.inv(hand), np.eye(4), atol=F64_TOL)
    assert np.allclose(g.UR_ROOT_TO_PRETRAIN @ g.UR_ROOT_TO_PRETRAIN.T,
                       np.eye(3), atol=F64_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_wire_pose_model_roundtrip(geom: UrSharpAWireGeometry, side: str) -> None:
    rng = np.random.default_rng(103)
    for _ in range(200):
        model = g.transform_to_row_pose9(random_transform(rng))
        wire = geom.model_pose_to_wire_pose(model, side)
        assert wire.shape == (9,)
        back = geom.wire_pose_to_model_pose(wire, side)
        assert np.allclose(back, model, atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_canonical_legacy_server_boundary_roundtrip(
    geom: UrSharpAWireGeometry, side: str
) -> None:
    rng = np.random.default_rng(107)
    for _ in range(200):
        canonical = g.transform_to_column_pose9(random_transform(rng))
        legacy = geom.canonical_pose_to_legacy_server_pose(canonical, side)
        assert not np.allclose(legacy, canonical)
        back = geom.legacy_server_pose_to_canonical_pose(legacy, side)
        assert np.allclose(back, canonical, atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_rtde_pose_wire_pose_roundtrip(geom: UrSharpAWireGeometry, side: str) -> None:
    """The full deploy path: RTDE TCP pose -> wire pose -> RTDE TCP pose."""
    rng = np.random.default_rng(104)
    for _ in range(200):
        rtde = g.transform_to_rtde_pose(random_transform(rng))
        wire = geom.rtde_pose_to_wire_pose(rtde, side)
        back = geom.wire_pose_to_rtde_pose(wire, side)
        assert np.allclose(back[:3], rtde[:3], atol=POSE9_TOL)
        # Compare rotations, not rotvecs: +-pi is the same rotation.
        assert np.allclose(g.rotvec_to_matrix(back[3:]),
                           g.rotvec_to_matrix(rtde[3:]), atol=POSE9_TOL)


@pytest.mark.parametrize("side", SIDES)
def test_capture_tcp_wire_pose_roundtrip(geom: UrSharpAWireGeometry, side: str) -> None:
    rng = np.random.default_rng(105)
    for _ in range(200):
        capture = random_transform(rng)
        back = geom.wire_pose_to_capture_tcp(
            geom.capture_tcp_to_wire_pose(capture, side), side
        )
        assert np.allclose(back, capture, atol=POSE9_TOL)


def test_left_and_right_bases_differ(geom: UrSharpAWireGeometry) -> None:
    """Left and right UR bases sit at different places under the capture root.

    Guards against a copy-paste that would make both arms share one calibration.
    """
    left = geom.capture_root_from_ur_base["left"]
    right = geom.capture_root_from_ur_base["right"]
    assert not np.allclose(left, right)
    # The two bases are roughly mirrored across the capture-root x axis.
    assert left[0, 3] > 0.0 > right[0, 3]
    assert np.allclose(left[1, 3], right[1, 3], atol=1.0e-3)


def test_hand_mount_offset_is_the_documented_29_5mm() -> None:
    """29.5 mm is UR-specific calibration, not shared interface geometry."""
    for side in SIDES:
        assert np.isclose(g.UR_TCP_FROM_SHARPA_HAND[side][0, 3], 0.0295)
        g.validate_transform(g.UR_TCP_FROM_SHARPA_HAND[side], f"{side} hand mount")


def test_side_validation() -> None:
    geom = UrSharpAWireGeometry()
    with pytest.raises(ValueError):
        geom.ur_base_tcp_to_capture_tcp(np.eye(4), "middle")
