"""Pure-math tests: rotation conversions and frame round-trips.

No hardware and no compiled bindings, so these run on any interpreter.

Tolerance note: the pose9 helpers cast to float32 because that is the interface
wire representation (docs/interface_contract.md), so any round-trip crossing
pose9 is bounded by float32 epsilon (~1e-7 relative), not float64. Round-trips
that stay in float64 are asserted much tighter.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur_common import geometry as g

POSE9_TOL = 1.0e-5   # crosses float32 pose9
F64_TOL = 1.0e-12    # stays in float64

SIDES = ("left", "right")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation via a normalised Gaussian quaternion."""
    q = rng.standard_normal(4)
    return g.quat_to_matrix(q / np.linalg.norm(q))


def random_transform(rng: np.random.Generator) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = random_rotation(rng)
    t[:3, 3] = rng.uniform(-0.8, 0.8, 3)
    return t


# ---------------------------------------------------------------------------
# rot6d <-> matrix <-> quaternion
# ---------------------------------------------------------------------------
def test_row_rot6d_matrix_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = random_rotation(rng)
        back = g.row_rot6d_to_matrix(g.matrix_to_row_rot6d(r))
        assert np.allclose(back, r, atol=POSE9_TOL)


def test_column_rot6d_matrix_roundtrip() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        r = random_rotation(rng)
        back = g.column_rot6d_to_matrix(g.matrix_to_column_rot6d(r))
        assert np.allclose(back, r, atol=POSE9_TOL)


def test_matrix_quat_roundtrip() -> None:
    rng = np.random.default_rng(2)
    for _ in range(500):
        r = random_rotation(rng)
        back = g.quat_to_matrix(g.matrix_to_quat(r))
        assert np.allclose(back, r, atol=F64_TOL)


def test_quat_matrix_quat_roundtrip_is_canonical() -> None:
    """quat -> matrix -> quat returns the same rotation, w >= 0 canonicalised."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        q = rng.standard_normal(4)
        q = q / np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        back = g.matrix_to_quat(g.quat_to_matrix(q))
        assert np.allclose(back, q, atol=1.0e-9)


def test_rot6d_quat_roundtrip() -> None:
    """The full rot6d <-> matrix <-> quaternion triangle."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        r = random_rotation(rng)
        rot6d = g.matrix_to_row_rot6d(r)
        quat = g.row_rot6d_to_quat(rot6d)
        back6d = g.quat_to_row_rot6d(quat)
        assert np.allclose(g.row_rot6d_to_matrix(back6d), r, atol=POSE9_TOL)


def test_matrix_quat_roundtrip_at_180_degrees() -> None:
    """Trace <= 0 branch: 180 degree rotations about each axis."""
    for axis in np.eye(3):
        r = g.rotvec_to_matrix(axis * np.pi)
        back = g.quat_to_matrix(g.matrix_to_quat(r))
        assert np.allclose(back, r, atol=1.0e-9)


def test_rotvec_matrix_roundtrip() -> None:
    rng = np.random.default_rng(5)
    for _ in range(300):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(1.0e-6, np.pi - 1.0e-6)
        rv = axis * angle
        back = g.matrix_to_rotvec(g.rotvec_to_matrix(rv))
        assert np.allclose(back, rv, atol=1.0e-9)


def test_rotvec_is_accurate_near_pi() -> None:
    """Regression: the arccos/sin form loses ~1e-3 rad as the angle nears pi.

    matrix_to_rotvec goes through a quaternion for exactly this reason. The
    original wire_geometry.py only special-cased pi - angle < 1e-6, but the
    standard branch divides by sin(angle) and degrades much earlier. This is on
    the live deploy path via transform_to_rtde_pose.
    """
    rng = np.random.default_rng(20)
    worst = 0.0
    for _ in range(2000):
        # Sample angles hugging pi, from ~0.3 rad away down to ~1e-6 away.
        angle = np.pi - 10.0 ** (-6.0 * rng.random() - 0.5)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        r = g.rotvec_to_matrix(axis * angle)
        back = g.rotvec_to_matrix(g.matrix_to_rotvec(r))
        worst = max(worst, float(np.linalg.norm(g.matrix_to_rotvec(r.T @ back))))
    assert worst < 1.0e-9, f"near-pi rotvec error {worst:.3e} rad is too large"


def test_rotvec_roundtrip_on_ill_conditioned_rotations() -> None:
    """Angles spanning the full range, including both degenerate ends."""
    for angle in (0.0, 1.0e-12, 1.0e-7, 1.0e-3, 1.0, 3.0,
                  np.pi - 1.0e-3, np.pi - 1.0e-9, np.pi):
        axis = np.asarray((0.3, -0.5, 0.81))
        axis /= np.linalg.norm(axis)
        r = g.rotvec_to_matrix(axis * angle)
        back = g.rotvec_to_matrix(g.matrix_to_rotvec(r))
        err = float(np.linalg.norm(g.matrix_to_rotvec(r.T @ back)))
        assert err < 1.0e-9, f"angle={angle}: error {err:.3e}"


def test_rotvec_identity_and_pi() -> None:
    assert np.allclose(g.rotvec_to_matrix(np.zeros(3)), np.eye(3))
    for axis in np.eye(3):
        r = g.rotvec_to_matrix(axis * np.pi)
        back = g.matrix_to_rotvec(r)
        # +-pi about an axis are the same rotation, so compare rotations.
        assert np.allclose(g.rotvec_to_matrix(back), r, atol=1.0e-7)


def test_euler_matrix_roundtrip() -> None:
    rng = np.random.default_rng(6)
    for _ in range(300):
        roll = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        yaw = rng.uniform(-np.pi, np.pi)
        euler = np.asarray((roll, pitch, yaw))
        back = g.matrix_to_euler_xyz(g.euler_xyz_to_matrix(euler))
        assert np.allclose(back, euler, atol=1.0e-9)


def test_pose_mat_pose_vector_roundtrip() -> None:
    """coord_transform.py ports: pose_mat <-> pose_vector."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        xyz = rng.uniform(-1.0, 1.0, 3)
        euler = np.asarray((rng.uniform(-np.pi, np.pi),
                            rng.uniform(-1.4, 1.4),
                            rng.uniform(-np.pi, np.pi)))
        vec = np.concatenate((xyz, euler))
        assert np.allclose(g.pose_vector(g.pose_mat(vec)), vec, atol=1.0e-9)


def test_pose_vector_quat_matches_matrix() -> None:
    rng = np.random.default_rng(8)
    for _ in range(200):
        t = random_transform(rng)
        pv = g.pose_vector_quat(t)
        assert pv.shape == (7,)
        assert np.allclose(pv[:3], t[:3, 3], atol=F64_TOL)
        assert np.allclose(g.quat_to_matrix(pv[3:]), t[:3, :3], atol=1.0e-9)


def test_rtde_pose_transform_roundtrip() -> None:
    rng = np.random.default_rng(9)
    for _ in range(300):
        t = random_transform(rng)
        back = g.rtde_pose_to_transform(g.transform_to_rtde_pose(t))
        assert np.allclose(back, t, atol=1.0e-9)


def test_row_pose9_transform_roundtrip() -> None:
    rng = np.random.default_rng(10)
    for _ in range(300):
        t = random_transform(rng)
        back = g.row_pose9_to_transform(g.transform_to_row_pose9(t))
        assert np.allclose(back, t, atol=POSE9_TOL)


def test_column_pose9_transform_roundtrip() -> None:
    rng = np.random.default_rng(11)
    for _ in range(300):
        t = random_transform(rng)
        back = g.column_pose9_to_transform(g.transform_to_column_pose9(t))
        assert np.allclose(back, t, atol=POSE9_TOL)


def test_row_and_column_rot6d_are_transposes() -> None:
    """Row and column conventions must not be silently interchangeable."""
    rng = np.random.default_rng(12)
    r = random_rotation(rng)
    row = g.row_rot6d_to_matrix(g.matrix_to_row_rot6d(r))
    col = g.column_rot6d_to_matrix(g.matrix_to_column_rot6d(r.T))
    assert np.allclose(row, col.T, atol=POSE9_TOL)


def test_validate_transform_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        g.validate_transform(np.zeros((4, 4)))
    bad_row = np.eye(4)
    bad_row[3] = (1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        g.validate_transform(bad_row)
    reflect = np.eye(4)
    reflect[0, 0] = -1.0  # determinant -1
    with pytest.raises(ValueError):
        g.validate_transform(reflect)
    with pytest.raises(ValueError):
        g.validate_transform(np.full((4, 4), np.nan))
