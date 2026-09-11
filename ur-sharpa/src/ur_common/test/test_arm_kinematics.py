"""UR5 arm FK/IK round-trip closure against the compiled ikfast bindings.

These tests need `ur_common.ikfastpy`, whose `.so` is built for CPython 3.10.
On any other interpreter the whole module skips with a clear reason rather than
stubbing the solver -- a faked result here would be worthless, since the point
is to validate the real compiled UR5 chain.

ACHIEVED TOLERANCE (measured, 3000 random reachable poses):
    position   median 6.5e-14   p99 2.3e-09   max 6.4e-07  metres
    rotation   median 6.8e-14   p99 1.1e-09   max 8.8e-06  rad

The bindings use float64 end to end. This matters near analytic branch
boundaries: float32 rounding previously made IKFast omit the local solution
even for IK(FK(q)), leaving only a remote shoulder/wrist branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur_common import geometry as g
from ur_common.kinematics import (
    UR5_NUM_JOINTS,
    SafetyConfig,
    UrCommonKinematics,
    ikfast_available,
)

pytestmark = pytest.mark.skipif(
    not ikfast_available(),
    reason=("compiled ur_common.ikfastpy unavailable: the bundled .so is "
            "cpython-310 and this interpreter is not CPython 3.10"),
)

# Measured bounds plus a safety factor. See the module docstring.
POS_TOL = 5.0e-05   # measured max 6.4e-07 m
ROT_TOL = 5.0e-05   # measured max 8.8e-06 rad
SIDES = ("left", "right")


@pytest.fixture(scope="module")
def kin() -> UrCommonKinematics:
    # Switches off: this exercises pure kinematics, not the safety layer.
    return UrCommonKinematics(SafetyConfig())


def random_reachable_joints(rng: np.random.Generator) -> np.ndarray:
    """Joint angles inside a comfortably reachable, non-singular region."""
    return np.array([
        rng.uniform(-2.8, 2.8),    # base
        rng.uniform(-2.4, -0.6),   # shoulder, kept below horizontal
        rng.uniform(0.3, 2.2),     # elbow, away from the fully-straight singularity
        rng.uniform(-2.5, 0.5),    # wrist 1
        rng.uniform(-2.5, 2.5),    # wrist 2
        rng.uniform(-2.8, 2.8),    # wrist 3
    ])


def pose_errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(position error in m, rotation error in rad) between two 4x4 transforms."""
    pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rel = a[:3, :3].T @ b[:3, :3]
    rot = float(np.linalg.norm(g.matrix_to_rotvec(rel)))
    return pos, rot


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------
def test_solver_is_six_dof(kin: UrCommonKinematics) -> None:
    assert kin.solver("left").n_joints == UR5_NUM_JOINTS


def test_arm_fk_is_a_valid_transform(kin: UrCommonKinematics) -> None:
    t = kin.arm_fk(np.zeros(6))
    # Validate the rigid-transform contract independently of IK selection.
    assert t.shape == (4, 4)
    assert np.allclose(t[3], (0, 0, 0, 1))
    r = t[:3, :3]
    assert np.allclose(r.T @ r, np.eye(3), atol=1.0e-6)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1.0e-6)


def test_arm_fk_is_ur5_not_ur5e(kin: UrCommonKinematics) -> None:
    """Guards against swapping in the UR5e chain.

    At the zero configuration the TCP position decomposes exactly into UR5 link
    constants. All three components differ on a UR5e, so this catches a wrong
    compiled model that would otherwise be a silent calibration error:

        x = a2 + a3   = -0.81725  (UR5e -0.81720)
        y = -(d4 + d6) = -0.19145 (UR5e -0.23290)
        z = d1 - d5    = -0.005491 (UR5e +0.062800)

    y and z are the decisive ones -- 41mm and 68mm apart respectively.
    """
    p = kin.arm_fk(np.zeros(6))[:3, 3]
    assert abs(p[0] - (-0.425 - 0.39225)) < 1.0e-5, f"x={p[0]:.6f}, expected a2+a3"
    assert abs(p[1] - (-(0.10915 + 0.0823))) < 1.0e-5, (
        f"y={p[1]:.6f}; UR5 expects -0.19145, UR5e would be -0.23290 "
        f"(wrong robot model compiled in)"
    )
    assert abs(p[2] - (0.089159 - 0.09465)) < 1.0e-5, (
        f"z={p[2]:.6f}; UR5 expects -0.005491, UR5e would be +0.062800 "
        f"(wrong robot model compiled in)"
    )


def test_arm_fk_rejects_wrong_joint_count(kin: UrCommonKinematics) -> None:
    with pytest.raises(ValueError):
        kin.arm_fk(np.zeros(7))


def test_arm_fk_rtde_matches_arm_fk(kin: UrCommonKinematics) -> None:
    rng = np.random.default_rng(200)
    for _ in range(50):
        q = random_reachable_joints(rng)
        t = kin.arm_fk(q)
        rtde = kin.arm_fk_rtde(q)
        assert np.allclose(rtde[:3], t[:3, 3], atol=1.0e-12)
        # matrix_to_rotvec normalises through a quaternion, so compare the
        # equivalent reconstructed rotation rather than raw vector values.
        assert np.allclose(g.rotvec_to_matrix(rtde[3:]), t[:3, :3], atol=1.0e-6)


# ---------------------------------------------------------------------------
# FK -> IK -> FK closure
# ---------------------------------------------------------------------------
def test_fk_ik_fk_roundtrip_closure(kin: UrCommonKinematics) -> None:
    """The core guarantee: IK reproduces the pose FK produced.

    Seeded at the true joints so a well-posed nearest solution exists; this
    measures solver closure, not global search.
    """
    rng = np.random.default_rng(1234)
    solver = kin.solver("left")
    pos_errs, rot_errs, no_solution = [], [], 0
    attempts = 400
    for _ in range(attempts):
        q = random_reachable_joints(rng)
        target = kin.arm_fk(q)
        # The solver deliberately carries `last_target_joint` across frames and
        # returns it on a hold -- correct for teleop, wrong for a stateless
        # round-trip. Reset so a hold falls back to `current_joint` and is
        # detectable via the stats counters.
        solver.reset()
        before = solver.stats["no_raw_solution"] + solver.stats["no_feasible"]
        solved = kin.arm_ik(target, q, side="left", ref_joint=q, allow_large=True)
        if solver.stats["no_raw_solution"] + solver.stats["no_feasible"] > before:
            no_solution += 1  # genuine ikfast blind spot, not a closure failure
            continue
        p, r = pose_errors(target, kin.arm_fk(solved))
        pos_errs.append(p)
        rot_errs.append(r)

    assert len(pos_errs) > attempts * 0.9, (
        f"only {len(pos_errs)}/{attempts} poses solved; "
        f"{no_solution} had no usable solution"
    )
    pos = np.asarray(pos_errs)
    rot = np.asarray(rot_errs)
    assert pos.max() < POS_TOL, (
        f"position closure max {pos.max():.3e} m exceeds {POS_TOL:.1e} "
        f"(median {np.median(pos):.3e})"
    )
    assert rot.max() < ROT_TOL, (
        f"rotation closure max {rot.max():.3e} rad exceeds {ROT_TOL:.1e} "
        f"(median {np.median(rot):.3e})"
    )
    # Typical closure should be near machine precision; rare analytic branch
    # degeneracies account for the looser worst-case tolerance above.
    assert np.median(pos) < 1.0e-6
    assert np.median(rot) < 1.0e-6


def test_fk_ik_returns_exact_seed_when_seeded_at_solution(kin: UrCommonKinematics) -> None:
    """Seeded at the true joints, the weighted-L1 pick should return them."""
    rng = np.random.default_rng(201)
    solver = kin.solver("left")
    exact = 0
    trials = 100
    for _ in range(trials):
        q = random_reachable_joints(rng)
        solver.reset()
        solved = kin.arm_ik(kin.arm_fk(q), q, side="left", ref_joint=q, allow_large=True)
        if np.allclose(solved, q, atol=1.0e-3):
            exact += 1
    assert exact > trials * 0.9, f"only {exact}/{trials} recovered the seed joints"


def test_fk_ik_preserves_vive_branch_at_reported_float32_failure() -> None:
    """Regression for the right-arm stutter observed near an IK branch edge.

    With the old vector[float] wrapper, IK(FK(q)) returned four solutions but
    omitted q's branch; the nearest result jumped 2.474 rad and was correctly
    rejected as WRIST_FLIP_HOLD. A double-precision binding must recover q.
    """
    q = np.array([
        -1.4261,
        -1.4635,
        -2.0773,
        -2.7544,
        -1.8597,
        2.5286,
    ])
    local_kin = UrCommonKinematics(SafetyConfig(enable_flip_hold=True))
    solved = local_kin.arm_ik(
        local_kin.arm_fk(q),
        q,
        side="right",
        ref_joint=q,
    )

    assert local_kin.solver("right").last_rejection is None
    np.testing.assert_allclose(solved, q, atol=1.0e-7)


@pytest.mark.parametrize("side", SIDES)
def test_eef9_roundtrip_through_the_interface_representation(
    kin: UrCommonKinematics, side: str
) -> None:
    """FK -> 9D xyz+rot6d -> IK -> FK, i.e. the actual deploy path.

    This crosses the float32 pose9 interface boundary, so it is the tolerance
    that matters in production.
    """
    rng = np.random.default_rng(202)
    solver = kin.solver(side)
    pos_errs, rot_errs = [], []
    for _ in range(200):
        q = random_reachable_joints(rng)
        eef9 = kin.arm_fk_eef9(q, side)
        assert eef9.shape == (9,)
        solver.reset()
        before = solver.stats["no_raw_solution"] + solver.stats["no_feasible"]
        solved = kin.arm_ik_eef9(eef9, q, side, ref_joint=q, allow_large=True)
        if solver.stats["no_raw_solution"] + solver.stats["no_feasible"] > before:
            continue  # ikfast blind spot, not a closure failure
        back = kin.arm_fk_eef9(solved, side)
        pos_errs.append(float(np.linalg.norm(back[:3] - eef9[:3])))
        rot_errs.append(float(np.linalg.norm(
            g.matrix_to_rotvec(g.column_rot6d_to_matrix(eef9[3:]).T
                               @ g.column_rot6d_to_matrix(back[3:])))))
    pos = np.asarray(pos_errs)
    rot = np.asarray(rot_errs)
    assert pos.max() < POS_TOL, f"eef9 position max {pos.max():.3e} m"
    assert rot.max() < ROT_TOL, f"eef9 rotation max {rot.max():.3e} rad"


@pytest.mark.parametrize("side", SIDES)
def test_wire_eef9_roundtrip_through_the_policy_boundary(
    kin: UrCommonKinematics, side: str
) -> None:
    rng = np.random.default_rng(205)
    solver = kin.solver(side)
    recovered = 0
    for _ in range(100):
        q = random_reachable_joints(rng)
        wire = kin.arm_fk_wire_eef9(q, side)
        manual = kin.geometry.capture_tcp_to_wire_pose(
            kin.geometry.ur_base_tcp_to_capture_tcp(kin.arm_fk(q), side), side
        )
        assert np.allclose(wire, manual, atol=1.0e-12)
        solver.reset()
        before = solver.stats["no_raw_solution"] + solver.stats["no_feasible"]
        solved = kin.arm_ik_wire_eef9(
            wire.astype(np.float32), q, side, ref_joint=q, allow_large=True
        )
        if solver.stats["no_raw_solution"] + solver.stats["no_feasible"] > before:
            continue
        back = kin.arm_fk_wire_eef9(solved, side)
        assert np.linalg.norm(back[:3] - wire[:3]) < POS_TOL
        assert np.linalg.norm(
            g.matrix_to_rotvec(
                g.column_rot6d_to_matrix(wire[3:]).T
                @ g.column_rot6d_to_matrix(back[3:])
            )
        ) < ROT_TOL
        recovered += 1
    assert recovered > 80


@pytest.mark.parametrize("side", SIDES)
def test_arm_fk_eef9_matches_the_manual_frame_chain(
    kin: UrCommonKinematics, side: str
) -> None:
    """arm_fk_eef9 must be exactly FK + the documented frame chain.

    If these diverge, the facade and the adapters disagree -- the drift this
    package exists to prevent.
    """
    rng = np.random.default_rng(203)
    for _ in range(50):
        q = random_reachable_joints(rng)
        manual = kin.geometry.capture_tcp_to_canonical_pose(
            kin.geometry.ur_base_tcp_to_capture_tcp(kin.arm_fk(q), side), side
        )
        assert np.allclose(kin.arm_fk_eef9(q, side), manual, atol=1.0e-12)


def test_arm_ik_accepts_both_target_formats(kin: UrCommonKinematics) -> None:
    rng = np.random.default_rng(204)
    q = random_reachable_joints(rng)
    t = kin.arm_fk(q)
    from_matrix = kin.arm_ik(t, q, ref_joint=q, allow_large=True)
    from_rtde = kin.arm_ik(g.transform_to_rtde_pose(t), q, ref_joint=q, allow_large=True)
    assert np.allclose(from_matrix, from_rtde, atol=1.0e-9)
    with pytest.raises(ValueError):
        kin.arm_ik(np.zeros(5), q)


def test_unreachable_target_holds_the_reference(kin: UrCommonKinematics) -> None:
    """5 metres out is unreachable (UR5 reach is ~1.0 m); hold, don't invent.

    reset() first: the solver intentionally remembers `last_target_joint`
    between frames and holds THAT on failure, so without a reset this would
    assert against a stale target left by an earlier test.
    """
    solver = kin.solver("left")
    solver.reset()
    q = np.array([0.0, -1.2, 1.0, -0.5, 0.3, 0.2])
    far = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = kin.arm_ik(far, q, ref_joint=q)
    assert np.allclose(out, q)
    assert solver.stats["no_raw_solution"] >= 1


def test_reset_clears_solver_reference(kin: UrCommonKinematics) -> None:
    solver = kin.solver("left")
    q = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0.2])
    kin.arm_ik(kin.arm_fk(q), q, ref_joint=q, allow_large=True)
    assert solver.last_target_joint is not None
    kin.reset("left")
    assert solver.last_target_joint is None


def test_flip_hold_on_against_the_real_solver(kin: UrCommonKinematics) -> None:
    """Switch 2 with the compiled solver: a large jump target holds."""
    hold_kin = UrCommonKinematics(SafetyConfig(enable_flip_hold=True, joint_vel_limit=0.35))
    ref = np.array([0.0, -1.2, 1.0, -0.5, 1.55, 0.0])
    # A far-away reachable pose: every solution is a big jump from ref.
    other = np.array([2.5, -0.8, 1.8, -2.0, -1.6, 2.0])
    target = hold_kin.arm_fk(other)
    out = hold_kin.arm_ik(target, ref, side="left", ref_joint=ref)
    assert np.allclose(out, ref), "flip hold ON should have held the reference"
    # Same target with the switch OFF moves.
    free_kin = UrCommonKinematics(SafetyConfig(enable_flip_hold=False))
    moved = free_kin.arm_ik(target, ref, side="left", ref_joint=ref)
    assert not np.allclose(moved, ref)
