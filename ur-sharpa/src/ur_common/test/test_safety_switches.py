"""The three safety switches: OFF is pass-through, ON actually gates/clips.

Switches 1 and 3 are pure math and always run. Switch 2 (enable_flip_hold)
lives in the IK solver, so it is exercised against a tiny fake ikfast stub --
the SELECTION logic is pure Python and needs no compiled bindings. The
compiled-solver behaviour is covered separately in test_arm_kinematics.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur_common.geometry import rotvec_to_matrix
from ur_common.ik_solver import FastIKSolver
from ur_common.safety import (
    DEFAULT_JOINT_VEL_LIMIT,
    DEFAULT_JUMP_MAX_REJECT,
    DEFAULT_JUMP_ROT_VEL,
    DEFAULT_JUMP_VEL,
    JointLimiter,
    JumpGate,
    SafetyConfig,
)


def transform(xyz, rotvec=(0.0, 0.0, 0.0)) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotvec_to_matrix(np.asarray(rotvec, dtype=np.float64))
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_all_three_switches_default_to_disabled() -> None:
    cfg = SafetyConfig()
    assert cfg.enable_jump_gate is False
    assert cfg.enable_flip_hold is False
    assert cfg.enable_joint_clip is False


def test_v1_defaults_enables_all_three() -> None:
    cfg = SafetyConfig.v1_defaults()
    assert cfg.enable_jump_gate is True
    assert cfg.enable_flip_hold is True
    assert cfg.enable_joint_clip is True


def test_documented_v1_thresholds() -> None:
    cfg = SafetyConfig()
    assert DEFAULT_JUMP_VEL == 3.5
    assert DEFAULT_JUMP_ROT_VEL == 1200.0
    assert DEFAULT_JUMP_MAX_REJECT == 5
    assert cfg.soft_flip_limit == 0.0  # raising this is what crashed the arm
    # flip threshold defaults to 6x the per-frame velocity limit
    assert cfg.resolved_flip_threshold() == pytest.approx(6.0 * DEFAULT_JOINT_VEL_LIMIT)
    # velocity thresholds convert with the control rate, not a fixed per-frame value
    assert cfg.resolved_jump_trans_thresh() == pytest.approx(3.5 / 30.0)
    assert cfg.resolved_jump_rot_thresh() == pytest.approx(1200.0 / 30.0)
    assert SafetyConfig(frequency=60.0).resolved_jump_trans_thresh() == pytest.approx(3.5 / 60.0)


def test_explicit_per_frame_thresholds_win_over_velocity() -> None:
    cfg = SafetyConfig(jump_trans_thresh=0.01, jump_rot_thresh=5.0)
    assert cfg.resolved_jump_trans_thresh() == pytest.approx(0.01)
    assert cfg.resolved_jump_rot_thresh() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# SWITCH 1: jump gate
# ---------------------------------------------------------------------------
def test_jump_gate_off_passes_everything_through_unchanged() -> None:
    gate = JumpGate(SafetyConfig(enable_jump_gate=False))
    gate.filter(transform((0.0, 0.0, 0.0)), "left")
    # A 10 metre teleport: absurd, and still passed through when OFF.
    absurd = transform((10.0, 0.0, 0.0))
    out, rejected = gate.filter(absurd, "left")
    assert rejected is False
    assert np.allclose(out, absurd)
    assert gate.stats["rejected"] == 0


def test_jump_gate_on_rejects_a_translation_spike() -> None:
    gate = JumpGate(SafetyConfig(enable_jump_gate=True, frequency=30.0))
    first = transform((0.0, 0.0, 0.0))
    out, rejected = gate.filter(first, "left")
    assert rejected is False  # first frame always accepted
    # 3.5 m/s @30Hz = 117mm/frame tolerance; 500mm is a tracker glitch.
    out, rejected = gate.filter(transform((0.5, 0.0, 0.0)), "left")
    assert rejected is True
    assert np.allclose(out, first)  # holds the previous accepted pose
    assert gate.stats["rejected"] == 1


def test_jump_gate_on_accepts_normal_fast_motion() -> None:
    """3.5 m/s is deliberately looser than any human hand.

    v1 used 0.05 m/frame (1.5 m/s @30Hz) and misjudged normal fast motion as
    bad frames: "fast motion stutters, slow motion is fine".
    """
    gate = JumpGate(SafetyConfig(enable_jump_gate=True, frequency=30.0))
    gate.filter(transform((0.0, 0.0, 0.0)), "left")
    # 80mm/frame @30Hz = 2.4 m/s, a fast but real human swing.
    out, rejected = gate.filter(transform((0.08, 0.0, 0.0)), "left")
    assert rejected is False
    assert gate.stats["rejected"] == 0


def test_jump_gate_on_rejects_a_rotation_spike() -> None:
    gate = JumpGate(SafetyConfig(enable_jump_gate=True, frequency=30.0))
    gate.filter(transform((0.0, 0.0, 0.0)), "left")
    # 1200 deg/s @30Hz = 40 deg/frame tolerance; 90 deg in one frame is a glitch.
    out, rejected = gate.filter(
        transform((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2)), "left")
    assert rejected is True


def test_jump_gate_force_passes_after_max_reject() -> None:
    """A genuine tracker relocation must not wedge the arm forever."""
    cfg = SafetyConfig(enable_jump_gate=True, frequency=30.0, jump_max_reject=5)
    gate = JumpGate(cfg)
    gate.filter(transform((0.0, 0.0, 0.0)), "left")
    far = transform((0.5, 0.0, 0.0))
    for i in range(cfg.jump_max_reject):
        _, rejected = gate.filter(far, "left")
        assert rejected is True, f"frame {i} should be rejected"
    out, rejected = gate.filter(far, "left")
    assert rejected is False  # forced through
    assert np.allclose(out, far)
    assert gate.stats["forced"] == 1
    # Counter reset, so the new position is now the reference.
    _, rejected = gate.filter(far, "left")
    assert rejected is False


def test_jump_gate_tracks_sides_independently() -> None:
    gate = JumpGate(SafetyConfig(enable_jump_gate=True, frequency=30.0))
    gate.filter(transform((0.0, 0.0, 0.0)), "left")
    gate.filter(transform((0.0, 0.0, 0.0)), "right")
    _, left_rejected = gate.filter(transform((0.5, 0.0, 0.0)), "left")
    _, right_rejected = gate.filter(transform((0.01, 0.0, 0.0)), "right")
    assert left_rejected is True
    assert right_rejected is False


def test_jump_gate_reset_clears_history() -> None:
    gate = JumpGate(SafetyConfig(enable_jump_gate=True, frequency=30.0))
    gate.filter(transform((0.0, 0.0, 0.0)), "left")
    gate.reset()
    # After reset this is a first frame again, so it is accepted.
    _, rejected = gate.filter(transform((0.5, 0.0, 0.0)), "left")
    assert rejected is False


def test_jump_gate_rejects_bad_side_and_shape() -> None:
    gate = JumpGate(SafetyConfig(enable_jump_gate=True))
    with pytest.raises(ValueError):
        gate.filter(np.eye(4), "middle")
    with pytest.raises(ValueError):
        gate.filter(np.eye(3), "left")


# ---------------------------------------------------------------------------
# SWITCH 3: joint clip
# ---------------------------------------------------------------------------
def test_joint_clip_off_passes_through_unchanged() -> None:
    lim = JointLimiter(SafetyConfig(enable_joint_clip=False))
    current = np.zeros(6)
    target = np.array([99.0, -99.0, 3.0, 0.0, 0.0, 0.0])  # wildly out of range
    assert np.allclose(lim.limit_joint_vel(target, current), target)
    assert np.allclose(lim.limit_joint_pos(target), target)
    assert np.allclose(lim.apply(target, current), target)


def test_joint_clip_on_limits_velocity() -> None:
    cfg = SafetyConfig(enable_joint_clip=True, joint_vel_limit=0.35)
    lim = JointLimiter(cfg)
    current = np.zeros(6)
    target = np.array([1.0, -1.0, 0.1, 0.0, 0.0, 0.0])
    out = lim.limit_joint_vel(target, current)
    assert np.allclose(out, [0.35, -0.35, 0.1, 0.0, 0.0, 0.0])
    # Never exceeds the per-frame step from current.
    assert np.all(np.abs(out - current) <= cfg.joint_vel_limit + 1.0e-12)


def test_joint_clip_on_limits_position() -> None:
    cfg = SafetyConfig(enable_joint_clip=True)
    lim = JointLimiter(cfg)
    target = np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
    out = lim.limit_joint_pos(target)
    assert np.allclose(out[:2], [6.28, -6.28])


def test_joint_clip_apply_order_matches_v1_move() -> None:
    """v1 move(): limit_joint_vel then limit_joint_pos."""
    cfg = SafetyConfig(enable_joint_clip=True, joint_vel_limit=0.35)
    lim = JointLimiter(cfg)
    current = np.full(6, 6.2)
    target = np.full(6, 100.0)
    out = lim.apply(target, current)
    # vel clip -> 6.55, then pos clip -> 6.28
    assert np.allclose(out, 6.28)


def test_joint_clip_does_not_change_direction() -> None:
    """The reason clipping is NOT a substitute for the wrist-flip hold.

    Clipping bounds step size only. The sign of every axis delta is preserved,
    so the arm still travels toward the flipped target -- just slower.
    """
    cfg = SafetyConfig(enable_joint_clip=True, joint_vel_limit=0.35)
    lim = JointLimiter(cfg)
    current = np.zeros(6)
    flipped_target = np.array([0.0, 0.0, 0.0, 0.0, -2.4, 0.0])  # a J4 wrist flip
    out = lim.limit_joint_vel(flipped_target, current)
    assert out[4] < 0.0  # still heading the same (wrong) way
    assert np.sign(out[4]) == np.sign(flipped_target[4])
    assert abs(out[4]) == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# SWITCH 2: wrist-flip hold (solver selection logic, stubbed ikfast)
# ---------------------------------------------------------------------------
class FakeKinematics:
    """Minimal ikfast stand-in: returns a fixed solution set.

    Only `getDOF` and `inverse` are used by the selection path under test.
    """

    def __init__(self, solutions) -> None:
        self._solutions = [np.asarray(s, dtype=np.float64) for s in solutions]

    def getDOF(self) -> int:  # noqa: N802 - matches the C++ binding name
        return 6

    def inverse(self, _ee_pose):
        if not self._solutions:
            return []
        return np.concatenate(self._solutions).tolist()


IDENTITY_CART = np.array([0.4, 0.1, 0.3, 0.0, 0.0, 0.0])


def test_flip_hold_off_returns_the_flipped_solution() -> None:
    """SWITCH OFF: the flipped solution is handed straight to the caller.

    This is the configuration whose real-robot result was a CRASH when combined
    with downstream clipping only. The test asserts the bypass works, not that
    it is safe.
    """
    ref = np.array([0.0, -1.2, 1.0, -0.5, 1.55, 0.0])
    flipped = ref.copy()
    flipped[4] = -0.85  # crosses the J4=0 wrist singularity plane, 2.4 rad
    solver = FastIKSolver(FakeKinematics([flipped]),
                          config=SafetyConfig(enable_flip_hold=False))
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert np.allclose(out, flipped)
    assert solver.stats["flip_held"] == 0
    assert solver.stats["solved"] == 1


def test_flip_hold_on_holds_instead_of_flipping() -> None:
    """SWITCH ON: the flipped solution is rejected and the reference held."""
    ref = np.array([0.0, -1.2, 1.0, -0.5, 1.55, 0.0])
    flipped = ref.copy()
    flipped[4] = -0.85
    solver = FastIKSolver(FakeKinematics([flipped]),
                          config=SafetyConfig(enable_flip_hold=True))
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert not np.allclose(out, flipped)
    assert np.allclose(out, ref)  # held
    assert solver.stats["flip_held"] == 1
    assert solver.stats["no_feasible"] == 1


def test_flip_hold_on_still_follows_normal_fast_motion() -> None:
    """Only flips are rejected; mild over-speed is passed to the joint clip.

    Rejecting normal fast motion was the earlier false-positive that showed up
    as teleop stutter.
    """
    cfg = SafetyConfig(enable_flip_hold=True, joint_vel_limit=0.35)
    ref = np.array([0.0, -1.2, 1.0, -0.5, 1.55, 0.0])
    nearby = ref.copy()
    nearby[1] += 0.9  # over vel_limit (0.35) but under flip_threshold (2.10)
    assert 0.35 < 0.9 < cfg.resolved_flip_threshold()
    solver = FastIKSolver(FakeKinematics([nearby]), config=cfg)
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert np.allclose(out, nearby)
    assert solver.stats["flip_held"] == 0


def test_flip_hold_on_prefers_the_same_j4_sign() -> None:
    """Among equally safe solutions, do not cross the wrist singularity."""
    cfg = SafetyConfig(enable_flip_hold=True, joint_vel_limit=0.35)
    ref = np.array([0.0, -1.2, 1.0, -0.5, 1.55, 0.0])
    same_sign = ref.copy()
    same_sign[4] = 1.20          # delta 0.35, same sign as ref
    cross_sign = ref.copy()
    cross_sign[4] = -0.60        # delta 2.15... must stay under flip_threshold
    cross_sign[4] = ref[4] - 2.0  # -0.45, delta 2.0 < 2.10
    solver = FastIKSolver(FakeKinematics([cross_sign, same_sign]), config=cfg)
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert out[4] > 0.0
    assert np.allclose(out, same_sign)
    assert solver.stats["j4_sign_kept"] == 1


def test_flip_hold_allow_large_bypasses_for_homing() -> None:
    """Homing mode intentionally permits a large excursion."""
    ref = np.array([0.0, -1.2, 1.0, -0.5, 1.55, 0.0])
    flipped = ref.copy()
    flipped[4] = -0.85
    solver = FastIKSolver(FakeKinematics([flipped]),
                          config=SafetyConfig(enable_flip_hold=True))
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref, allow_large=True)
    assert np.allclose(out, flipped)


def test_limit_filter_applies_regardless_of_flip_switch() -> None:
    """Joint-limit filtering is not part of switch 2 and always runs.

    The candidate must be out of limit even AFTER +-2pi unwrapping, or the
    unwrap legitimately rescues it. With a +-1.0 rad limit, J1=3.0 fails both
    as 3.0 and as 3.0-2pi = -3.28.
    """
    ref = np.zeros(6)
    out_of_limit = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tight = (1.0,) * 6
    for enable in (False, True):
        solver = FastIKSolver(
            FakeKinematics([out_of_limit]),
            config=SafetyConfig(enable_flip_hold=enable, joint_pos_limit=tight),
        )
        out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
        assert np.allclose(out, ref)  # held, out of limit
        assert solver.stats["no_feasible"] == 1


def test_unwrap_rescues_an_equivalent_angle_within_limits() -> None:
    """+-2pi unwrapping must normalise a wrapped solution, not reject it.

    A candidate at 2pi is the same pose as 0; v1 unwraps it rather than treating
    the -pi/+pi wrap as a huge jump.
    """
    ref = np.zeros(6)
    wrapped = np.array([2.0 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    solver = FastIKSolver(FakeKinematics([wrapped]), config=SafetyConfig())
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert np.allclose(out, ref, atol=1.0e-9)
    assert solver.stats["solved"] == 1


def test_no_solution_holds_previous_target() -> None:
    ref = np.array([0.1, -1.0, 1.0, -0.5, 0.5, 0.0])
    solver = FastIKSolver(FakeKinematics([]), config=SafetyConfig())
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert np.allclose(out, ref)
    assert solver.stats["no_raw_solution"] == 1


def test_unwrap_prefers_in_limit_equivalent_angle() -> None:
    """The regression v1 documented: sol=6.885 with limit +-6.28.

    The nearest equivalent is out of limit, but 6.885-2pi=0.602 is legal. A
    naive nearest-equivalent unwrap would get the whole set filtered out (HOLD,
    seen as teleop stutter).
    """
    ref = np.zeros(6)
    sol = np.array([6.885, 0.0, 0.0, 0.0, 0.0, 0.0])
    solver = FastIKSolver(FakeKinematics([sol]), config=SafetyConfig())
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert out[0] == pytest.approx(6.885 - 2.0 * np.pi, abs=1e-9)


def test_rtde_fallback_used_when_ikfast_has_nothing() -> None:
    ref = np.zeros(6)
    answer = np.array([0.2, -1.0, 0.9, -0.4, 0.3, 0.1])
    solver = FastIKSolver(FakeKinematics([]), config=SafetyConfig(),
                          rtde_ik_fn=lambda x, qnear: answer.tolist())
    out = solver.solve(IDENTITY_CART, ref, ref_joint=ref)
    assert np.allclose(out, answer)
    assert solver.stats["solved_rtde"] == 1


def test_solver_report_is_a_string() -> None:
    solver = FastIKSolver(FakeKinematics([]), config=SafetyConfig())
    solver.solve(IDENTITY_CART, np.zeros(6), ref_joint=np.zeros(6))
    assert "IK stats" in solver.report()
