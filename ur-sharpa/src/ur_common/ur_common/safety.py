"""The three v1 safety switches, as configurable parameters.

All three default to DISABLED. Each one is a single switch that fully restores
the original v1 behaviour when enabled. The logic is PRESENT but bypassed when
disabled -- nothing was deleted, because every one of these exists as a
response to a specific real-robot failure.

    switch              default   restores
    enable_jump_gate    False     tracker bad-frame gate (teleop input side)
    enable_flip_hold    False     wrist-flip hold in the IK solver
    enable_joint_clip   False     limit_joint_vel / limit_joint_pos

Signature note: v1's `limit_joint_vel` read the measured angle off the hardware
(`self.get_arm_position()`) inside the function. `ur_common` must stay
hardware-free so teleop and deploy can share it, so `current_joint` is now an
explicit argument. The clipping arithmetic is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import matrix_to_rotvec

# v1 defaults, from teleop_mocap_ur5_sharpa_hand_zed_ui.py and README_遥操作操作手册.md.
DEFAULT_JUMP_VEL = 3.5        # m/s
DEFAULT_JUMP_ROT_VEL = 1200.0  # deg/s
DEFAULT_JUMP_MAX_REJECT = 5
DEFAULT_JOINT_VEL_LIMIT = 0.35  # rad/frame
DEFAULT_JOINT_POS_LIMIT = (6.28, 6.28, 6.28, 6.28, 6.28, 6.28)
DEFAULT_FREQUENCY = 30.0
# Wrist-flip threshold defaults to 6x the per-frame velocity limit.
FLIP_THRESHOLD_VEL_MULTIPLE = 6.0


@dataclass
class SafetyConfig:
    """Every safety knob in one place. All switches default to OFF."""

    # ---- switch 1: tracker bad-frame jump gate -------------------------------
    # OFF consequence: a single bad tracker frame (a glitch of tens of cm) goes
    # straight into IK, which solves for that bogus pose and drives the arm at
    # it. This gate exists to stop exactly that.
    # ON restores: velocity-thresholded per-frame gate with forced pass-through
    # after `jump_max_reject` consecutive rejections.
    enable_jump_gate: bool = False

    # ---- switch 2: wrist-flip hold in the IK solver --------------------------
    # OFF consequence: REAL-ROBOT CRASH. See ik_solver.FastIKSolver and the
    # comment on `flip_threshold` below.
    # ON restores: reject any candidate whose per-axis delta exceeds
    # flip_threshold, i.e. hold the previous target instead of moving.
    enable_flip_hold: bool = False

    # ---- switch 3: joint velocity / position clipping ------------------------
    # OFF consequence: the raw IK output is sent to servoJ with no step-size or
    # range bound, so one large IK delta becomes one large single-frame motion.
    # ON restores: limit_joint_vel then limit_joint_pos, as in v1 move().
    enable_joint_clip: bool = False

    # ---- jump gate parameters ----
    frequency: float = DEFAULT_FREQUENCY
    jump_vel: float = DEFAULT_JUMP_VEL              # m/s
    jump_rot_vel: float = DEFAULT_JUMP_ROT_VEL      # deg/s
    jump_max_reject: int = DEFAULT_JUMP_MAX_REJECT
    # Explicit per-frame tolerances; when set they win over the velocity
    # conversion (v1 TELEOP_JUMP_TRANS / TELEOP_JUMP_ROT backward compat).
    jump_trans_thresh: float | None = None
    jump_rot_thresh: float | None = None

    # ---- flip hold parameters ----
    # None -> 6 x joint_vel_limit, matching v1's UR5_IK_FLIP_THRESH default.
    flip_threshold: float | None = None
    # Wrist-flip "soft ceiling". v1 default 0 = every over-threshold candidate
    # holds. DO NOT RAISE THIS. See ik_solver.py for the crash log.
    soft_flip_limit: float = 0.0
    prefer_same_wrist_sign: bool = True

    # ---- joint clip parameters ----
    joint_vel_limit: float = DEFAULT_JOINT_VEL_LIMIT
    joint_pos_limit: tuple[float, ...] = field(default_factory=lambda: DEFAULT_JOINT_POS_LIMIT)

    verbose: bool = False

    def resolved_flip_threshold(self) -> float:
        if self.flip_threshold is not None:
            return float(self.flip_threshold)
        return FLIP_THRESHOLD_VEL_MULTIPLE * float(self.joint_vel_limit)

    def resolved_jump_trans_thresh(self) -> float:
        """Per-frame translation tolerance in metres."""
        if self.jump_trans_thresh is not None:
            return float(self.jump_trans_thresh)
        return float(self.jump_vel) / max(1.0, float(self.frequency))

    def resolved_jump_rot_thresh(self) -> float:
        """Per-frame rotation tolerance in degrees."""
        if self.jump_rot_thresh is not None:
            return float(self.jump_rot_thresh)
        return float(self.jump_rot_vel) / max(1.0, float(self.frequency))

    def joint_pos_limit_array(self) -> np.ndarray:
        return np.asarray(self.joint_pos_limit, dtype=np.float64)

    @classmethod
    def v1_defaults(cls, **overrides: Any) -> "SafetyConfig":
        """All three switches ON with v1 thresholds -- the shipped v1 behaviour."""
        base = dict(enable_jump_gate=True, enable_flip_hold=True, enable_joint_clip=True)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_env(cls, **overrides: Any) -> "SafetyConfig":
        """Read the v1 environment variables.

        Note the polarity difference: v1 defaulted these gates ON
        (`TELEOP_JUMP_GATE` default "1"), whereas `SafetyConfig()` defaults them
        OFF. `from_env()` reproduces the v1 defaults so an existing deployment
        keeps behaving the same; explicit kwargs still win.
        """
        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw not in (None, "") else default

        cfg: dict[str, Any] = dict(
            enable_jump_gate=os.environ.get("TELEOP_JUMP_GATE", "1") == "1",
            enable_flip_hold=os.environ.get("UR5_IK_FLIP_HOLD", "1") == "1",
            enable_joint_clip=os.environ.get("UR5_JOINT_CLIP", "1") == "1",
            jump_vel=_f("TELEOP_JUMP_VEL", DEFAULT_JUMP_VEL),
            jump_rot_vel=_f("TELEOP_JUMP_ROT_VEL", DEFAULT_JUMP_ROT_VEL),
            jump_max_reject=int(_f("TELEOP_JUMP_MAX_REJECT", DEFAULT_JUMP_MAX_REJECT)),
            soft_flip_limit=_f("UR5_IK_SOFT_FLIP", 0.0),
            prefer_same_wrist_sign=os.environ.get("UR5_IK_PREFER_J4", "1") == "1",
            verbose=os.environ.get("UR5_IK_VERBOSE", "0") == "1",
        )
        if os.environ.get("TELEOP_JUMP_TRANS"):
            cfg["jump_trans_thresh"] = _f("TELEOP_JUMP_TRANS", 0.0)
        if os.environ.get("TELEOP_JUMP_ROT"):
            cfg["jump_rot_thresh"] = _f("TELEOP_JUMP_ROT", 0.0)
        if os.environ.get("UR5_IK_FLIP_THRESH"):
            cfg["flip_threshold"] = _f("UR5_IK_FLIP_THRESH", 0.0)
        cfg.update(overrides)
        return cls(**cfg)


class JumpGate:
    """SWITCH 1 -- tracker bad-frame jump gate (input side, per arm).

    Per-frame jump detection on the mapped wrist pose (4x4): if a single frame's
    translation or rotation jump exceeds threshold, the frame is a bad frame ->
    discard it and reuse the previous accepted target (hold). After
    `max_reject` consecutive rejections one frame is force-passed, so a genuine
    tracker relocation does not wedge the arm forever.

    REAL-ROBOT REASONING (from v1):
      - Thresholds are defined as VELOCITY (m/s, deg/s) and then converted with
        the actual control rate. A fixed "5cm per frame" means two different
        things at 30Hz and 60Hz.
      - The velocity ceiling is deliberately looser than any human hand: 3.5 m/s
        is well above normal teleop (a fast swing is ~1.5-2.5 m/s), so only real
        tracker spikes get caught. An earlier version used 0.05 m/frame (only
        1.5 m/s @30Hz) and misjudged normal fast motion as bad frames, holding
        the arm: "fast motion stutters, slow motion is fine". Same class of
        false positive as the IK velocity threshold incident.

    When `enable_jump_gate` is False every frame passes through unchanged and no
    state is tracked.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config if config is not None else SafetyConfig()
        self._prev: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._reject_count: dict[str, int] = {"left": 0, "right": 0}
        self.stats = {"calls": 0, "rejected": 0, "forced": 0}

    def reset(self, side: str | None = None) -> None:
        sides = ("left", "right") if side is None else (side,)
        for s in sides:
            self._prev[s] = None
            self._reject_count[s] = 0

    def filter(self, wrist: Any, side: str) -> tuple[np.ndarray, bool]:
        """Return (accepted_wrist_4x4, rejected).

        rejected=True means this frame was a bad frame and the returned pose is
        the previous accepted one (hold).
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        wrist = np.asarray(wrist, dtype=np.float64)
        if wrist.shape != (4, 4):
            raise ValueError("wrist must be a 4x4 transform")
        self.stats["calls"] += 1

        prev = self._prev[side]
        # SWITCH OFF, or first frame: accept as-is. Disabled means pass-through:
        # the caller gets exactly the pose it handed in.
        if not self.config.enable_jump_gate or prev is None:
            self._commit(wrist, side)
            return wrist, False

        trans_jump = float(np.linalg.norm(wrist[:3, 3] - prev[:3, 3]))
        rel_rot = prev[:3, :3].T @ wrist[:3, :3]
        rot_jump = float(np.degrees(np.linalg.norm(matrix_to_rotvec(rel_rot))))

        bad = (trans_jump > self.config.resolved_jump_trans_thresh()
               or rot_jump > self.config.resolved_jump_rot_thresh())
        if not bad:
            self._commit(wrist, side)
            return wrist, False

        reject_count = self._reject_count[side] + 1
        if reject_count > self.config.jump_max_reject:
            # Too many consecutive rejections: the tracker probably really did
            # move. Force-accept and reset, else the arm is wedged forever.
            if self.config.verbose:
                print(f"[jump_gate:{side}] {reject_count} consecutive rejects, forcing pass "
                      f"(trans={trans_jump*1000:.0f}mm rot={rot_jump:.0f}deg)")
            self._commit(wrist, side)
            self._reject_count[side] = 0
            self.stats["forced"] += 1
            return wrist, False

        self._reject_count[side] = reject_count
        self.stats["rejected"] += 1
        if self.config.verbose and (reject_count == 1 or reject_count % 5 == 0):
            print(f"[jump_gate:{side}] dropped jump frame #{reject_count} "
                  f"(trans={trans_jump*1000:.0f}mm/frame rot={rot_jump:.0f}deg/frame), holding")
        return prev.copy(), True

    def _commit(self, wrist: np.ndarray, side: str) -> None:
        self._prev[side] = wrist.copy()
        self._reject_count[side] = 0


class JointLimiter:
    """SWITCH 3 -- limit_joint_vel / limit_joint_pos.

    REAL-ROBOT REASONING (from v1 ur5.py move()): clipping bounds the per-frame
    STEP SIZE and the absolute range, and nothing else. Critically it does NOT
    change direction -- which is why clipping alone is not a substitute for the
    wrist-flip hold (see ik_solver.py). Keep both, or neither, but do not
    assume this one covers for that one.

    When `enable_joint_clip` is False both methods return the target unchanged.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config if config is not None else SafetyConfig()

    def limit_joint_vel(self, target_joint: Any, current_joint: Any) -> np.ndarray:
        """Clip the per-frame joint delta to +-joint_vel_limit."""
        target = np.asarray(target_joint, dtype=np.float64)
        # SWITCH OFF: pass-through, no clipping at all.
        if not self.config.enable_joint_clip:
            return target
        current = np.asarray(current_joint, dtype=np.float64)
        limit = self.config.joint_vel_limit
        return current + np.clip(target - current, -limit, limit)

    def limit_joint_pos(self, target_joint: Any) -> np.ndarray:
        """Clip absolute joint angles to +-joint_pos_limit."""
        target = np.asarray(target_joint, dtype=np.float64)
        # SWITCH OFF: pass-through.
        if not self.config.enable_joint_clip:
            return target
        limit = self.config.joint_pos_limit_array()
        return np.clip(target, -limit, limit)

    def apply(self, target_joint: Any, current_joint: Any) -> np.ndarray:
        """v1 move() order: limit_joint_vel then limit_joint_pos."""
        return self.limit_joint_pos(self.limit_joint_vel(target_joint, current_joint))
