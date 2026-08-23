"""Interpolated homing trajectories plus arrival checking, hardware-free.

`bias_node` is the only producer allowed to move the arms while UR is in the
`homed` state (see ur_node/arbitration.py), so this is the code path that runs
right after power-on. Everything here is a pure function of numbers so it can be
tested without RTDE, ROS, or a robot.

Shape comes from v3 `UrRtdePair.initialize_joints`
(the legacy hardware driver): 120 steps at 0.02 s with a
0.08 rad arrival tolerance, linear in joint space from the measured start pose.
The SharpA counterpart there used 60 steps at the same delay and tolerance.

The one behaviour that is NOT carried over is v3's blocking `time.sleep` loop.
This node streams the same samples as ROS messages on a timer instead, so a
homing move never blocks the executor. The step count, delay and tolerance are
unchanged, so the motion profile on the wire is identical.

TIMEOUT IS A FAILURE, NOT A SUCCESS. v3 raised `HardwareCommandError` when the
measured pose was still outside tolerance after the last step. Reporting success
on a timeout would tell `status_node` the arms are at home when they are
somewhere else, and the next transition to `running` would hand a teleop stream
an arm in an unknown pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# v3 initialize_joints defaults, kept verbatim.
UR_HOME_STEPS = 120
UR_HOME_STEP_DELAY_S = 0.02
UR_HOME_TOLERANCE_RAD = 0.08

SHARPA_HOME_STEPS = 60
SHARPA_HOME_STEP_DELAY_S = 0.02
SHARPA_HOME_TOLERANCE_RAD = 0.08

UR_JOINTS_PER_ARM = 6
SHARPA_JOINTS_TOTAL = 44
SHARPA_JOINTS_PER_HAND = 22

# SharpA home is the 44D zero vector: every hand joint at its neutral open pose.
SHARPA_HOME_Q44 = np.zeros(SHARPA_JOINTS_TOTAL, dtype=np.float64)

TWO_PI = 2.0 * np.pi


def unwrap_to_near(joint: Any, near: Any) -> np.ndarray:
    """Shift `joint` by whole turns to sit nearest `near`.

    Verbatim from v3 hardware_drivers._unwrap_to_near:99. Needed on the arrival
    check because RTDE can report an angle a full turn away from the commanded
    one while the arm is physically in exactly the commanded pose; comparing raw
    would fail a move that actually succeeded.
    """
    joint = np.asarray(joint, dtype=np.float64)
    near = np.asarray(near, dtype=np.float64)
    return joint + np.round((near - joint) / TWO_PI) * TWO_PI


def interpolate(start: Any, target: Any, steps: int) -> np.ndarray:
    """Linear joint-space ramp, shape (steps, n).

    v3 indexing: `alpha = step_index / steps` for step_index in 1..steps, so the
    first sample already moves and the last sample is exactly `target`. The
    start pose itself is never re-sent -- the arm is already there.
    """
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if start.shape != target.shape:
        raise ValueError(f"start {start.shape} and target {target.shape} disagree")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive int")
    alpha = (np.arange(1, steps + 1, dtype=np.float64) / float(steps))[:, None]
    return start[None, :] + alpha * (target - start)[None, :]


@dataclass(frozen=True)
class ArrivalResult:
    """Outcome of one arrival check. `reached=False` is a real failure."""

    reached: bool
    max_error_rad: float
    reason: str = ""
    per_side_error: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.reached


def check_arm_arrival(
    measured: dict,
    target: dict,
    tolerance_rad: float = UR_HOME_TOLERANCE_RAD,
) -> ArrivalResult:
    """Both arms within `tolerance_rad` of their home joints, per v3.

    `measured` / `target` are {'left': (6,), 'right': (6,)}. Measured angles are
    unwrapped toward the target first (see unwrap_to_near).
    """
    if tolerance_rad <= 0.0 or not np.isfinite(tolerance_rad):
        raise ValueError("tolerance_rad must be positive and finite")
    errors: dict = {}
    failing: list = []
    for side in ("left", "right"):
        if side not in measured or measured[side] is None:
            return ArrivalResult(False, float("inf"), f"{side} UR state unavailable")
        actual = unwrap_to_near(measured[side], target[side])
        err = float(np.max(np.abs(actual - np.asarray(target[side], dtype=np.float64))))
        errors[side] = err
        if not np.isfinite(err) or err > tolerance_rad:
            failing.append(side)
    worst = max(errors.values()) if errors else float("inf")
    if failing:
        return ArrivalResult(
            False,
            worst,
            f"{'/'.join(failing)} UR did not reach its initial joint target "
            f"(max error {worst:.4f} rad > {tolerance_rad:.4f})",
            errors,
        )
    return ArrivalResult(True, worst, "", errors)


def check_hand_arrival(
    measured: Any,
    target: Any = None,
    tolerance_rad: float = SHARPA_HOME_TOLERANCE_RAD,
) -> ArrivalResult:
    """44D hand pose within tolerance of the home pose (zeros by default).

    No unwrapping here: v3's SharpA arrival check compared raw, because hand
    joints have a limited range and never wrap.
    """
    if tolerance_rad <= 0.0 or not np.isfinite(tolerance_rad):
        raise ValueError("tolerance_rad must be positive and finite")
    if measured is None:
        return ArrivalResult(False, float("inf"), "SharpA state unavailable")
    target = SHARPA_HOME_Q44 if target is None else np.asarray(target, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64).reshape(-1)
    if measured.shape[0] != SHARPA_JOINTS_TOTAL:
        return ArrivalResult(
            False, float("inf"),
            f"SharpA state is {measured.shape[0]}D, expected {SHARPA_JOINTS_TOTAL}D",
        )
    diff = np.abs(measured - target.reshape(-1))
    errors = {
        "left": float(np.max(diff[:SHARPA_JOINTS_PER_HAND])),
        "right": float(np.max(diff[SHARPA_JOINTS_PER_HAND:])),
    }
    worst = float(np.max(diff)) if diff.size else float("inf")
    failing = [s for s, e in errors.items() if not np.isfinite(e) or e > tolerance_rad]
    if failing:
        return ArrivalResult(
            False, worst,
            f"{'/'.join(failing)} SharpA did not reach its initial joint target "
            f"(max error {worst:.4f} rad > {tolerance_rad:.4f})",
            errors,
        )
    return ArrivalResult(True, worst, "", errors)
