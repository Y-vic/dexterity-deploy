"""UR5 IK solver (FastIKSolver), ported from robot_control/ik_solver.py.

Solve pipeline:
  ikfast emits all analytic solutions -> unwrap toward the reference angle
  (+-2pi normalisation) -> joint-limit filter -> wrist-flip filter -> weighted
  L1 nearest pick. When ikfast has no solution at all, an optional numeric
  fallback (RTDE getInverseKinematics) fills the blind spot.

Key points carried over from v1:
  1) Solution selection references the PREVIOUS COMMANDED target joint angle
     (ref_joint), not the lagging measured angle. This removes false failures
     caused by servoJ following delay.
  2) All ikfast solutions are filtered against joint limits, and each axis is
     unwrapped by +-2pi to the equivalent angle nearest the reference. This
     removes pointless large jumps at the -pi/+pi wrap.
  3) Tiered safety: ONLY a wrist flip (any axis delta > flip_threshold) holds.
     Normal fast motion is not rejected. See the flip_threshold comment.
  4) Weighted L1 (shoulder-side joints weighted higher) picks among candidates.
  5) The wrist-flip filter is now behind SafetyConfig.enable_flip_hold and
     DEFAULTS TO OFF. Read the comment on that switch before leaving it off.

Hardware is UR5, not UR5e -- the compiled ikfast bindings in ur_common/ikfastpy
are generated for UR5 DH parameters.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from typing import Any

import numpy as np

from .geometry import matrix_to_rotvec, rotvec_to_matrix, transform_to_rtde_pose, rtde_pose_to_transform
from .safety import SafetyConfig

# Shoulder-side joints cost more when they deviate: base/shoulder/elbow get a
# larger weight, wrist joints a smaller one.
_DEFAULT_JOINT_WEIGHTS = np.array([1.5, 1.5, 1.3, 1.0, 1.0, 0.8])
_UNWRAP_OFFSETS = np.arange(-2, 3)[:, None, None]
_NATIVE_STDERR_LOCK = threading.Lock()
_GENERIC_IKFAST_ERROR = (
    b"Error: (inverse kinematics) failed to get ik solution\n"
)

__all__ = ["FastIKSolver", "cart_to_matrix"]


def cart_to_matrix(cart_pose: Any) -> np.ndarray:
    """[x,y,z, rx,ry,rz(rotvec)] -> 3x4 row-major, the ikfast.inverse input."""
    cart_pose = np.asarray(cart_pose, dtype=np.float64)
    ee = np.zeros((3, 4), dtype=np.float64)
    ee[:3, :3] = rotvec_to_matrix(cart_pose[3:])
    ee[:3, 3] = cart_pose[:3]
    return ee


def _inverse_with_filtered_stderr(kinematics: Any, pose: list[float]):
    """Drop the binding's generic error while preserving unrelated stderr.

    The bundled C++ extension writes directly to file descriptor 2, so Python's
    redirect_stderr cannot catch it. The redirection is process-wide for the
    few microseconds spent in IKFast; output from another thread is replayed
    unchanged after the call.
    """

    quiet = getattr(kinematics, 'inverse_quiet', None)
    if callable(quiet):
        return quiet(pose)

    with _NATIVE_STDERR_LOCK:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        saved_stderr = os.dup(2)
        try:
            with tempfile.TemporaryFile() as captured:
                os.dup2(captured.fileno(), 2)
                try:
                    result = kinematics.inverse(pose)
                finally:
                    try:
                        sys.stderr.flush()
                    except Exception:
                        pass
                    os.dup2(saved_stderr, 2)
                captured.seek(0)
                remaining = captured.read().replace(_GENERIC_IKFAST_ERROR, b"")
                if remaining:
                    os.write(saved_stderr, remaining)
                return result
        finally:
            os.close(saved_stderr)


class FastIKSolver:
    """Analytic UR5 IK with reference-angle solution selection.

    :param kinematics: an ikfastpy.PyKinematics instance
    :param config: SafetyConfig; supplies flip_threshold, joint limits, and the
        enable_flip_hold switch. Defaults to all-switches-OFF.
    :param joint_weights: weighted-L1 weights, shoulder-heavy by default
    :param rtde_ik_fn: optional numeric fallback getInverseKinematics(x, qnear),
        called only when ikfast finds nothing. Its result goes through the same
        limit (+ flip) checks.
    :param name: diagnostic tag, e.g. "left"/"right"
    """

    def __init__(self, kinematics: Any, config: SafetyConfig | None = None,
                 joint_weights: Any = None, rtde_ik_fn: Any = None,
                 name: str = "", verbose: bool | None = None) -> None:
        self.kin = kinematics
        self.n_joints = int(kinematics.getDOF())
        self.config = config if config is not None else SafetyConfig()
        self.rtde_ik_fn = rtde_ik_fn
        self.name = name

        limit = self.config.joint_pos_limit_array()
        if limit.shape[0] != self.n_joints:
            limit = np.full(self.n_joints, float(limit[0]))
        self.joint_pos_limit = limit
        self.joint_vel_limit = float(self.config.joint_vel_limit)

        # Wrist-flip threshold: a single-frame delta above this on ANY axis is
        # judged a wrist flip -> hold (crash protection). Normal fast motion
        # below this is handed to the downstream joint clip to follow at reduced
        # speed rather than being rejected. Default = 6x vel_limit.
        self.flip_threshold = self.config.resolved_flip_threshold()

        # ==== SWITCH 2: enable_flip_hold ==================================
        # LEAVING THIS OFF CRASHED THE REAL ROBOT. Verbatim reasoning from v1
        # robot_control/ik_solver.py (lines 55-69), which documents the attempt
        # to drop the hold and rely on downstream limit_joint_vel clipping only:
        #
        #   【真机教训，勿轻易开启】曾默认设 2.8，想让"只超一点"的换腕解放行、
        #   靠下游 limit_joint_vel(0.35rad/帧) 分帧过渡来消除卡顿。真机结果是**撞机**：
        #   一次 SOFT-PASS(min_jump=2.321) 后，左臂位置偏差 9mm→37.8mm→186.2mm，
        #   两帧就触发保护停。原因是 clip 只限制每帧步长、不改变方向——目标始终是那个
        #   换腕解，机械臂会以 0.35rad/帧**坚定地朝错误构型冲**，并不会"平滑过渡"。
        #   而 hold 之所以安全，正是因为它根本不朝那个方向动。
        #   2.3 的换腕和 3.9 的换腕都是换腕，危险性无本质差别，不能按数值远近放行。
        #
        # In English: clipping bounds the per-frame STEP SIZE but never changes
        # DIRECTION. The target stays the flipped configuration, so the arm
        # marches at it 0.35 rad/frame -- it does not "ease in". Position error
        # went 9mm -> 37.8mm -> 186.2mm and the protective stop fired in two
        # frames. hold is safe precisely because it does not move that way. A
        # 2.3 rad flip and a 3.9 rad flip are both flips; you cannot wave one
        # through because the number is smaller.
        self.enable_flip_hold = bool(self.config.enable_flip_hold)

        # Wrist-flip "soft ceiling". v1 default 0 = anything over
        # flip_threshold holds. Raising this is what caused the crash above.
        self.soft_flip_limit = float(self.config.soft_flip_limit)

        # J4 sign preference: prefer candidates whose J4 shares the reference
        # sign, i.e. do not cross the wrist singularity plane J4=0. This only
        # picks between EQUALLY SAFE solutions; it never relaxes a threshold.
        self.prefer_same_wrist_sign = bool(self.config.prefer_same_wrist_sign)

        self.joint_weights = (np.asarray(joint_weights, dtype=np.float64)
                              if joint_weights is not None
                              else _DEFAULT_JOINT_WEIGHTS[:self.n_joints])

        if verbose is None:
            verbose = self.config.verbose or os.environ.get("UR5_IK_VERBOSE", "0") == "1"
        self.verbose = bool(verbose)

        # Previous commanded target joint angle (the selection reference).
        # None on the first frame -> falls back to current_joint.
        self.last_target_joint: np.ndarray | None = None

        self.stats = {
            "calls": 0,
            "no_raw_solution": 0,   # ikfast found nothing (rtde fallback tried)
            "no_feasible": 0,       # had raw solutions, all filtered out
            "solved": 0,
            "solved_rtde": 0,
            "solved_local": 0,
            "soft_flip": 0,
            "j4_sign_kept": 0,
            "flip_held": 0,         # holds attributable to the flip filter
        }
        # Populated only when solve() returns a held target. Vive uses this to
        # distinguish an unreachable Cartesian pose from the two safety gates.
        self.last_rejection: dict[str, Any] | None = None
        self._last_filter_rejection: dict[str, Any] | None = None

    # ---- internals ----
    def _raw_solutions(self, cart_pose: Any) -> np.ndarray:
        """Call ikfast, return (n,6) analytic solutions; empty array if none."""
        target = np.asarray(cart_pose)
        ee = target[:3, :4] if target.shape == (4, 4) else cart_to_matrix(target)
        try:
            configs = _inverse_with_filtered_stderr(
                self.kin, ee.reshape(-1).tolist()
            )
        except Exception:
            return np.empty((0, self.n_joints))
        if configs is None or len(configs) == 0:
            return np.empty((0, self.n_joints))
        n = int(len(configs) / self.n_joints)
        return np.asarray(configs, dtype=np.float64).reshape(n, self.n_joints)

    def _unwrap_to_ref(self, sols: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Per solution per axis, add/subtract 2pi to reach the equivalent angle
        that is IN LIMIT and nearest the reference.

        Why not simply the nearest equivalent angle: the naive version takes
        k=round((ref-sol)/2pi). When that nearest equivalent lands outside the
        joint limit (measured on the real robot, right arm J0: sol=6.885,
        limit=+-6.28, while 6.885-2pi=0.602 is perfectly legal but ignored) the
        subsequent limit filter kills the whole solution set -> HOLD, which
        shows up as teleop stutter.

        Fix: among {sol + k*2pi} prefer in-limit AND nearest to ref. Only when
        every equivalent is out of limit (genuinely unreachable) fall back to
        the nearest one, so the limit filter can honestly reject it instead of
        us fabricating an out-of-range solution.
        """
        two_pi = 2.0 * np.pi
        limit = self.joint_pos_limit[None, :]
        k0 = np.round((ref[None, :] - sols) / two_pi)
        nearest = sols + k0 * two_pi

        # Scan +-2 turns around the nearest equivalent: with limits up to +-2pi
        # there are at most 3 in-limit equivalents. argmin keeps the first
        # offset on ties, exactly as the former strict-less-than scan did.
        candidates = sols[None] + (k0[None] + _UNWRAP_OFFSETS) * two_pi
        distances = np.where(np.abs(candidates) <= limit,
                             np.abs(candidates - ref), np.inf)
        indices = np.argmin(distances, axis=0)[None]
        best = np.take_along_axis(candidates, indices, axis=0)[0]
        found = np.isfinite(np.min(distances, axis=0))
        return np.where(found, best, nearest)

    def solve(self, cart_pose: Any, current_joint: Any, ref_joint: Any = None,
              allow_large: bool = False, local_refinement: bool = False) -> np.ndarray:
        """Solve and return 6 joint angles.

        On failure returns the previous target (hold) or current_joint.

        :param cart_pose: [x,y,z, rx,ry,rz(rotvec)] or a validated 4x4 TCP
            transform in the UR base frame. Matrix input avoids a redundant
            matrix -> rotvec -> matrix round-trip on EEF chunks.
        :param current_joint: measured joint angles (fallback + first reference)
        :param ref_joint: selection reference; defaults to last_target_joint,
            then current_joint
        :param allow_large: homing mode. True skips the wrist-flip filter and
            allows a large excursion to the (unwrapped) nearest solution, for
            returning home from an arbitrary pose. Limit filtering and unwrap
            still apply. Per-frame teleop following uses False.
        :param local_refinement: opt-in EEF recovery when IKFast omits a nearby
            branch. Bounded FK-based refinement must converge to the exact pose
            and pass the normal candidate filters; it never admits a remote flip.
        """
        self.stats["calls"] += 1
        current_joint = np.asarray(current_joint, dtype=np.float64)
        self.last_rejection = None
        self._last_filter_rejection = None

        if ref_joint is not None:
            ref = np.asarray(ref_joint, dtype=np.float64)
        elif self.last_target_joint is not None:
            ref = self.last_target_joint
        else:
            ref = current_joint

        sols = self._raw_solutions(cart_pose)

        if sols.shape[0] == 0:
            if np.asarray(cart_pose).shape == (4, 4):
                cart_pose = transform_to_rtde_pose(cart_pose)
            target = np.asarray(cart_pose, dtype=np.float64).reshape(-1)
            # ikfast found nothing (near singularity / workspace edge). This is
            # a genuine analytic blind spot, not a false failure. Use the
            # numeric fallback; its solution still goes through the same
            # limit + flip checks.
            self.stats["no_raw_solution"] += 1
            if local_refinement:
                best = self._try_local_refinement(cart_pose, ref, allow_large)
                if best is not None:
                    self.stats['solved_local'] += 1
                    self.last_target_joint = best.copy()
                    return best.copy()
            best = self._try_rtde_fallback(cart_pose, ref, current_joint, allow_large)
            if best is None:
                rejection = self._last_filter_rejection or {
                    "stage": "IKFAST_NO_SOLUTION",
                    "raw_candidates": 0,
                    "detail": (
                        "IKFast returned zero analytic solutions; joint-position, "
                        "wrist-flip and Vive step limits were not evaluated"
                    ),
                }
                self.last_rejection = {
                    **rejection,
                    "side": self.name,
                    "target_tcp": target.tolist(),
                    "reference_joint": ref.tolist(),
                }
                fallback = (self.last_target_joint if self.last_target_joint is not None
                            else current_joint)
                if self.verbose:
                    print(f"[IK:{self.name}] ikfast+rtde both unusable, HOLD. "
                          f"cart={np.round(np.asarray(cart_pose, dtype=np.float64), 3)}")
                return fallback.copy()
            self.stats["solved_rtde"] += 1
            self.last_target_joint = best.copy()
            return best.copy()

        sols = self._unwrap_to_ref(sols, ref)

        best = self._filter_and_pick(sols, ref, allow_large)
        if best is None and local_refinement:
            best = self._try_local_refinement(cart_pose, ref, allow_large)
            if best is not None:
                self.stats['solved_local'] += 1
                self.last_target_joint = best.copy()
                return best.copy()
        if best is None:
            if np.asarray(cart_pose).shape == (4, 4):
                cart_pose = transform_to_rtde_pose(cart_pose)
            target = np.asarray(cart_pose, dtype=np.float64).reshape(-1)
            self.stats["no_feasible"] += 1
            self.last_rejection = {
                **(self._last_filter_rejection or {
                    "stage": "IK_CANDIDATES_REJECTED",
                    "detail": "all analytic candidates were rejected",
                }),
                "side": self.name,
                "target_tcp": target.tolist(),
                "reference_joint": ref.tolist(),
            }
            fallback = (self.last_target_joint if self.last_target_joint is not None
                        else current_joint)
            if self.verbose:
                self._print_hold_diag(cart_pose, sols, ref, current_joint)
            return fallback.copy()

        self.stats["solved"] += 1
        self.last_target_joint = best.copy()
        return best.copy()

    def _try_local_refinement(self, target, ref, allow_large):
        """Recover only a small continuous branch using this same UR5 FK.

        IKFast's analytic branch tests can omit the local solution even for a
        sub-millimetre EEF step. Damped Newton iterations from the committed
        reference are bounded to 0.1 rad total / 0.05 rad per iteration and six
        iterations. A distant, singular, unreachable or out-of-limit result
        still returns None. No target projection or command clipping is used.
        """
        if not callable(getattr(self.kin, 'forward', None)):
            return None
        goal = np.asarray(target, dtype=np.float64)
        if goal.shape != (4, 4):
            goal = rtde_pose_to_transform(goal)
        q = np.asarray(ref, dtype=np.float64).copy()
        if q.shape != (self.n_joints,) or not np.isfinite(q).all():
            return None

        def fk(joints):
            return np.asarray(self.kin.forward(joints.tolist()), dtype=np.float64).reshape(3, 4)

        for iteration in range(7):
            actual = fk(q)
            error = np.r_[goal[:3, 3]-actual[:, 3],
                          matrix_to_rotvec(goal[:3, :3] @ actual[:, :3].T)]
            if not np.isfinite(error).all():
                return None
            if iteration == 0 and (np.linalg.norm(error[:3]) > .01
                                    or np.linalg.norm(error[3:]) > np.deg2rad(5)):
                return None
            if np.max(np.abs(error)) <= 1e-8:
                rejection = self._last_filter_rejection
                best = self._filter_and_pick(q[None], ref, allow_large)
                if best is None:
                    self._last_filter_rejection = rejection
                return best
            if iteration == 6:
                return None
            jacobian = np.empty((6, self.n_joints))
            epsilon = 1e-6
            for axis in range(self.n_joints):
                perturbed = q.copy()
                perturbed[axis] += epsilon
                moved = fk(perturbed)
                jacobian[:3, axis] = (moved[:, 3]-actual[:, 3])/epsilon
                rotation = moved[:, :3] @ actual[:, :3].T
                jacobian[3:, axis] = np.array([rotation[2, 1]-rotation[1, 2],
                    rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])/(2*epsilon)
            try:
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-10*np.eye(6), error)
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(delta).all() or np.max(np.abs(delta)) > .05:
                return None
            q += delta
            if np.max(np.abs(q-ref)) > .1 or np.any(np.abs(q) > self.joint_pos_limit):
                return None
        return None

    def _filter_and_pick(self, sols: Any, ref: np.ndarray,
                         allow_large: bool = False) -> np.ndarray | None:
        """Limit filter -> (wrist-flip filter) -> weighted L1 nearest.
        Returns None when nothing survives.

        Tiered safety:
        - Wrist-flip filter: drop candidates whose per-axis delta exceeds
          flip_threshold, to avoid driving the arm into a flipped configuration
          (crash). This is the only hard reject.
        - Mild over-speed (delta between vel_limit and flip_threshold, i.e.
          normal fast motion) is NOT rejected; the downstream joint clip
          follows it at reduced speed. Responsive, and still safe.

        Teleop (allow_large=False): apply the wrist-flip filter.
        Homing (allow_large=True): skip it, allow a large move to the nearest
        solution. Both keep the limit filter and the unwrap+nearest pick.
        """
        sols = np.atleast_2d(np.asarray(sols, dtype=np.float64))
        self._last_filter_rejection = None
        within_pos = np.all(np.abs(sols) <= self.joint_pos_limit[None, :], axis=1)
        cand = sols[within_pos]
        if cand.shape[0] == 0:
            overflow = np.maximum(
                np.abs(sols) - self.joint_pos_limit[None, :], 0.0
            )
            closest_index = int(np.argmin(np.max(overflow, axis=1)))
            closest = sols[closest_index]
            bad_axes = np.where(
                np.abs(closest) > self.joint_pos_limit + 1.0e-9
            )[0]
            self._last_filter_rejection = {
                "stage": "JOINT_POSITION_LIMIT",
                "raw_candidates": int(sols.shape[0]),
                "in_position_limit": 0,
                "limit_rad": self.joint_pos_limit.tolist(),
                "closest_candidate": closest.tolist(),
                "over_limit_axes": [int(axis) for axis in bad_axes],
                "detail": "all IK candidates exceeded the absolute joint range",
            }
            return None

        # ==== SWITCH 2 applied here: enable_flip_hold ======================
        # OFF -> the flip filter is skipped entirely and the weighted-L1 pick
        # below may return a wrist-flipped solution. The v1 real-robot result of
        # relying on downstream clipping instead of this hold was a CRASH:
        # clipping limits step size but never direction, so the arm drives
        # steadily at the flipped target (9mm -> 37.8mm -> 186.2mm deviation,
        # protective stop in two frames). See the __init__ comment.
        if self.enable_flip_hold and not allow_large:
            delta = np.abs(cand - ref[None, :])
            not_flip = np.all(delta <= self.flip_threshold, axis=1)
            cand_nf = cand[not_flip]
            if cand_nf.shape[0] > 0:
                cand = cand_nf
                # J4 sign preference (wrist singularity avoidance). Measured on
                # the real robot: 88 of 98 left-arm stutter cases had J4 as the
                # dominant axis, with ref J4 consistently positive
                # (+1.50~+1.63) and the chosen solution's J4 negative
                # (-0.92~-0.77) -- crossing the J4=0 wrist singularity plane,
                # 2.4 rad in one frame. At J4=0, J3 and J5 are collinear and
                # the two sides are mirror configurations with the same TCP
                # pose, so following on one side alone is fine. This only
                # chooses among already-safe solutions and relaxes nothing.
                if self.prefer_same_wrist_sign and abs(ref[4]) > 1e-6:
                    same = cand[cand[:, 4] * ref[4] > 0]
                    if same.shape[0] > 0 and same.shape[0] < cand.shape[0]:
                        cand = same
                        self.stats["j4_sign_kept"] += 1
            elif self.soft_flip_limit > self.flip_threshold:
                # All over threshold, but some only slightly. DO NOT ENABLE:
                # this is the exact path that crashed the arm. Kept only so the
                # v1 behaviour is reproducible, gated behind soft_flip_limit
                # which defaults to 0 (i.e. unreachable).
                max_ax = np.max(delta, axis=1)
                soft = cand[max_ax <= self.soft_flip_limit]
                if soft.shape[0] == 0:
                    self.stats["flip_held"] += 1
                    closest_index = int(np.argmin(max_ax))
                    per_axis_delta = delta[closest_index]
                    self._last_filter_rejection = {
                        "stage": "WRIST_FLIP_HOLD",
                        "raw_candidates": int(sols.shape[0]),
                        "in_position_limit": int(cand.shape[0]),
                        "flip_threshold_rad": float(self.flip_threshold),
                        "minimum_max_delta_rad": float(max_ax[closest_index]),
                        "dominant_axis": int(np.argmax(per_axis_delta)),
                        "per_axis_delta_rad": per_axis_delta.tolist(),
                        "closest_candidate": cand[closest_index].tolist(),
                        "detail": "every in-range candidate crossed the flip threshold",
                    }
                    return None
                cand = soft
                self.stats["soft_flip"] += 1
                if self.verbose:
                    print(f"[IK:{self.name}] SOFT-PASS wrist flip "
                          f"min_jump={float(np.min(max_ax)):.3f} "
                          f"(flip_thr={self.flip_threshold:.2f} "
                          f"soft={self.soft_flip_limit:.2f}), relying on downstream clip")
            else:
                # Every candidate is a wrist flip -> hold (crash protection).
                self.stats["flip_held"] += 1
                max_delta = np.max(delta, axis=1)
                closest_index = int(np.argmin(max_delta))
                closest = cand[closest_index]
                per_axis_delta = delta[closest_index]
                dominant_axis = int(np.argmax(per_axis_delta))
                self._last_filter_rejection = {
                    "stage": "WRIST_FLIP_HOLD",
                    "raw_candidates": int(sols.shape[0]),
                    "in_position_limit": int(cand.shape[0]),
                    "flip_threshold_rad": float(self.flip_threshold),
                    "minimum_max_delta_rad": float(max_delta[closest_index]),
                    "dominant_axis": dominant_axis,
                    "per_axis_delta_rad": per_axis_delta.tolist(),
                    "closest_candidate": closest.tolist(),
                    "detail": "every in-range candidate crossed the flip threshold",
                }
                return None

        dist = np.sum(self.joint_weights[None, :] * np.abs(cand - ref[None, :]), axis=1)
        return cand[int(np.argmin(dist))]

    def _try_rtde_fallback(self, cart_pose: Any, ref: np.ndarray, current_joint: np.ndarray,
                           allow_large: bool = False) -> np.ndarray | None:
        """Numeric fallback when ikfast has no solution; result is unwrapped and
        run through the same limit (+ flip) checks.

        RTDE getInverseKinematics iterates from qnear so it usually returns the
        nearest solution without flipping; if it does flip, the checks return
        None (hold). No callback or a solver error also returns None.
        """
        if self.rtde_ik_fn is None:
            return None
        try:
            q = self.rtde_ik_fn(list(np.asarray(cart_pose, dtype=np.float64)),
                                list(np.asarray(current_joint, dtype=np.float64)))
        except Exception as exc:
            if self.verbose:
                print(f"[IK:{self.name}] rtde fallback raised: {exc}")
            return None
        if q is None or len(q) != self.n_joints:
            return None
        q = np.asarray(q, dtype=np.float64)
        q = self._unwrap_to_ref(q[None, :], ref)
        return self._filter_and_pick(q, ref, allow_large)

    def _print_hold_diag(self, cart_pose: Any, sols: np.ndarray, ref: np.ndarray,
                         current_joint: np.ndarray) -> None:
        """Diagnostics for a HOLD. Pure printing, no control effect.

        Distinguishes the two HOLD causes -- limit filter wiped everything vs
        flip filter wiped everything -- because v1 printed "all wrist-flip"
        unconditionally and that misdirected debugging. Also classifies whether
        the dominant over-threshold axis is a shoulder joint (J0/1/2) or a
        wrist joint (J3/4/5): wrist-dominant is a true flip and the HOLD is
        correct protection; shoulder-dominant may be a false positive on motion
        that should have been allowed.
        """
        cart_pose = np.asarray(cart_pose, dtype=np.float64)
        within_pos = np.all(np.abs(sols) <= self.joint_pos_limit[None, :], axis=1)
        cand = sols[within_pos]
        min_jump = (float(np.min(np.max(np.abs(cand - ref[None, :]), axis=1)))
                    if cand.shape[0] > 0 else float("nan"))
        ref_gap = float(np.max(np.abs(ref - current_joint)))
        reason = "all out-of-limit" if cand.shape[0] == 0 else "all wrist-flip"
        jgrp = "n/a"
        if cand.shape[0] > 0:
            dmin = cand[int(np.argmin(np.max(np.abs(cand - ref[None, :]), axis=1)))]
            dom_ax = int(np.argmax(np.abs(dmin - ref)))
            jgrp = f"J{dom_ax}({'wrist' if dom_ax >= 3 else 'shoulder'})"
        print(f"[IK:{self.name}] HOLD({reason}) | flip_thr={self.flip_threshold:.2f} "
              f"min_jump={min_jump:.3f} dominant={jgrp} ref_vs_cur={ref_gap:.3f} "
              f"n_sols={sols.shape[0]} n_within_pos={cand.shape[0]}")
        if cand.shape[0] == 0:
            over = np.abs(sols) > self.joint_pos_limit[None, :]
            bad_axes = sorted(set(np.where(over)[1].tolist()))
            print(f"    ref          = {np.round(ref, 3)}")
            print(f"    cur          = {np.round(current_joint, 3)}")
            print(f"    over-limit   = J{bad_axes} (limit=+-{self.joint_pos_limit[0]:.2f})")
            return
        # Flip HOLD: record which axis forced the jump AND the target
        # ORIENTATION. v1 only had translation diagnostics, so orientation was a
        # blind spot; the logged positions each solved fine on their own
        # (min_jump < 2.0), so it is the position+orientation COMBINATION that
        # triggers the flip. Orientation must be logged.
        delta = np.abs(cand - ref[None, :])
        max_ax = np.max(delta, axis=1)
        b = int(np.argmin(max_ax))
        dom = int(np.argmax(delta[b]))
        rotvec = cart_pose[3:]
        try:
            rpy = np.degrees(matrix_to_rotvec(rotvec_to_matrix(rotvec)))
        except Exception:
            rpy = np.full(3, np.nan)
        print(f"    target pos   = {np.round(cart_pose[:3], 3)} "
              f"|p|={np.linalg.norm(cart_pose[:3]):.3f}m")
        print(f"    target rot   = rotvec{np.round(rotvec, 3)} deg{np.round(rpy, 1)}")
        print(f"    best sol     = {np.round(cand[b], 3)}")
        print(f"    per-axis d   = {np.round(delta[b], 3)} dominant=J{dom} "
              f"({delta[b][dom]:.3f} > flip_thr={self.flip_threshold:.2f})")

    def reset(self, ref_joint: Any = None) -> None:
        """Reset the reference angle, e.g. after homing, so a stale
        last_target_joint is not reused."""
        self.last_target_joint = (np.asarray(ref_joint, dtype=np.float64)
                                  if ref_joint is not None else None)

    def report(self) -> str:
        s = self.stats
        calls = max(1, s["calls"])
        return ("[IK stats] calls={calls} solved={solved}({sr:.1f}%) "
                "rtde_fallback={rtde} local_refined={local} no_raw={no_raw} no_feasible={no_feasible} "
                "flip_held={flip}").format(
            calls=s["calls"], solved=s["solved"], sr=100.0 * s["solved"] / calls,
            rtde=s["solved_rtde"], local=s["solved_local"], no_raw=s["no_raw_solution"],
            no_feasible=s["no_feasible"], flip=s["flip_held"])
