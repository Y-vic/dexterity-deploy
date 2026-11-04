"""THE single forward/inverse kinematics entry point for UR5 + SharpA.

Teleop and deploy both go through this module. Do not add a second kinematics
implementation anywhere: two implementations drift apart under calibration
changes, and the observation path and the execution path would then disagree.
Preventing that drift is the reason this package exists.

Hardware is UR5, NOT UR5e. The compiled ikfast bindings in `ur_common/ikfastpy`
are generated for UR5 DH parameters (shoulder offset 0.089159 m); UR5e values
differ and would silently produce a wrong TCP. The bundled `.so` is built for
CPython 3.10 and will not import on another minor version.

Surface:
    UrCommonKinematics.arm_fk(joints)                 -> 4x4 TCP in UR base
    UrCommonKinematics.arm_fk_rtde(joints)            -> [xyz, rotvec] (6,)
    UrCommonKinematics.arm_fk_eef9(joints, side)      -> canonical column pose9
    UrCommonKinematics.arm_ik(target, near_joints)    -> 6 joint angles
    UrCommonKinematics.arm_ik_eef9(eef9, ...)         -> 6 joint angles
    UrCommonKinematics.hand_fingertip_positions(q44)  -> {'left':(5,3),'right':(5,3)}
    UrCommonKinematics.hand_fingertip_transforms(q44) -> {'left':(5,4,4),...}
    UrCommonKinematics.geometry                       -> UrSharpAWireGeometry
    UrCommonKinematics.jump_gate / joint_limiter      -> safety switches 1 and 3
Rotation conversions are re-exported from `.geometry`.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import numpy as np

from . import geometry as _geom
from .geometry import (  # re-exported convenience surface
    UR_ROOT_TO_PRETRAIN,
    UR_TCP_FROM_SHARPA_HAND,
    UrSharpAWireGeometry,
    column_pose9_to_transform,
    column_rot6d_to_matrix,
    matrix_to_column_rot6d,
    matrix_to_euler_xyz,
    matrix_to_quat,
    matrix_to_rotvec,
    matrix_to_row_rot6d,
    pose_mat,
    pose_vector,
    pose_vector_quat,
    quat_to_matrix,
    quat_to_row_rot6d,
    rotvec_to_matrix,
    row_pose9_to_transform,
    row_rot6d_to_matrix,
    row_rot6d_to_quat,
    rtde_pose_to_transform,
    transform_to_column_pose9,
    transform_to_row_pose9,
    transform_to_rtde_pose,
    validate_transform,
    xyz_euler_to_transform,
)
from .ik_solver import FastIKSolver, cart_to_matrix
from .safety import JointLimiter, JumpGate, SafetyConfig

UR5_NUM_JOINTS = 6
IKFAST_PYTHON_VERSION = (3, 10)

__all__ = [
    "UrCommonKinematics",
    "SafetyConfig",
    "FastIKSolver",
    "JumpGate",
    "JointLimiter",
    "UrSharpAWireGeometry",
    "load_ikfast",
    "ikfast_available",
    "hand_fk_available",
    "UR5_NUM_JOINTS",
    "IKFAST_PYTHON_VERSION",
]


class IkfastUnavailable(ImportError):
    """The compiled ikfast bindings could not be imported."""


def load_ikfast():
    """Import the bundled UR5 ikfast bindings.

    The `.so` is `ikfastpy.cpython-310-*.so`, i.e. built for CPython 3.10. On
    any other minor version the import fails; that is a real environment
    problem, so it is raised rather than papered over with a stub.
    """
    try:
        return importlib.import_module("ur_common.ikfastpy")
    except ImportError as exc:
        here = os.path.dirname(os.path.abspath(__file__))
        sofiles = []
        ikdir = os.path.join(here, "ikfastpy")
        if os.path.isdir(ikdir):
            sofiles = sorted(f for f in os.listdir(ikdir) if f.endswith(".so"))
        raise IkfastUnavailable(
            f"cannot import ur_common.ikfastpy ({exc}). Running CPython "
            f"{sys.version_info.major}.{sys.version_info.minor}; the bundled "
            f"bindings require {IKFAST_PYTHON_VERSION[0]}.{IKFAST_PYTHON_VERSION[1]}. "
            f"Found: {sofiles or 'no .so'}. Rebuild via ur_common/ikfastpy/setup.py "
            f"for this interpreter, or run under python3.10."
        ) from exc


def ikfast_available() -> bool:
    """True when the compiled UR5 IK bindings import on this interpreter."""
    try:
        load_ikfast()
    except ImportError:
        return False
    return True


def hand_fk_available() -> bool:
    """True when pinocchio is importable, i.e. hand fingertip FK can run."""
    try:
        importlib.import_module("pinocchio")
    except ImportError:
        return False
    return True


class UrCommonKinematics:
    """Shared UR5 + SharpA kinematics facade.

    :param config: SafetyConfig. Defaults to ALL THREE SWITCHES OFF. Pass
        `SafetyConfig.v1_defaults()` to restore the shipped v1 behaviour, or
        `SafetyConfig.from_env()` to honour the v1 environment variables.
    :param rtde_ik_fn: per-side numeric IK fallback, e.g.
        `{"left": rtde_c_left.getInverseKinematics, ...}`. Optional.
    :param lazy: when True (default) the ikfast module loads on first use, so
        pure-geometry users can construct this on any interpreter.
    """

    def __init__(self, config: SafetyConfig | None = None,
                 rtde_ik_fn: dict | None = None, lazy: bool = True) -> None:
        self.config = config if config is not None else SafetyConfig()
        self.geometry = UrSharpAWireGeometry()
        # SWITCH 1 lives here (input side, teleop wrist stream).
        self.jump_gate = JumpGate(self.config)
        # SWITCH 3 lives here (output side, joint command clipping).
        self.joint_limiter = JointLimiter(self.config)
        self._rtde_ik_fn = dict(rtde_ik_fn or {})
        self._kin = None
        self._solvers: dict[str, FastIKSolver] = {}
        if not lazy:
            self._ensure_kin()

    # ---- ikfast plumbing ----
    def _ensure_kin(self):
        if self._kin is None:
            module = load_ikfast()
            self._kin = module.PyKinematics()
            dof = int(self._kin.getDOF())
            if dof != UR5_NUM_JOINTS:
                raise RuntimeError(f"expected a 6-DOF UR5 chain, ikfast reports {dof}")
        return self._kin

    def solver(self, side: str = "left") -> FastIKSolver:
        """The per-side FastIKSolver. SWITCH 2 (enable_flip_hold) lives inside it."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if side not in self._solvers:
            self._solvers[side] = FastIKSolver(
                self._ensure_kin(),
                config=self.config,
                rtde_ik_fn=self._rtde_ik_fn.get(side),
                name=side,
            )
        return self._solvers[side]

    # ---- arm FK ----
    def arm_fk(self, joints: Any) -> np.ndarray:
        """UR5 joint angles (6,) -> 4x4 TCP transform in the UR BASE frame."""
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.shape[0] != UR5_NUM_JOINTS:
            raise ValueError(f"expected {UR5_NUM_JOINTS} joint angles, got {q.shape[0]}")
        flat = self._ensure_kin().forward(q.tolist())
        tcp = np.asarray(flat, dtype=np.float64).reshape(3, 4)
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = tcp
        return out

    def arm_fk_rtde(self, joints: Any) -> np.ndarray:
        """UR5 joint angles -> [x,y,z, rx,ry,rz(rotvec)] (6,), the RTDE convention."""
        t = self.arm_fk(joints)
        return np.concatenate((t[:3, 3], matrix_to_rotvec(t[:3, :3])))

    def arm_fk_eef9(self, joints: Any, side: str) -> np.ndarray:
        """UR5 joint angles -> canonical PND wrist EEF.

        Chain: ikfast TCP in UR base -> capture root -> SharpA hand mount ->
        canonical root. Pose9 uses absolute xyz plus the first two columns of
        root-from-wrist R, exactly as frozen by `sharpa_interface.sharpa`.
        """
        ur_base_tcp = self.arm_fk(joints)
        return self.geometry.ur_base_tcp_to_canonical_pose(ur_base_tcp, side)

    def arm_fk_wire_eef9(self, joints: Any, side: str) -> np.ndarray:
        """UR5 joint angles -> column-major policy wire EEF (9,)."""
        ur_base_tcp = self.arm_fk(joints)
        capture_tcp = self.geometry.ur_base_tcp_to_capture_tcp(ur_base_tcp, side)
        return self.geometry.capture_tcp_to_wire_pose(capture_tcp, side)

    # ---- arm IK ----
    def arm_ik(self, target: Any, near_joints: Any, side: str = "left",
               ref_joint: Any = None, allow_large: bool = False,
               local_refinement: bool = False) -> np.ndarray:
        """Target TCP pose + seed joints -> UR5 joint angles (6,).

        :param target: 4x4 transform, or [xyz, rotvec] (6,), in the UR BASE frame
        :param near_joints: measured/seed joint angles; also the fallback on hold
        :param ref_joint: selection reference; defaults to the solver's previous
            commanded target, then near_joints
        :param allow_large: homing mode; skips the wrist-flip filter
        """
        target = np.asarray(target, dtype=np.float64)
        if target.shape == (4, 4):
            cart = validate_transform(target)
        elif target.shape == (6,):
            cart = target
        else:
            raise ValueError("target must be a 4x4 transform or a 6-vector [xyz, rotvec]")
        return self.solver(side).solve(cart, near_joints, ref_joint=ref_joint,
                                       allow_large=allow_large, local_refinement=local_refinement)

    def arm_ik_eef9(self, eef9: Any, near_joints: Any, side: str,
                    ref_joint: Any = None, allow_large: bool = False,
                    local_refinement: bool = False) -> np.ndarray:
        """Canonical PND wrist EEF -> UR5 joint angles (6,).

        Exact inverse of `arm_fk_eef9`, so the execution path consumes what the
        observation path produced.
        """
        ur_base_tcp = self.geometry.canonical_pose_to_ur_base_tcp(eef9, side)
        return self.arm_ik(ur_base_tcp, near_joints, side=side,
                           ref_joint=ref_joint, allow_large=allow_large, local_refinement=local_refinement)

    def arm_ik_wire_eef9(self, eef9: Any, near_joints: Any, side: str,
                         ref_joint: Any = None, allow_large: bool = False) -> np.ndarray:
        """Column-major policy wire EEF (9,) -> UR5 joint angles (6,)."""
        capture_tcp = self.geometry.wire_pose_to_capture_tcp(eef9, side)
        ur_base_tcp = self.geometry.capture_tcp_to_ur_base_tcp(capture_tcp, side)
        return self.arm_ik(ur_base_tcp, near_joints, side=side,
                           ref_joint=ref_joint, allow_large=allow_large)

    # ---- hand FK ----
    def hand_fingertip_positions(self, hand_joints: Any) -> dict:
        """44D hand joints -> {'left': (5,3), 'right': (5,3)} fingertip positions.

        Interface order: left 22 then right 22, thumb->pinky within each hand.
        Delegates to the ported `sharpa_fk`, cached Pinocchio models included.
        """
        from . import sharpa_fk

        left, right = sharpa_fk.split_interface_hand_joints(hand_joints)
        flat = sharpa_fk.hand_joints_to_fingertip_poses(left, right)
        arr = np.asarray(flat, dtype=np.float64).reshape(2, 5, 3)
        return {"left": arr[0], "right": arr[1]}

    def hand_fingertip_transforms(self, hand_joints: Any) -> dict:
        """44D hand joints -> {'left': (5,4,4), 'right': (5,4,4)} fingertip poses."""
        from . import sharpa_fk

        left, right = sharpa_fk.split_interface_hand_joints(hand_joints)
        return sharpa_fk.hand_joints_to_fingertip_transforms(left, right)

    # ---- geometry passthrough (named for readability at call sites) ----
    def ur_base_to_capture_root(self, ur_base_tcp: Any, side: str) -> np.ndarray:
        return self.geometry.ur_base_tcp_to_capture_tcp(ur_base_tcp, side)

    def capture_root_to_ur_base(self, capture_tcp: Any, side: str) -> np.ndarray:
        return self.geometry.capture_tcp_to_ur_base_tcp(capture_tcp, side)

    def capture_root_to_model(self, capture_tcp: Any, side: str) -> np.ndarray:
        return self.geometry.capture_tcp_to_model_pose(capture_tcp, side)

    def model_to_capture_root(self, eef9: Any, side: str) -> np.ndarray:
        return self.geometry.model_pose_to_capture_tcp(eef9, side)

    # ---- safety switch passthrough ----
    def gate_wrist(self, wrist: Any, side: str) -> tuple[np.ndarray, bool]:
        """SWITCH 1. Returns (accepted_wrist, rejected). No-op when disabled."""
        return self.jump_gate.filter(wrist, side)

    def limit_joints(self, target_joint: Any, current_joint: Any) -> np.ndarray:
        """SWITCH 3. limit_joint_vel then limit_joint_pos. No-op when disabled."""
        return self.joint_limiter.apply(target_joint, current_joint)

    def reset(self, side: str | None = None, ref_joint: Any = None) -> None:
        """Clear per-frame state: jump gate history and IK reference angles."""
        self.jump_gate.reset(side)
        sides = ("left", "right") if side is None else (side,)
        for s in sides:
            if s in self._solvers:
                self._solvers[s].reset(ref_joint)
