"""Numeric bridge between the canonical SharpA interface and MoT v5 (56D).

This module is the ONLY place the two representations meet, and it is
deliberately free of torch, ROS and any network code so it can be tested
offline against the training code it mirrors.

Two representations, two different things -- conflating them is the whole
class of bug this file exists to prevent:

    interface boundary   absolute pose9 per wrist, ``[x,y,z,r00,r10,r20,r01,r11,r21]``
            i.e. translation then the first two COLUMNS of the
    canonical-root-from-wrist rotation (rot6d).
    Frozen by ``sharpa_interface.sharpa``.

    MoT v5 boundary      relative 6D per wrist, ``[tx,ty,tz | rotvec(3)]`` measured
    in the wrist frame at an anchor, plus 22 ABSOLUTE hand
            qpos per hand. 56D total.
            Frozen by ``hand_action_repr`` (REPR_VERSION 5).

The wrist blocks are relative and the hand blocks are not. A joint angle has no
reference frame for an anchor to change, so it is copied through on both sides;
only the 24 wrist dims are ever transformed here.

WHY RELATIVE SURVIVES THE EMBODIMENT CHANGE. The model was trained on Vega-1
arms and this robot has UR5 arms. A constant ``A`` left-multiplied onto every
pose cancels in ``R_a^T R_i``, so the two base frames are the same quantity to
the policy. What does NOT cancel is a constant right-multiplied onto the TOOL
frame: that conjugates the relative transform, preserving rotation angle while
rotating the translation direction. See ``TOOL_ROLL_DEG`` below.

The functions here mirror ``hand_action_repr`` exactly, including its float32
casts and its small-angle guards. Where a formula is duplicated rather than
imported, it is because this runs on the workstation-facing side of the server
where the training package is not importable; ``test_repr_bridge.py`` asserts
the two agree on real data.
"""

from __future__ import annotations

import numpy as np


# --- MoT v5 layout, mirroring hand_action_repr.SLICES exactly -------------
WRIST_DIM = 6
HAND_DIM = 22
ACTION_DIM = 2 * (WRIST_DIM + HAND_DIM)  # 56
STATE_DIM = ACTION_DIM

SLICES = {
    "left_wrist_trans": slice(0, 3),
    "left_wrist_rotvec": slice(3, 6),
    "right_wrist_trans": slice(6, 9),
    "right_wrist_rotvec": slice(9, 12),
    "left_hand_joints": slice(12, 34),
    "right_hand_joints": slice(34, 56),
}
SIDES = ("left", "right")

# --- interface layout -----------------------------------------------------
POSE9_DIM = 9

# Training-time schedule. These are not free parameters at serving time: the
# action head emits exactly ``ACTION_HORIZON`` steps and the step embedding is
# sized for it, and a state row anchored at a different offset is a differently
# scaled quantity read through the same weights.
ACTION_HORIZON = 10
ACTION_STRIDE = 3
# FOUR rows, read from THIS run's hydra config -- NOT the dataclass default of
# (0,), which an earlier version of this file assumed. The expert rejects a
# wrong count at inference: each row owns a learned position, so a mismatch
# conditions it on a layout it never trained on, and the sampled chunk would
# still be well-shaped, so nothing downstream would notice.
#
# Oldest first, spanning 3..12 frames back from the window start, every row
# measured against the SINGLE anchor at start - STATE_ANCHOR_OFFSET.
STATE_HISTORY_OFFSETS = (9, 6, 3, 0)
STATE_ANCHOR_OFFSET = 12
TRAIN_FPS = 30.0

# The unresolved embodiment risk, exposed as a knob rather than baked in.
#
# The T-Rex corpus was converted with ``tool_frame=ur5_aligned``, which rotates
# the Vega ee frame so approach lands on +X as the UR5 recorded tool has it.
# But its ``conversion_info.json`` records ``tool_roll_deg {left: 0, right: 0}``
# while the converter's own default is ``{left: 180, right: 0}`` -- the file's
# comment states L_ee and R_ee differ by exactly 180 deg about the approach
# axis, so one of the two arms is rolled relative to the convention the UR5
# uses on both. That file also carries ``tool_roll_verified: false``.
#
# A roll about approach does not cancel under relative chunking (it is a
# right-multiplied constant), and it is invisible offline: both values produce
# finite, plausible trajectories. It can only be settled on hardware, in
# preview, by watching whether the left arm's translation direction matches
# intent. Default 0/0 reproduces what the checkpoint was actually trained on.
#
# SETTLED for the ur_unscrew_cap corpus (checked 2026-09-15). That corpus is NOT
# converted T-Rex data: its meta/info.json records
#
#     robot_type                    = ur_sharpa
#     wrist_source                  = ur/physical_12.npz:canonical_wrist_pose9
#     wrist_transform_at_conversion = none
#
# so the poses were recorded on this robot, in this robot's own tool frame, with
# no alignment step to disagree about. 0/0 is right by construction there, and the
# risk above applies only to a checkpoint trained on the converted Vega corpus.
TOOL_ROLL_DEG = {"left": 0.0, "right": 0.0}


def _as_f32(value, name: str, trailing: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != trailing:
        raise ValueError(f"{name} must have trailing dim {trailing}, got {array.shape}")
    return array


# --- rotation helpers: byte-for-byte mirrors of hand_action_repr ----------
def rot6d_to_rotation(rot6d: np.ndarray) -> np.ndarray:
    """Recover a rotation from its first-two-COLUMNS 6D form via Gram-Schmidt."""
    rot6d = _as_f32(rot6d, "rot6d", 6)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    a2_proj = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.clip(np.linalg.norm(a2_proj, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """First two COLUMNS of a rotation, flattened. Inverse of the above."""
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected trailing dims (3, 3), got {rotation.shape}")
    return rotation[..., :, :2].swapaxes(-1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Log map. Small-angle safe, not pi-safe -- as in training.

    The relative rotations this sees span p99 70 deg over 12 frames, so the
    theta -> pi branch a general implementation needs is unreachable. The guard
    is for theta -> 0, where sin(theta) cancels against theta.
    """
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected trailing dims (3, 3), got {rotation.shape}")
    skew = np.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        axis=-1,
    )
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    theta = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    small = theta < 1e-6
    denom = np.where(small, 1.0, 2.0 * np.sin(theta))
    factor = np.where(small, 0.5, theta / denom)
    return (skew * factor[..., None]).astype(np.float32)


def rotvec_to_rotation(rotvec: np.ndarray) -> np.ndarray:
    """Exp map. Inverse of ``rotation_to_rotvec``."""
    rotvec = _as_f32(rotvec, "rotvec", 3)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = rotvec / np.clip(theta, 1e-8, None)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    cross = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    angle = theta[..., None]
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), cross.shape)
    rotation = (
        eye
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.einsum("...ij,...jk->...ik", cross, cross)
    )
    return rotation.astype(np.float32)


# --- interface pose9 <-> (translation, rotation) --------------------------
def pose9_to_parts(pose9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical ``[...,9]`` -> translation ``[...,3]``, rotation ``[...,3,3]``.

    The interface order is ``x,y,z,r00,r10,r20,r01,r11,r21`` -- COLUMN major
    within the rot6d block, which is what ``rot6d_to_rotation`` expects. Reading
    it as rows would transpose every rotation and still produce a valid-looking
    orthonormal matrix, so this is asserted in the tests rather than trusted.
    """
    pose9 = _as_f32(pose9, "pose9", POSE9_DIM)
    return pose9[..., :3], rot6d_to_rotation(pose9[..., 3:])


def parts_to_pose9(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Inverse of ``pose9_to_parts``."""
    translation = _as_f32(translation, "translation", 3)
    return np.concatenate(
        [translation, rotation_to_rot6d(rotation)], axis=-1
    ).astype(np.float32)


def tool_roll_rotation(side: str) -> np.ndarray:
    """Right-multiplied tool correction ``C`` for one side, from TOOL_ROLL_DEG.

    Applied as ``R_corrected = R @ C``. At the default 0 deg this is identity
    and the whole thing is a no-op; it exists so the unresolved left-arm roll
    can be swept on hardware without touching any other code path.
    """
    theta = np.radians(float(TOOL_ROLL_DEG[side]))
    if theta == 0.0:
        return np.eye(3, dtype=np.float32)
    c, s = np.cos(theta), np.sin(theta)
    # Roll about the approach axis, which is +X in the UR5-aligned convention.
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32
    )


# --- absolute pose9 -> relative v5 wrist block ---------------------------
def wrist_relative_6d(
    anchor_translation: np.ndarray,
    anchor_rotation: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """``[R_a^T (t_i - t_a) | log(R_a^T R_i)]``, translation first.

    Mirrors ``hand_action_repr.wrist_pose_relative_6d``, but takes poses
    directly instead of indexing whole-episode arrays: at serving time there is
    no episode, only a ring buffer of what has been observed.

    Anchor-invariant by construction, which is why a UR5 base and a Vega base
    are the same quantity here: ``(A R_a)^T (A R_i) = R_a^T R_i`` and
    ``(A R_a)^T A (t_i - t_a) = R_a^T (t_i - t_a)``.
    """
    anchor_rotation_t = np.swapaxes(anchor_rotation, -1, -2)
    relative_translation = np.einsum(
    "...ij,...j->...i", anchor_rotation_t, translation - anchor_translation
    )
    relative_rotation = np.einsum("...ij,...jk->...ik", anchor_rotation_t, rotation)
    return np.concatenate(
        [relative_translation, rotation_to_rotvec(relative_rotation)], axis=-1
    ).astype(np.float32)


def wrist_absolute_from_relative(
    anchor_translation: np.ndarray,
    anchor_rotation: np.ndarray,
    relative6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a relative 6D block back onto an anchor pose.

    ``p = p_a + R_a @ dt`` and ``R = R_a @ exp(w)`` -- the exact inverse of
    ``wrist_relative_6d``, and the reason the representation is executable at
    all: the policy predicts from where the robot actually is, with no
    dependence on where an episode began or which corpus it came from.
    """
    relative6d = _as_f32(relative6d, "relative6d", WRIST_DIM)
    delta_translation = relative6d[..., :3]
    delta_rotation = rotvec_to_rotation(relative6d[..., 3:])
    translation = anchor_translation + np.einsum(
        "...ij,...j->...i", anchor_rotation, delta_translation
    )
    rotation = np.einsum("...ij,...jk->...ik", anchor_rotation, delta_rotation)
    return translation.astype(np.float32), rotation.astype(np.float32)


# --- whole-vector assembly ------------------------------------------------
def build_state_vector(
    anchor_pose9: dict[str, np.ndarray],
    current_pose9: dict[str, np.ndarray],
    hand_qpos: dict[str, np.ndarray],
) -> np.ndarray:
    """ONE 56D state row: wrists relative to ``anchor``, hands absolute.

    This is a single row. The expert wants ``len(STATE_HISTORY_OFFSETS)`` of
    them -- use ``build_state_rows`` for the tensor the model actually consumes;
    this stays public because it is the per-row unit both that function and the
    tests are written against.

    ``anchor_pose9`` is the pose ``STATE_ANCHOR_OFFSET`` frames back, which the
    server keeps in a ring buffer because the interface only ever sends the
    current frame.

    NOTE the state uses its OWN statistics, not the action's: since v5 the two
    are different quantities (a 12-frame backward span against a 30-frame
    forward one) and normalizing one with the other's scales is silent.
    """
    row = np.zeros(STATE_DIM, dtype=np.float32)
    for side in SIDES:
        anchor_t, anchor_R = pose9_to_parts(anchor_pose9[side])
        current_t, current_R = pose9_to_parts(current_pose9[side])
        correction = tool_roll_rotation(side)
        block = wrist_relative_6d(
            anchor_t, anchor_R @ correction, current_t, current_R @ correction
        )
        row[SLICES[f"{side}_wrist_trans"]] = block[:3]
        row[SLICES[f"{side}_wrist_rotvec"]] = block[3:]
    for side in SIDES:
        joints = _as_f32(hand_qpos[side], f"{side} hand qpos", HAND_DIM)
        row[SLICES[f"{side}_hand_joints"]] = joints
    return row


def build_state_rows(
    anchor_pose9: dict[str, np.ndarray],
    frames: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> np.ndarray:
    """``[len(STATE_HISTORY_OFFSETS), 56]``, oldest row first.

    ``frames`` is one ``(pose9_by_side, hand_qpos_by_side)`` pair per offset in
    ``STATE_HISTORY_OFFSETS`` order, i.e. oldest first. Every row shares the ONE
    anchor -- that is what makes the rows comparable, since the difference
    between two rows is then itself a physical displacement. Anchoring each row
    at its own frame would make every row identically zero.

    Mirrors ``hand_action_repr.build_state_rows``, which indexes whole-episode
    arrays; at serving time there is no episode, so the caller supplies the
    frames it has buffered.
    """
    if len(frames) != len(STATE_HISTORY_OFFSETS):
        raise ValueError(
            f"expected {len(STATE_HISTORY_OFFSETS)} frames for offsets "
            f"{STATE_HISTORY_OFFSETS}, got {len(frames)}"
        )
    rows = np.stack(
        [build_state_vector(anchor_pose9, pose, hand) for pose, hand in frames]
    ).astype(np.float32)
    if rows.shape != (len(STATE_HISTORY_OFFSETS), STATE_DIM):
        raise ValueError(f"expected {(len(STATE_HISTORY_OFFSETS), STATE_DIM)}, got {rows.shape}")
    return rows


def decode_action_chunk(
    chunk56: np.ndarray, anchor_pose9: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """``[horizon, 56]`` in v5 form -> absolute interface arrays.

    The anchor is the CURRENT measured pose, matching how the chunk was built
    in training (``build_action_chunk`` anchors every step on ``start``). Step
    ``k`` therefore lands ``(k+1) * ACTION_STRIDE`` frames ahead of now.

    Returns pose9 arrays ``[horizon, 9]`` per wrist and qpos ``[horizon, 22]``
    per hand, ready to be packed into the interface action dict.
    """
    chunk56 = _as_f32(chunk56, "chunk56", ACTION_DIM)
    if chunk56.ndim != 2:
        raise ValueError(f"Expected [horizon, {ACTION_DIM}], got {chunk56.shape}")
    horizon = chunk56.shape[0]
    out: dict[str, np.ndarray] = {}
    for side in SIDES:
        anchor_t, anchor_R = pose9_to_parts(anchor_pose9[side])
        correction = tool_roll_rotation(side)
        anchor_R_tool = anchor_R @ correction
        block = np.concatenate(
            [
                chunk56[:, SLICES[f"{side}_wrist_trans"]],
                chunk56[:, SLICES[f"{side}_wrist_rotvec"]],
            ],
            axis=-1,
        )
        translation, rotation = wrist_absolute_from_relative(
            np.broadcast_to(anchor_t, (horizon, 3)),
            np.broadcast_to(anchor_R_tool, (horizon, 3, 3)),
            block,
        )
        # Undo the tool correction so what leaves is in the interface's frame.
        rotation = np.einsum("...ij,jk->...ik", rotation, correction.T)
        out[f"{side}_eef"] = parts_to_pose9(translation, rotation)
        out[f"{side}_hand"] = chunk56[:, SLICES[f"{side}_hand_joints"]].copy()
    return out
