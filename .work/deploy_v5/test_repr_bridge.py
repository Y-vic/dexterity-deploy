"""Offline numeric tests for the v5 <-> interface bridge.

No GPU, no video, no network. Everything here is checkable against the training
code's own definitions, which is the point: the failures this guards against
(a transposed rotation, a swapped block, a relative chunk composed onto the
wrong anchor) all produce finite, plausible-looking numbers downstream.
"""

import numpy as np

import repr_bridge as rb


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation, via QR with the sign fix for det=+1."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def random_pose9(rng: np.random.Generator) -> np.ndarray:
    return rb.parts_to_pose9(
        rng.normal(scale=0.4, size=3).astype(np.float32), random_rotation(rng)
    )


# --- rotation primitives -------------------------------------------------
def test_rot6d_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rotation = random_rotation(rng)
        recovered = rb.rot6d_to_rotation(rb.rotation_to_rot6d(rotation))
        assert np.allclose(recovered, rotation, atol=1e-6)


def test_rotvec_roundtrip_over_the_trained_range():
    """p99 of a relative wrist rotation is ~70 deg; test well past it."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, np.radians(120.0))
        rotvec = (axis * angle).astype(np.float32)
        recovered = rb.rotation_to_rotvec(rb.rotvec_to_rotation(rotvec))
        assert np.allclose(recovered, rotvec, atol=1e-5)


def test_rotvec_is_small_angle_safe():
    """theta -> 0 is the branch the training implementation guards."""
    for scale in (0.0, 1e-9, 1e-7, 1e-5):
        rotvec = np.array([scale, -scale, scale], dtype=np.float32)
        recovered = rb.rotation_to_rotvec(rb.rotvec_to_rotation(rotvec))
        assert np.all(np.isfinite(recovered))
        assert np.allclose(recovered, rotvec, atol=1e-6)


def test_rot6d_block_is_columns_not_rows():
    """Guards the one error that still yields an orthonormal matrix.

    Reading the interface's r00,r10,r20,r01,r11,r21 as ROWS transposes every
    rotation. The result is still a valid rotation, so nothing downstream
    errors -- it just means something different.
    """
    rng = np.random.default_rng(2)
    rotation = random_rotation(rng)
    pose9 = rb.parts_to_pose9(np.zeros(3, dtype=np.float32), rotation)
    assert np.allclose(pose9[3:6], rotation[:, 0], atol=1e-6)
    assert np.allclose(pose9[6:9], rotation[:, 1], atol=1e-6)


def test_pose9_roundtrip():
    rng = np.random.default_rng(20)
    for _ in range(100):
        pose9 = random_pose9(rng)
        translation, rotation = rb.pose9_to_parts(pose9)
        assert np.allclose(rb.parts_to_pose9(translation, rotation), pose9, atol=1e-6)


# --- relative composition -------------------------------------------------
def relative_rotation_within(rng, max_deg: float):
    """An anchor and a target whose RELATIVE rotation is bounded.

    Two independent uniform rotations differ by a median ~127 deg, which is far
    outside anything this representation encodes: a relative wrist rotation
    measures p99 70 deg over 12 frames. Sampling uniformly would test the
    log map's unguarded theta -> pi branch, where float32 costs ~1e-4 rad --
    a real limit of the training implementation, but not one serving reaches.
    """
    anchor_R = random_rotation(rng)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, np.radians(max_deg))
    delta = rb.rotvec_to_rotation((axis * angle).astype(np.float32))
    anchor_t = rng.normal(scale=0.4, size=3).astype(np.float32)
    target_t = anchor_t + rng.normal(scale=0.05, size=3).astype(np.float32)
    return anchor_t, anchor_R, target_t, (anchor_R @ delta).astype(np.float32)


def test_relative_then_absolute_is_identity():
    """Exact to float32 over the range the representation actually encodes."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        anchor_t, anchor_R, target_t, target_R = relative_rotation_within(rng, 90.0)
        block = rb.wrist_relative_6d(anchor_t, anchor_R, target_t, target_R)
        got_t, got_R = rb.wrist_absolute_from_relative(anchor_t, anchor_R, block)
        assert np.allclose(got_t, target_t, atol=1e-6)
        assert np.allclose(got_R, target_R, atol=1e-6)


def test_log_map_degrades_only_near_pi():
    """Pins the known limit so it stays known.

    ``rotation_to_rotvec`` is documented small-angle safe but NOT pi-safe, and
    this bridge inherits that deliberately (it mirrors training byte for byte).
    Serving never gets near pi -- a 3-frame step is p99.9 35 deg. If a future
    change makes the near-pi case accurate, this test failing is the signal to
    check whether it still matches what the model was trained under.
    """
    rng = np.random.default_rng(30)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    anchor_R = random_rotation(rng)
    zero = np.zeros(3, dtype=np.float32)

    # Measured on this implementation, 2026-09-11. The serving range is the top
    # two rows; the bottom two are the documented degradation, recorded so a
    # future change to the log map is noticed rather than absorbed.
    #
    #    35 deg (p99.9 of a 3-frame step) .. 6e-08
    #    90 deg ........................... 1e-07
    #   170 deg ........................... 4e-06
    #   179.5 deg ......................... 1e-02
    measured = ((35.0, 1e-7), (90.0, 1e-6), (170.0, 1e-5), (179.5, 2e-2))
    for angle_deg, tolerance in measured:
        delta = rb.rotvec_to_rotation((axis * np.radians(angle_deg)).astype(np.float32))
        target_R = (anchor_R @ delta).astype(np.float32)
        block = rb.wrist_relative_6d(zero, anchor_R, zero, target_R)
        _, got_R = rb.wrist_absolute_from_relative(zero, anchor_R, block)
        error = float(np.abs(got_R - target_R).max())
        assert error < tolerance, f"{angle_deg} deg: {error:.2e} >= {tolerance:.0e}"

    # And the claim that matters: the serving range is orders of magnitude
    # better than the near-pi case, not merely "within tolerance".
    def error_at(angle_deg: float) -> float:
        delta = rb.rotvec_to_rotation((axis * np.radians(angle_deg)).astype(np.float32))
        target_R = (anchor_R @ delta).astype(np.float32)
        block = rb.wrist_relative_6d(zero, anchor_R, zero, target_R)
        _, got_R = rb.wrist_absolute_from_relative(zero, anchor_R, block)
        return float(np.abs(got_R - target_R).max())

    assert error_at(35.0) < error_at(179.5) / 1000.0


def test_relative_block_is_invariant_to_the_base_frame():
    """The property that lets a Vega-trained policy drive a UR5.

    A constant A applied to BOTH anchor and target must leave the relative
    block unchanged. If this ever fails, the embodiment gap is real and the
    whole approach is unsound -- so it is tested, not assumed.
    """
    rng = np.random.default_rng(4)
    for _ in range(100):
        anchor_t, anchor_R = rb.pose9_to_parts(random_pose9(rng))
        target_t, target_R = rb.pose9_to_parts(random_pose9(rng))
        offset_t, offset_R = rb.pose9_to_parts(random_pose9(rng))

        plain = rb.wrist_relative_6d(anchor_t, anchor_R, target_t, target_R)
        moved = rb.wrist_relative_6d(
            offset_R @ anchor_t + offset_t,
            offset_R @ anchor_R,
            offset_R @ target_t + offset_t,
            offset_R @ target_R,
        )
        assert np.allclose(plain, moved, atol=1e-4)


def test_zero_relative_block_reproduces_the_anchor():
    rng = np.random.default_rng(5)
    anchor_t, anchor_R = rb.pose9_to_parts(random_pose9(rng))
    got_t, got_R = rb.wrist_absolute_from_relative(
        anchor_t, anchor_R, np.zeros(6, dtype=np.float32)
    )
    assert np.allclose(got_t, anchor_t, atol=1e-6)
    assert np.allclose(got_R, anchor_R, atol=1e-6)


def test_a_tool_roll_does_not_cancel_under_relative_chunking():
    """The measured embodiment risk, asserted as a property.

    A roll about the approach axis is right-multiplied, so unlike a base frame
    it survives relativization and rotates the translation direction. This
    documents that TOOL_ROLL_DEG matters; if it ever passes with equal blocks,
    the knob has stopped doing anything.
    """
    rng = np.random.default_rng(6)
    anchor_t, anchor_R = rb.pose9_to_parts(random_pose9(rng))
    target_t, target_R = rb.pose9_to_parts(random_pose9(rng))
    theta = np.radians(180.0)
    c, s = np.cos(theta), np.sin(theta)
    roll = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

    plain = rb.wrist_relative_6d(anchor_t, anchor_R, target_t, target_R)
    rolled = rb.wrist_relative_6d(anchor_t, anchor_R @ roll, target_t, target_R @ roll)
    assert not np.allclose(plain[:3], rolled[:3], atol=1e-3)
    # The ANGLE is preserved, only the direction turns -- the precise claim in
    # convert_trex_joint_to_eef.py. Asserting both halves keeps this honest.
    assert np.allclose(
        np.linalg.norm(plain[3:]), np.linalg.norm(rolled[3:]), atol=1e-4
    )


# --- whole-vector assembly ------------------------------------------------
def test_state_vector_layout_and_hand_passthrough():
    """Hands must be copied verbatim; only the 24 wrist dims are transformed."""
    rng = np.random.default_rng(7)
    anchor = {side: random_pose9(rng) for side in rb.SIDES}
    current = {side: random_pose9(rng) for side in rb.SIDES}
    hands = {
        side: rng.uniform(-1.0, 1.0, size=rb.HAND_DIM).astype(np.float32)
        for side in rb.SIDES
    }
    row = rb.build_state_vector(anchor, current, hands)

    assert row.shape == (rb.STATE_DIM,)
    assert row.dtype == np.float32
    for side in rb.SIDES:
        assert np.allclose(
            row[rb.SLICES[f"{side}_hand_joints"]], hands[side], atol=0
        ), f"{side} hand block must be a byte-for-byte copy"


def test_state_vector_is_zero_when_current_equals_anchor():
    """No motion observed yet must read as no displacement, not as noise."""
    rng = np.random.default_rng(8)
    pose = {side: random_pose9(rng) for side in rb.SIDES}
    hands = {side: np.zeros(rb.HAND_DIM, dtype=np.float32) for side in rb.SIDES}
    row = rb.build_state_vector(pose, pose, hands)
    assert np.allclose(row[:12], 0.0, atol=1e-6)


def test_decode_action_chunk_shapes_and_hand_passthrough():
    rng = np.random.default_rng(9)
    horizon = rb.ACTION_HORIZON
    chunk = rng.normal(scale=0.02, size=(horizon, rb.ACTION_DIM)).astype(np.float32)
    anchor = {side: random_pose9(rng) for side in rb.SIDES}
    out = rb.decode_action_chunk(chunk, anchor)

    for side in rb.SIDES:
        assert out[f"{side}_eef"].shape == (horizon, rb.POSE9_DIM)
        assert out[f"{side}_hand"].shape == (horizon, rb.HAND_DIM)
        assert np.allclose(
            out[f"{side}_hand"], chunk[:, rb.SLICES[f"{side}_hand_joints"]], atol=0
        )


def test_decoded_pose9_rotations_are_orthonormal():
    """A decoded pose must be a valid rotation or IK will read garbage."""
    rng = np.random.default_rng(10)
    chunk = rng.normal(scale=0.05, size=(rb.ACTION_HORIZON, rb.ACTION_DIM))
    anchor = {side: random_pose9(rng) for side in rb.SIDES}
    out = rb.decode_action_chunk(chunk.astype(np.float32), anchor)

    for side in rb.SIDES:
        for pose9 in out[f"{side}_eef"]:
            _, rotation = rb.pose9_to_parts(pose9)
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)


def test_zero_chunk_holds_position_at_the_anchor():
    """The safety-relevant degenerate case: predict nothing, move nowhere."""
    rng = np.random.default_rng(11)
    anchor = {side: random_pose9(rng) for side in rb.SIDES}
    chunk = np.zeros((rb.ACTION_HORIZON, rb.ACTION_DIM), dtype=np.float32)
    out = rb.decode_action_chunk(chunk, anchor)

    for side in rb.SIDES:
        for pose9 in out[f"{side}_eef"]:
            assert np.allclose(pose9, anchor[side], atol=1e-5)


def test_state_and_decode_are_mutually_inverse():
    """Encode a known displacement, decode it, land back on the same pose.

    This is the round trip that matters end to end: it exercises the tool
    correction in both directions, so a correction applied on only one side
    would show up here even though each direction alone looks self-consistent.
    """
    rng = np.random.default_rng(12)
    for _ in range(50):
        anchor = {side: random_pose9(rng) for side in rb.SIDES}
        target = {side: random_pose9(rng) for side in rb.SIDES}
        hands = {side: np.zeros(rb.HAND_DIM, dtype=np.float32) for side in rb.SIDES}

        row = rb.build_state_vector(anchor, target, hands)
        out = rb.decode_action_chunk(row[None, :], anchor)
        for side in rb.SIDES:
            assert np.allclose(out[f"{side}_eef"][0], target[side], atol=1e-4)


def test_layout_matches_the_training_constants():
    """Guards against drift from hand_action_repr.SLICES."""
    assert rb.ACTION_DIM == 56
    assert rb.STATE_DIM == 56
    assert rb.SLICES == {
        "left_wrist_trans": slice(0, 3),
        "left_wrist_rotvec": slice(3, 6),
        "right_wrist_trans": slice(6, 9),
        "right_wrist_rotvec": slice(9, 12),
        "left_hand_joints": slice(12, 34),
        "right_hand_joints": slice(34, 56),
    }
    assert rb.ACTION_HORIZON == 10
    assert rb.ACTION_STRIDE == 3
    assert rb.STATE_ANCHOR_OFFSET == 12
    assert rb.STATE_ANCHOR_OFFSET > max(rb.STATE_HISTORY_OFFSETS)
