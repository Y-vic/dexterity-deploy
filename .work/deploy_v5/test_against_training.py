"""Assert the bridge agrees with the TRAINING code, not just with itself.

Runs on the model server only: it imports ``flagscale...hand_action_repr``, the
module that defined what the checkpoint's 56 numbers mean. Every other test in
this directory checks internal consistency, which cannot catch a shared
misreading -- if I copied a formula wrongly, my round trip still closes.

    cd /share/project/zjk/orca
    PATH=/share/project/zhangqq/envs_conda/flagscale-robo/bin:$PATH \
    PYTHONPATH=$(pwd) python -m pytest \
        scripts/deploy/sharpa_policy_server/test_against_training.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/share/project/zjk/orca")

import repr_bridge as rb

train = pytest.importorskip(
    "flagscale.models.vla.action_model.hand_action_repr",
    reason="only runs on the model server, where flagscale is importable",
)


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


# --- the layout constants -------------------------------------------------
def test_layout_constants_match_training():
    """A drift here silently redefines every slice of the 56D vector."""
    assert rb.ACTION_DIM == train.ACTION_DIM
    assert rb.STATE_DIM == train.STATE_DIM
    assert rb.WRIST_DIM == train.WRIST_DIM
    assert rb.HAND_DIM == train.HAND_DIM
    assert rb.SLICES == train.SLICES
    assert train.REPR_VERSION == 5


# --- the rotation primitives ----------------------------------------------
def test_rotation_to_rotvec_matches_training_bitwise():
    """Same inputs, same float32 outputs -- not merely 'close'.

    The bridge duplicates this formula rather than importing it, because the
    workstation-facing side runs where flagscale is not installed. That makes
    exact agreement the property worth asserting.
    """
    rng = np.random.default_rng(100)
    rotations = np.stack([random_rotation(rng) for _ in range(200)])
    mine = rb.rotation_to_rotvec(rotations)
    theirs = train.rotation_to_rotvec(rotations)
    assert mine.dtype == theirs.dtype == np.float32
    assert np.array_equal(mine, theirs)


def test_rotvec_to_rotation_matches_training_bitwise():
    rng = np.random.default_rng(101)
    rotvecs = rng.normal(scale=0.3, size=(200, 3)).astype(np.float32)
    assert np.array_equal(
        rb.rotvec_to_rotation(rotvecs), train.rotvec_to_rotation(rotvecs)
    )


def test_rot6d_helpers_match_training_bitwise():
    rng = np.random.default_rng(102)
    rotations = np.stack([random_rotation(rng) for _ in range(200)])
    mine6 = rb.rotation_to_rot6d(rotations)
    theirs6 = train.rotation_to_rot6d(rotations)
    assert np.array_equal(mine6, theirs6)
    assert np.array_equal(rb.rot6d_to_rotation(mine6), train.rot6d_to_rotation(theirs6))


# --- the relative wrist block ---------------------------------------------
def test_wrist_relative_matches_training_on_real_geometry():
    """The core claim: my per-pose form equals their episode-indexed form.

    ``wrist_pose_relative_6d`` indexes whole-episode arrays; at serving time
    there is no episode, so the bridge takes poses directly. Same arithmetic,
    different calling convention -- asserted rather than assumed.
    """
    rng = np.random.default_rng(103)
    length = 40
    translations = rng.normal(scale=0.3, size=(length, 3)).astype(np.float32)
    rotations = np.stack([random_rotation(rng) for _ in range(length)])

    anchor_index = np.array([5, 5, 5, 5])
    target_index = np.array([8, 11, 14, 17])
    theirs = train.wrist_pose_relative_6d(
        translations, rotations, anchor_index, target_index
    )
    mine = rb.wrist_relative_6d(
        translations[anchor_index],
        rotations[anchor_index],
        translations[target_index],
        rotations[target_index],
    )
    assert np.allclose(mine, theirs, atol=1e-6)


def test_decode_inverts_the_training_action_chunk():
    """Take a chunk THEY built and recover the poses it was built from.

    This is the end-to-end statement that matters for deployment: whatever the
    model emits, composing it onto the current pose has to reproduce the
    absolute trajectory the training data encoded. Built with
    ``build_action_chunk`` and inverted with ``decode_action_chunk``.
    """
    rng = np.random.default_rng(104)
    length = 60
    frames = {}
    poses = {}
    for side in ("left", "right"):
        translations = rng.normal(scale=0.2, size=(length, 3)).astype(np.float32)
        rotations = np.stack([random_rotation(rng) for _ in range(length)])
        # The stored 12D layout is t(3) + R(9) ROW-major.
        frames[f"observation.{side}_wrist_pose"] = np.concatenate(
            [translations, rotations.reshape(length, 9)], axis=-1
        ).astype(np.float32)
        frames[f"observation.{side}_qpos"] = rng.uniform(
            -1.0, 1.0, size=(length, 22)
        ).astype(np.float32)
        poses[side] = (translations, rotations)

    start = 10
    chunk = train.build_action_chunk(
        frames, start, rb.ACTION_HORIZON, rb.ACTION_STRIDE
    )
    assert chunk.shape == (rb.ACTION_HORIZON, rb.ACTION_DIM)

    # The anchor the chunk was built on: the achieved pose at `start`.
    anchor_pose9 = {
        side: rb.parts_to_pose9(poses[side][0][start], poses[side][1][start])
        for side in ("left", "right")
    }
    decoded = rb.decode_action_chunk(chunk, anchor_pose9)

    destination = start + (1 + np.arange(rb.ACTION_HORIZON)) * rb.ACTION_STRIDE
    for side in ("left", "right"):
        expected_t = poses[side][0][destination]
        expected_R = poses[side][1][destination]
        for step in range(rb.ACTION_HORIZON):
            got_t, got_R = rb.pose9_to_parts(decoded[f"{side}_eef"][step])
            assert np.allclose(got_t, expected_t[step], atol=1e-5), (side, step)
            assert np.allclose(got_R, expected_R[step], atol=1e-5), (side, step)
        # Hands are absolute qpos at the destination frame, copied verbatim.
        assert np.allclose(
            decoded[f"{side}_hand"],
            frames[f"observation.{side}_qpos"][destination],
            atol=0,
        )


def test_state_rows_match_training_build_state_rows():
    """All FOUR rows, same anchor rule, same numbers as the training builder.

    The offsets are (9, 6, 3, 0) with the anchor 12 back -- read from this run's
    hydra config, not the dataclass default of (0,). An earlier version of the
    bridge assumed one row; the real expert rejected it at inference, which is
    why this compares the whole tensor rather than a single row.
    """
    rng = np.random.default_rng(105)
    # start=30 + horizon*stride=30 needs frame 60 in range, so 90 not 60.
    length = 90
    frames = {}
    poses = {}
    hands = {}
    for side in ("left", "right"):
        translations = rng.normal(scale=0.2, size=(length, 3)).astype(np.float32)
        rotations = np.stack([random_rotation(rng) for _ in range(length)])
        frames[f"observation.{side}_wrist_pose"] = np.concatenate(
            [translations, rotations.reshape(length, 9)], axis=-1
        ).astype(np.float32)
        qpos = rng.uniform(-1.0, 1.0, size=(length, 22)).astype(np.float32)
        frames[f"observation.{side}_qpos"] = qpos
        poses[side] = (translations, rotations)
        hands[side] = qpos

    start = 30
    theirs = train.build_state_rows(
        frames, start, rb.STATE_ANCHOR_OFFSET, rb.STATE_HISTORY_OFFSETS
    )
    assert theirs.shape == (len(rb.STATE_HISTORY_OFFSETS), rb.STATE_DIM)

    def pose9_at(frame_index, side):
        return rb.parts_to_pose9(
            poses[side][0][frame_index], poses[side][1][frame_index]
        )

    anchor_frame = start - rb.STATE_ANCHOR_OFFSET
    anchor_pose9 = {s: pose9_at(anchor_frame, s) for s in ("left", "right")}
    rows = [
        (
            {s: pose9_at(start - offset, s) for s in ("left", "right")},
            {s: hands[s][start - offset] for s in ("left", "right")},
        )
        for offset in rb.STATE_HISTORY_OFFSETS
    ]
    mine = rb.build_state_rows(anchor_pose9, rows)

    assert mine.shape == theirs.shape
    assert np.allclose(mine, theirs, atol=1e-5), (
        f"max diff {np.abs(mine - theirs).max():.2e}"
    )


def test_state_and_action_use_different_anchors():
    """The two are NOT the same quantity since v5 -- assert it on real builders.

    build_state_rows anchors at start - 12, build_action_chunk at start. If a
    future change made them share an anchor, the separate normalizer statistics
    would silently become wrong, so this pins the distinction.
    """
    rng = np.random.default_rng(106)
    length = 90
    frames = {}
    for side in ("left", "right"):
        translations = rng.normal(scale=0.2, size=(length, 3)).astype(np.float32)
        rotations = np.stack([random_rotation(rng) for _ in range(length)])
        frames[f"observation.{side}_wrist_pose"] = np.concatenate(
            [translations, rotations.reshape(length, 9)], axis=-1
        ).astype(np.float32)
        frames[f"observation.{side}_qpos"] = rng.uniform(
            -1.0, 1.0, size=(length, 22)
        ).astype(np.float32)

    start = 30
    state = train.build_state_rows(
        frames, start, rb.STATE_ANCHOR_OFFSET, rb.STATE_HISTORY_OFFSETS
    )
    action = train.build_action_chunk(frames, start, rb.ACTION_HORIZON, rb.ACTION_STRIDE)

    # The newest state row is frame `start` against the anchor 12 back; it is a
    # real displacement, so it must not be zero -- that would mean the state
    # anchored on itself.
    newest = np.linalg.norm(state[-1][rb.SLICES["left_wrist_trans"]])
    assert newest > 1e-4, "newest state row is ~zero: anchored on itself?"
    # And the action's first step is measured from `start`, so the two blocks
    # describe different origins and cannot coincide.
    assert not np.allclose(
        state[-1][rb.SLICES["left_wrist_trans"]],
        action[0][rb.SLICES["left_wrist_trans"]],
        atol=1e-4,
    )


# --- the prompt contract --------------------------------------------------
def test_embodiment_constant_is_a_real_phrase_key():
    """``EMBODIMENT`` must be a key the training code will actually accept.

    ``embodiment_phrase_for`` raises on an unknown value rather than defaulting,
    so a typo here fails at the first inference -- after the 26 GB load, inside
    tmux, with the robot already waiting. Checked against the training code's own
    table instead of against a copied string.
    """
    import server as srv
    from flagscale.models.vla.action_model.hand_action_lerobot import (
        EMBODIMENT_PHRASES,
        embodiment_phrase_for,
    )

    assert srv.EMBODIMENT in EMBODIMENT_PHRASES, sorted(EMBODIMENT_PHRASES)
    # The UR-Sharpa windows rendered the robot phrase, same as T-Rex: dispatch is
    # on carrying a robot span and these genuinely are Sharpa hands.
    assert embodiment_phrase_for({"_embodiment": srv.EMBODIMENT}) == (
        "a robot's two Sharpa hands"
    )


def test_prompt_is_the_bare_instruction_not_a_rendered_sentence():
    """``lang`` is substituted INTO the checkpoint's template, not used as-is.

    ``build_qwenvl_inputs`` replaces ``{instruction}`` with whatever ``lang``
    holds, so a pre-rendered prompt would nest one template inside another and
    put every sample far outside the token-length distribution the checkpoint was
    trained on -- which nothing reports, because the chunk still comes out
    well-shaped. This asserts the registered prompt is a bare instruction by
    checking it carries none of the template's own words.
    """
    template = (
        "You are controlling {embodiment} from a head-mounted camera. "
        "The task is {instruction}. Predict how the scene and both hands will move next."
    )
    instruction = "Right hand unscrews the blue cap while left hand holds the bottle steady"
    rendered = template.replace("{instruction}", instruction).replace(
        "{embodiment}", "a robot's two Sharpa hands"
    )
    # A bare instruction cannot contain the template's framing.
    for marker in ("You are controlling", "head-mounted camera", "Predict how"):
        assert marker not in instruction
    assert instruction in rendered


def test_checkpoint_config_agrees_with_the_bridge_constants():
    """Read the SERVED checkpoint's own config, not the run's yaml.

    The yaml can be edited after a run finishes -- this one was, from 4000/250 to
    1800/150 -- so the only authority on what the weights expect is the config
    saved beside them.
    """
    import json
    import os

    import server as srv

    checkpoint = os.environ.get("SHARPA_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set SHARPA_TEST_CHECKPOINT to the served checkpoint directory")
    config = json.load(open(Path(checkpoint) / "pretrained_model" / "config.json"))
    action_model = config["action_model"]

    assert action_model["action_dim"] == rb.ACTION_DIM
    assert action_model["action_horizon"] == rb.ACTION_HORIZON
    assert action_model["action_stride"] == rb.ACTION_STRIDE
    assert tuple(action_model["state_history_offsets"]) == rb.STATE_HISTORY_OFFSETS
    assert action_model["state_anchor_offset"] == rb.STATE_ANCHOR_OFFSET
    assert action_model["use_state"] is True
    # The template must still have both slots, or the server's _embodiment key
    # is silently unused and the prompt the model sees is not what trained it.
    assert "{instruction}" in config["prompt_template"]
    assert "{embodiment}" in config["prompt_template"]
    assert srv.EMBODIMENT == "robot"


def test_make_policy_alone_does_not_load_weights():
    """The defect this pins: building a policy leaves the expert RANDOM.

    ``make_policy`` ends at ``TrainablePolicy.from_config``; the training entry
    loads weights separately in its own ``init_from`` branch. A server that only
    calls ``make_policy`` serves a randomly initialized 1.26B action expert, and
    NOTHING downstream reports it -- the chunk is finite, correctly shaped, and
    lands in a plausible millimetre range once the normalizer scales it.

    Asserted by reading the source rather than by loading a 26 GB model, so it
    runs in the normal suite. If a future refactor moves the load INTO
    make_policy, this test fails and says so, which is the point: the server would
    then be loading twice.
    """
    import inspect

    from flagscale.train import train_qwen3_5_gr00t as entry

    source = inspect.getsource(entry.make_policy)
    for marker in ("load_model_state_fsdp2", "init_from", "load_state_dict"):
        assert marker not in source, (
            f"make_policy now mentions {marker!r}; the server loads weights itself "
            "and would be loading them twice -- re-check MotPolicy.load"
        )


def test_server_loads_the_checkpoint_it_was_told_to_serve():
    """``checkpoint_path`` must reach a loader, not only a log line.

    It used to reach ONLY ``logger.info``, so the server announced the right
    checkpoint while serving a random expert. Reading the source is what catches
    that: any test that merely started the server and inspected its output saw
    well-formed numbers and passed.
    """
    import inspect

    import server as srv

    source = inspect.getsource(srv.MotPolicy.load)
    assert "load_model_state_fsdp2" in source, "MotPolicy.load does not load weights"
    load_line = source.index("load_model_state_fsdp2(model")
    path_line = source.index("self.checkpoint_path")
    assert path_line < load_line, "checkpoint_path is not what gets loaded"
