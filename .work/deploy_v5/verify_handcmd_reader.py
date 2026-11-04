"""Post-patch verification for the hand-command channel. Run on the server AFTER
patch_reader.py has been applied. Read-only against the dataset; imports the patched
corpus module and checks the three things that must hold:

  1. read_ur_episode now returns (frames, commands) with commands not None, and the
     hand blocks differ from achieved while wrist keys are the achieved arrays.
  2. build_action_chunk with these commands reproduces wrist dims 0:12 BYTE-EXACT
     versus the old single-trajectory call, and the hand dims 12:56 now equal the
     resampled command at the destination frames.
  3. The whole thing runs for a real span from discover_ur_episodes.
"""
import numpy as np

from flagscale.models.vla.action_model.hand_action_corpus import (
    discover_ur_episodes,
    read_ur_episode,
)
from flagscale.models.vla.action_model.hand_action_repr import build_action_chunk, SLICES

UR_ROOT = "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/lerobot_data/unscrew_cap"
HORIZON, STRIDE = 10, 3

spans = discover_ur_episodes(UR_ROOT, fps=30, min_frames=HORIZON * STRIDE + 1, task="x")
span = next(s for s in spans if s.episode_index == 79)
print("span ep", span.episode_index, "frames", span.num_frames, "corpus", span.corpus)

frames, commands = read_ur_episode(span)
assert commands is not None, "commands is None -- patch did not take"
for k in ("observation.left_wrist_pose", "observation.right_wrist_pose"):
    assert commands[k] is frames[k], f"{k} should alias achieved frames"
for k in ("observation.left_qpos", "observation.right_qpos"):
    same = np.array_equal(commands[k], frames[k])
    print(f"{k}: differs from achieved = {not same}  shape {commands[k].shape}")

S = 45  # grasp region for ep79
old = build_action_chunk(frames, S, HORIZON, STRIDE)
new = build_action_chunk(frames, S, HORIZON, STRIDE, command_frames=commands)

wrist_exact = np.array_equal(old[:, :12], new[:, :12])
print("wrist dims 0:12 byte-exact old-vs-new:", wrist_exact)
assert wrist_exact, "WRIST DIMS CHANGED -- this must never happen"

dst = S + (1 + np.arange(HORIZON)) * STRIDE
lh = commands["observation.left_qpos"][dst]
rh = commands["observation.right_qpos"][dst]
hand_from_cmd = (
 np.array_equal(new[:, SLICES["left_hand_joints"]], lh)
    and np.array_equal(new[:, SLICES["right_hand_joints"]], rh)
)
print("hand dims 12:56 equal resampled command at dst frames:", hand_from_cmd)
assert hand_from_cmd, "hand block did not come from the command"

hand_changed = not np.array_equal(old[:, 12:56], new[:, 12:56])
print("hand target changed vs achieved-label chunk:", hand_changed)
print()
print("left-hand dominant closure over the chunk (canonical col 0):")
print("  old (achieved label):", np.round(old[:, 12], 3))
print("  new (resampled cmd) :", np.round(new[:, 12], 3))
print()
print("ALL CHECKS PASSED")
