import os

F = "/share/project/zjk/orca/flagscale/models/vla/action_model/hand_action_corpus.py"
src = open(F).read()
lines = src.split("\n")

start = next(i for i, l in enumerate(lines) if l.startswith("def read_ur_episode"))
end = len(lines)

tmpl = r'''
def read_ur_episode(span: EpisodeSpan) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
>"""Read ONE UR-Sharpa episode as ``(achieved_frames, hand_command)``.
>
>WHAT CHANGED AND WHY. This reader used to return ``(achieved, None)`` -- the hand
>action target was the achieved joint q. But the Manus hand stream arrives at only
>~15-21 Hz while the LeRobot grid is 30 Hz filled by ``causal_latest_valid_hold``,
>so ~42-54% of hand rows on that grid are exact repeats of the previous row (arm
>rows: 0.3%). Grasp-onset timing is quantized into a ~10-frame window and roughly
>half of hand frames teach "the fingers did not move"; under MSE the model hedges
>the onset and closes the hand early and only ~two-thirds of the way -- the
>observed hardware failure. The parquet's own ``q_teleop`` (action[62:106]) is the
>SAME hold-filled 30 Hz stream (55% repeats), so it is not a fix.
>
>The fix is a sidecar produced by ``tools/resample_ur_hand_cmd.py``: the raw Manus
>command resampled onto the master timeline using its own ``q_cmd_timestamp_ns``.
>At the model's 10 Hz that removes the aliasing (whole-row repeat 0.42 -> 0.001).
>It is supplied ONLY as the hand block of a command channel. The wrist blocks of
>the command alias the achieved frames, so ``build_action_chunk`` reproduces the
>achieved wrist target bit-for-bit (its wrist anchor already comes from the
>achieved trajectory); and the state rows the policy conditions on are still the
>achieved q, because ``build_state_rows`` reads ``frames`` and never the commands.
>
>SIDECAR LAYOUT. ``<dataset_root>/hand_cmd/episode_%06d.npz`` with key
>``q_cmd_resampled`` of shape ``[num_frames, 44]`` in RAW thumb-first order, which
>is exactly the canonical order (verified parquet[:, UR_HAND_PERMUTATION] == raw at
>0.0 max error), so the 22+22 columns map straight onto the canonical qpos blocks
>with NO further permutation. ``<dataset_root>`` is recovered from the parquet path,
>whose layout is fixed at ``<root>/data/chunk-XXX/episode_XXXXXX.parquet``.
>
>A missing or mis-sized sidecar RAISES rather than silently falling back to the
>held labels -- that fallback is the exact defect this reader exists to remove.
>
>Only this episode's own parquet is read, so no differencing can cross an episode
>boundary -- see this module's header for why that matters.
>"""
>import pandas as pd
>
>table = pd.read_parquet(span.parquet_path, columns=["observation.state"])
>if len(table) != span.num_frames:
>>raise ValueError(
>>>f"{span.parquet_path} holds {len(table)} rows but the metadata claims "
>>>f"{span.num_frames} for episode {span.episode_index}. On this corpus one "
>>>"parquet is one episode, so a mismatch means the metadata and the data "
>>>"have drifted apart -- window start frames are indexed against the "
>>>"metadata length."
>>)
>rows = np.stack(table["observation.state"].to_numpy())
>frames = canonical_from_ur_62d(rows)
>
># <root>/data/chunk-XXX/episode_XXXXXX.parquet -> <root>
>dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(span.parquet_path)))
>sidecar_path = os.path.join(
>>dataset_root, "hand_cmd", "episode_%06d.npz" % span.episode_index
>)
>if not os.path.exists(sidecar_path):
>>raise FileNotFoundError(
>>>f"No hand-command sidecar at {sidecar_path}. This reader trains the hand "
>>>"target from the resampled Manus command, not the held achieved q; run "
>>>"tools/resample_ur_hand_cmd.py to produce it. Refusing to fall back to the "
>>>"held labels, which is the defect this change removes."
>>)
>hand = np.load(sidecar_path)["q_cmd_resampled"].astype(np.float32)
>if hand.shape != (span.num_frames, 2 * QPOS_DIM):
>>raise ValueError(
>>>f"{sidecar_path} holds hand commands of shape {hand.shape}, expected "
>>>f"{(span.num_frames, 2 * QPOS_DIM)} for episode {span.episode_index}."
>>)
>
># Wrist keys alias the achieved frames -> byte-exact wrist target. Hand keys carry
># the resampled command. Raw thumb-first order == canonical, so no permutation.
>commands = dict(frames)
>commands["observation.left_qpos"] = hand[:, :QPOS_DIM]
>commands["observation.right_qpos"] = hand[:, QPOS_DIM:]
>return frames, commands
'''

body = []
for line in tmpl.split("\n"):
    n = 0
    while n < len(line) and line[n] == ">":
        n += 1
    body.append("    " * n + line[n:])
new_func = "\n".join(body).strip("\n")

lines[start:end] = new_func.split("\n")
out = "\n".join(lines)
if not out.endswith("\n"):
    out += "\n"

import ast

ast.parse(out)
open(F, "w").write(out)
print("PATCHED, parsed OK. read_ur_episode at lines %d-%d" % (start + 1, start + len(new_func.split("\n"))))
