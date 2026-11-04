"""Export successful demonstration chunks without loading the policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closing_test as ct
import repr_bridge as rb


def main() -> None:
    from flagscale.models.vla.action_model import hand_action_repr as train
    from flagscale.models.vla.action_model.hand_action_corpus import discover_ur_episodes, read_ur_episode

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--starts", type=int, nargs="+", default=[120, 400, 620, 900, 1200, 1600])
    parser.add_argument("--output", default="/tmp/urscratch_gt_windows.npz")
    args = parser.parse_args()

    span = discover_ur_episodes(ct.ROOT, max_episodes=args.episode + 1)[args.episode]
    frames, _ = read_ur_episode(span)
    payload = {}
    for start in args.starts:
        gt = train.build_action_chunk(frames, start, rb.ACTION_HORIZON, rb.ACTION_STRIDE)
        pose = {
            "left": ct.pose9_from_pose12(frames["observation.left_wrist_pose"][start]),
            "right": ct.pose9_from_pose12(frames["observation.right_wrist_pose"][start]),
        }
        hand = {
            "left": np.asarray(frames["observation.left_qpos"][start], np.float32),
            "right": np.asarray(frames["observation.right_qpos"][start], np.float32),
        }
        key = f"s{start}"
        payload[f"{key}_gt_raw"] = gt
        payload[f"{key}_anchor_left"] = pose["left"]
        payload[f"{key}_anchor_right"] = pose["right"]
        payload[f"{key}_anchor_left_hand"] = hand["left"]
        payload[f"{key}_anchor_right_hand"] = hand["right"]
    np.savez(args.output, **payload)
    print(f"exported {len(args.starts)} windows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
