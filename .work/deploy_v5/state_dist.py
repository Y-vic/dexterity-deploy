"""Is the state the server sends in the distribution the model was trained on?

The last fork in the diagnosis. If the live state rows sit where training's do,
the out-of-distribution chunk is the checkpoint's doing. If they do not, the
server is feeding the expert something it never saw and the chunk is a symptom.

Both sides are measured in NORMALIZED units, because that is what the expert
actually receives.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import repr_bridge as rb


def describe(name: str, rows: np.ndarray) -> None:
    """One line per block, in sigma."""
    for label, sl in (
        ("wrist_trans", slice(0, 3)),
        ("wrist_rotvec", slice(3, 6)),
        ("hand", slice(12, 34)),
    ):
        block = rows[..., sl]
        print(
            f"  {name:14} {label:12} |mean|={np.abs(block).mean():7.3f} "
            f"max|.|={np.abs(block).max():7.3f}"
        )


def main() -> None:
    import torch
    from flagscale.models.vla.action_model import hand_action_repr as train
    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )

    from flagscale.models.vla.action_model.hand_action_norm import (
        HandActionNormalizer,
    )

    normalizer = HandActionNormalizer.load(
        "/share/project/zjk/orca/outputs/shared_mot_action_v5_ur_unscrew/hand_action_norm.json"
    )
    root = (
        "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
        "lerobot_data/unscrew_cap"
    )

    print("=== TRAINING state rows, normalized (what the expert saw) ===")
    collected = []
    for span in discover_ur_episodes(root, max_episodes=4):
        frames, _ = read_ur_episode(span)
        total = len(next(iter(frames.values())))
        for start in (total // 4, total // 2, (3 * total) // 4):
            if start < rb.STATE_ANCHOR_OFFSET:
                continue
            if start + rb.ACTION_HORIZON * rb.ACTION_STRIDE >= total:
                continue
            raw = train.build_state_rows(
                frames, start, rb.STATE_ANCHOR_OFFSET, rb.STATE_HISTORY_OFFSETS
            )
            collected.append(
                normalizer.normalize_state(torch.from_numpy(raw)).numpy()
            )
    stack = np.concatenate(collected, axis=0)
    print(f"  windows={len(collected)} rows={stack.shape}")
    describe("train", stack)


if __name__ == "__main__":
    main()
