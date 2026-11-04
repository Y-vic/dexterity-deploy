"""Ask the served model a REAL training window and compare against ground truth.

This is the test that separates "the model is jittery" from "the server feeds it
something out of distribution". Everything here comes from the training corpus
through the training code's own reader, so the only thing under test is the
checkpoint: if GT is smooth and the prediction is not, the weights are at fault;
if the prediction matches GT, the fault is in what the live server sends.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import repr_bridge as rb


def smoothness(block: np.ndarray) -> tuple[float, float, float]:
    """Path length, net displacement, ratio -- in mm, over a [steps, 3] block."""
    step = np.linalg.norm(np.diff(block, axis=0), axis=1) * 1000
    net = float(np.linalg.norm(block[-1] - block[0]) * 1000)
    return float(step.sum()), net, float(step.sum() / max(net, 1e-6))


def report(name: str, chunk: np.ndarray) -> None:
    print(f"--- {name} ---")
    for side in ("left", "right"):
        trans = chunk[:, rb.SLICES[f"{side}_wrist_trans"]]
        path, net, ratio = smoothness(trans)
        hand = chunk[:, rb.SLICES[f"{side}_hand_joints"]]
        hand_step = np.abs(np.diff(hand, axis=0)).max(axis=1)
        print(
            f"  {side}: path {path:6.1f}mm net {net:6.1f}mm ratio {ratio:5.2f} | "
            f"hand max|d|/step {hand_step.max():.3f} rad"
        )


def main() -> None:
    from flagscale.models.vla.action_model import hand_action_repr as train
    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )

    root = (
        "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
        "lerobot_data/unscrew_cap"
    )
    spans = discover_ur_episodes(root, max_episodes=3)
    print(f"episodes: {len(spans)}")

    for span in spans[:3]:
        frames, _ = read_ur_episode(span)
        total = len(next(iter(frames.values())))
        # A window well inside the episode: past the anchor, with a full horizon
        # of future left, so nothing is clamped at either end.
        start = max(rb.STATE_ANCHOR_OFFSET, total // 3)
        if start + rb.ACTION_HORIZON * rb.ACTION_STRIDE >= total:
            continue
        gt = train.build_action_chunk(
            frames, start, rb.ACTION_HORIZON, rb.ACTION_STRIDE
        )
        report(f"GT ep{span.episode_index} start={start} ({total} frames)", gt)


if __name__ == "__main__":
    main()
