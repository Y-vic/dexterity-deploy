"""What per-step motion does the CORPUS ever ask for, and how does it vary by phase?

Two questions, both needed to read the hardware report "the hands got faster once
the bottle was held, faster than during the descent":

  1. Is that speed-up real in the data? If GT itself moves several times faster in
     the unscrew phase than in the approach, then a faithful policy MUST speed up
     there and the fix is not to slow the policy down.
  2. What is the widest per-step motion the corpus ever contains? That is the
     envelope a serving-side safety clamp should be derived from. The clamps in
     actor_ik_node are ARM_MAX_STEP_RAD 0.25 rad and HAND_MAX_STEP_RAD 0.4 rad
     per 100 ms step -- 2.5 and 4.0 rad/s. Those were picked as crash protection,
     not from the data, and if the corpus never exceeds a small fraction of them
     they permit motion the task never needs.

Everything is read through the training code's own reader and chunk builder, so
these are the numbers the policy was fitted to. CPU only; no model, no GPU.
"""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import repr_bridge as rb

ROOT = (
    "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
    "lerobot_data/unscrew_cap"
)


def step_speeds(chunk: np.ndarray, side: str) -> np.ndarray:
    """Per-step wrist translation speed, mm per 100 ms, over one GT chunk."""
    block = np.asarray(
        chunk[:, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64
    ) * 1000.0
    with_anchor = np.vstack([np.zeros((1, 3)), block])
    return np.linalg.norm(np.diff(with_anchor, axis=0), axis=1)


def hand_steps(chunk: np.ndarray, side: str) -> np.ndarray:
    """Per-step max joint change of one hand, rad per 100 ms."""
    block = np.asarray(chunk[:, rb.SLICES[f"{side}_hand_joints"]], dtype=np.float64)
    return np.abs(np.diff(block, axis=0)).max(axis=1)


def main() -> None:
    import argparse

    from flagscale.models.vla.action_model import hand_action_repr as train
    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    spans = discover_ur_episodes(ROOT, max_episodes=args.episodes)
    print(f"episodes: {len(spans)}  bins: {args.bins}\n")

    span_needed = rb.ACTION_HORIZON * rb.ACTION_STRIDE
    per_bin = defaultdict(lambda: defaultdict(list))
    worst = defaultdict(float)

    for span in spans:
        frames, _ = read_ur_episode(span)
        total = len(next(iter(frames.values())))
        starts = range(rb.STATE_ANCHOR_OFFSET, total - span_needed, rb.ACTION_STRIDE)
        starts = list(starts)
        if not starts:
            continue
        for start in starts:
            gt = train.build_action_chunk(
                frames, start, rb.ACTION_HORIZON, rb.ACTION_STRIDE
            )
            frac = (start - starts[0]) / max(len(starts) - 1, 1)
            b = min(int(frac * args.bins), args.bins - 1)
            for side in ("left", "right"):
                sp = step_speeds(gt, side)
                hd = hand_steps(gt, side)
                per_bin[b][f"{side}_wrist"].append(sp.mean())
                per_bin[b][f"{side}_hand"].append(hd.mean())
                worst[f"{side}_wrist"] = max(worst[f"{side}_wrist"], sp.max())
                worst[f"{side}_hand"] = max(worst[f"{side}_hand"], hd.max())
    report(args, per_bin, worst)


def report(args, per_bin, worst) -> None:
    print("GT mean per-step motion by episode phase (bin 0 = start, "
          f"{args.bins - 1} = end)")
    print("wrist in mm per 100 ms step, hand in rad per 100 ms step\n")
    header = (f"{'bin':>4} {'L wrist':>9} {'R wrist':>9} "
              f"{'L hand':>9} {'R hand':>9}")
    print(header)
    print("-" * len(header))
    for b in sorted(per_bin):
        row = per_bin[b]
        print(f"{b:>4} "
              f"{np.mean(row['left_wrist']):>9.2f} "
              f"{np.mean(row['right_wrist']):>9.2f} "
              f"{np.mean(row['left_hand']):>9.4f} "
              f"{np.mean(row['right_hand']):>9.4f}")
    print()
    print("corpus-wide WORST single step (the envelope a clamp should respect):")
    print(f"  left  wrist {worst['left_wrist']:8.2f} mm/step")
    print(f"  right wrist {worst['right_wrist']:8.2f} mm/step")
    print(f"  left  hand  {worst['left_hand']:8.4f} rad/step")
    print(f"  right hand  {worst['right_hand']:8.4f} rad/step")
    print()
    print("against the serving clamps in actor_ik_node (per 100 ms step):")
    print(f"  HAND_MAX_STEP_RAD = 0.4  -> {0.4 / max(worst['left_hand'], worst['right_hand']):.1f}x "
          "the widest hand step the corpus contains")
    print("  ARM_MAX_STEP_RAD  = 0.25 rad is a JOINT limit; the wrist numbers "
          "above are Cartesian, so compare via IK, not directly.")
    peak = max(
        (np.mean(per_bin[b]['right_wrist']), b) for b in per_bin
    )
    quiet = min(
        (np.mean(per_bin[b]['right_wrist']), b) for b in per_bin
    )
    print()
    print(f"right wrist fastest phase = bin {peak[1]} at {peak[0]:.2f} mm/step; "
          f"slowest = bin {quiet[1]} at {quiet[0]:.2f} mm/step")
    print(f"  ratio {peak[0] / max(quiet[0], 1e-9):.1f}x -- a policy that tracks "
          "the corpus MUST vary its speed by about this much")


if __name__ == "__main__":
    main()
