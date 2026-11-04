"""EXECUTE_START decided by a CLOSED-LOOP simulation on ground truth.

`es_on_gt.py` re-anchored each cycle on the corpus pose, so the simulated arm was
teleported back onto the reference track every cycle and no per-cycle bias could
accumulate -- which is precisely the quantity in question. Its ranking is
therefore not evidence and is superseded by this file.

Here the arm's own state is carried forward. Each cycle:

  1. the observation is taken at the arm's CURRENT simulated pose,
  2. a GT chunk is built at the corpus frame nearest the arm's progress and
     decoded onto that pose, so the chunk is correct but the anchor is the arm's,
  3. steps [es, es+L) are executed, moving the arm,
  4. the next observation happens L periods later.

A slice that is one step early re-commands ground already covered, so the arm
falls progressively behind the corpus; one step late and it runs ahead. The
diagnostic is the arm's PROGRESS RATE against the corpus's over the same wall
clock: 1.0 means time-consistent.

Using GT chunks means the policy's own underprediction cannot contaminate the
answer: any deviation from 1.0 belongs to the slice arithmetic alone.

CPU only.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import repr_bridge as rb


def pose9_at(frames, side: str, index: int) -> np.ndarray:
    return ct.pose9_from_pose12(frames[f"observation.{side}_wrist_pose"][index])


def simulate(frames, total, execute_start: int, rtc: int, first: int,
             cycles: int) -> dict:
    """Run the serving loop against GT chunks, carrying the arm's own state.

    Returns progress rate per side plus the number of cycles actually run.
    ``progress`` is the arm's NET DISPLACEMENT divided by the corpus's over the
    same number of periods. Net, not path length: path length counts jitter as
    progress, and a slice that oscillates would score above 1.0 while getting
    nowhere -- which is what an earlier version of this file did.
    """
    from flagscale.models.vla.action_model import hand_action_repr as train

    stride = rb.ACTION_STRIDE
    length = rb.ACTION_HORIZON - execute_start
    if length <= 0 or not (0 <= rtc < length):
        return {}
    # The arm starts on the corpus track.
    arm = {s: pose9_at(frames, s, first).astype(np.float64) for s in ("left", "right")}
    origin = {s: arm[s][:3].copy() for s in ("left", "right")}
    obs_frame = first
    ran = 0
    for _ in range(cycles):
        # The chunk the policy would be asked for at this observation.
        if obs_frame + rb.ACTION_HORIZON * stride >= total:
            break
        chunk = train.build_action_chunk(frames, obs_frame, rb.ACTION_HORIZON, stride)
        # Decoded onto the ARM's pose, which is what serving does.
        anchor = {s: arm[s].astype(np.float32) for s in ("left", "right")}
        decoded = rb.decode_action_chunk(chunk, anchor)
        for side in ("left", "right"):
            world = np.asarray(decoded[f"{side}_eef"], dtype=np.float64)
            executed = world[execute_start:execute_start + length]
            for row in executed:
                arm[side] = row.copy()
        obs_frame += length * stride
        ran += 1
    if ran == 0:
        return {}
    periods = ran * length
    out = {"cycles": ran, "periods": periods}
    end_frame = min(first + periods * stride, total - 1)
    for side in ("left", "right"):
        arm_net = float(np.linalg.norm(arm[side][:3] - origin[side])) * 1000.0
        corpus_net = float(
            np.linalg.norm(
                pose9_at(frames, side, end_frame)[:3] - pose9_at(frames, side, first)[:3]
            )
        ) * 1000.0
        out[side] = arm_net / max(corpus_net, 1e-9)
        out[f"{side}_mm"] = arm_net
        out[f"{side}_corpus_mm"] = corpus_net
    return out


def main() -> None:
    import argparse

    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--rtc", type=int, default=4)
    parser.add_argument("--candidates", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--cycles", type=int, default=15)
    parser.add_argument("--starts", type=int, nargs="+", default=[150, 600, 1000])
    args = parser.parse_args()

    spans = discover_ur_episodes(ct.ROOT, max_episodes=args.episodes)
    cover = args.rtc + 1
    print(f"\n=== closed-loop slice check on GT chunks, rtc={args.rtc} ===")
    print(f"    cover = rtc+1 = {cover}; derived EXECUTE_START = {cover}")
    print("    progress = arm travel / corpus travel over the same wall clock")
    print("    1.00 = time-consistent, <1 = losing ground every cycle\n")
    header = f"{'ES':>4} {'L':>3} {'left':>8} {'right':>8} {'runs':>6}"
    print(header)
    print("-" * len(header))
    for es in args.candidates:
        left, right, runs = [], [], 0
        for span in spans:
            frames, _ = read_ur_episode(span)
            total = len(next(iter(frames.values())))
            for first in args.starts:
                if first + rb.ACTION_HORIZON * rb.ACTION_STRIDE >= total:
                    continue
                out = simulate(frames, total, es, args.rtc, first, args.cycles)
                if out:
                    left.append(out["left"])
                    right.append(out["right"])
                    runs += 1
        if not runs:
            print(f"{es:>4} {rb.ACTION_HORIZON - es:>3}   (no valid runs)")
            continue
        tag = " <== derived" if es == cover else ""
        print(f"{es:>4} {rb.ACTION_HORIZON - es:>3} {np.mean(left):>8.3f} "
              f"{np.mean(right):>8.3f} {runs:>6}{tag}")
    print()
    print("  GT chunks, so a value below 1 is the SLICE losing ground, not the")
    print("  policy underpredicting. The closest to 1.00 is the right slice.")


if __name__ == "__main__":
    main()
