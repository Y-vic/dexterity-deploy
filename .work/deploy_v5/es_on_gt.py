"""SUPERSEDED -- THIS SCRIPT'S CONCLUSION IS WRONG. Kept only as a record.

Which EXECUTE_START is time-consistent? Decided on GROUND TRUTH alone.

No model, no sampling noise, no skew from dividing by a near-zero GT step. This
simulates the serving loop over the corpus itself: at each cycle it takes the GT
chunk the policy WOULD have been asked for, applies the slice, and asks where the
arm ends up against where the corpus says it should be at that same instant.

If EXECUTE_START is one step too early, every cycle commands a pose the corpus
already passed and the simulated arm falls behind the corpus track. Too late and
it runs ahead. The right value is the one whose tracking error stays flat over
many cycles -- an error that GROWS is the signature of a per-cycle bias, which is
exactly what a retraction or a lurch is.

CPU only; no GPU, no model load.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import repr_bridge as rb


def gt_xyz(frames, side: str, index: int) -> np.ndarray:
    return np.asarray(
        ct.pose9_from_pose12(frames[f"observation.{side}_wrist_pose"][index])[:3],
        dtype=np.float64,
    ) * 1000.0


def simulate(frames, total, side: str, execute_start: int, rtc: int,
             first: int, cycles: int) -> tuple[float, float]:
    """Track the corpus for `cycles` cycles. Returns (mean |error|, drift/cycle).

    The chunk is GT, so any error is attributable to the SLICE, not the policy.
    Each cycle: observe at frame `obs`, execute steps [es, es+L), then the next
    observation is L periods later.
    """
    from flagscale.models.vla.action_model import hand_action_repr as train

    stride = rb.ACTION_STRIDE
    length = rb.ACTION_HORIZON - execute_start
    if length <= 0 or not (0 <= rtc < length):
        return float("nan"), float("nan")
    obs = first
    errors = []
    for _ in range(cycles):
        if obs + rb.ACTION_HORIZON * stride >= total:
            break
        chunk = train.build_action_chunk(frames, obs, rb.ACTION_HORIZON, stride)
        anchor = {
            s: ct.pose9_from_pose12(frames[f"observation.{s}_wrist_pose"][obs])
            for s in ("left", "right")
        }
        world = rb.decode_action_chunk(chunk, anchor)[f"{side}_eef"][:, :3] * 1000.0
        cover = rtc + 1
        # The step commanded AT take-over must be reached one period later.
        due_frame = obs + (cover + 1) * stride
        if due_frame >= total:
            break
        commanded = np.asarray(world[execute_start], dtype=np.float64)
        errors.append(float(np.linalg.norm(commanded - gt_xyz(frames, side, due_frame))))
        obs += length * stride
    if not errors:
        return float("nan"), float("nan")
    arr = np.asarray(errors)
    drift = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 2 else 0.0
    return float(arr.mean()), drift


def main() -> None:
    import argparse

    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--rtc", type=int, default=4)
    parser.add_argument("--candidates", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--cycles", type=int, default=12)
    args = parser.parse_args()

    spans = discover_ur_episodes(ct.ROOT, max_episodes=args.episodes)
    cover = args.rtc + 1
    print(f"\n=== EXECUTE_START on GROUND-TRUTH chunks, rtc={args.rtc} ===")
    print(f"    cover = rtc+1 = {cover} steps, so the derived EXECUTE_START is {cover}")
    print("    error = |commanded pose - corpus pose due at that instant|, mm")
    print("    a value that is flat across cycles is time-consistent; drift is bias\n")

    per_es = {es: [] for es in args.candidates}
    for span in spans:
        frames, _ = read_ur_episode(span)
        total = len(next(iter(frames.values())))
        for side in ("left", "right"):
            for first in (150, 600, 1000):
                if first + rb.ACTION_HORIZON * rb.ACTION_STRIDE >= total:
                    continue
                for es in args.candidates:
                    mean_err, drift = simulate(
                        frames, total, side, es, args.rtc, first, args.cycles
                    )
                    if np.isfinite(mean_err):
                        per_es[es].append((mean_err, drift))
    header = f"{'ES':>4} {'mean error':>11} {'drift/cycle':>12} {'n':>5}"
    print(header)
    print("-" * len(header))
    for es in args.candidates:
        rows = per_es[es]
        if not rows:
            print(f"{es:>4}   (no valid runs)")
            continue
        arr = np.asarray(rows)
        tag = " <== derived" if es == cover else ""
        print(f"{es:>4} {arr[:, 0].mean():>11.2f} {arr[:, 1].mean():>12.3f} "
              f"{len(rows):>5}{tag}")
    valid = [(np.asarray(v)[:, 0].mean(), es) for es, v in per_es.items() if v]
    if valid:
        best = min(valid)
        print(f"\n  lowest mean error: EXECUTE_START = {best[1]} ({best[0]:.2f} mm)")


if __name__ == "__main__":
    main()
