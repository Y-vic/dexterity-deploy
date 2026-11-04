"""Does the arm's COMMANDED step distribution match the corpus's own?

The earlier per-window ratios were misleading. The corpus's per-100 ms wrist step
is strongly skewed -- left p50 0.43 mm, p90 6.2 mm, max 18.8 mm -- so it alternates
between nearly stationary and fast bursts. Dividing one commanded step by the GT
step at that same instant therefore produces a huge ratio whenever GT happens to
be in a pause, which says nothing about whether the policy is too fast.

The honest comparison is DISTRIBUTIONAL: over many windows, the set of steps the
serving slice would command against the set of steps the corpus contains. If the
upper tail matches, the policy is not running hot and the hardware's "too fast"
is the task's own fast phase. If the tail is heavier, there is a real excursion
and the ceiling in `speed_guard.py` is what bounds it.

Runs on the remote with its own model copy.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import phase_sweep as ps
import repr_bridge as rb
import server as srv


def executed_steps(chunk: np.ndarray, side: str) -> np.ndarray:
    """The per-step motion the serving slice would actually command, in mm."""
    block = np.asarray(
        chunk[:, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64
    ) * 1000.0
    with_anchor = np.vstack([np.zeros((1, 3)), block])
    steps = np.linalg.norm(np.diff(with_anchor, axis=0), axis=1)
    return steps[srv.EXECUTE_START:srv.EXECUTE_START + srv.EXECUTE_LENGTH]


def hand_steps(chunk: np.ndarray, side: str) -> np.ndarray:
    block = np.asarray(chunk[:, rb.SLICES[f"{side}_hand_joints"]], dtype=np.float64)
    steps = np.abs(np.diff(block, axis=0)).max(axis=1)
    return steps[srv.EXECUTE_START:srv.EXECUTE_START + srv.EXECUTE_LENGTH]


def main() -> None:
    import argparse
    import io as _io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer
    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--first", type=int, default=120)
    parser.add_argument("--last", type=int, default=1700)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/004000")
    args = parser.parse_args()

    span = discover_ur_episodes(ct.ROOT, max_episodes=args.episode + 1)[args.episode]
    frames, _ = read_ur_episode(span)

    model = ps.ct_load(args.checkpoint)
    normalizer = HandActionNormalizer.load(ps.NORM)

    def sample(state56, jpeg, noise):
        image = Image.open(_io.BytesIO(jpeg)).convert("RGB").resize((256, 256))
        state = normalizer.normalize_state(torch.from_numpy(state56))
        batch = [{"image": [image], "lang": ct.UR_TASK, "state": state,
                  "_embodiment": "robot"}]
        with torch.no_grad():
            out = model._predict_action_mot(batch, noise=noise)
        chunk = out[next(iter(out))].float().cpu()
        chunk = normalizer.denormalize(chunk)
        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return chunk.numpy().astype(np.float32)

    torch.manual_seed(0)
    noises = [
        torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")
        for _ in range(args.draws)
    ]
    report(args, frames, sample, noises)


def report(args, frames, sample, noises) -> None:
    import speed_guard as sg

    model_w = {"left": [], "right": []}
    model_h = {"left": [], "right": []}
    guard_hits = 0
    chunk_count = 0

    for start in range(args.first, args.last + 1, args.stride):
        try:
            anchor, hrows, jpeg, _gt = ct.build_window(args.episode, start)
        except Exception:
            continue
        state56 = rb.build_state_rows(anchor, hrows)
        for noise in noises:
            chunk = sample(state56, jpeg, noise)
            chunk_count += 1
            _, rep = sg.limit_chunk(chunk)
            if rep["engaged"]:
                guard_hits += 1
            for side in ("left", "right"):
                model_w[side].append(executed_steps(chunk, side))
                model_h[side].append(hand_steps(chunk, side))

    # Corpus reference: every per-period step in the episode.
    print(f"\n=== commanded vs corpus step distribution, episode {args.episode} ===")
    print(f"    {chunk_count} chunks, slice [{srv.EXECUTE_START}, "
          f"{srv.EXECUTE_START + srv.EXECUTE_LENGTH})\n")
    header = (f"{'channel':>12} {'source':>7} {'p50':>7} {'p90':>7} {'p99':>7} "
              f"{'max':>7}")
    print(header)
    print("-" * len(header))
    for side in ("left", "right"):
        pose = np.asarray(
            [ct.pose9_from_pose12(r)[:3]
             for r in frames[f"observation.{side}_wrist_pose"]], dtype=float
        ) * 1000.0
        gt = np.linalg.norm(pose[rb.ACTION_STRIDE:] - pose[:-rb.ACTION_STRIDE], axis=1)
        m = np.concatenate(model_w[side])
        for name, arr in (("model", m), ("corpus", gt)):
            print(f"{side + ' wrist':>12} {name:>7} "
                  f"{np.percentile(arr, 50):>7.2f} {np.percentile(arr, 90):>7.2f} "
                  f"{np.percentile(arr, 99):>7.2f} {arr.max():>7.2f}")
        qpos = np.asarray(frames[f"observation.{side}_qpos"], dtype=float)
        gth = np.abs(qpos[rb.ACTION_STRIDE:] - qpos[:-rb.ACTION_STRIDE]).max(axis=1)
        mh = np.concatenate(model_h[side])
        for name, arr in (("model", mh), ("corpus", gth)):
            print(f"{side + ' hand':>12} {name:>7} "
                  f"{np.percentile(arr, 50):>7.4f} {np.percentile(arr, 90):>7.4f} "
                  f"{np.percentile(arr, 99):>7.4f} {arr.max():>7.4f}")
    print()
    print(f"  speed_guard engaged on {guard_hits}/{chunk_count} chunks "
          f"(caps: {sg.MAX_WRIST_STEP_MM:.1f} mm, {sg.MAX_HAND_STEP_RAD:.3f} rad)")
    print("  A ceiling should be near-silent here: these are in-distribution")
    print("  windows, so anything it catches would be a sampling excursion.")


if __name__ == "__main__":
    main()
