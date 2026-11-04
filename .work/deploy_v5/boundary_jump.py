"""SUPERSEDED -- THIS SCRIPT'S CONCLUSION IS WRONG. Kept only as a record.

The per-cycle jump the slice fix leaves behind, and its sign per phase.

`EXECUTE_START = RTC_STEPS` drops the expired steps, which is right, but it
throws them away by ASSUMING the arm covered exactly `d(EXECUTE_START)` during
the lag -- the chunk's own prediction of its first 5 steps. What the arm actually
covered is the PREVIOUS chunk's executed tail, `d(9) - d(4)`. Those two are equal
only if the chunk's velocity is uniform over the horizon.

At the boundary the arm sits at   P + (d_N(9) - d_N(4))
and the new chunk commands       P + d_{N+1}(4)
so every cycle commands a step of

    jump = d_{N+1}(4) - (d_N(9) - d_N(4))   ~=   2*d(4) - d(9)

for consecutive chunks of the same phase. Front-loaded chunk (fast early, slow
late) -> jump points FORWARD: a periodic lurch on top of the commanded speed.
Back-loaded -> jump points BACKWARD: the residual retraction. One mechanism, and
the sign is a property of the fitted velocity field, so it can differ by phase.

This measures, per phase: the within-chunk speed profile, the jump vector, its
size against one normal step, and its sign along the direction of travel.

Runs on the remote with its own model copy. Does not touch the serving process.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import phase_sweep as ps
import repr_bridge as rb
import server as srv


def trans(chunk: np.ndarray, side: str) -> np.ndarray:
    """[horizon, 3] wrist translation, relative to the chunk's anchor, in mm."""
    block = chunk[:, rb.SLICES[f"{side}_wrist_trans"]]
    return np.asarray(block, dtype=np.float64) * 1000.0


def hand(chunk: np.ndarray, side: str) -> np.ndarray:
    """[horizon, 22] ABSOLUTE hand qpos -- no anchor, so no lag arithmetic."""
    return np.asarray(chunk[:, rb.SLICES[f"{side}_hand_joints"]], dtype=np.float64)


def cover_steps(execute_start: int, rtc: int, horizon: int = rb.ACTION_HORIZON) -> int:
    """Steps from the observation instant to the next chunk's take-over.

    Chunk N takes over at t=0 and publishes local step i at i*period. The fetch
    fires once ``executed_in_slice == L - rtc``, i.e. after local step
    ``L-rtc-1``, so the observation is at ``(L-rtc-1)*period``. The successor
    takes over at ``L*period``. The difference is ``(rtc+1)`` periods -- it does
    NOT depend on execute_start.
    """
    length = horizon - execute_start
    return (length - (length - rtc - 1))  # == rtc + 1


def speed_profile(chunk: np.ndarray, side: str) -> np.ndarray:
    """Per-step speed, mm per 100 ms period, over the whole horizon.

    Step 0 is measured from the anchor, which is one period before it, so the
    profile has exactly ``horizon`` entries and entry k is the speed the arm is
    asked to hold while moving into step k.
    """
    block = trans(chunk, side)
    with_anchor = np.vstack([np.zeros((1, 3)), block])
    return np.linalg.norm(np.diff(with_anchor, axis=0), axis=1)


def boundary_increment(chunk: np.ndarray, side: str, execute_start: int, rtc: int):
    """What the take-over step commands, against where the arm actually is.

    Both chunks are assumed to be in the same phase, so one chunk's profile
    stands in for its successor's. Returns ``(increment_mm, normal_step_mm)``.

    The arm at take-over sits at ``d(H-1)`` of the OLD chunk's anchor. The new
    anchor is the pose measured at the observation instant, which is the old
    chunk's step ``H-1-rtc-1``, and the new chunk commands ``d(execute_start)``
    past it. Positive increment = forward lurch, negative = retraction.
    """
    d = trans(chunk, side)
    horizon = d.shape[0]
    g_at_obs = horizon - rtc - 2
    arm_at_takeover = d[horizon - 1]
    commanded = d[g_at_obs] + d[execute_start]
    increment = float(np.linalg.norm(commanded - arm_at_takeover))
    # Signed along the direction the chunk is travelling.
    heading = d[horizon - 1] - d[g_at_obs]
    norm = np.linalg.norm(heading)
    if norm > 1e-9:
        increment = float(np.dot(commanded - arm_at_takeover, heading / norm))
    profile = speed_profile(chunk, side)
    return increment, float(profile[execute_start:].mean())


def main() -> None:
    import argparse
    import io as _io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--starts", type=int, nargs="+",
                        default=[120, 600, 900, 1200, 1600])
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--rtc", type=int, default=srv.RTC_STEPS)
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/004000")
    args = parser.parse_args()

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
    report(args, sample, noises)


def report(args, sample, noises) -> None:
    cover = args.rtc + 1
    print(f"\n=== within-chunk speed profile and boundary step, episode {args.episode} ===")
    print(f"    rtc={args.rtc} -> observation-to-take-over cover = {cover} steps "
          f"({cover * 100} ms)")
    print(f"    so the time-consistent EXECUTE_START is {cover}; "
          f"server.py currently uses {srv.EXECUTE_START}")
    print(f"    {args.draws} noise draws per cell; speeds in mm per 100 ms step\n")

    for start in args.starts:
        try:
            anchor, hrows, jpeg, gt = ct.build_window(args.episode, start)
        except Exception as exc:  # a window at the very end of an episode
            print(f"  --- start {start}: skipped ({exc}) ---")
            continue
        state56 = rb.build_state_rows(anchor, hrows)
        chunks = [sample(state56, jpeg, n) for n in noises]
        print(f"  --- start {start} ---")
        for side in ("left", "right"):
            gt_profile = speed_profile(gt, side)
            profile = np.mean([speed_profile(c, side) for c in chunks], axis=0)
            print(f"    {side} model speed/step: "
                  + " ".join(f"{v:5.1f}" for v in profile))
            print(f"    {side} GT    speed/step: "
                  + " ".join(f"{v:5.1f}" for v in gt_profile))
            front = profile[:cover].mean()
            back = profile[cover:].mean()
            shape = "front-loaded" if front > back else "back-loaded"
            print(f"    {side} first {cover} steps {front:5.2f} vs rest {back:5.2f}"
                  f"  -> {shape}")
            for es in sorted({srv.EXECUTE_START, cover}):
                inc = np.mean([
                    boundary_increment(c, side, es, args.rtc)[0] for c in chunks
                ])
                step = np.mean([
                    boundary_increment(c, side, es, args.rtc)[1] for c in chunks
                ])
                verdict = ("retraction" if inc < -0.05 * max(step, 1e-9)
                           else "lurch" if inc > 1.5 * max(step, 1e-9)
                           else "smooth")
                tag = " (current)" if es == srv.EXECUTE_START else " (derived)"
                print(f"      EXECUTE_START={es}{tag}: boundary step "
                      f"{inc:+6.2f}mm vs normal {step:5.2f}mm -> {verdict}")
            hand_gt = np.abs(np.diff(hand(gt, side), axis=0)).max(axis=1)
            hand_m = np.mean(
                [np.abs(np.diff(hand(c, side), axis=0)).max(axis=1) for c in chunks],
                axis=0,
            )
            print(f"    {side} hand max|d|/step model {hand_m.max():.4f} "
                  f"GT {hand_gt.max():.4f} rad")
        print()


if __name__ == "__main__":
    main()
