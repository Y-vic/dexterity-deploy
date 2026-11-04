"""SUPERSEDED -- THIS SCRIPT'S CONCLUSION IS WRONG. Kept only as a record.

The real boundary: TWO chunks from two consecutive observations, in world frame.

`boundary_jump.py` used one chunk as a stand-in for its own successor, which
folds the successor's re-anchoring into the number and makes the result useless
for choosing EXECUTE_START -- it reported multi-mm boundary steps in both
settings because the two chunks it differenced were the same chunk.

This does it properly. Chunk N is sampled at observation frame `start`; the
successor is sampled at `start + cover*ACTION_STRIDE`, i.e. the frame the arm's
observation is actually taken at one cycle later, with the CORPUS pose at that
frame as its anchor. Both chunks are then decoded to ABSOLUTE world poses through
`repr_bridge.decode_action_chunk`, so the comparison happens in one frame and no
anchor arithmetic is left for the reader to get wrong.

The number that matters: the size of the step the arm is commanded at take-over,
against the median commanded step inside a chunk. Smooth means those are equal.

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

ROOT = ct.ROOT


def world_trans(chunk56: np.ndarray, anchor_pose9: dict, side: str) -> np.ndarray:
    """[horizon, 3] ABSOLUTE wrist translation in mm, via the serving decoder."""
    decoded = rb.decode_action_chunk(chunk56, anchor_pose9)
    return np.asarray(decoded[f"{side}_eef"][:, :3], dtype=np.float64) * 1000.0


def cover_steps(rtc: int) -> int:
    """Steps from an observation to the take-over of the chunk it produced.

    The fetch fires after local step ``L-rtc-1`` of the running slice and the
    successor takes over after local step ``L-1``, so the gap is ``rtc+1``
    periods regardless of EXECUTE_START or EXECUTE_LENGTH.
    """
    return int(rtc) + 1


def gt_world(frames, side: str, frame_index: int) -> np.ndarray:
    """The corpus wrist translation at one frame, mm, in the interface frame."""
    key = f"observation.{side}_wrist_pose"
    pose9 = ct.pose9_from_pose12(frames[key][frame_index])
    return np.asarray(pose9[:3], dtype=np.float64) * 1000.0


def anchor_at(frames, frame_index: int) -> dict:
    """The pose9 pair the serving decoder would anchor on at one frame."""
    return {
        side: ct.pose9_from_pose12(frames[f"observation.{side}_wrist_pose"][frame_index])
        for side in ("left", "right")
    }


def evaluate(frames, chunk56, obs_frame: int, side: str, rtc: int, candidates):
    """Per-candidate EXECUTE_START error against ground truth, in world frame.

    Chunk step ``k`` is trained to BE the pose at ``obs_frame + (k+1)*stride``.
    At take-over the arm sits at ``obs_frame + cover*stride`` and the command it
    is given must be reached one period later, i.e. at
    ``obs_frame + (cover+1)*stride``. So the correct first executed step is the
    ``k`` with ``k+1 == cover+1``.

    Returns, per candidate: the world distance between the commanded pose and the
    ground-truth pose that is due at that instant, and the size of the commanded
    step against the ground truth's own motion over one period.
    """
    stride = rb.ACTION_STRIDE
    cover = cover_steps(rtc)
    world = world_trans(chunk56, anchor_at(frames, obs_frame), side)
    arm_now = gt_world(frames, side, obs_frame + cover * stride)
    due_next = gt_world(frames, side, obs_frame + (cover + 1) * stride)
    gt_step = float(np.linalg.norm(due_next - arm_now))
    out = {}
    for es in candidates:
        commanded = world[es]
        out[es] = {
            "error_mm": float(np.linalg.norm(commanded - due_next)),
            "step_mm": float(np.linalg.norm(commanded - arm_now)),
        }
    return out, gt_step


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
    parser.add_argument("--starts", type=int, nargs="+",
                        default=[120, 600, 900, 1200, 1500])
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--rtc", type=int, default=srv.RTC_STEPS)
    parser.add_argument("--candidates", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/004000")
    args = parser.parse_args()

    span = discover_ur_episodes(ROOT, max_episodes=args.episode + 1)[args.episode]
    frames, _ = read_ur_episode(span)
    total = len(next(iter(frames.values())))

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
    report(args, frames, total, sample, noises)


def report(args, frames, total, sample, noises) -> None:
    cover = cover_steps(args.rtc)
    stride = rb.ACTION_STRIDE
    print(f"\n=== take-over command vs ground truth, world frame, episode {args.episode} ===")
    print(f"    rtc={args.rtc} -> cover={cover} steps; the arm is at "
          f"obs+{cover * stride} frames when the chunk takes over,")
    print(f"    and the command must be REACHED at obs+{(cover + 1) * stride} frames.")
    print(f"    So the time-consistent EXECUTE_START is {cover}; "
          f"server.py uses {srv.EXECUTE_START}.")
    print(f"    error_mm = |commanded - GT due at take-over+1|, lower is better.\n")

    totals = {es: [] for es in args.candidates}
    for start in args.starts:
        need = start + (cover + 1) * stride
        if need >= total or start + rb.ACTION_HORIZON * stride >= total:
            print(f"  --- start {start}: skipped (needs frame {need} of {total}) ---")
            continue
        anchor, hrows, jpeg, _gt = ct.build_window(args.episode, start)
        state56 = rb.build_state_rows(anchor, hrows)
        chunks = [sample(state56, jpeg, n) for n in noises]
        print(f"  --- start {start} ---")
        for side in ("left", "right"):
            per_draw = [
                evaluate(frames, c, start, side, args.rtc, args.candidates)
                for c in chunks
            ]
            gt_step = float(np.mean([g for _, g in per_draw]))
            print(f"    {side}: GT moves {gt_step:5.2f}mm in the period after take-over")
            for es in args.candidates:
                err = float(np.mean([d[es]["error_mm"] for d, _ in per_draw]))
                step = float(np.mean([d[es]["step_mm"] for d, _ in per_draw]))
                totals[es].append(err)
                tag = ""
                if es == srv.EXECUTE_START:
                    tag += " (current)"
                if es == cover:
                    tag += " (derived)"
                print(f"      ES={es}: error {err:6.2f}mm  commanded step "
                      f"{step:6.2f}mm vs GT {gt_step:5.2f}mm{tag}")
        print()
    print("  === mean error across windows ===")
    ranked = sorted((np.mean(v), es) for es, v in totals.items() if v)
    for err, es in ranked:
        tag = ""
        if es == srv.EXECUTE_START:
            tag += " (current)"
        if es == cover:
            tag += " (derived)"
        print(f"    ES={es}: {err:6.2f}mm{tag}")
    if ranked:
        print(f"\n  best measured EXECUTE_START = {ranked[0][1]}")


if __name__ == "__main__":
    main()
