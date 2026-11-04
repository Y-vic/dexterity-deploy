"""What the arm actually advances per cycle, before and after the slice fix.

Before: the chunk executed steps 0..9 anchored on a pose the arm had already
left, so the first steps retracted. After: it executes steps EXECUTE_START..9,
which are the steps whose horizons lie ahead of the take-over moment.

The comparison that matters is against GROUND TRUTH OVER THE SAME INTERVAL. A
cycle covers EXECUTE_LENGTH steps of wall clock, so the GT leg to compare with is
GT's own displacement over that same span -- not GT's whole chunk, which would
make any fix look worse than it is.

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


def leg(chunk: np.ndarray, side: str, a: int, b: int) -> float:
    trans = np.asarray(chunk[:, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64)
    return float(np.linalg.norm(trans[b] - trans[a]) * 1000)


def main() -> None:
    import argparse
    import io as _io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--starts", type=int, nargs="+", default=[120, 900, 1600])
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/001500")
    args = parser.parse_args()

    model = ps.ct_load(args.checkpoint)
    normalizer = HandActionNormalizer.load(ps.NORM)
    start_step, length = srv.EXECUTE_START, srv.EXECUTE_LENGTH

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
    report(args, sample, noises, start_step, length)


def report(args, sample, noises, start_step: int, length: int) -> None:
    print(f"\n=== per-cycle advance, episode {args.episode} ===")
    print(f"    slice = steps [{start_step}, {start_step + length}) of {rb.ACTION_HORIZON}")
    print(f"    GT leg compared over the SAME {length}-step span\n")
    for start in args.starts:
        anchor, hrows, jpeg, gt = ct.build_window(args.episode, start)
        state56 = rb.build_state_rows(anchor, hrows)
        chunks = [sample(state56, jpeg, n) for n in noises]
        print(f"  --- start {start} ---")
        for side in ("left", "right"):
            # GT advance over the executed span, measured from the same offset.
            gt_leg = leg(gt, side, start_step, start_step + length - 1)
            # BEFORE: executed 0..9 from a pose already `start_step` steps stale,
            # so net progress was the whole-chunk net minus the retraction.
            before = np.mean(
                [leg(c, side, 0, -1) - leg(c, side, 0, start_step) for c in chunks]
            )
            # AFTER: executes the tail; every step of it lies ahead of take-over.
            after = np.mean(
                [leg(c, side, start_step, start_step + length - 1) for c in chunks]
            )
            print(
                f"    {side:5} GT {gt_leg:6.1f}mm | before {before:6.1f}mm "
                f"({before / max(gt_leg, 1e-6) * 100:5.1f}%)  "
                f"after {after:6.1f}mm ({after / max(gt_leg, 1e-6) * 100:5.1f}%)"
            )


if __name__ == "__main__":
    main()
