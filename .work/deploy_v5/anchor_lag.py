"""How much of each chunk's progress does the re-anchoring throw away?

Measured on hardware: inference latency 385 ms, inter-chunk gap 0.87 ms. So RTC
is doing its job -- the arm never waits. But that is also what makes the timing
error invisible:

    step 5 of chunk A   observation captured, pose P measured, fetch sent
    steps 6..9 of A     the arm keeps moving, ~385 ms, ~4 steps past P
    chunk B arrives     its step 0 is P + (relative step 0) -- i.e. back at P

Chunk B is anchored where the arm WAS when the observation was taken, not where
it is when B starts executing. Its first steps therefore command a return to a
stale pose. Net progress per cycle is not "the chunk's displacement" but "the
chunk's displacement minus the retraction", and the retraction is whatever the arm
covered during those 4 steps.

This script puts a number on it from real windows: displacement over the first
`lag` steps (thrown away) against net displacement over the chunk (intended).

Runs on the remote with its own model copy.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import phase_sweep as ps
import repr_bridge as rb

# 385 ms of inference at 10 Hz. Rounded down: the arm publishes on period, so a
# partial period has not produced a step yet.
LAG_STEPS = 3


def leg(chunk: np.ndarray, side: str, a: int, b: int) -> float:
    """Displacement in mm between two steps of a relative chunk."""
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
    parser.add_argument("--lag", type=int, default=LAG_STEPS)
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/001500")
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
    """Per window: what the chunk intends, and what survives the re-anchoring.

    `retract` is the displacement over the first `lag` steps. Those steps are
    commanded relative to a pose the arm has already left, so they undo motion
    already performed rather than adding to it. `effective` is what is left.
    """
    print(f"\n=== anchor lag, episode {args.episode}, lag={args.lag} steps "
          f"({args.lag * 100} ms at 10 Hz) ===")
    print(f"    {args.draws} noise draws per cell\n")
    for start in args.starts:
        anchor, hrows, jpeg, gt = ct.build_window(args.episode, start)
        state56 = rb.build_state_rows(anchor, hrows)
        chunks = [sample(state56, jpeg, n) for n in noises]
        print(f"  --- start {start} ---")
        for side in ("left", "right"):
            net = np.mean([leg(c, side, 0, -1) for c in chunks])
            retract = np.mean([leg(c, side, 0, args.lag) for c in chunks])
            gt_net = leg(gt, side, 0, -1)
            effective = net - retract
            print(
                f"    {side:5} GT {gt_net:6.1f}mm | chunk net {net:6.1f}mm  "
                f"first {args.lag} steps {retract:5.1f}mm  "
                f"effective {effective:6.1f}mm  "
                f"= {effective / max(gt_net, 1e-6) * 100:5.1f}% of GT"
            )


if __name__ == "__main__":
    main()
