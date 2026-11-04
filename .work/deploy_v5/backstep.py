"""Does the EXECUTED SLICE itself contain backward steps? Model vs corpus.

The slice arithmetic is settled: a closed-loop simulation of the serving loop
(wall clock, arm state carried forward) shows EXECUTE_START == RTC_STEPS gives a
boundary increment of exactly one period, and late inference costs progress RATE
without breaking alignment. So a retraction visible on hardware DURING a descent
is not a boundary artefact -- it has to be inside the chunk.

This measures, over the executed slice only, the component of each step that
points AGAINST the slice's overall direction of travel. GT is measured the same
way from the corpus, so "the data does this too" is distinguishable from "the
policy adds it".

back_frac = summed backward component / summed absolute step length. 0 = strictly
forward, 0.5 = as much back as forward.

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


def back_fraction(chunk: np.ndarray, side: str, start: int, length: int):
    """Backward-motion fraction of the executed slice, and its net/path ratio."""
    block = np.asarray(
        chunk[:, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64
    ) * 1000.0
    with_anchor = np.vstack([np.zeros((1, 3)), block])
    steps = np.diff(with_anchor, axis=0)[start:start + length]
    net = steps.sum(axis=0)
    norm = np.linalg.norm(net)
    if norm < 1e-9:
        return 0.0, 0.0
    unit = net / norm
    along = steps @ unit
    back = float(-along[along < 0].sum())
    path = float(np.abs(along).sum())
    return back / max(path, 1e-9), float(norm / max(
        np.linalg.norm(steps, axis=1).sum(), 1e-9))


def main() -> None:
    import argparse
    import io as _io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--first", type=int, default=120)
    parser.add_argument("--last", type=int, default=1700)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--draws", type=int, default=3)
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
    es, length = srv.EXECUTE_START, srv.EXECUTE_LENGTH
    print(f"\n=== backward motion inside the executed slice [{es}, {es+length}) ===")
    print("    back_frac: 0 = strictly forward, 0.5 = as much back as forward")
    print("    straight: net/path, 1.0 = a straight line\n")
    header = (f"{'start':>6} {'side':>6} {'model back':>11} {'GT back':>8} "
              f"{'model straight':>15} {'GT straight':>12}")
    print(header)
    print("-" * len(header))
    rows = []
    for start in range(args.first, args.last + 1, args.stride):
        try:
            anchor, hrows, jpeg, gt = ct.build_window(args.episode, start)
        except Exception:
            continue
        state56 = rb.build_state_rows(anchor, hrows)
        chunks = [sample(state56, jpeg, n) for n in noises]
        for side in ("left", "right"):
            mb = np.mean([back_fraction(c, side, es, length)[0] for c in chunks])
            ms = np.mean([back_fraction(c, side, es, length)[1] for c in chunks])
            gb, gs = back_fraction(gt, side, es, length)
            rows.append((mb, gb, ms, gs))
            print(f"{start:>6} {side:>6} {mb:>11.3f} {gb:>8.3f} {ms:>15.3f} "
                  f"{gs:>12.3f}")
    if rows:
        arr = np.asarray(rows)
        print("-" * len(header))
        print(f"{'mean':>6} {'':>6} {arr[:,0].mean():>11.3f} {arr[:,1].mean():>8.3f} "
              f"{arr[:,2].mean():>15.3f} {arr[:,3].mean():>12.3f}")
        print()
        print(f"  model backward fraction {arr[:,0].mean():.3f} vs GT "
              f"{arr[:,1].mean():.3f}")
        print(f"  model straightness {arr[:,2].mean():.3f} vs GT {arr[:,3].mean():.3f}")


if __name__ == "__main__":
    main()
