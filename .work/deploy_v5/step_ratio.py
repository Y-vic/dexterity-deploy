"""Why the arm looks FAST while the chunk underpredicts: the profile is wrong.

Two facts that seem to contradict each other, both measured:

  net displacement per chunk   0.5-0.66x GT   (underprediction, known)
  commanded step at take-over  2-7x GT        (two_chunk.py, every window)

They are the same fact. The model's chunks are FRONT-LOADED: they put most of
their motion into the first few steps and flatten out, while the corpus keeps a
steadier speed across the horizon. Serving only ever executes a slice, so what
reaches the arm is a sample of the profile, not the net -- and with EXECUTE_START
near the front, the executed steps are the fast ones. The arm therefore moves
FASTER per step than the corpus while getting LESS FAR per chunk.

That is what "faster during the grasp than during the descent" reads as: the
grasp phase is where the corpus itself is fastest, so the same front-loading
multiplier lands on a bigger base.

This measures the profile ratio (first step / mean step) for model and GT, over a
dense sweep, and the executed-slice speed against the GT speed over the same
wall-clock span. It is the number a retrain has to move; no slice choice fixes it,
which is why the serving-side answer is a ceiling, not a re-slice.

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


def profile(chunk: np.ndarray, side: str) -> np.ndarray:
    """Per-step speed in mm, anchor-relative, over the whole horizon."""
    block = np.asarray(
        chunk[:, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64
    ) * 1000.0
    with_anchor = np.vstack([np.zeros((1, 3)), block])
    return np.linalg.norm(np.diff(with_anchor, axis=0), axis=1)


def main() -> None:
    import argparse
    import io as _io

    import torch
    from PIL import Image

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--first", type=int, default=150)
    parser.add_argument("--last", type=int, default=1700)
    parser.add_argument("--stride", type=int, default=150)
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
    report(args, sample, noises)


def report(args, sample, noises) -> None:
    es, length = srv.EXECUTE_START, srv.EXECUTE_LENGTH
    print(f"\n=== profile shape, model vs GT, episode {args.episode} ===")
    print(f"    executed slice = steps [{es}, {es + length}) of {rb.ACTION_HORIZON}")
    print("    front = first step / mean step over the horizon. GT ~1 means a flat")
    print("    profile; model >1 means front-loaded, so the executed slice runs hot.\n")
    header = (f"{'start':>6} {'side':>6} {'model front':>12} {'GT front':>9} "
              f"{'exec mm/step':>13} {'GT mm/step':>11} {'ratio':>7}")
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
            mp = np.mean([profile(c, side) for c in chunks], axis=0)
            gp = profile(gt, side)
            m_front = float(mp[0] / max(mp.mean(), 1e-9))
            g_front = float(gp[0] / max(gp.mean(), 1e-9))
            m_exec = float(mp[es:es + length].mean())
            g_exec = float(gp[es:es + length].mean())
            ratio = m_exec / max(g_exec, 1e-9)
            rows.append((m_front, g_front, ratio))
            print(f"{start:>6} {side:>6} {m_front:>12.2f} {g_front:>9.2f} "
                  f"{m_exec:>13.2f} {g_exec:>11.2f} {ratio:>7.2f}")
    if rows:
        arr = np.asarray(rows)
        print("-" * len(header))
        print(f"{'mean':>6} {'':>6} {arr[:,0].mean():>12.2f} {arr[:,1].mean():>9.2f} "
              f"{'':>13} {'':>11} {arr[:,2].mean():>7.2f}")
        print()
        print(f"  model front-loading {arr[:,0].mean():.2f}x vs GT "
              f"{arr[:,1].mean():.2f}x")
        print(f"  executed-slice speed vs GT over the same span: "
              f"{arr[:,2].mean():.2f}x (median {np.median(arr[:,2]):.2f}x)")
        hot = int((arr[:, 2] > 1.5).sum())
        print(f"  windows where the executed slice runs >1.5x GT speed: "
              f"{hot}/{len(arr)}")


if __name__ == "__main__":
    main()
