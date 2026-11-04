"""Is the 2-5x under-prediction real, and does the step count cause it?

Two confounds have to be removed from the phase sweep's numbers before the
under-prediction can be called real:

  ONE NOISE DRAW. The sweep pinned noise so windows could be compared, but a
  single draw estimates nothing about the expected magnitude. Flow matching maps
  noise to output, so a small draw yields a small chunk. Averaged over draws here.

  TEN INTEGRATION STEPS. Too few Euler steps on a curved probability path
  systematically undershoots -- the integrator cuts the corner. That is exactly
  the shape of the observed error (direction right, magnitude short), so it is
  the first thing to rule in or out. Swept here.

Reported per window: GT net, then mean+-std of the model's net over draws, at each
step count. If the ratio is flat in steps, the integrator is not the cause and the
shortfall is in the fitted velocity field itself -- which would point at the action
loss weight or the step budget of training, not at anything serving does.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import phase_sweep as ps
import repr_bridge as rb


def beta_times(steps: int, a: float = 1.5, b: float = 1.0) -> list[float]:
    """Node times placed by the inverse CDF of the TRAINING flow-time density.

    Training draws t ~ Beta(a, b) * 0.999. With b == 1 the CDF is u = t**a, so the
    inverse is t = u**(1/a) and no special function is needed. Nodes are taken at
    the midpoints of `steps` equal-probability bins and sorted descending, so the
    integrator spends its steps where the velocity field was actually fitted
    instead of crowding them into t~0, which training barely visited.
    """
    quantiles = [(i + 0.5) / steps for i in range(steps)]
    nodes = [(q ** (1.0 / a)) * 0.999 for q in quantiles]
    return sorted(nodes, reverse=True)


def net_mm(chunk: np.ndarray, side: str) -> float:
    trans = chunk[:, rb.SLICES[f"{side}_wrist_trans"]]
    return float(np.linalg.norm(np.asarray(trans[-1])) * 1000)


def main() -> None:
    import argparse

    import torch
    from PIL import Image
    import io as _io

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--starts", type=int, nargs="+", default=[120, 900, 1600])
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--checkpoint", default=f"{ps.RUN}/checkpoints/001500")
    args = parser.parse_args()

    model = ps.ct_load(args.checkpoint)
    normalizer = HandActionNormalizer.load(ps.NORM)

    def sample(state56, jpeg, noise, times=None):
        image = Image.open(_io.BytesIO(jpeg)).convert("RGB").resize((256, 256))
        state = normalizer.normalize_state(torch.from_numpy(state56))
        batch = [{"image": [image], "lang": ct.UR_TASK, "state": state,
                  "_embodiment": "robot"}]
        with torch.no_grad():
            out = model._predict_action_mot(batch, noise=noise, times=times)
        chunk = out[next(iter(out))].float().cpu()
        chunk = normalizer.denormalize(chunk)
        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return chunk.numpy().astype(np.float32)

    # (label, times) -- None means the config's own uniform 10-step grid.
    configs = [
        ("uniform 10 (served)", None),
        ("uniform 30", [1.0 - i / 30 for i in range(30)]),
        ("beta 10", beta_times(10)),
        ("beta 30", beta_times(30)),
    ]
    run_sweep(args, sample, configs)


def run_sweep(args, sample, configs) -> None:
    """For each window, GT net beside the model's mean net over `draws` noises."""
    import torch

    print(f"\n=== magnitude vs integration schedule, episode {args.episode} ===")
    print(f"    {args.draws} noise draws per cell; ratio = OUT mean / GT\n")

    windows = []
    for start in args.starts:
        anchor, hrows, jpeg, gt = ct.build_window(args.episode, start)
        windows.append((start, rb.build_state_rows(anchor, hrows), jpeg, gt))

    torch.manual_seed(0)
    noises = [
        torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")
        for _ in range(args.draws)
    ]

    for label, times in configs:
        print(f"  {label}")
        for start, state56, jpeg, gt in windows:
            for side in ("left", "right"):
                gt_mm = net_mm(gt, side)
                outs = [net_mm(sample(state56, jpeg, n, times), side) for n in noises]
                mean, std = float(np.mean(outs)), float(np.std(outs))
                print(
                    f"    start {start:5} {side:5}  GT {gt_mm:6.1f}mm   "
                    f"OUT {mean:6.1f} +-{std:5.1f}mm   ratio {mean / max(gt_mm, 1e-6):5.2f}"
                )
        print()


if __name__ == "__main__":
    main()
