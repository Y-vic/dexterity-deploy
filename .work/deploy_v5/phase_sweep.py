"""Does the model's output TRACK THE TASK PHASE, or is every chunk the same chunk?

The symptom on hardware is not wrong magnitude -- it is absence of progress. Both
hands jitter around a point instead of the left hand descending onto the bottle
and the fingers closing. Two very different causes produce that:

  (a) the live observation is out of distribution, so the model is extrapolating
  (b) the model ignores its conditioning and emits ~the training mean every time

They are distinguished by feeding windows the model definitely trained on, drawn
from DIFFERENT PHASES of one episode, and asking whether the output changes with
the phase. Ground truth for each window comes from the same corpus, so "changes"
can be checked against "changes the way it should".

Three things get measured per window:

  net displacement per side          -- is there a trend at all, or only jitter
  cosine(model, GT) on that vector   -- does the trend point where it should
  net hand change per side           -- do the fingers move toward closing

and one thing across windows: the spread of model chunks versus the spread of GT
chunks. If GT varies by centimetres between phases while the model varies by
millimetres, the model has collapsed to the mean regardless of how plausible any
single chunk looks. That is the failure mode a single-window test cannot see --
and the one the closing test missed by reporting only per-window magnitude.

Runs on the REMOTE with its own model copy, so it needs a free GPU. It does not
touch the serving process.
"""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import repr_bridge as rb

RUN = (
    "/share/project/zjk/orca/outputs/"
    "mot_action_v5_ur_unscrew_16gpu_alr3p5e4_vlr3p5e5_aw0p15_bs8_wu150_st4000"
)
NORM = (
    "/share/project/zjk/orca/outputs/shared_mot_action_v5_ur_unscrew/"
    "hand_action_norm.json"
)


def trend(chunk: np.ndarray) -> dict:
    """Net motion of one chunk: where it ends up relative to where it started.

    Net, not path length. Path length says "something moved"; jitter has plenty of
    it. Net says "it got somewhere", which is exactly the property the hardware is
    missing, so it is what has to be compared against GT.
    """
    out = {}
    for side in ("left", "right"):
        trans = chunk[:, rb.SLICES[f"{side}_wrist_trans"]]
        hand = chunk[:, rb.SLICES[f"{side}_hand_joints"]]
        # The chunk is relative to the action anchor, so row -1 IS the net vector.
        net = np.asarray(trans[-1], dtype=np.float64)
        out[side] = {
            "net": net,
            "net_mm": float(np.linalg.norm(net) * 1000),
            "dz_mm": float(net[2] * 1000),
            # Mean over the 22 joints: closing is a coordinated move, and a single
            # joint's delta is dominated by whichever finger happens to lead.
            "hand_net_rad": float(np.asarray(hand[-1]).mean()),
        }
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def make_sampler(model, normalizer):
    """Return sample(state56, jpeg, noise) -> denormalized 56D chunk.

    Noise is a parameter rather than drawn inside: comparing two windows means
    holding it fixed, or the 1.5 mm sampling spread measured earlier would be
    mixed into every difference reported below.
    """
    import torch
    from PIL import Image

    def sample(state56: np.ndarray, jpeg: bytes, noise) -> np.ndarray:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((256, 256))
        state = normalizer.normalize_state(torch.from_numpy(state56))
        batch = [
            {
                "image": [image],
                "lang": ct.UR_TASK,
                "state": state,
                "_embodiment": "robot",
            }
        ]
        with torch.no_grad():
            out = model._predict_action_mot(batch, noise=noise)
        chunk = out[next(iter(out))].float().cpu()
        chunk = normalizer.denormalize(chunk)
        chunk = chunk[0] if chunk.ndim == 3 else chunk
        return chunk.numpy().astype(np.float32)

    return sample


def main() -> None:
    import argparse

    import torch

    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--starts",
        type=int,
        nargs="+",
        # Spread over an 1860-frame episode: early approach, two mid-task points,
        # and late. Whatever the phase boundaries actually are, these cannot all
        # be the same phase, which is all this test needs.
        default=[120, 400, 620, 900, 1200, 1600],
    )
    parser.add_argument("--checkpoint", default=f"{RUN}/checkpoints/001500")
    args = parser.parse_args()

    model = ct_load(args.checkpoint)
    normalizer = HandActionNormalizer.load(NORM)
    sample = make_sampler(model, normalizer)

    torch.manual_seed(0)
    noise = torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")

    print(f"\n=== phase sweep, episode {args.episode}, noise pinned ===")
    print(f"    checkpoint {args.checkpoint}")
    rows = []
    for start in args.starts:
        try:
            anchor, hrows, jpeg, gt = ct.build_window(args.episode, start)
        except Exception as exc:  # a start past the end, or an unreadable frame
            print(f"  start {start:5}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        state56 = rb.build_state_rows(anchor, hrows)
        pred = sample(state56, jpeg, noise)
        rows.append({"start": start, "gt": gt, "pred": pred, "jpeg": jpeg,
                     "state56": state56})
        report(start, gt, pred)

    if len(rows) >= 2:
        collapse(rows)
        image_ablation(rows, sample, noise)


def report(start: int, gt: np.ndarray, pred: np.ndarray) -> None:
    """One window: GT trend beside model trend, with the direction agreement."""
    g, p = trend(gt), trend(pred)
    print(f"\n  --- start {start} ---")
    for side in ("left", "right"):
        cos = cosine(g[side]["net"], p[side]["net"])
        print(
            f"    {side:5} GT  net {g[side]['net_mm']:6.1f}mm  dz {g[side]['dz_mm']:+7.1f}mm"
            f"  hand {g[side]['hand_net_rad']:+.4f}rad"
        )
        print(
            f"    {side:5} OUT net {p[side]['net_mm']:6.1f}mm  dz {p[side]['dz_mm']:+7.1f}mm"
            f"  hand {p[side]['hand_net_rad']:+.4f}rad   cos={cos:+.3f}"
        )


def collapse(rows: list) -> None:
    """Compare how much GT varies across phases with how much the model varies.

    This is the decisive number. A model that has collapsed to the training mean
    still produces a plausible chunk for every window -- it just produces nearly
    the SAME one, so the ratio below comes out far below 1. A healthy model tracks
    its conditioning and lands near 1.
    """
    print("\n  --- variation across phases ---")
    for side in ("left", "right"):
        gt_nets = np.stack([trend(r["gt"])[side]["net"] for r in rows])
        pr_nets = np.stack([trend(r["pred"])[side]["net"] for r in rows])
        # Spread of the net vectors around their own mean, in mm.
        gt_spread = float(np.linalg.norm(gt_nets - gt_nets.mean(0), axis=1).mean() * 1000)
        pr_spread = float(np.linalg.norm(pr_nets - pr_nets.mean(0), axis=1).mean() * 1000)
        gt_hand = np.array([trend(r["gt"])[side]["hand_net_rad"] for r in rows])
        pr_hand = np.array([trend(r["pred"])[side]["hand_net_rad"] for r in rows])
        print(
            f"    {side:5} wrist spread  GT {gt_spread:6.1f}mm  OUT {pr_spread:6.1f}mm"
            f"   ratio {pr_spread / max(gt_spread, 1e-6):5.2f}"
        )
        print(
            f"    {side:5} hand  spread  GT {gt_hand.std():.4f}rad  "
            f"OUT {pr_hand.std():.4f}rad   ratio "
            f"{pr_hand.std() / max(gt_hand.std(), 1e-9):5.2f}"
        )


def image_ablation(rows: list, sample, noise) -> None:
    """Hold the state, swap the image: does the picture change the decision?

    The state channel was already shown to work. The image channel never was, and
    it is the one that has to carry "the cap is still on" versus "the cap is off".
    Same construction as the state test: noise pinned, one input varied, so any
    difference is attributable.

    A near-zero difference here means the model is driving on state alone -- which
    would explain jitter with no progress exactly, because the state of a hand that
    is holding still looks the same at every phase of the task.
    """
    first, last = rows[0], rows[-1]
    print("\n  --- image ablation (state held from start"
          f" {first['start']}, image swapped) ---")
    base = sample(first["state56"], first["jpeg"], noise)
    swapped = sample(first["state56"], last["jpeg"], noise)
    wrist = float(np.abs(base[:, 0:12] - swapped[:, 0:12]).max())
    hand = float(np.abs(base[:, 12:56] - swapped[:, 12:56]).max())
    print(f"    image from start {last['start']}:  wrist max|d|={wrist*1000:8.3f}mm"
          f"  hand max|d|={hand:7.4f}rad")
    print(f"    image changes the chunk: {wrist > 1e-5 or hand > 1e-5}")


def ct_load(checkpoint: str):
    """Reuse the state test's loader: same build, same explicit weight load."""
    import state_sensitivity as ss

    return ss.load_model(f"{RUN}/hydra/.hydra/config.yaml", checkpoint)


if __name__ == "__main__":
    main()
