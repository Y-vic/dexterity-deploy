"""Offline checkpoint-vs-ground-truth comparison for the current UR-SharpA run.

This process loads a second copy on a separate GPU.  It never connects to the
robot or to the resident policy server.  Sampling noise is pinned so variation
between windows is caused by the observation rather than the diffusion seed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closing_test as ct
import repr_bridge as rb
import state_sensitivity as ss


RUN = Path(
    "/share/project/zjk/orca/outputs/"
    "mot_action_v5_urscratch_16gpu_alr3p5e4_vlr3p5e5_aw0p15_bs8_wu400_st4000"
)
CHECKPOINT = RUN / "checkpoints/001750"
NORMALIZER = Path(
    "/share/project/zjk/orca/outputs/shared_mot_action_v5_urscratch/"
    "hand_action_norm.json"
)
EMBODIMENT = "robot"


def endpoint(chunk: np.ndarray, side: str) -> np.ndarray:
    return np.asarray(chunk[-1, rb.SLICES[f"{side}_wrist_trans"]], dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def main() -> None:
    import argparse
    import torch
    from PIL import Image
    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--starts", type=int, nargs="+", default=[120, 400, 620, 900, 1200, 1600])
    parser.add_argument("--export", default="/tmp/urscratch_gt_windows.npz")
    args = parser.parse_args()

    print(f"loading checkpoint {CHECKPOINT}", flush=True)
    model = ss.load_model(str(RUN / "hydra/.hydra/config.yaml"), str(CHECKPOINT))
    normalizer = HandActionNormalizer.load(str(NORMALIZER))
    print("model ready", flush=True)

    windows = []
    for start in args.starts:
        anchor, history, jpeg, gt = ct.build_window(args.episode, start)
        current_pose, current_hand = history[-1]
        state56 = rb.build_state_rows(anchor, history)
        image = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((256, 256))
        windows.append((start, current_pose, current_hand, state56, image, jpeg, gt))

    torch.manual_seed(20260917)
    fixed_noise = torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")

    results = []
    print("\n=== current checkpoint vs successful demonstration ===", flush=True)
    for start, current_pose, current_hand, state56, image, jpeg, gt in windows:
        state = normalizer.normalize_state(torch.from_numpy(state56), embodiment=EMBODIMENT)
        batch = [{"image": [image], "lang": ct.UR_TASK, "state": state, "_embodiment": EMBODIMENT}]
        with torch.no_grad():
            output = model._predict_action_mot(batch, noise=fixed_noise)
        raw = output[next(iter(output))].float().cpu()
        pred = normalizer.denormalize(raw, embodiment=EMBODIMENT)
        pred = (pred[0] if pred.ndim == 3 else pred).numpy().astype(np.float32)

        item = {"start": start, "gt": gt, "pred": pred, "pose": current_pose, "hand": current_hand}
        results.append(item)
        print(f"\nstart={start}")
        for side in ("left", "right"):
            g, p = endpoint(gt, side), endpoint(pred, side)
            rmse = float(np.sqrt(np.mean((pred[:, rb.SLICES[f'{side}_wrist_trans']] - gt[:, rb.SLICES[f'{side}_wrist_trans']]) ** 2)) * 1000)
            gh = gt[-1, rb.SLICES[f"{side}_hand_joints"]] - gt[0, rb.SLICES[f"{side}_hand_joints"]]
            ph = pred[-1, rb.SLICES[f"{side}_hand_joints"]] - pred[0, rb.SLICES[f"{side}_hand_joints"]]
            print(
                f"  {side:5} GT={np.round(g*1000,1)}mm OUT={np.round(p*1000,1)}mm "
                f"cos={cosine(g,p):+.3f} trans_rmse={rmse:.1f}mm "
                f"hand_delta_norm GT={np.linalg.norm(gh):.3f} OUT={np.linalg.norm(ph):.3f}",
                flush=True,
            )

    print("\n=== phase-conditioning spread ===")
    for side in ("left", "right"):
        gs = np.stack([endpoint(x["gt"], side) for x in results])
        ps = np.stack([endpoint(x["pred"], side) for x in results])
        gspread = float(np.linalg.norm(gs - gs.mean(0), axis=1).mean() * 1000)
        pspread = float(np.linalg.norm(ps - ps.mean(0), axis=1).mean() * 1000)
        print(f"  {side:5} GT spread={gspread:.1f}mm OUT spread={pspread:.1f}mm ratio={pspread/max(gspread,1e-9):.2f}")

    payload = {}
    for item in results:
        key = f"s{item['start']}"
        decoded = rb.decode_action_chunk(item["gt"], item["pose"])
        payload[f"{key}_gt_raw"] = item["gt"]
        payload[f"{key}_left_eef"] = decoded["left_eef"]
        payload[f"{key}_right_eef"] = decoded["right_eef"]
        payload[f"{key}_hand"] = np.concatenate([decoded["left_hand"], decoded["right_hand"]], axis=1)
        payload[f"{key}_anchor_left"] = item["pose"]["left"]
        payload[f"{key}_anchor_right"] = item["pose"]["right"]
        payload[f"{key}_anchor_left_hand"] = item["hand"]["left"]
        payload[f"{key}_anchor_right_hand"] = item["hand"]["right"]
    np.savez(args.export, **payload)
    print(f"\nexported GT deployment inputs to {args.export}", flush=True)


if __name__ == "__main__":
    main()
