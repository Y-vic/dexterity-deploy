"""Teacher-forcing diagnosis for the UR-SharpA ``urhandcmd`` checkpoint.

The script is intentionally offline: it loads a checkpoint on a spare GPU, reads
one successful training episode through the *training* reader, and never opens a
robot or policy-server connection.  Unlike the older diagnostics, the ground-truth
chunk passes ``command_frames`` into ``build_action_chunk``.  Therefore its hand
targets are the timestamp-resampled Manus q_teleop commands used by urhandcmd,
while state remains achieved q exactly as it did during training.

Besides ordinary wrist and hand errors, the report measures the right-wrist pose
relative to the left wrist.  That is the quantity most directly related to the
hardware symptom: the left hand holds the bottle, then the right hand collides
instead of arriving over the cap.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO = Path("/share/project/zjk/orca")
DEPLOY = REPO / "scripts/deploy/sharpa_policy_server"
RUN = REPO / "outputs/mot_action_v5_urhandcmd_16gpu_alr3p5e4_vlr3p5e5_aw0p064_bs8_wu400_st4000"
CHECKPOINT = RUN / "checkpoints/002500"
NORMALIZER = REPO / "outputs/shared_mot_action_v5_urhandcmd/hand_action_norm.json"
DATASET = Path(
    "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
    "lerobot_data/unscrew_cap"
)
TASK = "Right hand unscrews the blue cap while left hand holds the bottle steady"
EMBODIMENT = "robot"
FIXED_NOISE_SEED = 20260917

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DEPLOY))

import repr_bridge as rb  # noqa: E402
import state_sensitivity as ss  # noqa: E402


def pose9_from_pose12(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32)
    return np.concatenate([row[:3], row[3:9]]).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan")


def _valid_start(value: int, length: int) -> int:
    lo = rb.STATE_ANCHOR_OFFSET
    hi = length - rb.ACTION_HORIZON * rb.ACTION_STRIDE - 1
    return int(np.clip(value, lo, hi))


def discover_phase_starts(frames: dict, commands: dict) -> tuple[list[int], dict]:
    """Locate grip and right-approach landmarks from the demonstration itself."""

    left_p = np.asarray(frames["observation.left_wrist_pose"], dtype=np.float64)[:, :3]
    right_p = np.asarray(frames["observation.right_wrist_pose"], dtype=np.float64)[:, :3]
    left_h = np.asarray(commands["observation.left_qpos"], dtype=np.float64)
    length = len(left_p)

    # q_teleop is smooth at 30 Hz now.  A 12-frame difference is robust to small
    # per-frame noise and localizes the main grip transition rather than one joint.
    lag = 12
    hand_motion = np.zeros(length)
    hand_motion[lag:] = np.linalg.norm(left_h[lag:] - left_h[:-lag], axis=1)
    grip = int(np.argmax(hand_motion))

    wrist_distance = np.linalg.norm(right_p - left_p, axis=1)
    horizon = rb.ACTION_HORIZON * rb.ACTION_STRIDE
    distance_drop = np.full(length, np.nan)
    distance_drop[:-horizon] = wrist_distance[horizon:] - wrist_distance[:-horizon]

    # Search after the main left-hand grip.  The most negative change in inter-wrist
    # distance is the right hand's strongest approach toward the held bottle.
    search_lo = min(max(grip + 1, rb.STATE_ANCHOR_OFFSET), length - horizon - 1)
    search_hi = length - horizon
    segment = distance_drop[search_lo:search_hi]
    approach = search_lo + int(np.nanargmin(segment))
    closest = search_lo + int(np.argmin(wrist_distance[search_lo:]))

    # Cover the transition densely, plus one window before grip and near closest
    # approach.  These are start frames on the native 30 Hz recording; each action
    # chunk spans 30 frames = ten 10 Hz commands.
    candidates = [
        grip - 90,
        grip - 30,
        grip,
        grip + 30,
        approach - 90,
        approach - 60,
        approach - 30,
        approach,
        approach + 30,
        approach + 60,
        closest - 60,
        closest - 30,
        closest,
    ]
    starts = sorted({_valid_start(x, length) for x in candidates})
    def separated_extrema(values: np.ndarray, *, smallest: bool = False) -> list[int]:
        finite = np.flatnonzero(np.isfinite(values))
        order = finite[np.argsort(values[finite])]
        if not smallest:
            order = order[::-1]
        picked: list[int] = []
        for index in order:
            if all(abs(int(index) - old) >= 45 for old in picked):
                picked.append(int(index))
            if len(picked) == 8:
                break
        return sorted(picked)

    hand_peaks = separated_extrema(hand_motion)
    approach_peaks = separated_extrema(distance_drop, smallest=True)
    events = {
        "episode_frames": length,
        "grip_frame": grip,
        "grip_motion_norm_12f": float(hand_motion[grip]),
        "right_approach_frame": approach,
        "distance_drop_30f_mm": float(distance_drop[approach] * 1000.0),
        "closest_frame": closest,
        "closest_wrist_distance_mm": float(wrist_distance[closest] * 1000.0),
        "selected_starts": starts,
        "left_hand_motion_peaks": [
            {"frame": index, "norm_12f": float(hand_motion[index])}
            for index in hand_peaks
        ],
        "inter_wrist_approach_peaks": [
            {"frame": index, "drop_30f_mm": float(distance_drop[index] * 1000.0)}
            for index in approach_peaks
        ],
    }
    return starts, events


def write_overview(span, frames: dict, commands: dict, output: Path) -> None:
    """Write a labeled contact sheet so phase selection is visually auditable."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flagscale.models.vla.action_model.hand_action_video import decode_frames_at

    length = span.num_frames
    sample_frames = sorted(set(np.linspace(20, length - 35, 20, dtype=int).tolist()))
    images = decode_frames_at(
        span.video.path,
        sample_frames,
        size=(256, 256),
        frame_offset=span.video.frame_offset,
        num_frames=span.video.num_frames,
    )
    left_p = np.asarray(frames["observation.left_wrist_pose"], dtype=np.float64)[:, :3]
    right_p = np.asarray(frames["observation.right_wrist_pose"], dtype=np.float64)[:, :3]
    left_h = np.asarray(commands["observation.left_qpos"], dtype=np.float64)
    hand_from_start = np.linalg.norm(left_h - left_h[0], axis=1)
    wrist_distance = np.linalg.norm(right_p - left_p, axis=1) * 1000.0

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for axis, index in zip(axes.flat, sample_frames):
        axis.imshow(images[index])
        axis.set_title(
            f"f={index}  Lhand={hand_from_start[index]:.2f}\n"
            f"wrist dist={wrist_distance[index]:.0f} mm",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle("Episode 79 phase overview (training ego frames)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def current_pose(frames: dict, start: int) -> dict[str, np.ndarray]:
    return {
        side: pose9_from_pose12(frames[f"observation.{side}_wrist_pose"][start])
        for side in ("left", "right")
    }


def metric_row(start: int, gt: np.ndarray, pred: np.ndarray, pose: dict) -> dict:
    gt_dec = rb.decode_action_chunk(gt, pose)
    pr_dec = rb.decode_action_chunk(pred, pose)
    row: dict[str, float | int] = {"start": start}

    for side in ("left", "right"):
        gt_xyz = np.asarray(gt_dec[f"{side}_eef"], dtype=np.float64)[:, :3]
        pr_xyz = np.asarray(pr_dec[f"{side}_eef"], dtype=np.float64)[:, :3]
        error_mm = np.linalg.norm(pr_xyz - gt_xyz, axis=1) * 1000.0
        gt_rel = gt[:, rb.SLICES[f"{side}_wrist_trans"]]
        pr_rel = pred[:, rb.SLICES[f"{side}_wrist_trans"]]
        hand_gt = gt[:, rb.SLICES[f"{side}_hand_joints"]]
        hand_pr = pred[:, rb.SLICES[f"{side}_hand_joints"]]
        row[f"{side}_wrist_mean_err_mm"] = float(error_mm.mean())
        row[f"{side}_wrist_endpoint_err_mm"] = float(error_mm[-1])
        row[f"{side}_wrist_max_err_mm"] = float(error_mm.max())
        row[f"{side}_wrist_endpoint_cos"] = cosine(gt_rel[-1], pr_rel[-1])
        row[f"{side}_hand_mean_mae_rad"] = float(np.abs(hand_pr - hand_gt).mean())
        row[f"{side}_hand_endpoint_mae_rad"] = float(
            np.abs(hand_pr[-1] - hand_gt[-1]).mean()
        )
        row[f"{side}_hand_endpoint_max_rad"] = float(
            np.abs(hand_pr[-1] - hand_gt[-1]).max()
        )

    gt_left = np.asarray(gt_dec["left_eef"], dtype=np.float64)[:, :3]
    gt_right = np.asarray(gt_dec["right_eef"], dtype=np.float64)[:, :3]
    pr_left = np.asarray(pr_dec["left_eef"], dtype=np.float64)[:, :3]
    pr_right = np.asarray(pr_dec["right_eef"], dtype=np.float64)[:, :3]
    gt_pair = gt_right - gt_left
    pr_pair = pr_right - pr_left
    pair_error = np.linalg.norm(pr_pair - gt_pair, axis=1) * 1000.0
    endpoint_vector_mm = (pr_pair[-1] - gt_pair[-1]) * 1000.0
    row["pair_mean_err_mm"] = float(pair_error.mean())
    row["pair_endpoint_err_mm"] = float(pair_error[-1])
    row["pair_endpoint_dx_mm"] = float(endpoint_vector_mm[0])
    row["pair_endpoint_dy_mm"] = float(endpoint_vector_mm[1])
    row["pair_endpoint_dz_mm"] = float(endpoint_vector_mm[2])
    row["gt_endpoint_wrist_distance_mm"] = float(np.linalg.norm(gt_pair[-1]) * 1000.0)
    row["pred_endpoint_wrist_distance_mm"] = float(np.linalg.norm(pr_pair[-1]) * 1000.0)
    return row


def write_plot(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["start"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes[0, 0].plot(x, [r["right_wrist_endpoint_err_mm"] for r in rows], "o-")
    axes[0, 0].set_ylabel("Right wrist endpoint error (mm)")
    axes[0, 1].plot(x, [r["pair_endpoint_err_mm"] for r in rows], "o-", color="C1")
    axes[0, 1].set_ylabel("Inter-wrist endpoint error (mm)")
    axes[1, 0].plot(x, [r["right_hand_endpoint_mae_rad"] for r in rows], "o-", color="C2")
    axes[1, 0].set_ylabel("Right hand endpoint MAE (rad)")
    axes[1, 1].plot(
        x,
        [r["gt_endpoint_wrist_distance_mm"] for r in rows],
        "o-",
        label="GT",
    )
    axes[1, 1].plot(
        x,
        [r["pred_endpoint_wrist_distance_mm"] for r in rows],
        "o-",
        label="model",
    )
    axes[1, 1].set_ylabel("Endpoint wrist distance (mm)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("episode frame")
    fig.suptitle("urhandcmd_002500 teacher-forcing, episode 79")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    import torch
    from PIL import Image
    from flagscale.models.vla.action_model import hand_action_repr as train
    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )
    from flagscale.models.vla.action_model.hand_action_norm import HandActionNormalizer
    from flagscale.models.vla.action_model.hand_action_video import decode_frames_at

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=79)
    parser.add_argument("--starts", type=int, nargs="+")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output-dir", default="/tmp/urhandcmd_ep79_teacher_forcing")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spans = discover_ur_episodes(str(DATASET), max_episodes=args.episode + 1)
    span = spans[args.episode]
    frames, commands = read_ur_episode(span)
    starts, events = discover_phase_starts(frames, commands)
    if args.starts:
        starts = sorted({_valid_start(x, span.num_frames) for x in args.starts})
        events["selected_starts"] = starts
    print("=== demonstration phase landmarks ===", flush=True)
    print(json.dumps(events, indent=2), flush=True)
    write_overview(span, frames, commands, output_dir / "episode_overview.png")
    (output_dir / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    if args.scan_only:
        print(f"scan artifacts: {output_dir}", flush=True)
        return

    decoded = decode_frames_at(
        span.video.path,
        starts,
        size=(256, 256),
        frame_offset=span.video.frame_offset,
        num_frames=span.video.num_frames,
    )

    print(f"\nloading checkpoint {args.checkpoint}", flush=True)
    model = ss.load_model(str(RUN / "hydra/.hydra/config.yaml"), args.checkpoint)
    normalizer = HandActionNormalizer.load(str(NORMALIZER))
    torch.manual_seed(FIXED_NOISE_SEED)
    fixed_noise = torch.randn(1, rb.ACTION_HORIZON, rb.ACTION_DIM, device="cuda")
    print("model ready; deterministic diffusion noise pinned", flush=True)

    rows: list[dict] = []
    payload: dict[str, np.ndarray] = {}
    for start in starts:
        gt = train.build_action_chunk(
            frames,
            start,
            rb.ACTION_HORIZON,
            rb.ACTION_STRIDE,
            command_frames=commands,
        )
        state56 = train.build_state_rows(
            frames,
            start,
            rb.STATE_ANCHOR_OFFSET,
            rb.STATE_HISTORY_OFFSETS,
        )
        image = Image.fromarray(decoded[start]).convert("RGB")
        state = normalizer.normalize_state(
            torch.from_numpy(state56), embodiment=EMBODIMENT
        )
        batch = [
            {
                "image": [image],
                "lang": TASK,
                "state": state,
                "_embodiment": EMBODIMENT,
            }
        ]
        with torch.no_grad():
            output = model._predict_action_mot(batch, noise=fixed_noise)
        raw = output[next(iter(output))].float().cpu()
        pred = normalizer.denormalize(raw, embodiment=EMBODIMENT)
        pred = (pred[0] if pred.ndim == 3 else pred).numpy().astype(np.float32)

        pose = current_pose(frames, start)
        row = metric_row(start, gt, pred, pose)
        rows.append(row)
        print(
            f"frame {start:4d} | R wrist end {row['right_wrist_endpoint_err_mm']:6.1f} mm "
            f"| pair end {row['pair_endpoint_err_mm']:6.1f} mm "
            f"dxyz=[{row['pair_endpoint_dx_mm']:+5.1f},"
            f"{row['pair_endpoint_dy_mm']:+5.1f},{row['pair_endpoint_dz_mm']:+5.1f}] mm "
            f"| R hand end MAE {row['right_hand_endpoint_mae_rad']:.4f} rad",
            flush=True,
        )
        key = f"s{start}"
        payload[f"{key}_state"] = state56
        payload[f"{key}_gt"] = gt
        payload[f"{key}_pred"] = pred
        payload[f"{key}_image"] = np.asarray(decoded[start], dtype=np.uint8)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "windows.npz", **payload)
    write_plot(rows, output_dir / "metrics.png")

    worst = max(rows, key=lambda row: row["pair_endpoint_err_mm"])
    best = min(rows, key=lambda row: row["pair_endpoint_err_mm"])
    print("\n=== summary ===", flush=True)
    print(
        f"pair endpoint error: median={np.median([r['pair_endpoint_err_mm'] for r in rows]):.1f} mm "
        f"best={best['pair_endpoint_err_mm']:.1f} mm@{best['start']} "
        f"worst={worst['pair_endpoint_err_mm']:.1f} mm@{worst['start']}",
        flush=True,
    )
    print(f"artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
