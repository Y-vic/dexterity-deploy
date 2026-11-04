#!/usr/bin/env python3
"""Resample the Manus hand teleop command onto the training time axis.

WHY THIS EXISTS
    unscrew_cap trains its hand action target from observation.state (achieved joint
    q). The Sharpa/Manus hand streams arrive at only ~15-21 Hz, but the LeRobot
    dataset is a 30 Hz grid filled by causal_latest_valid_hold. On that grid ~42-54%
    of hand rows are exact repeats of the previous row (arm rows: 0.3%), so grasp
    onset timing is quantized into a ~10-frame window and roughly half of hand
    frames teach "the fingers did not move". Under MSE the model hedges the onset --
    it closes the hand early and only ~two-thirds of the way, which is the observed
    hardware failure: touch, not grip.

    The parquet already carries q_teleop (action[62:106]), but that is the SAME
    hold-filled 30 Hz stream (55% repeats, holds up to 57 frames), so switching to it
    as-is is worse, not better. The fix goes back to the raw recording, where the
    command has its own arrival timestamps (q_cmd_timestamp_ns), and resamples that
    onto the master timeline. The model runs at 10 Hz -- below the ~17 Hz hand rate --
    so linear interpolation on true timestamps removes the aliasing, and the repeat
    fraction drops to ~0.04.

WHAT IT WRITES
    One sidecar per episode at <dataset_root>/hand_cmd/episode_%06d.npz holding
    q_cmd_resampled (N x 44, float32), row-aligned with that episode's parquet so
    N == output_rows. The 44 columns stay in RAW thumb-first order, which IS the
    canonical order build_action_chunk consumes -- verified as
    parquet[:, UR_HAND_PERMUTATION] == raw at 0.0 max error on episodes 79/0/3/8/9 --
    so NO permutation is applied here. The parquet and videos are never touched.

ALIGNMENT (meta/source_manifest.json + timeline/clock.npz)
    Each manifest entry maps episode_index -> source_episode with a
    crop_start_timeline_row and output_rows. clock.npz time_unix_ns is the 30 Hz
    master grid, built by the recorder; the converter only crops. So: interpolate
    onto the full grid, then slice [crop_start : crop_start + output_rows].

VALIDITY / NaN
    Only samples passing q_cmd_valid & ~isnan(q_cmd).any(axis=1) & (q_cmd_source == 2)
    feed the interpolation. Measured: NaNs occur ONLY in frames already flagged
    invalid, and q_cmd_source is uniformly 2 (Manus) on this corpus. np.interp clamps
    to the first/last valid sample outside their range, which is the desired
    hold-at-boundary behaviour.

USAGE
    python resample_ur_hand_cmd.py --check      # verify only, no write
    python resample_ur_hand_cmd.py              # all 115 episodes, real write
    python resample_ur_hand_cmd.py --limit 5    # first 5 only
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DATASET_ROOT = "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/lerobot_data/unscrew_cap"
RAW_ROOT = "/share/project/yenaisheng/datasets/postrain/ur-sharpa/raw_data/unscrew_cap"
HAND_DIM = 44
SOURCE_MANUS = 2


def load_manifest(dataset_root):
    path = os.path.join(dataset_root, "meta", "source_manifest.json")
    with open(path) as fh:
        return json.load(fh)["episodes"]


def resample_one(entry, raw_root):
    """Return (q_cmd_resampled [output_rows, 44] float32, stats dict)."""
    src = entry["source_episode"]
    crop = int(entry["crop_start_timeline_row"])
    n_out = int(entry["output_rows"])
    base = os.path.join(raw_root, src)

    joints = np.load(os.path.join(base, "sharpa", "joints_44.npz"))
    clock = np.load(os.path.join(base, "timeline", "clock.npz"))

    q_cmd = joints["q_cmd"].astype(np.float64)
    ts = joints["q_cmd_timestamp_ns"].astype(np.int64)
    valid = joints["q_cmd_valid"].astype(bool)
    source = joints["q_cmd_source"].astype(np.uint8)
    grid = clock["time_unix_ns"].astype(np.int64)

    good = valid & ~np.isnan(q_cmd).any(axis=1) & (source == SOURCE_MANUS)
    if int(good.sum()) < 2:
        raise ValueError(f"{src}: only {int(good.sum())} valid q_cmd samples")

    tv = ts[good].astype(np.float64)
    order = np.argsort(tv, kind="stable")
    tv = tv[order]
    vv = q_cmd[good][order]

    cols = [np.interp(grid.astype(np.float64), tv, vv[:, k]) for k in range(HAND_DIM)]
    full = np.stack(cols, axis=1)

    stop = crop + n_out
    if stop > full.shape[0]:
        raise ValueError(f"{src}: crop[{crop}:{stop}] exceeds master grid {full.shape[0]}")
    out = full[crop:stop].astype(np.float32)

    dur_s = (grid[-1] - grid[0]) / 1e9
    stats = {
        "source_episode": src,
        "crop_start": crop,
        "output_rows": n_out,
        "master_rows": int(full.shape[0]),
        "valid_sample_count": int(good.sum()),
        "raw_frames": int(q_cmd.shape[0]),
        "source_hz": round(float((len(np.unique(tv)) - 1) / dur_s), 2),
    }
    return out, stats


def repeat_fraction_stride(a, stride=3):
    """Whole-row repeat fraction sampled every `stride` rows (the chunk cadence)."""
    s = a[::stride]
    if s.shape[0] < 2:
        return 0.0
    return float((np.abs(np.diff(s, axis=0)).max(axis=1) < 1e-9).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    episodes = load_manifest(args.dataset_root)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    out_dir = os.path.join(args.dataset_root, "hand_cmd")
    if not args.check:
        os.makedirs(out_dir, exist_ok=True)

    reps = []
    for entry in episodes:
        ei = int(entry["episode_index"])
        out, stats = resample_one(entry, args.raw_root)
        rep = repeat_fraction_stride(out, stride=3)
        reps.append(rep)
        if not args.check:
            dst = os.path.join(out_dir, "episode_%06d.npz" % ei)
            np.savez(
                dst,
                q_cmd_resampled=out,
                source_episode=stats["source_episode"],
                crop_start=stats["crop_start"],
                valid_sample_count=stats["valid_sample_count"],
                source_hz=stats["source_hz"],
                method="linear_on_q_cmd_timestamp_ns",
            )
        tag = "" if args.check else "  -> written"
        print(
            "ep %3d %-20s rows %5d  validHz %4.1f  stride3_repeat %.3f%s"
            % (ei, stats["source_episode"][-20:], stats["output_rows"], stats["source_hz"], rep, tag)
        )

    arr = np.array(reps)
    print()
    print(
        "episodes: %d   stride-3 repeat fraction: mean %.3f  min %.3f  max %.3f"
        % (len(arr), arr.mean(), arr.min(), arr.max())
    )
    print("(reference: current achieved-q labels ~0.42; target after resample ~0.04)")


if __name__ == "__main__":
    main()
