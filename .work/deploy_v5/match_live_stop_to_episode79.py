"""Match a stopped hardware snapshot and its last chunk to episode 79.

Run in the ORCA training environment.  This is offline/read-only: it loads the
training reader and recorded deployment JSON, but does not load the model or
connect to the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/share/project/zjk/orca")
DEPLOY = REPO / "scripts/deploy/sharpa_policy_server"
DATASET = Path(
    "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
    "lerobot_data/unscrew_cap"
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DEPLOY))

import repr_bridge as rb  # noqa: E402
from flagscale.models.vla.action_model import hand_action_repr as train  # noqa: E402
from flagscale.models.vla.action_model.hand_action_corpus import (  # noqa: E402
    discover_ur_episodes,
    read_ur_episode,
)
from flagscale.models.vla.action_model.hand_action_video import (  # noqa: E402
    decode_frames_at,
)


def rot_from_pose9(pose: np.ndarray) -> np.ndarray:
    """Deployment pose9 stores the first two rows of canonical_from_TCP R."""

    rows = np.asarray(pose, dtype=np.float64)[3:9].reshape(2, 3)
    a = rows[0] / np.linalg.norm(rows[0])
    b = rows[1] - np.dot(rows[1], a) * a
    b /= np.linalg.norm(b)
    return np.stack((a, b, np.cross(a, b)), axis=0)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    value = np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def pose9_from_pose12(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32)
    return np.concatenate((row[:3], row[3:9])).astype(np.float32)


def demo_pose(frames: dict, side: str, index: int) -> dict:
    value = np.asarray(frames[f"observation.{side}_wrist_pose"][index])
    return {"p": value[:3].astype(np.float64), "r": value[3:12].reshape(3, 3)}


def local_pair(left_eef: np.ndarray, right_eef: np.ndarray) -> np.ndarray:
    result = []
    for left, right in zip(left_eef, right_eef):
        result.append(rot_from_pose9(left).T @ (right[:3] - left[:3]))
    return np.asarray(result)


def local_motion(eef: np.ndarray) -> np.ndarray:
    rotation = rot_from_pose9(eef[0])
    return (rotation.T @ (eef[:, :3] - eef[0, :3]).T).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-id", type=int, default=11)
    parser.add_argument("--phase-start", type=int, default=50)
    parser.add_argument("--phase-end", type=int, default=400)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    live = np.load(args.snapshot)
    trajectories = [json.loads(line) for line in Path(args.trajectory).open()]
    trace = next(t for t in trajectories if t["request_id"] == args.request_id)
    live_left = np.asarray(trace["left_eef"], dtype=np.float64).reshape(-1, 9)
    live_right = np.asarray(trace["right_eef"], dtype=np.float64).reshape(-1, 9)
    live_hand = np.asarray(trace["hand_joint"], dtype=np.float64).reshape(-1, 44)

    spans = discover_ur_episodes(str(DATASET), max_episodes=80)
    span = spans[79]
    frames, commands = read_ur_episode(span)
    lo = max(args.phase_start, rb.STATE_ANCHOR_OFFSET)
    hi = min(
        args.phase_end,
        span.num_frames - rb.ACTION_HORIZON * rb.ACTION_STRIDE - 1,
    )

    # Match the exact stopped state to achieved demonstration states.
    state_rows = []
    for index in range(lo, hi + 1):
        lp, rp = demo_pose(frames, "left", index), demo_pose(frames, "right", index)
        lpos = np.linalg.norm(lp["p"] - live["left_eef"][:3]) * 1000.0
        rpos = np.linalg.norm(rp["p"] - live["right_eef"][:3]) * 1000.0
        lrot = angle_deg(lp["r"], rot_from_pose9(live["left_eef"]))
        rrot = angle_deg(rp["r"], rot_from_pose9(live["right_eef"]))
        lh = float(np.mean(np.abs(frames["observation.left_qpos"][index] - live["left_hand"])))
        rh = float(np.mean(np.abs(frames["observation.right_qpos"][index] - live["right_hand"])))
        score = (lpos + rpos) / 20.0 + (lrot + rrot) / 15.0 + (lh + rh) / 0.10
        state_rows.append(
            {
                "frame": index,
                "score": score,
                "left_pos_mm": lpos,
                "right_pos_mm": rpos,
                "left_rot_deg": lrot,
                "right_rot_deg": rrot,
                "left_hand_mae_rad": lh,
                "right_hand_mae_rad": rh,
                "demo_inter_wrist_mm": float(np.linalg.norm(rp["p"] - lp["p"]) * 1000.0),
                "live_inter_wrist_mm": float(
                    np.linalg.norm(live["right_eef"][:3] - live["left_eef"][:3]) * 1000.0
                ),
            }
        )
    state_rows.sort(key=lambda row: row["score"])

    # Match the whole final live chunk to every possible 1-second demo chunk.
    live_pair = local_pair(live_left, live_right)
    live_lmotion = local_motion(live_left)
    live_rmotion = local_motion(live_right)
    chunk_rows = []
    decoded_by_start = {}
    for start in range(lo, hi + 1):
        gt = train.build_action_chunk(
            frames,
            start,
            rb.ACTION_HORIZON,
            rb.ACTION_STRIDE,
            command_frames=commands,
        )
        pose = {
            side: pose9_from_pose12(frames[f"observation.{side}_wrist_pose"][start])
            for side in ("left", "right")
        }
        decoded = rb.decode_action_chunk(gt, pose)
        dl = np.asarray(decoded["left_eef"], dtype=np.float64)
        dr = np.asarray(decoded["right_eef"], dtype=np.float64)
        demo_pair = local_pair(dl, dr)
        pair_errors = np.linalg.norm(live_pair - demo_pair, axis=1) * 1000.0
        lmotion_errors = np.linalg.norm(live_lmotion - local_motion(dl), axis=1) * 1000.0
        rmotion_errors = np.linalg.norm(live_rmotion - local_motion(dr), axis=1) * 1000.0
        demo_lh = gt[:, rb.SLICES["left_hand_joints"]]
        demo_rh = gt[:, rb.SLICES["right_hand_joints"]]
        lhand = float(np.mean(np.abs(live_hand[:, :22] - demo_lh)))
        rhand = float(np.mean(np.abs(live_hand[:, 22:] - demo_rh)))
        score = (
            float(pair_errors.mean()) / 15.0
            + float(lmotion_errors.mean() + rmotion_errors.mean()) / 20.0
            + (lhand + rhand) / 0.10
        )
        chunk_rows.append(
            {
                "start": start,
                "end_frame": start + rb.ACTION_HORIZON * rb.ACTION_STRIDE,
                "score": score,
                "pair_mean_err_mm": float(pair_errors.mean()),
                "pair_endpoint_err_mm": float(pair_errors[-1]),
                "left_motion_mean_err_mm": float(lmotion_errors.mean()),
                "right_motion_mean_err_mm": float(rmotion_errors.mean()),
                "left_hand_mae_rad": lhand,
                "right_hand_mae_rad": rhand,
            }
        )
        decoded_by_start[start] = (dl, dr, gt)
    chunk_rows.sort(key=lambda row: row["score"])

    best_state = state_rows[0]
    best_chunk = chunk_rows[0]
    selected_frames = sorted(
        {
            best_state["frame"],
            best_chunk["start"],
            best_chunk["end_frame"],
            160,
            190,
            220,
            250,
        }
    )
    images = decode_frames_at(
        span.video.path,
        selected_frames,
        size=(640, 480),
        frame_offset=span.video.frame_offset,
        num_frames=span.video.num_frames,
    )
    from PIL import Image

    for index in selected_frames:
        Image.fromarray(images[index]).save(output / f"episode79_frame_{index:04d}.jpg")

    report = {
        "snapshot": str(args.snapshot),
        "request_id": args.request_id,
        "searched_phase": [lo, hi],
        "best_state_match": best_state,
        "top_state_matches": state_rows[:15],
        "best_chunk_match": best_chunk,
        "top_chunk_matches": chunk_rows[:15],
        "note": (
            "State matching uses achieved q and pose. Chunk matching compares the "
            "live published targets with q_teleop demo targets and compares wrist "
            "geometry in the left-wrist frame, so global table offsets do not dominate."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
