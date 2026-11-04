"""Feed the SERVED model a real training window and compare against ground truth.

This removes the live observation pipeline from the loop: the image, the poses and
the hand qpos all come from the training corpus through the training code's own
reader, and the GT action chunk comes from the same window. The only thing left
under test is the checkpoint plus the sampling.

Runs on the REMOTE, against the model already resident on GPU 0, so it needs no
second 26 GB load. Writes a .npz the local side can compare.
"""

import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca")
sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import repr_bridge as rb

ROOT = (
    "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
    "lerobot_data/unscrew_cap"
)
# The instruction the run trained on, verbatim from the yaml's `ur_task`. The
# prompt is part of what is under test: a different string is a different
# conditioning, so it must not be paraphrased here.
UR_TASK = "Right hand unscrews the blue cap while left hand holds the bottle steady"


def pose9_from_pose12(row: np.ndarray) -> np.ndarray:
    """The corpus stores xyz + rot6d as [12]; the interface wants pose9.

    canonical_from_ur_62d is what the reader uses, so the 12D block is already
    canonical -- the first 3 are translation and the next 6 are the two columns.
    The trailing 3 are the redundant third column, dropped here.
    """
    row = np.asarray(row, dtype=np.float32)
    return np.concatenate([row[:3], row[3:9]]).astype(np.float32)


def build_window(episode: int, start: int):
    """Return (anchor, rows, jpeg, gt_chunk) for one real training window."""
    import io

    from PIL import Image

    from flagscale.models.vla.action_model import hand_action_repr as train
    from flagscale.models.vla.action_model.hand_action_corpus import (
        discover_ur_episodes,
        read_ur_episode,
    )
    from flagscale.models.vla.action_model.hand_action_video import decode_frames_at

    span = discover_ur_episodes(ROOT, max_episodes=episode + 1)[episode]
    frames, _ = read_ur_episode(span)

    gt = train.build_action_chunk(frames, start, rb.ACTION_HORIZON, rb.ACTION_STRIDE)

    def at(offset: int):
        index = start - offset
        pose = {
            "left": pose9_from_pose12(frames["observation.left_wrist_pose"][index]),
            "right": pose9_from_pose12(frames["observation.right_wrist_pose"][index]),
        }
        hand = {
            "left": np.asarray(frames["observation.left_qpos"][index], dtype=np.float32),
            "right": np.asarray(frames["observation.right_qpos"][index], dtype=np.float32),
        }
        return pose, hand

    anchor, _ = at(rb.STATE_ANCHOR_OFFSET)
    rows = [at(offset) for offset in rb.STATE_HISTORY_OFFSETS]

    # The ego frame at the window start, decoded exactly as training decodes it:
    # same size, same swscale path, so the pixels the model sees are the pixels it
    # trained on rather than a PIL re-encode of them.
    decoded = decode_frames_at(
        span.video.path,
        [start],
        size=(256, 256),
        frame_offset=span.video.frame_offset,
        num_frames=span.video.num_frames,
    )
    buffer = io.BytesIO()
    Image.fromarray(decoded[start]).save(buffer, format="JPEG", quality=95)
    return anchor, rows, buffer.getvalue(), gt


def metrics(chunk: np.ndarray) -> dict:
    """Path length, net displacement and hand jitter, per side, in metric units."""
    out = {}
    for side in ("left", "right"):
        trans = chunk[:, rb.SLICES[f"{side}_wrist_trans"]]
        step = np.linalg.norm(np.diff(trans, axis=0), axis=1)
        hand = chunk[:, rb.SLICES[f"{side}_hand_joints"]]
        out[side] = {
            "path_mm": float(step.sum() * 1000),
            "net_mm": float(np.linalg.norm(trans[-1] - trans[0]) * 1000),
            "hand_max_dstep_rad": float(np.abs(np.diff(hand, axis=0)).max()),
        }
        out[side]["ratio"] = out[side]["path_mm"] / max(out[side]["net_mm"], 1e-6)
    return out


def show(name: str, chunk: np.ndarray) -> None:
    m = metrics(chunk)
    print(f"  {name}")
    for side in ("left", "right"):
        s = m[side]
        print(
            f"    {side:5} path {s['path_mm']:7.1f}mm  net {s['net_mm']:7.1f}mm  "
            f"ratio {s['ratio']:6.2f}  hand max|d|/step {s['hand_max_dstep_rad']:.3f} rad"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=int, default=620)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    anchor, rows, jpeg, gt = build_window(args.episode, args.start)
    print(f"=== episode {args.episode} start {args.start} ===")
    show("GROUND TRUTH", gt)

    state56 = rb.build_state_rows(anchor, rows)
    print(
        f"  state rows {state56.shape}  wrist_trans |mean|="
        f"{np.abs(state56[:, 0:3]).mean():.4f}m"
    )
    np.savez(
        "/tmp/closing_window.npz", state56=state56, gt=gt, jpeg=np.frombuffer(jpeg, np.uint8)
    )
    print(f"  saved /tmp/closing_window.npz  (jpeg {len(jpeg)} bytes)")


if __name__ == "__main__":
    main()
