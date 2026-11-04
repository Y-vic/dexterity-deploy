"""How much do the hands actually MOVE during a training episode?

Two hypotheses predict "fingers barely close" on hardware and this separates them:

  A. the robot starts outside the data, so the policy holds still
  B. the fingers barely move in the DATA either, and the live behaviour is correct

If the corpus's own hand joints are nearly static across a whole episode, then a
policy that keeps them static is reproducing what it was taught, and no amount of
repositioning will produce a grasp.
"""

import numpy as np

from flagscale.models.vla.action_model.hand_action_corpus import (
    discover_ur_episodes,
    read_ur_episode,
)

ROOT = (
    "/share/project/yenaisheng/dreamzero/datasets/postrain/ur-sharpa/"
    "lerobot_data/unscrew_cap"
)


def main() -> None:
    for span in discover_ur_episodes(ROOT, max_episodes=4):
        frames, _ = read_ur_episode(span)
        print(f"--- episode {span.episode_index} ({span.num_frames} frames) ---")
        for side in ("left", "right"):
            qpos = np.asarray(frames[f"observation.{side}_qpos"], dtype=float)
            span_per_joint = np.ptp(qpos, axis=0)
            # Per-step change at the training stride, which is what the action
            # chunk actually has to reproduce.
            step = np.abs(np.diff(qpos[::3], axis=0)).max(axis=1)
            print(
                f"  {side}: whole-episode range max={span_per_joint.max():.3f} "
                f"mean={span_per_joint.mean():.3f} rad | per-3-frame step "
                f"p50={np.percentile(step,50):.4f} p99={np.percentile(step,99):.4f}"
            )
        wrist = np.asarray(frames["observation.left_wrist_pose"], dtype=float)[:, :3]
        print(f"  left wrist total travel: {np.linalg.norm(np.diff(wrist,axis=0),axis=1).sum()*1000:.0f} mm")


if __name__ == "__main__":
    main()
