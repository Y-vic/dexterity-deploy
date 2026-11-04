"""How far is the robot's current pose from where every training episode began?

The unscrew_cap corpus has no approach phase: all 115 episodes start with the left
hand ALREADY holding the bottle. So if the robot starts from anywhere else, the
policy is being asked about a state it never saw, and its learned behaviour
("hold steady") is exactly the small drift with idle fingers that shows up on
hardware. This measures that gap instead of assuming it.
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

# Read off /api/status while the robot was idling in the state the user describes.
LIVE_EEF = {
    "left": np.array([0.2614, 0.4242, 0.2216]),
    "right": np.array([-0.2428, 0.4011, 0.1964]),
}
LIVE_HAND_MEAN = {"left": 0.080, "right": 0.106}


def main() -> None:
    starts = {"left": [], "right": []}
    hands = {"left": [], "right": []}
    for span in discover_ur_episodes(ROOT, max_episodes=30):
        frames, _ = read_ur_episode(span)
        for side in ("left", "right"):
            starts[side].append(frames[f"observation.{side}_wrist_pose"][0][:3])
            hands[side].append(float(frames[f"observation.{side}_qpos"][0].mean()))

    for side in ("left", "right"):
        arr = np.asarray(starts[side], dtype=float)
        live = LIVE_EEF[side]
        distance = np.linalg.norm(arr - live, axis=1)
        print(f"--- {side} ---")
        print("  train start xyz mean : " + " ".join(f"{v:+.4f}" for v in arr.mean(0)))
        print("  live        xyz      : " + " ".join(f"{v:+.4f}" for v in live))
        print(
            f"  gap to nearest start : {distance.min()*1000:6.0f} mm"
            f"   median {np.median(distance)*1000:6.0f} mm"
        )
        print(
            f"  hand mean  train     : {np.mean(hands[side]):.3f} rad"
            f"   live {LIVE_HAND_MEAN[side]:.3f} rad"
        )


if __name__ == "__main__":
    main()
