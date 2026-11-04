"""Pass successful-demo GT through the deployed v5 protocol and Actor IK offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import repr_bridge as rb
from time_stretch import stretch_chunk


BIAS = Path("/home/yns/.config/ur_sharpa/teleop_bias.json")


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    ra = rb.rot6d_to_rotation(np.asarray(a[3:9], np.float32))
    rbm = rb.rot6d_to_rotation(np.asarray(b[3:9], np.float32))
    return float(np.linalg.norm(rb.rotation_to_rotvec(ra.T @ rbm)))


def main() -> None:
    from actor_ik_node.actor_ik_node import ActorIkCore
    from sharpa_interface.server.action import parse_policy_action
    from ur_common.kinematics import UrCommonKinematics
    from ur_common.safety import SafetyConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--time-scale", type=float, default=0.3)
    args = parser.parse_args()

    data = np.load(args.npz)
    bias = json.loads(BIAS.read_text())
    initial_seed = {
        "left": np.asarray(bias["left_joint"], np.float64),
        "right": np.asarray(bias["right_joint"], np.float64),
    }
    starts = sorted({key.split("_")[0] for key in data.files if key.endswith("_gt_raw")}, key=lambda x: int(x[1:]))
    kin = UrCommonKinematics(config=SafetyConfig.v1_defaults())
    core = ActorIkCore(kin)

    print("=== successful GT -> v5 protocol -> deployed Actor IK (offline) ===")
    print(f"time_scale={args.time_scale} (10 GT knots -> 34 execution knots)")
    for key in starts:
        raw = data[f"{key}_gt_raw"]
        anchor = {"left": data[f"{key}_anchor_left"], "right": data[f"{key}_anchor_right"]}
        hands = {
            "left": data[f"{key}_anchor_left_hand"],
            "right": data[f"{key}_anchor_right_hand"],
        }

        # Representation closure before any protocol or kinematics.
        native = rb.decode_action_chunk(raw, anchor)
        rebuilt = []
        for i in range(raw.shape[0]):
            pose = {"left": native["left_eef"][i], "right": native["right_eef"][i]}
            hand = {"left": native["left_hand"][i], "right": native["right_hand"][i]}
            rebuilt.append(rb.build_state_vector(anchor, pose, hand))
        repr_err = float(np.max(np.abs(np.stack(rebuilt) - raw)))

        stretched = stretch_chunk(raw, hands, args.time_scale)
        decoded = rb.decode_action_chunk(stretched, anchor)
        length = int(stretched.shape[0])
        payload = {
            "schema": "sharpa_policy_action.v5",
            "session_id": "offline-gt",
            "request_id": 1,
            "action_id": f"gt-{key}",
            "revision": 0,
            "timestamp_ns": 1,
            "execution": {
                "frequency_hz": 10.0,
                "action_length": length,
                "execute_start": 0,
                "execute_length": length,
                "rtc": 0,
            },
            "action": {
                "left_wrist": {"joint": None, "eef": decoded["left_eef"].astype(np.float32), "eef_def": "absolute"},
                "right_wrist": {"joint": None, "eef": decoded["right_eef"].astype(np.float32), "eef_def": "absolute"},
                "hand_joint": {
                    "left": decoded["left_hand"].astype(np.float32),
                    "right": decoded["right_hand"].astype(np.float32),
                },
            },
            "auxiliary": {"video": {"ego": None, "left_wrist": None, "right_wrist": None}, "tactile": {"deformation": None, "wrench": None, "hand_tau": None}},
            "diagnostics": {"offline_gt": True},
            "next_metadata_format": None,
        }
        parsed = parse_policy_action(payload, expected_session_id="offline-gt", expected_request_id=1, expected_schema="sharpa_policy_action.v5")

        # The corpus stores no arm joints. Reconstruct the joint seed at the
        # observed anchor pose using homing-mode IK, then test normal Actor IK.
        seeds = {}
        anchor_residual = {}
        for side in ("left", "right"):
            seeds[side] = kin.arm_ik_eef9(anchor[side], initial_seed[side], side, ref_joint=initial_seed[side], allow_large=True)
            back = kin.arm_fk_eef9(seeds[side], side)
            anchor_residual[side] = float(np.linalg.norm(back[:3] - anchor[side][:3]) * 1000)
            kin.solver(side).reset(seeds[side])

        before = {side: dict(kin.solver(side).stats) for side in ("left", "right")}
        plan = core.make_plan(
            left_eef=parsed.left_wrist.eef,
            right_eef=parsed.right_wrist.eef,
            hand_joint=np.concatenate([
                np.asarray(parsed.hand_joint.left).reshape(length, 22),
                np.asarray(parsed.hand_joint.right).reshape(length, 22),
            ], axis=1),
            execute_length=length,
            left_seed=seeds["left"],
            right_seed=seeds["right"],
        )

        print(f"\n{key}: protocol accepted; repr max_err={repr_err:.2e}; endpoint_preserved={np.array_equal(stretched[-1], raw[-1])}")
        for side, joints, targets in (
            ("left", plan.left_joint, decoded["left_eef"]),
            ("right", plan.right_joint, decoded["right_eef"]),
        ):
            pos, rot = [], []
            for q, target in zip(joints, targets):
                back = kin.arm_fk_eef9(q, side)
                pos.append(np.linalg.norm(back[:3] - target[:3]) * 1000)
                rot.append(rotation_error(back, target))
            after = kin.solver(side).stats
            held = (after["no_raw_solution"] - before[side]["no_raw_solution"] +
                    after["no_feasible"] - before[side]["no_feasible"] +
                    after["flip_held"] - before[side]["flip_held"])
            print(
                f"  {side:5} anchor_seed_residual={anchor_residual[side]:.3f}mm "
                f"IK_target_error max={max(pos):.3f}mm/{max(rot):.4f}rad "
                f"solver_holds={held}/{length} joint_finite={np.isfinite(joints).all()}"
            )


if __name__ == "__main__":
    main()
