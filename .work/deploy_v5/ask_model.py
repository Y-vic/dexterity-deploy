"""Ask the RESIDENT server a real training window over its own /infer socket.

Runs on the remote against 127.0.0.1 -- no tunnel, and no second 26 GB load, so
the GPU contention on this node does not block it. The observation is built by
test_contract.make_observation, which is already validated against the
workstation's parser, with only the state history and the ego frame swapped for
the real window's.
"""

import socket
import sys

import numpy as np

sys.path.insert(0, "/share/project/zjk/orca/scripts/deploy/sharpa_policy_server")

import closing_test as ct
import repr_bridge as rb
import server as srv
from test_contract import make_observation


def real_observation(session, request_id, anchor, rows, jpeg):
    """make_observation, with the real window's frames substituted in."""
    obs = make_observation(session, request_id)
    obs["prompt"] = ct.UR_TASK
    obs["image"]["ego_cam"]["current"] = {"data": jpeg}

    history = obs["state"]["history"]
    frames = rb.STATE_ANCHOR_OFFSET
    # The server reads offset k at index (frames - k), and the anchor at index 0.
    # Fill every slot so nothing it touches is left as the fixture's synthetic
    # ramp; the offsets it actually reads get the real poses.
    for side in ("left", "right"):
        eef = np.stack([anchor[side]] * frames).astype(np.float32)
        qpos = np.stack([rows[-1][1][side]] * frames).astype(np.float32)
        for row_index, offset in enumerate(rb.STATE_HISTORY_OFFSETS):
            if offset == 0:
                continue
            eef[frames - offset] = rows[row_index][0][side]
            qpos[frames - offset] = rows[row_index][1][side]
        history[f"{side}_wrist"]["eef"] = eef
        history["hand_joint"][side] = qpos
        # offset 0 is `current`, not history.
        obs["state"]["current"][f"{side}_wrist"]["eef"] = rows[-1][0][side]
        obs["state"]["current"]["hand_joint"][side] = rows[-1][1][side]
    return obs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=int, default=620)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    anchor, rows, jpeg, gt = ct.build_window(args.episode, args.start)
    print(f"=== episode {args.episode} start {args.start} ===")
    ct.show("GROUND TRUTH", gt)

    # The state the model will receive, for the record: this is the same builder
    # the live server uses, so a mismatch against training would show here.
    state56 = rb.build_state_rows(anchor, rows)
    print(f"  state wrist_trans |mean|={np.abs(state56[:, 0:3]).mean():.5f}m "
          f"max={np.abs(state56[:, 0:3]).max():.5f}m")
    print()

    from sharpa_interface.server.transport import SharpaV3PolicyClient

    client = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{args.port}/infer", timeout_s=180.0)
    try:
        session = client.reset("closing-test")["session_id"]
        for index in range(args.repeats):
            obs = real_observation(session, index, anchor, rows, jpeg)
            result = client.infer(obs)
            payload = result.payload
            if payload.get("schema") == srv.ERROR_SCHEMA:
                print("  ERROR:", payload["error"])
                break
            chunk = np.zeros((rb.ACTION_HORIZON, rb.ACTION_DIM), dtype=np.float32)
            for side in ("left", "right"):
                eef = payload["action"][f"{side}_wrist"]["eef"]
                # Back to relative, against the same anchor the server used, so the
                # metrics are comparable with GT rather than absolute positions.
                a_t, a_R = rb.pose9_to_parts(rows[-1][0][side])
                for step in range(rb.ACTION_HORIZON):
                    t, R = rb.pose9_to_parts(eef[step])
                    blk = rb.wrist_relative_6d(a_t, a_R, t, R)
                    chunk[step, rb.SLICES[f"{side}_wrist_trans"]] = blk[:3]
                    chunk[step, rb.SLICES[f"{side}_wrist_rotvec"]] = blk[3:]
                chunk[:, rb.SLICES[f"{side}_hand_joints"]] = payload["action"]["hand_joint"][side]
            ct.show(f"MODEL sample {index} ({result.latency_s:.2f}s)", chunk)
    finally:
        client.close()


if __name__ == "__main__":
    main()
