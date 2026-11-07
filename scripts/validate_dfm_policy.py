"""Run recorded canonical observations through native DFM; never publish actions.

The prepared input must contain current_canonical_observations.npz, rebuilt from
recorded TCP feedback using the current UR geometry and inverse-permuted hands.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from sharpa_interface.dfm.client import NativePolicyClient
from sharpa_interface.server.client import ServerConfig


def recorded_observation(root, metadata, canonical, index, *, session_id, request_id, active_format, prompt=None):
    from PIL import Image

    info = metadata["observations"][index]
    state = np.asarray(canonical["states"][index], np.float32)
    timestamp_ns = int(canonical["time_unix_ns"][index])
    images = {}
    for view, camera in (("ego", "ego_cam"), ("left_wrist", "left_wrist_cam"), ("right_wrist", "right_wrist_cam")):
        buffer = io.BytesIO()
        with Image.open(root / info["images"][view]) as image:
            image.convert("RGB").save(buffer, format="JPEG", quality=95)
        images[camera] = {
            "history": [],
            "current": {
                "encoding": "jpeg",
                "data": buffer.getvalue(),
                "timestamp_ns": info["sources"][view]["image_timestamp_unix_ns"],
                "valid": True,
            },
        }
    return {
        "schema": "sharpa_policy_observation.v3",
        "metadata_format_id": active_format["format_id"],
        "session_id": session_id,
        "request_id": request_id,
        "timestamp_ns": timestamp_ns,
        "prompt": metadata["instruction"] if prompt is None else prompt,
        "image": images,
        "state": {
            "history": None,
            "current": {
                "timestamp_ns": max(info["measured_arm_timestamp_unix_ns"], info["measured_hand_timestamp_unix_ns"]),
                "left_wrist": {"joint": None, "eef": state[:9], "eef_def": "absolute"},
                "right_wrist": {"joint": None, "eef": state[9:18], "eef_def": "absolute"},
                "hand_joint": {"left": state[18:40], "right": state[40:62]},
                "valid": True,
            },
        },
        "sensor": {name: {"history": None, "current": None} for name in ("tau", "wrench", "deformation")},
        "execution_feedback": {"last_action_id": None, "executed_steps": 0, "success": True},
    }


def validate_recorded_policy(client, episode_root, output, *, prompt=None):
    episode_root, output = Path(episode_root), Path(output)
    metadata = json.loads((episode_root / "observation_metadata.json").read_text())
    canonical = np.load(episode_root / "current_canonical_observations.npz")
    session_id = "native-offline-validation"
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("Explicit prompt must be a nonempty instruction")
    instruction = metadata["instruction"] if prompt is None else prompt
    ready = client.connect()
    if ready["prompt"] != instruction:
        raise ValueError("Validation instruction differs from selected checkpoint's deployment prompt")
    client.reset(session_id, request_id=0)
    original = ready["native_interface"]
    proposed = json.loads(json.dumps(original))
    proposed["observation"]["state"].update(eef_frame="camera", eef_camera=proposed["camera_bindings"]["left_wrist"])
    proposed["action"].update(eef_frame="root", eef_camera=None)
    proposed["execution"].update(execute_start=5, execute_length=10)
    report = {
        "schema": "ur_sharpa.dfm_offline_validation.v1",
        "metadata": ready,
        "recorded_episode": metadata["episode_id"],
        "recorded_instruction": metadata["instruction"],
        "effective_instruction": instruction,
        "observation_boundary": "current UR canonical wrist + wire hand order, JPEG quality95",
        "requests": [],
        "proposal": None,
    }
    outputs = {}
    for request_id in range(3):
        index = min(request_id, len(canonical["states"]) - 1)
        active = client.current_metadata
        value = recorded_observation(
            episode_root,
            metadata,
            canonical,
            index,
            session_id=session_id,
            request_id=request_id,
            active_format=active["metadata_format"],
            prompt=instruction,
        )
        before = time.perf_counter()
        action, latency = client.infer_action(value, expected_session_id=session_id, expected_request_id=request_id)
        elapsed = time.perf_counter() - before
        full = np.concatenate(
            (action.left_wrist.eef, action.right_wrist.eef, action.hand_joint.left, action.hand_joint.right), axis=-1
        )
        if full.shape != (30, 62) or not np.isfinite(full).all():
            raise ValueError("Native result must retain the full finite 30x62 action chunk")
        expected_revision = 1 if request_id < 2 else 2
        if action.diagnostics["native_interface_revision"] != expected_revision:
            raise ValueError("Interface proposal did not commit at the old-action/new-observation boundary")
        if request_id == 2 and (action.execution.execute_start, action.execution.execute_length) != (5, 10):
            raise ValueError("New execution window was not retained in the original full horizon")
        outputs[f"request_{request_id}_actions"] = full
        outputs[f"request_{request_id}_wire_state"] = canonical["states"][index]
        outputs[f"request_{request_id}_camera_from_root"] = np.stack(
            list(action.diagnostics["camera_from_root"].values())
        )
        report["requests"].append(
            {
                "request_id": request_id,
                "source_observation": index,
                "latency_s": latency,
                "roundtrip_s": elapsed,
                "action_id": action.action_id,
                "revision": action.revision,
                "execute_start": action.execution.execute_start,
                "execute_length": action.execution.execute_length,
                "action_length": action.execution.action_length,
                "diagnostics": action.diagnostics,
            }
        )
        if request_id == 0:
            report["proposal"] = client.update_interface(proposed)
            if client.current_metadata["interface_revision"] != 1:
                raise ValueError("Proposal acknowledgment changed the current action declaration")
    report["final_interface"] = client.current_metadata["native_interface"]
    report["average_latency_s"] = float(np.mean([item["latency_s"] for item in report["requests"]]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **outputs)
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:5509/infer")
    parser.add_argument("--ssh-host", default="")
    parser.add_argument("--ssh-remote-port", type=int, default=5509)
    parser.add_argument("--python", required=True)
    parser.add_argument("--dfm-root", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", help="Explicit selected-checkpoint prompt; defaults to the recorded instruction")
    args = parser.parse_args(argv)
    client = NativePolicyClient(
        ServerConfig(
            url=args.url,
            timeout_s=120,
            ssh_host=args.ssh_host,
            ssh_remote_host="127.0.0.1",
            ssh_remote_port=args.ssh_remote_port,
        ),
        python=args.python,
        source_root=args.dfm_root,
        calibration_path=args.calibration,
        data_contract_path=args.data_contract,
    )
    try:
        report = validate_recorded_policy(client, args.episode_root, args.output, prompt=args.prompt)
        print(
            json.dumps(
                {
                    "run_id": report["metadata"]["run_id"],
                    "requests": 3,
                    "latency_s": [item["latency_s"] for item in report["requests"]],
                    "average_latency_s": report["average_latency_s"],
                    "output": str(Path(args.output).with_suffix(".npz")),
                },
                ensure_ascii=False,
            )
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
