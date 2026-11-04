"""Run fixed-noise image/state A/B inference from a safe live capture.

Conditions:
  live_live: captured image + captured state
  demo_live: matched demonstration image + captured state
  live_demo: captured image + matched demonstration state
  demo_demo: matched demonstration image + matched demonstration state

No ROS publishers or hardware APIs are used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time
import uuid
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _interpolate_nan_columns(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    for column in range(output.shape[1]):
        vector = output[:, column]
        valid = np.isfinite(vector)
        if not np.any(valid):
            raise ValueError(f"demonstration hand joint {column} is entirely invalid")
        missing = ~valid
        if np.any(missing):
            vector[missing] = np.interp(
                np.flatnonzero(missing), np.flatnonzero(valid), vector[valid]
            )
    return output


def _component(current: dict[str, Any], *path: str) -> np.ndarray | None:
    value: Any = current
    for key in path:
        value = value[key]
    return None if value is None else np.asarray(value, dtype=np.float32)


def _match_demo_anchor(
    observation: dict[str, Any],
    demo_wrist: np.ndarray,
    demo_arm: np.ndarray,
    demo_hand: np.ndarray,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, dict[str, float]]:
    current = observation["state"]["current"]
    live_left_eef = _component(current, "left_wrist", "eef")
    live_right_eef = _component(current, "right_wrist", "eef")
    live_left_joint = _component(current, "left_wrist", "joint")
    live_right_joint = _component(current, "right_wrist", "joint")
    live_left_hand = _component(current, "hand_joint", "left")
    live_right_hand = _component(current, "hand_joint", "right")

    scores = np.zeros(maximum - minimum + 1, dtype=np.float64)
    terms: dict[str, np.ndarray] = {}
    rows = slice(minimum, maximum + 1)
    if live_left_eef is not None and live_right_eef is not None:
        position = np.linalg.norm(
            np.concatenate(
                (
                    demo_wrist[rows, 0, :3] - live_left_eef[:3],
                    demo_wrist[rows, 1, :3] - live_right_eef[:3],
                ),
                axis=1,
            ),
            axis=1,
        )
        terms["wrist_position_m"] = position
        scores += position / 0.02
    if live_left_joint is not None and live_right_joint is not None:
        arm = np.linalg.norm(
            demo_arm[rows]
            - np.concatenate((live_left_joint, live_right_joint))[None, :],
            axis=1,
        )
        terms["arm_joint_rad"] = arm
        scores += arm / 0.10
    if live_left_hand is not None and live_right_hand is not None:
        hand = np.linalg.norm(
            demo_hand[rows]
            - np.concatenate((live_left_hand, live_right_hand))[None, :],
            axis=1,
        )
        terms["hand_joint_rad"] = hand
        scores += hand / 0.20
    offset = int(np.argmin(scores))
    anchor = minimum + offset
    detail = {name: float(values[offset]) for name, values in terms.items()}
    detail["score"] = float(scores[offset])
    return anchor, detail


def _demo_state(
    template: dict[str, Any],
    *,
    rows: np.ndarray,
    demo_arm: np.ndarray,
    demo_wrist: np.ndarray,
    demo_hand: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    state = deepcopy(template)
    history_rows = rows[:-1]
    current_row = int(rows[-1])
    history = state["history"]
    current = state["current"]
    if history is not None:
        history["timestamp_ns"] = timestamps[history_rows].astype(np.int64)
        history["valid"] = np.ones(len(history_rows), dtype=np.bool_)
        if history["left_wrist"]["joint"] is not None:
            history["left_wrist"]["joint"] = demo_arm[history_rows, :6]
        if history["right_wrist"]["joint"] is not None:
            history["right_wrist"]["joint"] = demo_arm[history_rows, 6:]
        if history["left_wrist"]["eef"] is not None:
            history["left_wrist"]["eef"] = demo_wrist[history_rows, 0]
        if history["right_wrist"]["eef"] is not None:
            history["right_wrist"]["eef"] = demo_wrist[history_rows, 1]
        if history["hand_joint"]["left"] is not None:
            history["hand_joint"]["left"] = demo_hand[history_rows, :22]
        if history["hand_joint"]["right"] is not None:
            history["hand_joint"]["right"] = demo_hand[history_rows, 22:]
    if current is not None:
        current["timestamp_ns"] = int(timestamps[current_row])
        current["valid"] = True
        if current["left_wrist"]["joint"] is not None:
            current["left_wrist"]["joint"] = demo_arm[current_row, :6]
        if current["right_wrist"]["joint"] is not None:
            current["right_wrist"]["joint"] = demo_arm[current_row, 6:]
        if current["left_wrist"]["eef"] is not None:
            current["left_wrist"]["eef"] = demo_wrist[current_row, 0]
        if current["right_wrist"]["eef"] is not None:
            current["right_wrist"]["eef"] = demo_wrist[current_row, 1]
        if current["hand_joint"]["left"] is not None:
            current["hand_joint"]["left"] = demo_hand[current_row, :22]
        if current["hand_joint"]["right"] is not None:
            current["hand_joint"]["right"] = demo_hand[current_row, 22:]
    return state


def _demo_ego_jpeg(demo_dir: Path, anchor: int) -> tuple[bytes, int]:
    index = np.load(demo_dir / "cameras/ego/index.npz")
    valid = np.asarray(index["valid"], dtype=bool)
    timeline_rows = np.asarray(index["timeline_row"], dtype=np.int64)
    candidates = np.flatnonzero(valid)
    selected = int(candidates[np.argmin(np.abs(timeline_rows[candidates] - anchor))])
    frame_index = int(index["frame_index"][selected])
    select_filter = f"select=eq(n\\,{frame_index})"
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(demo_dir / "cameras/ego/rgb.mp4"),
            "-vf",
            select_filter,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not process.stdout.startswith(b"\xff\xd8"):
        raise RuntimeError(f"failed to decode demo ego frame {frame_index}")
    return process.stdout, int(index["source_timestamp_ns"][selected])


def _replace_ego(payload: dict[str, Any], jpeg: bytes, timestamp_ns: int) -> None:
    frame = payload["image"]["ego_cam"]["current"]
    if frame is None:
        raise ValueError("active model format has no current ego frame")
    frame["data"] = bytes(jpeg)
    frame["timestamp_ns"] = int(timestamp_ns)
    frame["encoding"] = "jpeg"
    frame["valid"] = True


def _action_arrays(parsed: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_eef = np.asarray(parsed.left_wrist.eef, dtype=np.float32)
    right_eef = np.asarray(parsed.right_wrist.eef, dtype=np.float32)
    left_hand = np.asarray(parsed.hand_joint.left, dtype=np.float32)
    right_hand = np.asarray(parsed.hand_joint.right, dtype=np.float32)
    return left_eef, right_eef, np.concatenate((left_hand, right_hand), axis=1)


def _metrics(
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    gt: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    left, right, hand = predicted
    gt_left, gt_right, gt_hand = gt
    horizon = min(len(left), len(gt_left))
    left = left[:horizon]
    right = right[:horizon]
    hand = hand[:horizon]
    gt_left = gt_left[:horizon]
    gt_right = gt_right[:horizon]
    gt_hand = gt_hand[:horizon]
    return {
        "horizon": horizon,
        "left_xyz_rmse_mm": float(
            np.sqrt(np.mean((left[:, :3] - gt_left[:, :3]) ** 2)) * 1000.0
        ),
        "right_xyz_rmse_mm": float(
            np.sqrt(np.mean((right[:, :3] - gt_right[:, :3]) ** 2)) * 1000.0
        ),
        "left_hand_rmse_rad": float(
            np.sqrt(np.mean((hand[:, :22] - gt_hand[:, :22]) ** 2))
        ),
        "right_hand_rmse_rad": float(
            np.sqrt(np.mean((hand[:, 22:] - gt_hand[:, 22:]) ** 2))
        ),
        "left_endpoint_mm": (left[-1, :3] * 1000.0).tolist(),
        "left_gt_endpoint_mm": (gt_left[-1, :3] * 1000.0).tolist(),
        "right_endpoint_mm": (right[-1, :3] * 1000.0).tolist(),
        "right_gt_endpoint_mm": (gt_right[-1, :3] * 1000.0).tolist(),
    }


def main() -> None:
    from sharpa_interface.server.action import parse_policy_action
    from sharpa_interface.server.client import ServerClient, ServerConfig
    from sharpa_interface.server.serialization import packb, unpackb

    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--demo-anchor", type=int)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    demo_dir = Path(args.demo_dir).expanduser().resolve()
    observation = unpackb((capture_dir / "observation.msgpack").read_bytes())
    capture_manifest = json.loads((capture_dir / "manifest.json").read_text())
    metadata = json.loads((capture_dir / "metadata.json").read_text())

    ur = np.load(demo_dir / "ur/physical_12.npz")
    hand_archive = np.load(demo_dir / "sharpa/joints_44.npz")
    demo_arm = np.asarray(ur["q"], dtype=np.float32)
    demo_wrist = np.asarray(ur["canonical_wrist_pose9"], dtype=np.float32)
    demo_hand = _interpolate_nan_columns(hand_archive["q"])
    timestamps = np.maximum(
        np.asarray(ur["timestamp_ns"], dtype=np.int64),
        np.asarray(hand_archive["q_timestamp_ns"], dtype=np.int64),
    )

    history = observation["state"]["history"]
    history_len = 0 if history is None else len(history["timestamp_ns"])
    maximum = len(demo_arm) - 32
    if args.demo_anchor is None:
        anchor, match = _match_demo_anchor(
            observation,
            demo_wrist,
            demo_arm,
            demo_hand,
            minimum=history_len,
            maximum=maximum,
        )
    else:
        anchor = int(args.demo_anchor)
        match = {"forced": True}
    if anchor < history_len or anchor > maximum:
        raise ValueError(
            f"demo anchor {anchor} cannot supply {history_len} history rows and future GT"
        )

    rows = np.arange(anchor - history_len, anchor + 1, dtype=np.int64)
    demo_state = _demo_state(
        observation["state"],
        rows=rows,
        demo_arm=demo_arm,
        demo_wrist=demo_wrist,
        demo_hand=demo_hand,
        timestamps=timestamps,
    )
    demo_jpeg, demo_image_timestamp = _demo_ego_jpeg(demo_dir, anchor)
    (capture_dir / "demo_matched_ego.jpg").write_bytes(demo_jpeg)

    server = capture_manifest["server"]
    config = ServerConfig(
        url=str(server["url"]),
        configured_prompt="",
        ssh_host=str(server["ssh_host"]),
        ssh_remote_host=str(server["ssh_remote_host"]),
        ssh_remote_port=int(server["ssh_remote_port"]),
    )
    client = ServerClient(config)
    conditions: dict[str, dict[str, Any]] = {}
    output_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    try:
        current_metadata = client.connect()
        for name, use_demo_image, use_demo_state in (
            ("live_live", False, False),
            ("demo_live", True, False),
            ("live_demo", False, True),
            ("demo_demo", True, True),
        ):
            payload = deepcopy(observation)
            session_id = f"ur-sharpa-ab-{name}-{uuid.uuid4().hex[:10]}"
            payload["session_id"] = session_id
            payload["request_id"] = 0
            payload["timestamp_ns"] = time.time_ns()
            payload["execution_feedback"] = None
            if use_demo_image:
                _replace_ego(payload, demo_jpeg, demo_image_timestamp)
            if use_demo_state:
                payload["state"] = deepcopy(demo_state)
            reset = client.reset(session_id, request_id=0)
            expected_format = str(reset["metadata_format"]["format_id"])
            if payload["metadata_format_id"] != expected_format:
                raise RuntimeError(
                    f"format mismatch for {name}: {payload['metadata_format_id']} != {expected_format}"
                )
            result = client.infer(payload)
            parsed = parse_policy_action(
                result.payload,
                expected_session_id=session_id,
                expected_request_id=0,
                expected_schema=str(current_metadata["action_schema"]),
            )
            (capture_dir / f"observation_{name}.msgpack").write_bytes(packb(payload))
            (capture_dir / f"action_{name}.msgpack").write_bytes(packb(result.payload))
            output_arrays[name] = _action_arrays(parsed)
            conditions[name] = {"latency_s": result.latency_s}
    finally:
        client.close()

    horizon = min(len(output_arrays["live_live"][0]), 10)
    gt_rows = anchor + 3 * np.arange(1, horizon + 1, dtype=np.int64)
    gt = (
        demo_wrist[gt_rows, 0],
        demo_wrist[gt_rows, 1],
        demo_hand[gt_rows],
    )
    for name, predicted in output_arrays.items():
        conditions[name].update(_metrics(predicted, gt))

    baseline = output_arrays["live_live"]
    for name, predicted in output_arrays.items():
        left, right, hand = predicted
        conditions[name]["delta_from_live_live"] = {
            "left_xyz_rmse_mm": float(
                np.sqrt(np.mean((left[:, :3] - baseline[0][:, :3]) ** 2)) * 1000.0
            ),
            "right_xyz_rmse_mm": float(
                np.sqrt(np.mean((right[:, :3] - baseline[1][:, :3]) ** 2)) * 1000.0
            ),
            "hand_rmse_rad": float(np.sqrt(np.mean((hand - baseline[2]) ** 2))),
        }

    report = {
        "schema": "ur_sharpa.observation_ab.v1",
        "capture_dir": str(capture_dir),
        "demo_dir": str(demo_dir),
        "demo_anchor": anchor,
        "demo_time_s": anchor / float(np.asarray(np.load(demo_dir / "timeline/clock.npz")["sample_rate_hz"])),
        "match": match,
        "gt_rows": gt_rows.tolist(),
        "conditions": conditions,
    }
    report_path = capture_dir / "ab_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
