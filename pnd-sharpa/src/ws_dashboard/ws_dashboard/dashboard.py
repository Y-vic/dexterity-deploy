#!/usr/bin/env python3
"""Six-panel live dashboard for the workstation DreamZero pipeline."""

from __future__ import annotations

import html
import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import rclpy
from deploy_common.joints import (
    ADAM_COMMAND_JOINTS_19,
    ADAM_PHYSICAL_JOINTS_31,
    SHARPA_JOINT_NAMES,
)
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ws_core.kinematics import (
    PND_TO_MUJOCO_JOINT,
    resolve_model_xml,
)
from ws_msgs.msg import (
    ModelImage,
    PndAction,
    PolicyObs,
    PolicyPred,
    RobotState,
    RobotTactile,
    Status,
)


DEFAULT_STATUS_TOPICS = [
    "/ws/status",
    "/ws/robot_states/status",
    "/ws/robot_tactile/status",
    "/ws/robot_vision/status",
    "/ws/obs_sync/status",
    "/ws/policy_client/status",
    "/ws/action_execute/status",
]

PANEL_ROUTES = {
    "/zed.mjpg": "zed",
    "/robot.mjpg": "robot",
    "/current_hands.mjpg": "current_hands",
    "/pred_video.mjpg": "pred_video",
    "/pred_robot.mjpg": "pred_robot",
    "/pred_hands.mjpg": "pred_hands",
}

PANEL_IMAGE_ROUTES = {
    route.removesuffix(".mjpg") + ".jpg": panel
    for route, panel in PANEL_ROUTES.items()
}

PANEL_TITLES = {
    "zed": "Current Videos",
    "robot": "Current PND-SharpA",
    "current_hands": "Current Waist Hands",
    "pred_video": "Predicted Video",
    "pred_robot": "Predicted PND-SharpA",
    "pred_hands": "Predicted Waist Hands",
}

PRED_VIDEO_SUPPORTED_PROVIDERS = frozenset(
    {"dreamzero", "dreamzero_sharpa62", "dz", "sharpa62"}
)
PRED_VIDEO_UNSUPPORTED_PROVIDERS = frozenset(
    {
        "cgp",
        "cgp_n17",
        "cgp_n17_sharpa62",
        "gcc",
        "gcc_n17",
        "gcc_n17_sharpa62",
        "groot",
        "groot_n17",
        "groot_n17_mot",
        "groot_n17_mot_sharpa62",
        "groot_n17_sharpa62",
        "pace",
        "pace_n17",
        "pace_n17_sharpa62",
        "t_rex",
        "t-rex",
        "trex",
        "trex_sharpa62",
        "vitac",
        "vitac_former",
        "vitacformer",
        "vitacformer_sharpa62",
    }
)


def _pred_video_capability(provider: str) -> str:
    normalized = str(provider).strip().lower()
    if normalized in PRED_VIDEO_SUPPORTED_PROVIDERS:
        return "supported"
    if normalized in PRED_VIDEO_UNSUPPORTED_PROVIDERS:
        return "unsupported"
    return "unknown"


def _prediction_video_path(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    containers = [payload]
    for key in ("debug", "diagnostics"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        value = container.get("server_video_pred_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

MODEL_IMAGE_WIDTH = 320
MODEL_IMAGE_HEIGHT = 160
LEG_PND_JOINTS = [
    name for name in ADAM_PHYSICAL_JOINTS_31 if name not in ADAM_COMMAND_JOINTS_19
]
DASHBOARD_PND_TO_MUJOCO_JOINT = {
    **PND_TO_MUJOCO_JOINT,
    **{name: name.removeprefix("dof_pos/") for name in LEG_PND_JOINTS},
}

_HAND_EDGES = [(0, start) for start in (1, 5, 9, 13, 17)]
for _finger_start in (1, 5, 9, 13, 17):
    _HAND_EDGES.extend(
        [
            (_finger_start, _finger_start + 1),
            (_finger_start + 1, _finger_start + 2),
            (_finger_start + 2, _finger_start + 3),
        ]
    )

_HAND_BODIES = {
    "left": [
        "wristRollLeft",
        "left_thumb_MC",
        "left_thumb_PP",
        "left_thumb_DP",
        "left_thumb_DP",
        "left_index_PP",
        "left_index_MP",
        "left_index_DP",
        "left_index_DP",
        "left_middle_PP",
        "left_middle_MP",
        "left_middle_DP",
        "left_middle_DP",
        "left_ring_PP",
        "left_ring_MP",
        "left_ring_DP",
        "left_ring_DP",
        "left_pinky_MC",
        "left_pinky_PP",
        "left_pinky_MP",
        "left_pinky_DP",
    ],
    "right": [
        "wristRollRight",
        "right_thumb_MC",
        "right_thumb_PP",
        "right_thumb_DP",
        "right_thumb_DP",
        "right_index_PP",
        "right_index_MP",
        "right_index_DP",
        "right_index_DP",
        "right_middle_PP",
        "right_middle_MP",
        "right_middle_DP",
        "right_middle_DP",
        "right_ring_PP",
        "right_ring_MP",
        "right_ring_DP",
        "right_ring_DP",
        "right_pinky_MC",
        "right_pinky_PP",
        "right_pinky_MP",
        "right_pinky_DP",
    ],
}

_PND_TO_BAAI = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

_HAND_COLORS = ("#2468B4", "#D94A4A")
_HIP_COLOR = "#2C2C2C"
_BG_COLOR = "#FAFAFA"
_GRID_COLOR = (0.86, 0.86, 0.86, 1.0)
_PANE_COLOR = (0.97, 0.97, 0.97, 1.0)
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_json(value: str) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value, "parse_error": "invalid_json"}


def _header_dict(header: Any) -> dict[str, Any]:
    stamp = getattr(header, "stamp", None)
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return {
        "frame_id": str(getattr(header, "frame_id", "")),
        "stamp": {
            "sec": sec,
            "nanosec": nanosec,
            "stamp_ns": sec * 1_000_000_000 + nanosec,
        },
    }


def _bytes_summary(data: Any) -> dict[str, Any]:
    return {"bytes": len(data) if data is not None else 0}


def _topic_key(topic: str) -> str:
    return topic.strip("/").replace("/", ".") or "root"


def _status_frame(width: int, height: int, title: str, detail: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 22, 25)
    cv2.rectangle(frame, (0, 0), (width, 76), (28, 34, 38), -1)
    cv2.putText(
        frame,
        title,
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (235, 242, 246),
        2,
        cv2.LINE_AA,
    )
    for row, line in enumerate(detail.splitlines()[:8]):
        cv2.putText(
            frame,
            line[:80],
            (24, 104 + row * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (174, 190, 199),
            1,
            cv2.LINE_AA,
        )
    return frame


def _encode_jpeg(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("failed to encode dashboard JPEG")
    return encoded.tobytes()


class SharedFrame:
    def __init__(self, width: int, height: int, title: str, quality: int) -> None:
        self.quality = quality
        self.condition = threading.Condition()
        self.revision = 0
        self.jpeg = _encode_jpeg(
            _status_frame(width, height, title, "starting"), quality
        )
        self.status: dict[str, Any] = {
            "title": title,
            "state": "starting",
            "updated_unix_s": time.time(),
        }

    def update(self, frame_rgb: np.ndarray, **status: Any) -> None:
        self.update_jpeg(_encode_jpeg(frame_rgb, self.quality), **status)

    def update_jpeg(self, jpeg: bytes, **status: Any) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.revision += 1
            self.status = {
                "title": self.status.get("title", ""),
                **status,
                "updated_unix_s": time.time(),
            }
            self.condition.notify_all()

    def snapshot(self) -> tuple[bytes, dict[str, Any], int]:
        with self.condition:
            return self.jpeg, dict(self.status), self.revision

    def wait_next(self, revision: int, timeout: float = 1.0) -> tuple[bytes, int]:
        with self.condition:
            if revision == self.revision:
                self.condition.wait(timeout=timeout)
            return self.jpeg, self.revision


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler_factory(
    get_snapshot: Callable[[], dict[str, Any]],
    frames: dict[str, SharedFrame],
) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "PndSharpaDashboard/2.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/dashboard", "/index.html"):
                body = Path(__file__).with_name("dashboard.html").read_bytes()
                initial_subtitle = (
                    f"Task: {get_snapshot().get('_meta', {}).get('prompt', '')}"
                    if get_snapshot().get("_meta", {}).get("prompt")
                    else "Task: unset"
                )
                body = body.replace(
                    b"__INITIAL_SUBTITLE__",
                    html.escape(initial_subtitle).encode("utf-8"),
                )
                if "snapshot" in urlparse(self.path).query:
                    body = body.replace(b".mjpg", b".jpg")
                self._send_bytes(
                    body,
                    "text/html; charset=utf-8",
                )
                return
            if path in ("/status.json", "/dashboard/status.json"):
                self._send_bytes(
                    json.dumps(get_snapshot(), sort_keys=True).encode("utf-8"),
                    "application/json",
                )
                return
            panel = PANEL_ROUTES.get(path)
            if panel is not None:
                self._stream_mjpeg(frames[panel])
                return
            panel = PANEL_IMAGE_ROUTES.get(path)
            if panel is not None:
                self._send_bytes(frames[panel].snapshot()[0], "image/jpeg")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream_mjpeg(self, frame: SharedFrame) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            revision = -1
            try:
                while True:
                    jpeg, revision = frame.wait_next(revision)
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

    return DashboardHandler


def _colorize_robot(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        geom_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
        name = f"{body_name} {geom_name}"
        lower = name.lower()
        if any(
            token in lower
            for token in (
                "hand",
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
                "_pp",
                "_mp",
                "_dp",
            )
        ):
            model.geom_rgba[geom_id] = np.array(
                [0.94, 0.61, 0.24, 1.0], dtype=np.float32
            )
        elif "head" in lower:
            model.geom_rgba[geom_id] = np.array(
                [0.88, 0.82, 0.62, 1.0], dtype=np.float32
            )
        elif "Left" in name or "left_" in lower:
            model.geom_rgba[geom_id] = np.array(
                [0.18, 0.48, 0.95, 1.0], dtype=np.float32
            )
        elif "Right" in name or "right_" in lower:
            model.geom_rgba[geom_id] = np.array(
                [0.20, 0.72, 0.42, 1.0], dtype=np.float32
            )
        elif "neck" in lower or "torso" in lower or "waist" in lower:
            model.geom_rgba[geom_id] = np.array(
                [0.78, 0.78, 0.84, 1.0], dtype=np.float32
            )
        elif "pelvis" in lower:
            model.geom_rgba[geom_id] = np.array(
                [0.55, 0.56, 0.62, 1.0], dtype=np.float32
            )


class RobotVisualizer:
    def __init__(
        self, model_xml: Path, width: int, height: int, waist_body: str
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        _colorize_robot(self.model)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.lookat[:] = np.array([0.02, 0.0, 0.82], dtype=np.float64)
        self.camera.distance = 2.35
        self.camera.azimuth = 142.0
        self.camera.elevation = -10.0
        self.joint_addr: dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if name:
                self.joint_addr[name] = int(self.model.jnt_qposadr[joint_id])
        self.body_ids = {
            name: self._body_id(name)
            for name in {
                waist_body,
                *(_HAND_BODIES["left"]),
                *(_HAND_BODIES["right"]),
            }
        }
        self.waist_body = waist_body

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body not found: {name}")
        return int(body_id)

    @staticmethod
    def _section_values(section: Any, *, current: bool) -> dict[str, float]:
        if not isinstance(section, dict):
            return {}
        names_key = "name" if current else "joint_names"
        names = section.get(names_key)
        positions = section.get("q")
        if not isinstance(names, list) or not isinstance(positions, list):
            return {}
        return {
            str(name): float(position)
            for name, position in zip(names, positions, strict=False)
        }

    def set_pose(
        self,
        current_state: dict[str, Any],
    ) -> None:
        adam_values = self._section_values(current_state.get("adam"), current=True)
        sharpa_values = self._section_values(
            current_state.get("sharpa"), current=True
        )
        self._set_pose_values(adam_values, sharpa_values)

    def set_predicted_pose(
        self,
        action: dict[str, Any],
    ) -> None:
        action_adam = self._section_values(action.get("adam"), current=False)
        action_sharpa = self._section_values(action.get("sharpa"), current=False)
        missing_adam = [
            name for name in ADAM_COMMAND_JOINTS_19 if name not in action_adam
        ]
        missing_hands = [
            name for name in SHARPA_JOINT_NAMES if name not in action_sharpa
        ]
        if missing_adam or missing_hands:
            raise ValueError(
                "predicted command is incomplete: "
                f"missing adam={missing_adam}, missing hands={missing_hands}"
            )
        adam_values = {name: 0.0 for name in LEG_PND_JOINTS}
        adam_values.update(
            {name: action_adam[name] for name in ADAM_COMMAND_JOINTS_19}
        )
        self._set_pose_values(adam_values, action_sharpa)

    def _set_pose_values(
        self,
        adam_values: dict[str, float],
        sharpa_values: dict[str, float],
    ) -> None:
        if "dof_pos/neckYaw" in adam_values and "dof_pos/neckPitch" in adam_values:
            adam_values["dof_pos/neckYaw"], adam_values["dof_pos/neckPitch"] = (
                adam_values["dof_pos/neckPitch"],
                adam_values["dof_pos/neckYaw"],
            )
        qpos = self.model.qpos0.copy()
        for pnd_name, mujoco_name in DASHBOARD_PND_TO_MUJOCO_JOINT.items():
            if pnd_name in adam_values and mujoco_name in self.joint_addr:
                qpos[self.joint_addr[mujoco_name]] = adam_values[pnd_name]
        for joint_name in SHARPA_JOINT_NAMES:
            if joint_name in sharpa_values and joint_name in self.joint_addr:
                qpos[self.joint_addr[joint_name]] = sharpa_values[joint_name]
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def robot_frame(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def hand_frame(
        self, width: int, height: int, title: str, sample_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        waist_id = self.body_ids[self.waist_body]
        waist_pos = self.data.xpos[waist_id].astype(np.float32)
        waist_rot = self.data.xmat[waist_id].reshape(3, 3).astype(np.float32)
        hands = []
        for side in ("left", "right"):
            points = []
            for body_name in _HAND_BODIES[side]:
                body_pos = self.data.xpos[self.body_ids[body_name]].astype(np.float32)
                points.append(waist_rot.T @ (body_pos - waist_pos))
            hands.append(_pnd_points_to_baai(np.asarray(points, dtype=np.float32)))
        keypoints = np.stack([hands[0], hands[1]], axis=0)
        wrist18 = np.zeros(18, dtype=np.float32)
        for body_name, out_offset in (
            ("wristRollLeft", 0),
            ("wristRollRight", 9),
        ):
            body_id = self.body_ids[body_name]
            body_pos = self.data.xpos[body_id].astype(np.float32)
            body_rot = self.data.xmat[body_id].reshape(3, 3).astype(np.float32)
            rel_pos = waist_rot.T @ (body_pos - waist_pos)
            rel_rot = waist_rot.T @ body_rot
            rel_pos_baai = _PND_TO_BAAI @ rel_pos
            rel_rot_baai = _PND_TO_BAAI @ rel_rot @ _PND_TO_BAAI.T
            wrist18[out_offset : out_offset + 3] = rel_pos_baai
            wrist18[out_offset + 3 : out_offset + 9] = np.concatenate(
                [rel_rot_baai[:, 0], rel_rot_baai[:, 1]]
            )
        frame = _render_baai_hand_motion(
            keypoints[None, ...],
            width=width,
            height=height,
            title_prefix=f"{title} · waist={self.waist_body}",
            frame_index=0,
            sample_count=sample_count,
        )
        return frame, wrist18[:9], wrist18[9:18]

    def close(self) -> None:
        self.renderer.close()


def _pnd_points_to_baai(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return (pts @ _PND_TO_BAAI.T).astype(np.float32)


def _render_baai_hand_motion(
    keypoints: np.ndarray,
    *,
    width: int,
    height: int,
    title_prefix: str,
    frame_index: int,
    sample_count: int | None = None,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 4 or kp.shape[1:] != (2, 21, 3):
        raise ValueError(f"keypoints must be (T, 2, 21, 3), got {kp.shape}")
    frame_index = max(0, min(int(frame_index), kp.shape[0] - 1))
    def to_plot(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return points[..., 0], points[..., 2], points[..., 1]

    visible_points = np.concatenate(
        [kp[frame_index].reshape(-1, 3), np.zeros((1, 3), dtype=np.float32)],
        axis=0,
    )
    plot_points = visible_points[:, [0, 2, 1]]
    centers = (plot_points.min(axis=0) + plot_points.max(axis=0)) * 0.5
    plot_span = max(float(np.ptp(plot_points, axis=0).max()) * 1.25, 0.5)
    half_span = plot_span * 0.5

    fig = plt.figure(figsize=(8.0, 8.0), dpi=100, facecolor=_BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_BG_COLOR)
    ax.set_xlim(centers[0] - half_span, centers[0] + half_span)
    ax.set_ylim(centers[1] - half_span, centers[1] + half_span)
    ax.set_zlim(centers[2] - half_span, centers[2] + half_span)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=20.0, azim=60.0)
    ax.set_xlabel("x  (m)", fontsize=8, labelpad=2, color="#666666")
    ax.set_ylabel("z  (m)", fontsize=8, labelpad=2, color="#666666")
    ax.set_zlabel("y  (m)", fontsize=8, labelpad=2, color="#666666")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(_PANE_COLOR)
        axis.pane.set_edgecolor((0.83, 0.83, 0.83, 1.0))
        axis._axinfo["grid"]["color"] = _GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.set_tick_params(labelsize=6, colors="#888888")

    hip_x, hip_y, hip_z = to_plot(np.zeros((1, 3), dtype=np.float32))
    ax.scatter(
        hip_x,
        hip_y,
        hip_z,
        s=180,
        c=_HIP_COLOR,
        marker="*",
        edgecolors="white",
        linewidths=0.7,
        depthshade=False,
    )

    for hand in range(2):
        color = _HAND_COLORS[hand]
        hand_kp = kp[frame_index, hand]
        kx, ky, kz = to_plot(hand_kp)
        for first, second in _HAND_EDGES:
            ax.plot(
                [kx[first], kx[second]],
                [ky[first], ky[second]],
                [kz[first], kz[second]],
                color=color,
                linewidth=1.6,
                alpha=0.92,
                solid_capstyle="round",
            )
        ax.scatter(
            kx[1:],
            ky[1:],
            kz[1:],
            color=color,
            s=14,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.25,
            depthshade=False,
        )
        ax.scatter(
            [kx[0]],
            [ky[0]],
            [kz[0]],
            color=color,
            s=16,
            edgecolors="black",
            linewidths=0.55,
            depthshade=False,
            marker=("o" if hand == 0 else "s"),
        )

    ax.scatter(
        [], [], [], color=_HAND_COLORS[0], s=18, marker="o", edgecolors="black",
        linewidths=0.4, label="L hand"
    )
    ax.scatter(
        [], [], [], color=_HAND_COLORS[1], s=18, marker="s", edgecolors="black",
        linewidths=0.4, label="R hand"
    )
    ax.scatter(
        [], [], [], color=_HIP_COLOR, s=90, marker="*", edgecolors="white",
        linewidths=0.5, label="waist"
    )
    legend = ax.legend(
        loc="upper left",
        fontsize=7,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        borderpad=0.4,
        labelspacing=0.35,
    )
    for text in legend.get_texts():
        text.set_color("#333333")
    suffix = (
        f"sample {sample_count}"
        if sample_count is not None
        else f"frame {frame_index:02d}/{kp.shape[0]}"
    )
    ax.set_title(
        f"{title_prefix}  ·  {suffix}", fontsize=11, color="#222222", pad=8
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    if rgb.shape[1] != width or rgb.shape[0] != height:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


class DashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("dashboard")

        self.declare_parameter("http_host", "127.0.0.1")
        self.declare_parameter("http_port", 8088)
        self.declare_parameter(
            "status_file", "deploy/runs/ws_dashboard/latest_status.json"
        )
        self.declare_parameter("robot_states_topic", "/ws/robot_states")
        self.declare_parameter("robot_tactile_topic", "/ws/robot_tactile")
        self.declare_parameter("model_image_topic", "/ws/robot_vision")
        self.declare_parameter("obs_topic", "/ws/obs")
        self.declare_parameter("pred_topic", "/ws/pred")
        self.declare_parameter("action_topic", "/ws/action")
        self.declare_parameter("status_topics", DEFAULT_STATUS_TOPICS)
        self.declare_parameter(
            "model_xml",
            "",
        )
        self.declare_parameter("task_prompt", "把苹果放在盘子里")
        self.declare_parameter("baai_ssh_host", "BAAI2")
        self.declare_parameter(
            "pred_video_dir", "deploy/runs/ws_dashboard/pred_videos"
        )
        self.declare_parameter("render_width", 640)
        self.declare_parameter("render_height", 480)
        self.declare_parameter("render_fps", 5.0)
        self.declare_parameter("jpeg_quality", 82)
        self.declare_parameter("waist_body", "pelvis")

        self.http_host = str(self.get_parameter("http_host").value)
        self.http_port = int(self.get_parameter("http_port").value)
        self.status_file = Path(str(self.get_parameter("status_file").value))
        self.model_xml = resolve_model_xml(str(self.get_parameter("model_xml").value))
        self.task_prompt = str(self.get_parameter("task_prompt").value)
        self.baai_ssh_host = str(self.get_parameter("baai_ssh_host").value)
        self.pred_video_dir = Path(
            str(self.get_parameter("pred_video_dir").value)
        ).resolve()
        self.render_width = int(self.get_parameter("render_width").value)
        self.render_height = int(self.get_parameter("render_height").value)
        self.render_fps = float(self.get_parameter("render_fps").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.waist_body = str(self.get_parameter("waist_body").value)
        self.started_monotonic = time.monotonic()
        self.lock = threading.Lock()
        self.policy_provider = ""
        self.pred_video_capability = "unknown"
        self.topics: dict[str, dict[str, Any]] = {}
        self.statuses: dict[str, dict[str, Any]] = {}
        self._latest_model_image: dict[str, Any] | None = None
        self._latest_robot_state: tuple[int, dict[str, Any]] | None = None
        self._latest_action: tuple[int, dict[str, Any]] | None = None
        self._pending_pred_video: tuple[int, str] | None = None
        self._pred_video_condition = threading.Condition()
        self._stop_event = threading.Event()
        self._http_server: _ReusableThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._pred_video_thread: threading.Thread | None = None
        self.frames = {
            key: SharedFrame(
                (
                    MODEL_IMAGE_WIDTH
                    if key in {"zed", "pred_video"}
                    else self.render_width
                ),
                (
                    MODEL_IMAGE_HEIGHT
                    if key in {"zed", "pred_video"}
                    else self.render_height
                ),
                title,
                self.jpeg_quality,
            )
            for key, title in PANEL_TITLES.items()
        }
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._subscribe(
            "robot_states",
            str(self.get_parameter("robot_states_topic").value),
            RobotState,
            self._robot_state_summary,
            qos,
        )
        self._subscribe(
            "robot_tactile",
            str(self.get_parameter("robot_tactile_topic").value),
            RobotTactile,
            self._robot_tactile_summary,
            qos,
        )
        self._subscribe(
            "model_image",
            str(self.get_parameter("model_image_topic").value),
            ModelImage,
            self._model_image_summary,
            qos,
        )
        self._subscribe(
            "obs",
            str(self.get_parameter("obs_topic").value),
            PolicyObs,
            self._policy_obs_summary,
            qos,
        )
        self._subscribe(
            "pred",
            str(self.get_parameter("pred_topic").value),
            PolicyPred,
            self._policy_pred_summary,
            qos,
        )
        self._subscribe(
            "action",
            str(self.get_parameter("action_topic").value),
            PndAction,
            self._action_summary,
            qos,
        )
        for topic in self._status_topics():
            self.create_subscription(
                Status,
                topic,
                lambda msg, status_topic=topic: self._on_status(status_topic, msg),
                qos,
            )

        self._start_workers()
        self._start_http()
        self.create_timer(1.0, self._write_latest_status)
        self._write_latest_status()
        self.get_logger().info(
            f"PND-SharpA six-panel dashboard: http://{self.http_host}:{self.http_port}/"
        )

    def destroy_node(self) -> bool:
        self._stop_workers()
        self._stop_http()
        return super().destroy_node()

    def _status_topics(self) -> list[str]:
        topics: list[str] = []
        for item in self.get_parameter("status_topics").value:
            topic = str(item)
            if topic and topic not in topics:
                topics.append(topic)
        return topics

    def _subscribe(
        self,
        key: str,
        topic: str,
        msg_type: Any,
        summarize: Callable[[Any], dict[str, Any]],
        qos: QoSProfile,
    ) -> None:
        self.create_subscription(
            msg_type,
            topic,
            lambda msg: self._record_topic(key, topic, summarize(msg)),
            qos,
        )
        with self.lock:
            self.topics[key] = {
                "topic": topic,
                "received": 0,
                "last_received_at": None,
                "last_age_s": None,
                "last": None,
            }

    def _record_topic(self, key: str, topic: str, summary: dict[str, Any]) -> None:
        now = time.time()
        with self.lock:
            current = self.topics[key]
            current["topic"] = topic
            current["received"] = int(current.get("received", 0)) + 1
            current["last_received_at"] = _utc_now()
            current["_last_received_unix"] = now
            current["last"] = summary

    def _set_policy_provider(self, provider: str) -> None:
        normalized = str(provider).strip().lower()
        if not normalized:
            return
        capability = _pred_video_capability(normalized)
        with self.lock:
            previous_provider = self.policy_provider
            previous_capability = self.pred_video_capability
            self.policy_provider = normalized
            self.pred_video_capability = capability
        if (
            capability == previous_capability
            and normalized == previous_provider
        ):
            return
        if capability == "unsupported":
            self.frames["pred_video"].update(
                _status_frame(
                    MODEL_IMAGE_WIDTH,
                    MODEL_IMAGE_HEIGHT,
                    PANEL_TITLES["pred_video"],
                    f"provider {normalized} has no prediction video",
                ),
                state="unsupported",
                capability=capability,
                provider=normalized,
            )
        elif capability == "supported":
            self.frames["pred_video"].update(
                _status_frame(
                    MODEL_IMAGE_WIDTH,
                    MODEL_IMAGE_HEIGHT,
                    PANEL_TITLES["pred_video"],
                    "waiting for server prediction video",
                ),
                state="starting",
                capability=capability,
                provider=normalized,
            )

    def _on_status(self, topic: str, msg: Status) -> None:
        node_name = msg.node or _topic_key(topic)
        status_key = {
            "/ws/obs/debug": "obs_sync_debug",
            "/ws/action_execute/plan_debug": "action_execute_plan",
            "/ws/action_execute/safety": "action_execute_safety",
        }.get(topic, node_name)
        summary = {
            "topic": topic,
            "node": node_name,
            "ok": bool(msg.ok),
            "header": _header_dict(msg.header),
            "payload": _parse_json(msg.payload_json),
        }
        payload = summary["payload"]
        provider = str(payload.get("provider") or "") if isinstance(payload, dict) else ""
        if status_key == "policy_client":
            self._set_policy_provider(provider)
        now = time.time()
        with self.lock:
            current = self.statuses.setdefault(
                status_key,
                {
                    "topic": topic,
                    "received": 0,
                    "last_received_at": None,
                    "last_age_s": None,
                    "ok": None,
                    "last": None,
                },
            )
            current["topic"] = topic
            current["received"] = int(current.get("received", 0)) + 1
            current["last_received_at"] = _utc_now()
            current["_last_received_unix"] = now
            current["ok"] = bool(msg.ok)
            current["last"] = summary

    def _robot_state_summary(self, msg: RobotState) -> dict[str, Any]:
        payload = _parse_json(msg.payload_json)
        if isinstance(payload, dict):
            with self.lock:
                self._latest_robot_state = (int(msg.seq), payload)
        return {
            "header": _header_dict(msg.header),
            "seq": int(msg.seq),
            "stamp_ns": int(msg.stamp_ns),
            "recv_time_ns": int(msg.recv_time_ns),
            "payload": payload,
        }

    def _robot_tactile_summary(self, msg: RobotTactile) -> dict[str, Any]:
        return {
            "header": _header_dict(msg.header),
            "seq": int(msg.seq),
            "nearest_obs_seq": int(msg.nearest_obs_seq),
            "stamp_ns": int(msg.stamp_ns),
            "recv_time_ns": int(msg.recv_time_ns),
            "metadata": _parse_json(msg.metadata_json),
            "data": _bytes_summary(msg.data),
        }

    def _model_image_summary(self, msg: ModelImage) -> dict[str, Any]:
        image = {
            "seq": int(msg.frame_seq),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "data": bytes(msg.data),
        }
        with self.lock:
            self._latest_model_image = image
        return {
            "header": _header_dict(msg.header),
            "frame_seq": int(msg.frame_seq),
            "stamp_ns": int(msg.stamp_ns),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "data": _bytes_summary(msg.data),
        }

    def _policy_obs_summary(self, msg: PolicyObs) -> dict[str, Any]:
        return {
            "header": _header_dict(msg.header),
            "seq": int(msg.seq),
            "provider": str(msg.provider),
            "payload": _parse_json(msg.payload_json),
            "image_rgb": _bytes_summary(msg.image_rgb),
        }

    def _policy_pred_summary(self, msg: PolicyPred) -> dict[str, Any]:
        payload = _parse_json(msg.payload_json)
        provider = str(msg.provider)
        if not provider and isinstance(payload, dict):
            provider = str(payload.get("provider") or "")
        self._set_policy_provider(provider)
        if isinstance(payload, dict):
            remote_path = _prediction_video_path(payload)
            request_index = int(payload.get("request_index", int(msg.seq)))
            if remote_path:
                with self._pred_video_condition:
                    self._pending_pred_video = (request_index, remote_path)
                    self._pred_video_condition.notify_all()
        return {
            "header": _header_dict(msg.header),
            "seq": int(msg.seq),
            "provider": str(msg.provider),
            "payload": payload,
        }

    def _action_summary(self, msg: PndAction) -> dict[str, Any]:
        payload = _parse_json(msg.payload_json)
        if isinstance(payload, dict):
            with self.lock:
                self._latest_action = (int(msg.seq), payload)
        return {
            "header": _header_dict(msg.header),
            "seq": int(msg.seq),
            "stamp_ns": int(msg.stamp_ns),
            "payload": payload,
        }

    def _start_workers(self) -> None:
        self._render_thread = threading.Thread(
            target=self._render_loop, name="dashboard-render", daemon=True
        )
        self._pred_video_thread = threading.Thread(
            target=self._pred_video_loop,
            name="dashboard-pred-video",
            daemon=True,
        )
        self._render_thread.start()
        self._pred_video_thread.start()

    def _stop_workers(self) -> None:
        self._stop_event.set()
        with self._pred_video_condition:
            self._pred_video_condition.notify_all()
        for thread in (self._render_thread, self._pred_video_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)

    @staticmethod
    def _image_rgb(image: dict[str, Any]) -> np.ndarray:
        width = int(image["width"])
        height = int(image["height"])
        data = np.frombuffer(image["data"], dtype=np.uint8)
        if width <= 0 or height <= 0 or data.size != width * height * 3:
            raise ValueError("model image size does not match RGB payload")
        frame = data.reshape(height, width, 3)
        encoding = str(image["encoding"]).lower()
        if encoding == "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif encoding != "rgb8":
            raise ValueError(f"unsupported model image encoding: {encoding}")
        return frame

    def _render_loop(self) -> None:
        current_view: RobotVisualizer | None = None
        predicted_view: RobotVisualizer | None = None
        try:
            current_view = RobotVisualizer(
                self.model_xml,
                self.render_width,
                self.render_height,
                self.waist_body,
            )
            predicted_view = RobotVisualizer(
                self.model_xml,
                self.render_width,
                self.render_height,
                self.waist_body,
            )
            last_image_seq = -1
            last_state_seq = -1
            last_action_seq = -1
            period_s = 1.0 / max(1.0, self.render_fps)
            while not self._stop_event.is_set():
                started = time.monotonic()
                with self.lock:
                    image = self._latest_model_image
                    state_entry = self._latest_robot_state
                    action_entry = self._latest_action
                if image is not None and int(image["seq"]) != last_image_seq:
                    try:
                        self.frames["zed"].update(
                            self._image_rgb(image),
                            state="ok",
                            frame_seq=int(image["seq"]),
                            width=int(image["width"]),
                            height=int(image["height"]),
                        )
                        last_image_seq = int(image["seq"])
                    except Exception as exc:
                        self._update_error("zed", exc)
                if state_entry is not None and state_entry[0] != last_state_seq:
                    try:
                        state_seq, state = state_entry
                        current_view.set_pose(state)
                        self.frames["robot"].update(
                            current_view.robot_frame(),
                            state="ok",
                            seq=state_seq,
                            joints=len(state.get("adam", {}).get("q", [])),
                            hand_joints=len(state.get("sharpa", {}).get("q", [])),
                        )
                        hand_frame, left_wrist, right_wrist = (
                            current_view.hand_frame(
                                self.render_width,
                                self.render_height,
                                "Current Waist Hands",
                                state_seq,
                            )
                        )
                        self.frames["current_hands"].update(
                            hand_frame,
                            state="ok",
                            seq=state_seq,
                            samples=state_seq,
                            waist_body=self.waist_body,
                            q44_source="measured_sharpa",
                            left_wrist_9d_waist=left_wrist.tolist(),
                            right_wrist_9d_waist=right_wrist.tolist(),
                        )
                        last_state_seq = state_seq
                    except Exception as exc:
                        self._update_error("robot", exc)
                        self._update_error("current_hands", exc)
                if action_entry is not None and action_entry[0] != last_action_seq:
                    try:
                        action_seq, action = action_entry
                        predicted_view.set_predicted_pose(action)
                        policy_pred = action.get("policy_pred", {})
                        policy_json = (
                            policy_pred.get("json", {})
                            if isinstance(policy_pred, dict)
                            else {}
                        )
                        request_index = policy_json.get(
                            "request_index",
                            action.get("source", {}).get("policy_pred_seq", -1),
                        )
                        step = int(action.get("selected_action_step", 0))
                        meta = {
                            "state": "ok",
                            "seq": action_seq,
                            "request_index": int(request_index),
                            "step_index": step,
                            "joints": len(action.get("adam", {}).get("q", [])),
                            "hand_joints": len(
                                action.get("sharpa", {}).get("q", [])
                            ),
                        }
                        self.frames["pred_robot"].update(
                            predicted_view.robot_frame(), **meta
                        )
                        hand_frame, left_wrist, right_wrist = (
                            predicted_view.hand_frame(
                                self.render_width,
                                self.render_height,
                                "Predicted Waist Hands",
                                step,
                            )
                        )
                        self.frames["pred_hands"].update(
                            hand_frame,
                            **meta,
                            waist_body=self.waist_body,
                            left_wrist_9d_waist=left_wrist.tolist(),
                            right_wrist_9d_waist=right_wrist.tolist(),
                        )
                        last_action_seq = action_seq
                    except Exception as exc:
                        self._update_error("pred_robot", exc)
                        self._update_error("pred_hands", exc)
                elapsed = time.monotonic() - started
                self._stop_event.wait(max(0.01, period_s - elapsed))
        except Exception as exc:
            for panel in ("robot", "current_hands", "pred_robot", "pred_hands"):
                self._update_error(panel, exc)
            self.get_logger().error(f"dashboard renderer failed: {exc}")
        finally:
            for view in (current_view, predicted_view):
                if view is not None:
                    view.close()

    def _update_error(self, panel: str, exc: Exception) -> None:
        width = (
            MODEL_IMAGE_WIDTH
            if panel in {"zed", "pred_video"}
            else self.render_width
        )
        height = (
            MODEL_IMAGE_HEIGHT
            if panel in {"zed", "pred_video"}
            else self.render_height
        )
        self.frames[panel].update(
            _status_frame(
                width,
                height,
                PANEL_TITLES[panel],
                repr(exc),
            ),
            state="error",
            error=repr(exc),
        )

    def _pred_video_loop(self) -> None:
        self.pred_video_dir.mkdir(parents=True, exist_ok=True)
        active_frames: list[np.ndarray] = []
        active_request = -1
        loaded_monotonic = time.monotonic()
        displayed_index = -1
        while not self._stop_event.is_set():
            with self.lock:
                capability = self.pred_video_capability
            if capability == "unsupported":
                active_frames = []
                active_request = -1
                displayed_index = -1
                with self._pred_video_condition:
                    self._pending_pred_video = None
                self._stop_event.wait(0.08)
                continue
            pending: tuple[int, str] | None
            with self._pred_video_condition:
                pending = self._pending_pred_video
                self._pending_pred_video = None
                if pending is None:
                    self._pred_video_condition.wait(timeout=0.08)
                    pending = self._pending_pred_video
                    self._pending_pred_video = None
            if pending is not None:
                request_index, remote_path = pending
                try:
                    local_path = self._sync_pred_video(request_index, remote_path)
                    frames = self._decode_video(local_path)
                    if not frames:
                        raise RuntimeError(f"predicted video has no frames: {local_path}")
                    active_frames = frames
                    active_request = request_index
                    loaded_monotonic = time.monotonic()
                    displayed_index = -1
                except Exception as exc:
                    if not active_frames:
                        self._update_error("pred_video", exc)
                    else:
                        jpeg, status, _ = self.frames["pred_video"].snapshot()
                        self.frames["pred_video"].update_jpeg(
                            jpeg,
                            **{
                                **status,
                                "state": "waiting",
                                "sync_error": repr(exc),
                                "pending_request_index": request_index,
                            },
                        )
            if active_frames:
                frame_index = int(
                    (time.monotonic() - loaded_monotonic) * 7.5
                ) % len(active_frames)
                if frame_index != displayed_index:
                    self.frames["pred_video"].update(
                        active_frames[frame_index],
                        state="ok",
                        request_index=active_request,
                        step_index=frame_index,
                        video_frame_index=frame_index,
                        frame_count=len(active_frames),
                        source="BAAI2 server_video_pred_path",
                    )
                    displayed_index = frame_index

    def _sync_pred_video(self, request_index: int, remote_path: str) -> Path:
        local_path = self.pred_video_dir / f"request_{request_index:06d}.mp4"
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        tmp_path = local_path.with_suffix(".mp4.tmp")
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            self.baai_ssh_host,
            f"cat -- {shlex.quote(remote_path)}",
        ]
        ssh_environment = os.environ.copy()
        for variable in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "OPENSSL_CONF",
            "OPENSSL_MODULES",
        ):
            ssh_environment.pop(variable, None)
        try:
            with tmp_path.open("wb") as output:
                result = subprocess.run(
                    command,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    env=ssh_environment,
                    timeout=30.0,
                    check=False,
                )
            if result.returncode != 0:
                error = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"BAAI2 predicted-video sync failed: {error}")
            if tmp_path.stat().st_size <= 0:
                raise RuntimeError("BAAI2 predicted-video sync returned an empty file")
            tmp_path.replace(local_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        videos = sorted(
            self.pred_video_dir.glob("request_*.mp4"),
            key=lambda path: path.stat().st_mtime,
        )
        for stale in videos[:-12]:
            stale.unlink(missing_ok=True)
        return local_path

    @staticmethod
    def _decode_video(path: Path) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while len(frames) < 64:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        return frames

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            topics = self._public_entries(self.topics, now)
            statuses = self._public_entries(self.statuses, now)
            policy_provider = self.policy_provider
            pred_video_capability = self.pred_video_capability
        panels = {
            key: frame.snapshot()[1] for key, frame in self.frames.items()
        }
        return {
            "generated_at": _utc_now(),
            "node": "dashboard",
            "uptime_s": round(time.monotonic() - self.started_monotonic, 3),
            "http": {
                "host": self.http_host,
                "port": self.http_port,
                "routes": [
                    "/",
                    "/status.json",
                    *PANEL_ROUTES.keys(),
                    *PANEL_IMAGE_ROUTES.keys(),
                ],
            },
            "latest_file": str(self.status_file),
            "_meta": {
                "prompt": self.task_prompt,
                "layout": "3x2",
                "panel_count": 6,
                "model_xml": str(self.model_xml),
                "pred_video_source": self.baai_ssh_host,
                "policy_provider": policy_provider,
                "pred_video_capability": pred_video_capability,
                "predicted_pose_source": {
                    "legs": "zero",
                    "waist_neck_arms_hands": "/ws/action",
                },
                "waist_body": self.waist_body,
            },
            **panels,
            "panels": panels,
            "topics": topics,
            "statuses": statuses,
        }

    @staticmethod
    def _public_entries(
        entries: dict[str, dict[str, Any]], now: float
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for key, value in entries.items():
            public = dict(value)
            last_unix = public.pop("_last_received_unix", None)
            public["last_age_s"] = (
                round(now - float(last_unix), 3) if last_unix is not None else None
            )
            result[key] = public
        return result

    def _write_latest_status(self) -> None:
        try:
            snapshot = self.snapshot()
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.status_file.with_name(self.status_file.name + ".tmp")
            tmp_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, self.status_file)
        except Exception as exc:
            self.get_logger().warning(f"failed to write status file: {exc}")

    def _start_http(self) -> None:
        self._http_server = _ReusableThreadingHTTPServer(
            (self.http_host, self.http_port),
            _handler_factory(self.snapshot, self.frames),
        )
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="ws-dashboard-http",
            daemon=True,
        )
        self._http_thread.start()

    def _stop_http(self) -> None:
        server = self._http_server
        thread = self._http_thread
        self._http_server = None
        self._http_thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
