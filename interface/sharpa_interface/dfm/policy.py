"""Translate the existing SharpA wire contract to a checkpoint-backed DFM Policy."""

from __future__ import annotations

import io
import json
import math
import threading
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
from sharpa_interface.server.metadata import validate_server_metadata
from sharpa_interface.server.observation import validate_policy_observation


class DFMDeploymentPolicy:
    def __init__(self, policy, calibration, *, checkpoint, task, run, port, execute_steps=30):
        self.policy, self.calibration = policy, calibration
        capabilities = policy.capabilities
        if (
            capabilities.state_dim != 62
            or capabilities.action_dim != 62
            or capabilities.horizon != 30
            or capabilities.action_hz != 30
            or capabilities.cameras != ("zed_left", "left_wrist", "right_wrist")
            or capabilities.reference_camera != "zed_left"
            or capabilities.action_format != "sharpa62_camera_absolute"
        ):
            raise ValueError("Deployment requires the trained three-camera 30Hz/30-step SharpA62 checkpoint")
        if type(execute_steps) is not int or not 1 <= execute_steps <= 30:
            raise ValueError("execute_steps must be within the saved 30-step horizon")
        self.execute_steps = execute_steps
        metadata_format = {
            "schema": "sharpa_policy_metadata_format.v1",
            "format_id": "dfm_sim_real_canonical.v1",
            "image": {
                name: {"history_len": 0, "current": True} for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
            },
            "state": {
                "history_len": 0,
                "current": True,
                "left_wrist": {"joint": False, "eef": True},
                "right_wrist": {"joint": False, "eef": True},
                "hand_joint": {"left": True, "right": True},
            },
            "sensor": {name: {"history_len": 0, "current": False} for name in ("tau", "wrench", "deformation")},
        }
        self.metadata = validate_server_metadata(
            {
                "schema": "sharpa_policy_server.v3",
                "policy_family": "dfm",
                "model_id": run["model"],
                "checkpoint_id": run["run_id"],
                "checkpoint_path": str(checkpoint),
                "task_id": task["id"],
                "run_id": run["run_id"],
                "dataset_path": task.get("dataset", ""),
                "prompt": task["prompt"],
                "transport": "websocket+binary_msgpack",
                "observation_schema": "sharpa_policy_observation.v3",
                "action_schema": "sharpa_policy_action.v5",
                "host": "127.0.0.1",
                "port": port,
                "infer_path": "/infer",
                "health_path": "/healthz",
                "metadata_path": "/metadata",
                "reset_path": "/reset",
                "max_message_size": 67108864,
                "metadata_format": metadata_format,
                "eef_frame": "canonical",
                "calibration_id": calibration.calibration_id,
                "session_owner_token_required": True,
                "interface": {
                    "eef_contract": {
                        "schema": "sharpa62_direct_absolute_eef.v1",
                        "observation_eef_def": "absolute",
                        "action_eef_def": "absolute",
                        "eef_pose_layout": "xyz,r00,r10,r20,r01,r11,r21",
                        "server_xyz_direction_transform": "none",
                    }
                },
                "execution": {
                    "frequency_hz": 30,
                    "action_length": 30,
                    "execute_start": 0,
                    "execute_length": execute_steps,
                    "rtc": 0,
                },
            }
        )
        self._lock = threading.Lock()
        self.session = None

    def warmup(self):
        """Prepare both inference paths, then restore the initial episode and seed.

        Inputs are deterministic compile probes at the saved three-camera pixel
        budgets. The measured calibration is used unchanged, and returned targets
        stay inside this process. This is not a robot execution or an evaluation.
        """
        from PIL import Image
        from sharpa_interface.server.serialization import packb, unpackb

        with self._lock:
            if self.session is not None:
                raise RuntimeError("Warmup requires an unowned policy")
        processor = json.loads((Path(self.metadata["checkpoint_path"]) / "dfm_processor.json").read_text())
        if processor.get("camera_views") != ["ego", "left_wrist", "right_wrist"]:
            raise ValueError("Warmup requires the saved three-camera processor")
        images = {}
        image_sizes = {}
        # Native ego (1280x720) and wrist (480x270) inputs share 16:9. These
        # factor-32 sizes match their saved Qwen resize grids without allocating
        # full camera frames or depending on a dataset at service startup.
        aspect = 16 / 9
        for camera, view in (
            ("ego_cam", "ego"),
            ("left_wrist_cam", "left_wrist"),
            ("right_wrist_cam", "right_wrist"),
        ):
            budget = processor["image_sizes"][view]
            minimum, maximum = budget["min_pixels"], budget["max_pixels"]
            if type(minimum) is not int or type(maximum) is not int or not 0 < minimum <= maximum <= 16_777_216:
                raise ValueError("Invalid saved camera pixel budget")
            height = int(math.sqrt(maximum / aspect) // 32) * 32
            width = int(math.sqrt(maximum * aspect) // 32) * 32
            if not minimum <= height * width <= maximum:
                raise ValueError("Saved camera budget cannot represent the native 16:9 resize grid")
            buffer = io.BytesIO()
            Image.new("RGB", (width, height), (127, 127, 127)).save(buffer, format="JPEG")
            images[camera] = buffer.getvalue()
            image_sizes[view] = [width, height]
        model_state = np.zeros(62, dtype=np.float32)
        model_state[3:9] = model_state[12:18] = [1, 0, 0, 0, 1, 0]
        wire_state = self.calibration.convert(model_state, to_model=False)
        session = "dfm-startup-warmup"
        request_seconds = []
        request_diagnostics = []
        started = time.perf_counter()
        try:
            self.reset({"session_id": session})
            for request_id in (1, 2):
                timestamp_ns = request_id * 1_000_000_000
                value = {
                    "schema": "sharpa_policy_observation.v3",
                    "metadata_format_id": self.metadata["metadata_format"]["format_id"],
                    "session_id": session,
                    "request_id": request_id,
                    "timestamp_ns": timestamp_ns,
                    "prompt": self.metadata["prompt"],
                    "image": {
                        camera: {
                            "history": [],
                            "current": {"encoding": "jpeg", "data": data, "timestamp_ns": timestamp_ns, "valid": True},
                        }
                        for camera, data in images.items()
                    },
                    "state": {
                        "history": None,
                        "current": {
                            "timestamp_ns": timestamp_ns,
                            "left_wrist": {"joint": None, "eef": wire_state[:9], "eef_def": "absolute"},
                            "right_wrist": {"joint": None, "eef": wire_state[9:18], "eef_def": "absolute"},
                            "hand_joint": {"left": wire_state[18:40], "right": wire_state[40:62]},
                            "valid": True,
                        },
                    },
                    "sensor": {name: {"history": None, "current": None} for name in ("tau", "wrench", "deformation")},
                    "execution_feedback": {"last_action_id": None, "executed_steps": 0, "success": True},
                }
                before = time.perf_counter()
                result = self.infer(unpackb(packb(value)))
                packb(result)
                request_diagnostics.append(deepcopy(result["diagnostics"]))
                request_seconds.append(time.perf_counter() - before)
        finally:
            # reset clears generated plans and restores seed 42; invalidate leaves
            # no client session. This also runs after a failed compile/predict.
            try:
                self.reset({"session_id": session})
            finally:
                self.invalidate()
        return {
            "requests": 2,
            "seconds": time.perf_counter() - started,
            "request_seconds": request_seconds,
            "request_diagnostics": request_diagnostics,
            "image_sizes": image_sizes,
        }

    def reset(self, value):
        session = value.get("session_id")
        request_id = value.get("request_id")
        if (
            not isinstance(session, str)
            or not session.strip()
            or (request_id is not None and (type(request_id) is not int or request_id < 0))
        ):
            raise ValueError("Reset requires session_id and an optional nonnegative request_id")
        with self._lock:
            self.session = None
            self.policy.set_seed(42)
            self.policy.reset(self.metadata["prompt"])
            self.session = session
            return {
                "schema": "sharpa_policy_reset.v1",
                "session_id": session,
                "request_id": request_id,
                "reset": True,
                "metadata_format": deepcopy(self.metadata["metadata_format"]),
            }

    def invalidate(self):
        with self._lock:
            self.session = None

    def infer(self, value):
        from PIL import Image

        from dfm.contracts import Observation, ObservationContext

        value = validate_policy_observation(value, self.metadata["metadata_format"])
        with self._lock:
            if self.session is None or value["session_id"] != self.session:
                raise ValueError("Reset and use the active session before infer")
            if value["prompt"] != self.metadata["prompt"]:
                raise ValueError("Observation prompt disagrees with the selected task")
            state = value["state"]["current"]
            if not state["valid"]:
                raise ValueError("Measured state is invalid")
            wire_state = np.concatenate(
                (
                    state["left_wrist"]["eef"],
                    state["right_wrist"]["eef"],
                    state["hand_joint"]["left"],
                    state["hand_joint"]["right"],
                )
            )
            images = {}
            for name, view in (
                ("ego_cam", "ego"),
                ("left_wrist_cam", "left_wrist"),
                ("right_wrist_cam", "right_wrist"),
            ):
                frame = value["image"][name]["current"]
                if not frame["valid"]:
                    raise ValueError(f"Invalid camera frame: {name}")
                with Image.open(io.BytesIO(frame["data"])) as image:
                    if image.width * image.height > 16_777_216:
                        raise ValueError("Decoded camera frame exceeds pixel limit")
                    images[view] = np.array(image.convert("RGB"), dtype=np.uint8)
            model_state = self.calibration.convert(wire_state, to_model=True)
            observation = Observation(
                images["ego"],
                model_state,
                images=images,
                context=ObservationContext(
                    observation_id=f"{self.session}:{value['request_id']}", timestamp_s=value["timestamp_ns"] / 1e9
                ),
            )
            try:
                plan = self.policy.predict(observation)
                if plan.actions.shape != (30, 62) or plan.action_hz != 30:
                    raise ValueError("Policy action shape/frequency differs from the saved contract")
                actions = self.calibration.convert(plan.actions, to_model=False)
            except Exception:
                self.session = None
                raise
            diagnostics = plan.diagnostics_dict()
            diagnostics["calibration_id"] = self.calibration.calibration_id
            return {
                "schema": "sharpa_policy_action.v5",
                "session_id": self.session,
                "request_id": value["request_id"],
                "action_id": uuid4().hex,
                "revision": 0,
                "timestamp_ns": time.time_ns(),
                "execution": deepcopy(self.metadata["execution"]),
                "action": {
                    "left_wrist": {"joint": None, "eef": actions[:, :9], "eef_def": "absolute"},
                    "right_wrist": {"joint": None, "eef": actions[:, 9:18], "eef_def": "absolute"},
                    "hand_joint": {"left": actions[:, 18:40], "right": actions[:, 40:62]},
                },
                "auxiliary": {
                    "video": {"ego": None, "left_wrist": None, "right_wrist": None},
                    "tactile": {"deformation": None, "wrench": None, "hand_tau": None},
                },
                "diagnostics": diagnostics,
                "next_metadata_format": None,
            }
