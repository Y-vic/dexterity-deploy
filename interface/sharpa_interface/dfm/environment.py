"""Canonical ROS wrist ABI to native physical hand/root declarations.

This adapter only prepares observations and targets. It never executes hardware.
Per-arm measured A/B transforms and hand joint order belong to the UR boundary;
native camera declarations are converted using one capture-time snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sharpa_interface.dfm.calibration import DFMCalibration, _pose, _transform
from sharpa_interface.server.metadata import validate_metadata_format
from sharpa_interface.server.observation import validate_policy_observation

WIRE_CAMERAS = {"head": "ego_cam", "ego": "ego_cam", "left_wrist": "left_wrist_cam", "right_wrist": "right_wrist_cam"}


class NativeEnvironmentAdapter:
    def __init__(self, calibration_path, data_contract_path):
        self.calibration = DFMCalibration.load(calibration_path)
        payload = json.loads(Path(data_contract_path).read_text())
        geometry = payload["geometry"]
        expected = {
            "position_unit": "m",
            "joint_unit": "rad",
            "rotation": "column_rot6d",
            "eef": "hand_C_MC",
            "reference_camera": "ego",
        }
        if any(geometry.get(key) != value for key, value in expected.items()):
            raise ValueError("Native adapter requires physical ego-optical hand_C_MC training geometry")
        camera = payload["calibration"]
        self.ego_from_root = _transform(camera["camera_from_root"], "ego.camera_from_root")
        mounts = camera["hand_from_wrist_optical"]
        if len(mounts) != 2:
            raise ValueError("Native calibration requires both wrist camera mounts")
        self.hand_from_optical = tuple(_transform(mount, "hand_from_wrist_optical") for mount in mounts)

    @staticmethod
    def validate_interface(interface):
        if set(interface.camera_bindings) - set(WIRE_CAMERAS):
            raise ValueError("Native interface declares an unavailable UR camera view")
        if interface.action_selection.indices != list(range(62)):
            raise ValueError(
                "UR Actor requires both EEFs and both hands; partial native action execution is unsupported"
            )

    def metadata_format(self, interface):
        self.validate_interface(interface)
        enabled = {WIRE_CAMERAS[view] for view in interface.camera_bindings}
        return validate_metadata_format(
            {
                "schema": "sharpa_policy_metadata_format.v1",
                "format_id": "dfm_native_canonical.v1",
                "image": {
                    name: {"history_len": 0, "current": name in enabled}
                    for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
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
        )

    def capture(self, wire_state, interface):
        """Return native input state and optical-from-root camera snapshots."""
        from dfm.inference.serving.adapter import convert_values, transform_values

        self.validate_interface(interface)
        model_ego = self.calibration.convert(wire_state, to_model=True)
        root = transform_values(model_ego, np.linalg.inv(self.ego_from_root))
        snapshots = {}
        for view, camera in interface.camera_bindings.items():
            if view in ("head", "ego"):
                snapshots[camera] = self.ego_from_root.copy()
            else:
                side = 0 if view == "left_wrist" else 1
                root_from_hand = _pose(root[side * 9 : side * 9 + 9])
                snapshots[camera] = np.linalg.inv(root_from_hand @ self.hand_from_optical[side])
        selection = interface.state_selection
        state = convert_values(
            root,
            source_frame="root",
            source_camera=None,
            target_frame=selection.eef_frame,
            target_camera=selection.eef_camera,
            calibration=snapshots,
        )
        return state, snapshots

    def observation(self, value, interface):
        from dfm.inference.serving.protocol import EncodedImage

        value = validate_policy_observation(value, self.metadata_format(interface))
        measured = value["state"]["current"]
        if not measured["valid"]:
            raise ValueError("Measured state is invalid")
        wire = np.concatenate(
            (
                measured["left_wrist"]["eef"],
                measured["right_wrist"]["eef"],
                measured["hand_joint"]["left"],
                measured["hand_joint"]["right"],
            )
        )
        state, snapshot = self.capture(wire, interface)
        images = {}
        for view, camera in interface.camera_bindings.items():
            frame = value["image"][WIRE_CAMERAS[view]]["current"]
            if not frame["valid"]:
                raise ValueError(f"Invalid camera frame: {camera}")
            images[camera] = EncodedImage(data=frame["data"], codec="jpeg")
        return value, images, state, snapshot

    def action(self, prediction, captured_interface):
        """Convert the full old-contract plan using that prediction's calibration."""
        from dfm.inference.serving.adapter import convert_values, transform_values

        self.validate_interface(captured_interface)
        selection = captured_interface.action_selection
        if (prediction.eef_frame, prediction.eef_camera) != (selection.eef_frame, selection.eef_camera):
            raise ValueError("Prediction frame differs from the captured action declaration")
        root = convert_values(
            prediction.actions,
            source_frame=prediction.eef_frame,
            source_camera=prediction.eef_camera,
            target_frame="root",
            target_camera=None,
            calibration=prediction.camera_from_root,
        )
        model_ego = transform_values(root, self.ego_from_root)
        return self.calibration.convert(model_ego, to_model=False)
