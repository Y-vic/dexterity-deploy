"""Python 3.12 native InferenceClient worker; stdin/stdout carry only framed IPC."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict

from dfm.inference.serving.client import InferenceClient
from dfm.inference.serving.connection import ConnectionConfig, SSHConfig
from dfm.inference.serving.interface import PolicyInterface
from sharpa_interface.dfm.environment import NativeEnvironmentAdapter
from sharpa_interface.dfm.identity import native_identity
from sharpa_interface.dfm.ipc import read_frame, write_frame
from sharpa_interface.server.action import empty_auxiliary


class NativePolicySession:
    def __init__(self, connection, environment, *, expected_identity=None):
        self.client = InferenceClient(connection)
        self.environment = environment
        self.session_id = None
        self.last_request_id = -1
        self.pending_interface = None
        self.owns_pending_interface = False
        self.ready_identity = tuple(expected_identity) if expected_identity is not None else None

    def metadata(self):
        interface = self.client.interface
        self.environment.validate_interface(interface)
        config = self.client.config
        return {
            "schema": "sharpa_policy_server.v3",
            "policy_family": "dfm",
            "model_id": "dfm_flow",
            "checkpoint_id": config.task.run_id,
            "checkpoint_path": config.model.checkpoint_path,
            "task_id": config.task.task_id,
            "run_id": config.task.run_id,
            "prompt": config.task.prompt,
            "dataset_path": "",
            "observation_schema": "sharpa_policy_observation.v3",
            "action_schema": "sharpa_policy_action.v5",
            "policy_transport": "dfm_native",
            "metadata_origin": "native_ready",
            "eef_frame": "canonical",
            "native_interface": interface.to_dict(),
            "interface_revision": self.client.interface_revision,
            "pending_interface_update": self.pending_interface,
            "metadata_format": self.environment.metadata_format(interface),
            "calibration_id": self.environment.calibration.calibration_id,
            "execution": {**asdict(interface.execution), "rtc": 0},
            "interface": {
                "eef_contract": {
                    "schema": "sharpa62_direct_absolute_eef.v1",
                    "observation_eef_def": "absolute",
                    "action_eef_def": "absolute",
                    "eef_pose_layout": "xyz,r00,r10,r20,r01,r11,r21",
                    "server_xyz_direction_transform": "none",
                }
            },
        }

    def connect(self):
        reconnecting = self.client.socket is None
        self.client.connect()
        metadata = self.metadata()
        identity = native_identity(metadata)
        if self.ready_identity is not None and identity != self.ready_identity:
            self.client.close()
            raise ValueError("Native ready identity changed during explicit reconnect")
        self.ready_identity = identity
        if reconnecting:
            # Uncommitted wire proposals belong to the old native owner.
            self.pending_interface = None
            self.owns_pending_interface = False
            metadata["pending_interface_update"] = None
        return metadata

    def reset(self, session_id, request_id=None):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Reset requires a nonempty local session_id")
        if request_id is not None and (type(request_id) is not int or request_id < 0):
            raise ValueError("Reset request_id must be nonnegative")
        self.session_id = None
        self.connect()
        if self.owns_pending_interface:
            # A local rollout reset retires its uncommitted wire proposal.
            # Only this owner's queue is canceled; active/model-owned settings persist.
            self.client.update_interface(self.client.interface)
            self.pending_interface = None
            self.owns_pending_interface = False
        self.client.reset(self.client.config.task.prompt)
        self.session_id = session_id
        self.last_request_id = -1
        return {
            "schema": "sharpa_policy_reset.v1",
            "session_id": session_id,
            "request_id": request_id,
            "reset": True,
            "metadata_format": self.metadata()["metadata_format"],
        }

    def update_interface(self, interface):
        interface = PolicyInterface.from_dict(interface)
        self.environment.validate_interface(interface)
        result = self.client.update_interface(interface)
        self.pending_interface = (
            {"revision": result["pending_revision"], "interface": result["interface"]} if result["queued"] else None
        )
        self.owns_pending_interface = result["pending_owned"]
        return result

    def infer(self, payload):
        worker_started = time.perf_counter()
        if self.session_id is None or payload.get("session_id") != self.session_id:
            raise ValueError("Reset and use the active local session before inference")
        if type(payload.get("request_id")) is not int or payload["request_id"] <= self.last_request_id:
            raise ValueError("Local inference request_id must increase")
        if payload.get("prompt") != self.client.config.task.prompt:
            raise ValueError("Observation prompt differs from the selected real deployment task")
        captured_interface = self.client.interface
        value, images, state, snapshot = self.environment.observation(payload, captured_interface)
        started = time.perf_counter()
        try:
            prediction = self.client.infer(
                images,
                state,
                camera_from_root=snapshot,
                timestamp_s=value["timestamp_ns"] / 1e9,
                observation_id=f"{self.session_id}:{value['request_id']}",
            )
            predicted_at = time.perf_counter()
            actions = self.environment.action(prediction, captured_interface)
            # A server-originated declaration must also fit this embodiment.
            if prediction.interface_update is not None:
                self.environment.validate_interface(prediction.interface_update)
                self.pending_interface = None
                self.owns_pending_interface = False
        except Exception:
            self.session_id = None
            self.client.close()
            raise
        converted_at = time.perf_counter()
        latency = converted_at - started
        self.last_request_id = value["request_id"]
        diagnostics = dict(prediction.diagnostics)
        diagnostics.update(
            native_observation_prepare_s=started - worker_started,
            native_policy_predict_s=predicted_at - started,
            native_action_convert_s=converted_at - predicted_at,
            calibration_id=self.environment.calibration.calibration_id,
            native_session_id=self.client.session_id,
            native_episode_id=self.client.episode_id,
            native_plan_id=prediction.plan_id,
            native_observation_id=prediction.observation_id,
            native_interface_revision=prediction.interface_revision,
            camera_from_root={name: matrix.tolist() for name, matrix in prediction.camera_from_root.items()},
        )
        result = {
            "schema": "sharpa_policy_action.v5",
            "session_id": self.session_id,
            "request_id": value["request_id"],
            "action_id": prediction.plan_id,
            "revision": prediction.interface_revision,
            "timestamp_ns": time.time_ns(),
            "execution": {
                "frequency_hz": prediction.action_hz,
                "action_length": len(actions),
                "execute_start": prediction.execute_start,
                "execute_length": prediction.execute_length,
                "rtc": 0,
            },
            "action": {
                "left_wrist": {"joint": None, "eef": actions[:, :9], "eef_def": "absolute"},
                "right_wrist": {"joint": None, "eef": actions[:, 9:18], "eef_def": "absolute"},
                "hand_joint": {"left": actions[:, 18:40], "right": actions[:, 40:62]},
            },
            "auxiliary": empty_auxiliary(),
            "diagnostics": diagnostics,
            "next_metadata_format": self.metadata()["metadata_format"]
            if prediction.interface_update is not None
            else None,
        }
        diagnostics["native_worker_infer_s"] = time.perf_counter() - worker_started
        return {"action": result, "latency_s": latency}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    values = json.loads(args.config)
    ssh = (
        SSHConfig(values["ssh_host"], values["ssh_remote_host"], values["ssh_remote_port"])
        if values["ssh_host"]
        else None
    )
    session = NativePolicySession(
        ConnectionConfig(url=values["url"], timeout_s=values["timeout_s"], ssh=ssh),
        NativeEnvironmentAdapter(values["calibration_path"], values["data_contract_path"]),
        expected_identity=values.get("expected_identity"),
    )
    output = sys.stdout.buffer
    # Third-party libraries may log to stdout; keep the binary channel isolated.
    sys.stdout = sys.stderr

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        while True:
            try:
                request = read_frame(sys.stdin.buffer)
            except EOFError:
                break
            try:
                kind = request["type"]
                if kind == "connect":
                    result = session.connect()
                elif kind == "reset":
                    result = session.reset(request["session_id"], request.get("request_id"))
                elif kind == "infer":
                    result = session.infer(request["payload"])
                elif kind == "update_interface":
                    result = session.update_interface(request["interface"])
                else:
                    raise ValueError("Unknown native worker operation")
                response = {"ok": True, "result": result, "metadata": session.metadata()}
            except Exception as error:
                response = {"ok": False, "error": str(error), "error_type": type(error).__name__}
            write_frame(output, response)
    finally:
        session.client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
