"""ROS-Python-compatible lifecycle around an isolated native DFM client worker."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from sharpa_interface.dfm.identity import native_identity
from sharpa_interface.dfm.ipc import read_frame, write_frame
from sharpa_interface.server.action import parse_policy_action
from sharpa_interface.server.metadata import validate_reset_response


class NativePolicyClient:
    def __init__(self, config, *, python, source_root, calibration_path, data_contract_path):
        self.config = config
        self.python = python
        self.source_root = source_root
        self.calibration_path = calibration_path
        self.data_contract_path = data_contract_path
        self._process = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self.current_metadata = None
        self._ready_identity = None

    def _require_process(self):
        with self._state_lock:
            if self._closed:
                raise ConnectionError("Native policy client is shutting down")
            if self._process is None:
                interface_root = str(Path(__file__).resolve().parents[2])
                values = {
                    "url": self.config.url,
                    "timeout_s": self.config.timeout_s,
                    "ssh_host": self.config.ssh_host,
                    "ssh_remote_host": self.config.ssh_remote_host or "127.0.0.1",
                    "ssh_remote_port": self.config.ssh_remote_port,
                    "calibration_path": self.calibration_path,
                    "data_contract_path": self.data_contract_path,
                    "expected_identity": self._ready_identity,
                }
                env = dict(os.environ)
                # Only the worker imports DFM; the ROS process retains its interpreter/dependencies.
                env["PYTHONPATH"] = os.pathsep.join((self.source_root, interface_root))
                self._process = subprocess.Popen(
                    [self.python, "-m", "sharpa_interface.dfm.worker", "--config", json.dumps(values)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    bufsize=0,
                )
            return self._process

    def _request(self, kind, *, _timings=None, **values):
        with self._operation_lock:
            started = time.perf_counter()
            process = self._require_process()
            prepared_at = time.perf_counter()
            try:
                write_frame(process.stdin, {"type": kind, **values})
                written_at = time.perf_counter()
                response = read_frame(process.stdout, timeout_s=self.config.timeout_s + 5)
                received_at = time.perf_counter()
                with self._state_lock:
                    if self._closed or self._process is not process:
                        raise ConnectionError("Native result belongs to a retired worker")
                if response.get("ok") is not True:
                    raise RuntimeError(response.get("error", "Native worker failed"))
                identity = native_identity(response["metadata"])
                if self._ready_identity is not None and identity != self._ready_identity:
                    self._retire_process(process)
                    raise ValueError("Native ready identity changed during explicit worker recovery")
                self._ready_identity = identity
                self.current_metadata = response["metadata"]
                if _timings is not None:
                    completed_at = time.perf_counter()
                    _timings.update(
                        native_worker_prepare_s=prepared_at - started,
                        native_ipc_write_s=written_at - prepared_at,
                        native_ipc_wait_s=received_at - written_at,
                        native_metadata_validate_s=completed_at - received_at,
                        native_ipc_roundtrip_s=completed_at - started,
                    )
                return response["result"]
            except (EOFError, TimeoutError, BrokenPipeError, OSError, ValueError):
                self._retire_process(process)
                raise

    def connect(self):
        result = self._request("connect")
        if self.config.configured_prompt and self.config.configured_prompt != result["prompt"]:
            self.close()
            raise ValueError("Configured prompt differs from the native task prompt")
        return result

    def reset(self, session_id, request_id=None):
        return validate_reset_response(
            self._request("reset", session_id=session_id, request_id=request_id),
            expected_session_id=session_id,
            expected_request_id=request_id,
        )

    def infer_action(self, payload, *, expected_session_id, expected_request_id):
        started = time.perf_counter()
        timings = {}
        result = self._request("infer", _timings=timings, payload=dict(payload))
        received_at = time.perf_counter()
        action = parse_policy_action(
            result["action"],
            expected_session_id=expected_session_id,
            expected_request_id=expected_request_id,
            expected_schema="sharpa_policy_action.v5",
        )
        parsed_at = time.perf_counter()
        latency = parsed_at - started
        action.diagnostics.update(timings)
        action.diagnostics["native_action_parse_s"] = parsed_at - received_at
        action.diagnostics["native_rpc_latency_s"] = result["latency_s"]
        action.diagnostics["client_request_latency_s"] = latency
        return action, latency

    def update_interface(self, interface):
        return self._request("update_interface", interface=interface)

    def close(self):
        # Shutdown must interrupt a blocked read; do not acquire the operation lock.
        with self._state_lock:
            self._closed = True
            process = self._process
        self._retire_process(process)

    def _retire_process(self, process):
        with self._state_lock:
            if self._process is process:
                self._process = None
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()
