"""Pure ROS-compatible calibration and native-worker lifecycle boundaries."""

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from sharpa_interface.dfm.calibration import DFMCalibration
from sharpa_interface.dfm.client import NativePolicyClient
from sharpa_interface.dfm.ipc import read_frame, write_frame
from sharpa_interface.server.client import ServerConfig


def calibration_payload():
    left = np.array([[0, -1, 0, 0.3], [1, 0, 0, -0.2], [0, 0, 1, 0.7], [0, 0, 0, 1]])
    right = left.copy()
    right[0, 3] += 0.17
    mount = np.array([[1, 0, 0, 0.01], [0, 0, -1, 0.02], [0, 1, 0, 0.03], [0, 0, 0, 1]])
    return {
        "schema": "ur_sharpa.dfm_calibration.v1",
        "calibration_id": "test-measured",
        "source": {"training_manifest": "test"},
        "model_joint_to_wire_indices": list(range(5, 22)) + list(range(5)),
        "sides": {
            side: {"model_from_canonical": transform.tolist(), "canonical_wrist_from_model_hand": mount.tolist()}
            for side, transform in (("left", left), ("right", right))
        },
    }


def physical_state():
    values = np.arange(62, dtype=np.float32) * 0.01
    values[3:9] = values[12:18] = [1, 0, 0, 0, 1, 0]
    return values


def test_per_arm_calibration_roundtrip_retains_mount_and_joint_permutation():
    calibration = DFMCalibration(calibration_payload())
    values = np.repeat(physical_state()[None], 30, axis=0)
    model = calibration.convert(values, to_model=True)
    np.testing.assert_array_equal(model[:, 18:40], values[:, 18 + calibration.indices])
    np.testing.assert_array_equal(model[:, 40:62], values[:, 40 + calibration.indices])
    assert not np.allclose(model[:, :18], values[:, :18])
    np.testing.assert_allclose(calibration.convert(model, to_model=False), values, atol=1e-7)
    broken = physical_state()
    broken[3:9] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        calibration.convert(broken, to_model=True)


def test_ipc_handles_large_fragmented_payload_without_truncating():
    incoming, outgoing = os.pipe()
    received = []
    with os.fdopen(incoming, "rb", buffering=0) as reader, os.fdopen(outgoing, "wb", buffering=0) as writer:
        thread = threading.Thread(target=lambda: received.append(read_frame(reader, timeout_s=3)))
        thread.start()
        value = {"rgb": bytes(range(256)) * 20000, "state": physical_state()}
        write_frame(writer, value)
        thread.join(timeout=5)
    assert not thread.is_alive() and received[0]["rgb"] == value["rgb"]
    np.testing.assert_array_equal(received[0]["state"], value["state"])


def test_thin_client_import_keeps_dfm_and_torch_out_of_ros_interpreter():
    import subprocess

    code = "from sharpa_interface.dfm.client import NativePolicyClient; import sys; assert 'dfm' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        check=True,
    )


def test_client_close_can_retire_pending_worker_without_operation_lock():
    class Process:
        def __init__(self):
            from types import SimpleNamespace

            self.terminated = False
            self.stdin = self.stdout = SimpleNamespace(close=lambda: None)

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    client = NativePolicyClient(
        ServerConfig(),
        python=sys.executable,
        source_root="unused",
        calibration_path="unused",
        data_contract_path="unused",
    )
    process = Process()
    client._process = process
    client._operation_lock.acquire()
    try:
        client.close()
        assert process.terminated and client._process is None
    finally:
        client._operation_lock.release()
    with pytest.raises(ConnectionError, match="shutting down"):
        client.connect()
