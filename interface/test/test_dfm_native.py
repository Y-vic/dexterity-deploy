"""Real native WebSocket and isolated worker tests; no weights or hardware."""

import importlib.util
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sharpa_interface.dfm.client import NativePolicyClient
from sharpa_interface.dfm.environment import NativeEnvironmentAdapter
from sharpa_interface.dfm.model_runtime import SerializedPolicy
from sharpa_interface.server.client import ServerConfig
from test_dfm_deployment import calibration_payload, physical_state

NATIVE_AVAILABLE = sys.version_info >= (3, 11) and importlib.util.find_spec("dfm") is not None
pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native DFM runs in the isolated Python >=3.11 worker")

if NATIVE_AVAILABLE:
    from sharpa_interface.dfm.server import DeploymentInferenceServer, warmup
    from sharpa_interface.dfm.worker import NativePolicySession

    from dfm.contracts import ActionPlan, PolicyCapabilities
    from dfm.inference.serving.configuration import DeploymentConfig
    from dfm.inference.serving.connection import ConnectionConfig
    from dfm.inference.serving.server import InferenceServer


class Policy:
    @property
    def capabilities(self):
        return PolicyCapabilities(
            cameras=("zed_left", "left_wrist", "right_wrist"), reference_camera="zed_left", action_hz=30, horizon=30
        )

    def __init__(self):
        self.calls, self.history = [], []
        self.resets = []
        self.fail = False
        self.operation_threads = [threading.get_ident()]

    def reset(self, instruction, *, demonstrations=None):
        self.operation_threads.append(threading.get_ident())
        self.instruction = instruction
        self.resets.append(instruction)
        self.history.clear()

    def set_seed(self, seed):
        self.operation_threads.append(threading.get_ident())
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict(self, observation):
        self.operation_threads.append(threading.get_ident())
        self.calls.append(observation)
        if self.fail:
            raise RuntimeError("synthetic model failure")
        actions = np.repeat(observation.state[None], 30, axis=0)
        actions[:, 0] += 0.005
        actions[:, 18:] += len(self.history) * 0.01
        actions[:, 18:] += self.rng.normal(0, 0.001, (30, 44)).astype(np.float32)
        self.history.append(observation.context.observation_id)
        return ActionPlan(actions, 30, 30, {"history_semantics": "generated_plans"})

    def close(self):
        self.operation_threads.append(threading.get_ident())


@pytest.fixture
def prepared(tmp_path):
    import dfm

    source_root = Path(dfm.__file__).resolve().parent.parent
    (tmp_path / "calibration.json").write_text(json.dumps(calibration_payload()))
    ego = np.eye(4)
    ego[:3, 3] = [0.1, 0.2, 0.3]
    mount = np.eye(4)
    mount[0, 3] = 0.08
    contract = {
        "geometry": {
            "position_unit": "m",
            "joint_unit": "rad",
            "rotation": "column_rot6d",
            "eef": "hand_C_MC",
            "reference_camera": "ego",
        },
        "calibration": {"camera_from_root": ego.tolist(), "hand_from_wrist_optical": [mount.tolist(), mount.tolist()]},
    }
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    environment = NativeEnvironmentAdapter(tmp_path / "calibration.json", tmp_path / "contract.json")
    config = DeploymentConfig.load(source_root / "configs/inference/dfm_sharpa62.yaml")
    config = replace(
        config,
        model=replace(config.model, checkpoint_path=str(tmp_path), device="cpu"),
        task=replace(config.task, task_id="ur_duck", run_id="immutable-run", prompt="Pick ducks."),
    )
    processor = {
        "camera_views": ["ego", "left_wrist", "right_wrist"],
        "image_sizes": {
            "ego": {"min_pixels": 57344, "max_pixels": 73728},
            "left_wrist": {"min_pixels": 24576, "max_pixels": 32768},
            "right_wrist": {"min_pixels": 24576, "max_pixels": 32768},
        },
    }
    (tmp_path / "dfm_processor.json").write_text(json.dumps(processor))
    return tmp_path, source_root, environment, config


@contextmanager
def serving(policy, config):
    service = InferenceServer(policy, config)
    with service.create_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/infer", service
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def observation(request_id=0, session_id="test-session"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), (20, 40, 60)).save(buffer, format="JPEG")
    state = physical_state()
    return {
        "schema": "sharpa_policy_observation.v3",
        "metadata_format_id": "dfm_native_canonical.v1",
        "session_id": session_id,
        "request_id": request_id,
        "timestamp_ns": 123000000,
        "prompt": "Pick ducks.",
        "image": {
            camera: {
                "history": [],
                "current": {"encoding": "jpeg", "data": buffer.getvalue(), "timestamp_ns": 123000000, "valid": True},
            }
            for camera in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
        },
        "state": {
            "history": None,
            "current": {
                "timestamp_ns": 123000000,
                "left_wrist": {"joint": None, "eef": state[:9], "eef_def": "absolute"},
                "right_wrist": {"joint": None, "eef": state[9:18], "eef_def": "absolute"},
                "hand_joint": {"left": state[18:40], "right": state[40:62]},
                "valid": True,
            },
        },
        "sensor": {name: {"history": None, "current": None} for name in ("tau", "wrench", "deformation")},
        "execution_feedback": {"last_action_id": None, "executed_steps": 0, "success": True},
    }


def test_environment_retains_per_arm_ab_and_dynamic_camera_snapshot(prepared):
    _, _, environment, config = prepared
    state, snapshot = environment.capture(physical_state(), config.interface)
    expected_ego = environment.calibration.convert(physical_state(), to_model=True)
    from dfm.inference.serving.adapter import transform_values

    np.testing.assert_allclose(
        state, transform_values(expected_ego, np.linalg.inv(environment.ego_from_root)), atol=1e-7
    )
    state_before = state.copy()
    original = snapshot["left_wrist_cam"].copy()
    moved = physical_state()
    moved[0] += 0.2
    _, next_snapshot = environment.capture(moved, config.interface)
    assert not np.allclose(original, next_snapshot["left_wrist_cam"])
    np.testing.assert_array_equal(snapshot["left_wrist_cam"], original)
    np.testing.assert_array_equal(state, state_before)


def test_environment_forwards_exact_validated_camera_bytes_without_rgb_decode(prepared, monkeypatch):
    from PIL import Image

    from dfm.inference.serving.protocol import (
        EncodedImage,
        message,
        pack_message,
        unpack_message,
    )

    _, _, environment, config = prepared
    payload = observation()

    def fail_decode(*args, **kwargs):
        raise AssertionError("Camera JPEG RGB must decode only after native WebSocket transport")

    with monkeypatch.context() as patch:
        patch.setattr(Image.Image, "load", fail_decode)
        _, images, state, snapshot = environment.observation(payload, config.interface)
        for camera, image in images.items():
            assert isinstance(image, EncodedImage)
            assert image.data == payload["image"][camera]["current"]["data"]
        encoded = pack_message(message("test", images=images))
    decoded = unpack_message(encoded)["images"]
    for camera in images:
        with Image.open(io.BytesIO(payload["image"][camera]["current"]["data"])) as prior:
            np.testing.assert_array_equal(np.asarray(decoded[camera]), np.asarray(prior.convert("RGB")))
    expected_state, expected_snapshot = environment.capture(physical_state(), config.interface)
    np.testing.assert_array_equal(state, expected_state)
    for camera in snapshot:
        np.testing.assert_array_equal(snapshot[camera], expected_snapshot[camera])


@pytest.mark.parametrize("invalid", ["bad_header", "codec_mismatch", "pixel_limit"])
def test_environment_rejects_invalid_encoded_camera_contracts(prepared, monkeypatch, invalid):
    from PIL import Image

    from dfm.inference.serving import protocol

    _, _, environment, config = prepared
    payload = observation()
    if invalid == "bad_header":
        payload["image"]["ego_cam"]["current"]["data"] = b"not a jpeg"
    elif invalid == "codec_mismatch":
        buffer = io.BytesIO()
        Image.new("RGB", (12, 8)).save(buffer, format="PNG")
        payload["image"]["ego_cam"]["current"]["data"] = buffer.getvalue()
    else:
        monkeypatch.setattr(protocol, "MAX_IMAGE_PIXELS", 95)
    with pytest.raises(ValueError, match="image|codec|pixel"):
        environment.observation(payload, config.interface)


def test_real_native_worker_full_chunk_nonzero_slice_and_interface_boundary(prepared):
    temporary, source_root, environment, config = prepared
    policy = Policy()
    with serving(policy, config) as (url, service):
        client = NativePolicyClient(
            ServerConfig(url=url, timeout_s=5),
            python=sys.executable,
            source_root=str(source_root),
            calibration_path=str(temporary / "calibration.json"),
            data_contract_path=str(temporary / "contract.json"),
        )
        try:
            ready = client.connect()
            assert ready["model_id"] == "dfm_flow" and ready["run_id"] == "immutable-run"
            client.reset("test-session", request_id=0)
            first, latency = client.infer_action(
                observation(), expected_session_id="test-session", expected_request_id=0
            )
            assert first.left_wrist.eef.shape == (30, 9) and first.execution.execute_length == 30
            assert latency >= first.diagnostics["native_rpc_latency_s"]
            assert latency == first.diagnostics["client_request_latency_s"]
            diagnostics = first.diagnostics
            assert diagnostics["native_worker_infer_s"] >= diagnostics["native_rpc_latency_s"]
            assert diagnostics["native_policy_predict_s"] >= diagnostics["client_websocket_roundtrip_s"]
            assert diagnostics["native_ipc_wait_s"] >= diagnostics["native_worker_infer_s"]
            assert diagnostics["native_ipc_roundtrip_s"] >= diagnostics["native_ipc_wait_s"]
            for phase in ("observation_prepare", "action_convert", "action_parse", "ipc_write"):
                assert diagnostics[f"native_{phase}_s"] >= 0
            from PIL import Image

            with Image.open(io.BytesIO(observation()["image"]["ego_cam"]["current"]["data"])) as original:
                np.testing.assert_array_equal(
                    np.asarray(policy.calls[0].images["ego"]), np.asarray(original.convert("RGB"))
                )
            proposal = config.interface.to_dict()
            proposal["observation"]["state"].update(eef_frame="camera", eef_camera="capture_left")
            proposal["camera_bindings"]["left_wrist"] = "capture_left"
            proposal["observation"]["image"]["capture_left"] = proposal["observation"]["image"].pop("left_wrist_cam")
            proposal["action"].update(eef_frame="root", eef_camera=None)
            proposal["execution"].update(execute_start=5, execute_length=10)
            assert client.update_interface(proposal)["queued"]
            assert client.current_metadata["interface_revision"] == service.interface_revision == 1
            second, _ = client.infer_action(observation(1), expected_session_id="test-session", expected_request_id=1)
            assert second.execution.execute_start == 0 and second.diagnostics["native_interface_revision"] == 1
            third, _ = client.infer_action(observation(2), expected_session_id="test-session", expected_request_id=2)
            assert (
                third.execution.action_length == 30
                and third.execution.execute_start == 5
                and third.execution.execute_length == 10
            )
            assert third.left_wrist.eef.shape == (30, 9) and third.executable_action()["left_wrist"]["eef"].shape == (
                10,
                9,
            )
            assert third.diagnostics["native_interface_revision"] == client.current_metadata["interface_revision"] == 2
            # EEF targets represent the same physical output across camera/root declarations.
            np.testing.assert_allclose(first.left_wrist.eef, third.left_wrist.eef, atol=2e-7)
            assert second.next_metadata_format is not None and third.next_metadata_format is None
            assert len(policy.calls) == 3
            assert tuple(policy.calls[2].images) == ("ego", "left_wrist", "right_wrist")
            assert "left_wrist_cam" in first.diagnostics["camera_from_root"]
            assert "capture_left" in third.diagnostics["camera_from_root"]
            assert "capture_left" not in first.diagnostics["camera_from_root"]
        finally:
            client.close()


def test_native_rejects_partial_actions_stale_identity_and_bad_inputs_without_predict(prepared):
    _, _, environment, config = prepared
    policy = Policy()
    with serving(policy, config) as (url, service):
        session = NativePolicySession(ConnectionConfig(url=url), environment)
        try:
            session.connect()
            session.reset("test-session")
            partial = config.interface.to_dict()
            partial["action"]["hand_joint"]["left"] = False
            with pytest.raises(ValueError, match="partial"):
                session.update_interface(partial)
            with pytest.raises(ValueError, match="active local session"):
                session.infer(observation(session_id="retired"))
            invalid = observation()
            invalid["image"]["left_wrist_cam"]["current"]["valid"] = False
            with pytest.raises(ValueError, match="Invalid camera"):
                session.infer(invalid)
            assert not policy.calls and service.interface_revision == 1
            session.infer(observation())
            with pytest.raises(ValueError, match="increase"):
                session.infer(observation())
            assert len(policy.calls) == 1
        finally:
            session.client.close()


def test_native_warmup_restores_seed_history_and_never_creates_an_owner(prepared):
    _, _, environment, config = prepared
    policy = Policy()
    service = DeploymentInferenceServer(InferenceServer(policy, config), environment)
    result = warmup(service)
    assert result["requests"] == 2 and result["image_sizes"] == {
        "ego": [352, 192],
        "left_wrist": [224, 128],
        "right_wrist": [224, 128],
    }
    assert policy.seed == 42 and not policy.history and len(policy.calls) == 2
    assert not service.health()["session_active"] and service.interface_revision == 1


def test_load_warmup_and_websocket_operations_share_one_model_worker_and_seeded_history(prepared):
    temporary, source_root, environment, config = prepared
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-model") as executor:
        policy = executor.submit(Policy).result()
        serialized = SerializedPolicy(policy, executor)
        service = DeploymentInferenceServer(InferenceServer(serialized, config), environment)
        _, images, state, snapshot = environment.observation(observation(), config.interface)
        from dfm.inference.serving.protocol import message, pack_message, unpack_message

        images = unpack_message(pack_message(message("test", images=images)))["images"]
        native = service.adapter.observation(
            {
                "format_id": config.interface.observation.format_id,
                "images": images,
                "state": state,
                "observation_id": "baseline",
            },
            snapshot,
        )
        serialized.set_seed(42)
        serialized.reset(config.task.prompt)
        baseline = [serialized.predict(native).actions for _ in range(2)]
        service.warmup_result = warmup(service)
        assert not policy.history
        with serving(serialized, config) as (url, _):
            client = NativePolicyClient(
                ServerConfig(url=url, timeout_s=5),
                python=sys.executable,
                source_root=str(source_root),
                calibration_path=str(temporary / "calibration.json"),
                data_contract_path=str(temporary / "contract.json"),
            )
            try:
                client.connect()
                client.reset("test-session")
                for index in range(2):
                    action, _ = client.infer_action(
                        observation(index), expected_session_id="test-session", expected_request_id=index
                    )
                    wire = np.concatenate(
                        (
                            action.left_wrist.eef,
                            action.right_wrist.eef,
                            action.hand_joint.left,
                            action.hand_joint.right,
                        ),
                        axis=-1,
                    )
                    np.testing.assert_allclose(
                        wire, environment.calibration.convert(baseline[index], to_model=False), atol=2e-7
                    )
            finally:
                client.close()
        serialized.close()
        assert len(set(policy.operation_threads)) == 1
        assert policy.operation_threads[0] != threading.get_ident()


@pytest.mark.parametrize("failure", ["model", "ipc"])
def test_explicit_reset_recovers_failed_native_transport_without_inference_retry(prepared, failure):
    temporary, source_root, environment, config = prepared
    policy = Policy()
    with serving(policy, config) as (url, service):
        client = NativePolicyClient(
            ServerConfig(url=url, timeout_s=5),
            python=sys.executable,
            source_root=str(source_root),
            calibration_path=str(temporary / "calibration.json"),
            data_contract_path=str(temporary / "contract.json"),
        )
        try:
            client.connect()
            client.reset("test-session")
            process = client._process
            if failure == "model":
                policy.fail = True
            else:
                process.kill()
                process.wait(timeout=3)
            with pytest.raises((RuntimeError, EOFError, BrokenPipeError)):
                client.infer_action(observation(), expected_session_id="test-session", expected_request_id=0)
            calls = len(policy.calls)
            assert calls == (1 if failure == "model" else 0) and not client._closed
            policy.fail = False
            deadline = time.monotonic() + 2
            while service._owner.locked() and time.monotonic() < deadline:
                time.sleep(0.001)
            client.reset("recovered-session")
            action, _ = client.infer_action(
                observation(session_id="recovered-session"),
                expected_session_id="recovered-session",
                expected_request_id=0,
            )
            assert action.session_id == "recovered-session" and len(policy.calls) == calls + 1
            if failure == "ipc":
                assert client._process is not process
        finally:
            client.close()
        with pytest.raises(ConnectionError, match="shutting down"):
            client.reset("retired-session")


def test_respawn_revalidates_native_ready_before_resetting_another_checkpoint(prepared):
    temporary, source_root, environment, config = prepared
    policy = Policy()
    with serving(policy, config) as (url, service):
        client = NativePolicyClient(
            ServerConfig(url=url, timeout_s=5),
            python=sys.executable,
            source_root=str(source_root),
            calibration_path=str(temporary / "calibration.json"),
            data_contract_path=str(temporary / "contract.json"),
        )
        try:
            client.connect()
            client.reset("test-session")
            client._process.kill()
            client._process.wait(timeout=3)
            with pytest.raises((EOFError, BrokenPipeError)):
                client.infer_action(observation(), expected_session_id="test-session", expected_request_id=0)
            deadline = time.monotonic() + 2
            while service._owner.locked() and time.monotonic() < deadline:
                time.sleep(0.001)
            reset_count = len(policy.resets)
            service.config = replace(
                service.config, model=replace(service.config.model, checkpoint_path="another-checkpoint")
            )
            with pytest.raises(RuntimeError, match="identity changed"):
                client.reset("recovered-session")
            assert len(policy.resets) == reset_count and not policy.calls
        finally:
            client.close()


def test_explicit_rollout_reset_discards_its_uncommitted_wire_proposal(prepared):
    _, _, environment, config = prepared
    policy = Policy()
    with serving(policy, config) as (url, service):
        session = NativePolicySession(ConnectionConfig(url=url), environment)
        try:
            session.connect()
            session.reset("first-rollout")
            old_native_owner = session.client.session_id
            proposal = config.interface.to_dict()
            proposal["execution"]["execute_length"] = 10
            session.update_interface(proposal)
            session.reset("next-rollout")
            assert session.client.session_id == old_native_owner
            assert service._pending_interface is None and session.pending_interface is None
            assert service.interface_revision == 1 and not policy.calls
            result = session.infer(observation(session_id="next-rollout"))
            assert result["action"]["execution"]["execute_length"] == 30
            assert result["action"]["next_metadata_format"] is None
        finally:
            session.client.close()


def test_rollout_reset_preserves_external_model_owned_pending_declaration(prepared):
    _, _, environment, config = prepared
    policy = Policy()
    with serving(policy, config) as (url, service):
        session = NativePolicySession(ConnectionConfig(url=url), environment)
        try:
            session.connect()
            session.reset("first-rollout")
            proposal = config.interface.to_dict()
            proposal["execution"]["execute_length"] = 10
            service.update_interface(proposal)
            assert not session.update_interface(proposal)["pending_owned"]
            session.reset("next-rollout")
            assert service._pending_interface is not None and service.interface_revision == 1
            result = session.infer(observation(session_id="next-rollout"))
            assert result["action"]["execution"]["execute_length"] == 30
            assert result["action"]["next_metadata_format"] is not None
            assert session.client.interface_revision == service.interface_revision == 2
        finally:
            session.client.close()
