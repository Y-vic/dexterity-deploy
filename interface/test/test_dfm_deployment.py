"""Physical ABI conversion and real deployment transport without model or hardware."""

import asyncio
import io
import json
import sys
import threading
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sharpa_interface.dfm.calibration import DFMCalibration
from sharpa_interface.dfm.policy import DFMDeploymentPolicy
from sharpa_interface.dfm.server import create_app
from sharpa_interface.server.action import PolicyServerError, parse_policy_action
from sharpa_interface.server.client import ServerClient, ServerConfig
from sharpa_interface.server.serialization import packb, unpackb


@pytest.fixture
def calibration():
    rotation = np.array([[0, -1, 0, 0.3], [1, 0, 0, -0.2], [0, 0, 1, 0.7], [0, 0, 0, 1]])
    mount = np.array([[1, 0, 0, 0.01], [0, 0, -1, 0.02], [0, 1, 0, 0.03], [0, 0, 0, 1]])
    return DFMCalibration(
        {
            "schema": "ur_sharpa.dfm_calibration.v1",
            "calibration_id": "test-measured",
            "source": {"training_manifest_sha256": "test"},
            "model_joint_to_wire_indices": list(range(5, 13))
            + list(range(17, 22))
            + list(range(13, 17))
            + list(range(5)),
            "sides": {
                side: {"model_from_canonical": rotation, "canonical_wrist_from_model_hand": mount}
                for side in ("left", "right")
            },
        }
    )


def state_values():
    values = np.arange(62, dtype=np.float32) * 0.01
    values[3:9] = values[12:18] = [1, 0, 0, 0, 1, 0]
    return values


def test_calibration_roundtrip_and_joint_order(calibration):
    state = state_values()
    chunk = np.repeat(state[None], 30, axis=0)
    chunk[:, :3] += np.arange(30)[:, None] * 0.02
    model = calibration.convert(chunk, to_model=True)
    np.testing.assert_array_equal(model[:, 18:40], chunk[:, 18 + calibration.indices])
    np.testing.assert_array_equal(model[:, 40:62], chunk[:, 40 + calibration.indices])
    assert not np.allclose(model[:, :18], chunk[:, :18])
    np.testing.assert_allclose(calibration.convert(model, to_model=False), chunk, atol=1e-7)


def test_calibration_rejects_degenerate_pose(calibration):
    state = state_values()
    state[3:9] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        calibration.convert(state, to_model=True)


def observation():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (100, 20, 30)).save(buffer, format="JPEG")
    state = state_values()

    def wrist(start):
        return {"joint": None, "eef": state[start : start + 9], "eef_def": "absolute"}

    return {
        "schema": "sharpa_policy_observation.v3",
        "metadata_format_id": "dfm_sim_real_canonical.v1",
        "session_id": "session-test",
        "request_id": 0,
        "timestamp_ns": 123_000_000,
        "prompt": "Pick ducks.",
        "image": {
            name: {
                "history": [],
                "current": {"encoding": "jpeg", "data": buffer.getvalue(), "timestamp_ns": 123_000_000, "valid": True},
            }
            for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
        },
        "state": {
            "history": None,
            "current": {
                "timestamp_ns": 123_000_000,
                "left_wrist": wrist(0),
                "right_wrist": wrist(9),
                "hand_joint": {"left": state[18:40], "right": state[40:62]},
                "valid": True,
            },
        },
        "sensor": {name: {"history": None, "current": None} for name in ("tau", "wrench", "deformation")},
        "execution_feedback": {"last_action_id": None, "executed_steps": 0, "success": True},
    }


@pytest.fixture
def adapter(calibration, monkeypatch, tmp_path):
    # The deployment workstation deliberately has no DFM/Torch installation.
    contracts = ModuleType("dfm.contracts")
    contracts.Observation = lambda image, state, **kw: SimpleNamespace(image=image, state=state, **kw)
    contracts.ObservationContext = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "dfm", ModuleType("dfm"))
    monkeypatch.setitem(sys.modules, "dfm.contracts", contracts)

    class Policy:
        capabilities = SimpleNamespace(
            state_dim=62,
            action_dim=62,
            horizon=30,
            action_hz=30,
            cameras=("zed_left", "left_wrist", "right_wrist"),
            reference_camera="zed_left",
            action_format="sharpa62_camera_absolute",
        )
        predictions = 0
        fail = False

        def __init__(self):
            self.operation_threads = []
            self.history = []

        def set_seed(self, seed):
            self.operation_threads.append(threading.get_ident())
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        def reset(self, prompt):
            self.operation_threads.append(threading.get_ident())
            self.prompt = prompt
            self.history.clear()

        def predict(self, value):
            self.operation_threads.append(threading.get_ident())
            self.predictions += 1
            self.observation = value
            if self.fail:
                raise RuntimeError("model failed")
            self.history.append(value.context.observation_id)
            return SimpleNamespace(
                actions=np.repeat(value.state[None], 30, axis=0),
                action_hz=30,
                diagnostics_dict=lambda: {"history_semantics": "generated_plans"},
            )

    (tmp_path / "dfm_processor.json").write_text(
        json.dumps(
            {
                "camera_views": ["ego", "left_wrist", "right_wrist"],
                "image_sizes": {
                    "ego": {"min_pixels": 57344, "max_pixels": 73728},
                    "left_wrist": {"min_pixels": 24576, "max_pixels": 32768},
                    "right_wrist": {"min_pixels": 24576, "max_pixels": 32768},
                },
            }
        )
    )
    return DFMDeploymentPolicy(
        Policy(),
        calibration,
        checkpoint=tmp_path,
        task={"id": "ur_duck", "prompt": "Pick ducks."},
        run={"run_id": "immutable-training-run", "model": "dfm_flow"},
        port=5500,
        execute_steps=15,
    )


def test_policy_returns_wire_physical_actions_and_execution_prefix(adapter):
    adapter.reset({"session_id": "session-test"})
    result = adapter.infer(observation())
    action = parse_policy_action(result, expected_session_id="session-test", expected_request_id=0)
    assert action.execution.execute_length == 15
    assert action.execution.action_length == 30
    assert action.execution.frequency_hz == 30
    np.testing.assert_allclose(result["action"]["left_wrist"]["eef"][0], state_values()[:9], atol=1e-7)
    np.testing.assert_array_equal(result["action"]["hand_joint"]["left"][0], state_values()[18:40])
    assert tuple(adapter.policy.observation.images) == ("ego", "left_wrist", "right_wrist")
    assert adapter.policy.observation.context.timestamp_s == 0.123
    assert result["diagnostics"]["calibration_id"] == "test-measured"


def test_policy_requires_valid_inputs_and_reset_after_failure(adapter):
    adapter.reset({"session_id": "session-test"})
    value = observation()
    value["image"]["left_wrist_cam"]["current"]["valid"] = False
    with pytest.raises(ValueError, match="Invalid camera"):
        adapter.infer(value)
    value = observation()
    value["state"]["current"]["valid"] = False
    with pytest.raises(ValueError, match="state is invalid"):
        adapter.infer(value)
    adapter.policy.fail = True
    with pytest.raises(RuntimeError, match="model failed"):
        adapter.infer(observation())
    adapter.policy.fail = False
    with pytest.raises(ValueError, match="Reset"):
        adapter.infer(observation())


def test_real_http_websocket_reset_infer_duplicate_and_ownership(adapter):
    from aiohttp import ClientSession, WSServerHandshakeError
    from aiohttp.test_utils import TestServer

    async def run():
        server = TestServer(create_app(adapter, warmup=False))
        await server.start_server()
        url = str(server.make_url("/infer")).replace("http:", "ws:")
        client = ServerClient(ServerConfig(url=url))
        try:
            metadata = await asyncio.to_thread(client.connect)
            assert metadata["run_id"] == "immutable-training-run"
            assert metadata["model_id"] == "dfm_flow"
            await asyncio.to_thread(client.reset, "session-test")
            first = await asyncio.to_thread(client.infer, observation())
            second = await asyncio.to_thread(client.infer, observation())
            assert first.payload["action_id"] == second.payload["action_id"]
            assert adapter.policy.predictions == 1
            await asyncio.to_thread(client.reset, "next-rollout")
            next_observation = observation()
            next_observation["session_id"] = "next-rollout"
            await asyncio.to_thread(client.infer, next_observation)
            assert adapter.policy.predictions == 2
            async with ClientSession() as http:
                response = await http.get(server.make_url("/healthz"))
                assert (await response.json())["ready"] is True
                with pytest.raises(WSServerHandshakeError) as exc:
                    await http.ws_connect(server.make_url("/infer"))
                assert exc.value.status == 409
                rejected = await http.post(server.make_url("/reset"), data=packb({"session_id": "other-session"}))
                assert rejected.status == 409
                assert "owns" in unpackb(await rejected.read())["error"]["message"]
            changed = deepcopy(next_observation)
            changed["timestamp_ns"] += 1
            error = await asyncio.to_thread(client.infer, changed)
            assert "Request IDs" in error.payload["error"]["message"]
            with pytest.raises(PolicyServerError, match="Reset"):
                await asyncio.to_thread(
                    client.infer_action, changed, expected_session_id="session-test", expected_request_id=0
                )
            await asyncio.to_thread(client.reset, "failure-rollout")
            successful = observation()
            successful["session_id"] = "failure-rollout"
            await asyncio.to_thread(client.infer, successful)
            failed = deepcopy(successful)
            failed["request_id"] = 1
            adapter.policy.fail = True
            error = await asyncio.to_thread(client.infer, failed)
            assert error.payload["error"]["message"] == "model failed"
            adapter.policy.fail = False
            replayed = await asyncio.to_thread(client.infer, successful)
            assert "Reset" in replayed.payload["error"]["message"]
            assert adapter.policy.predictions == 4
        finally:
            await asyncio.to_thread(client.close)
            await server.close()

    asyncio.run(run())


def test_disconnect_invalidates_session_before_reconnect(adapter):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    async def run():
        server = TestServer(create_app(adapter, warmup=False))
        await server.start_server()
        try:
            async with ClientSession() as client:
                token = "a" * 32
                async with client.ws_connect(server.make_url("/infer?owner_token=" + token)):
                    reply = await client.post(
                        server.make_url("/reset"), data=packb({"session_id": "session-test", "owner_token": token})
                    )
                    assert reply.status == 200
                for _ in range(100):
                    health = await client.get(server.make_url("/healthz"))
                    if not (await health.json())["session_active"]:
                        break
                    await asyncio.sleep(0.001)
                assert adapter.session is None
                async with client.ws_connect(server.make_url("/infer?owner_token=" + "b" * 32)) as socket:
                    await socket.send_bytes(packb(observation()))
                    result = unpackb((await socket.receive()).data)
                    assert "Reset" in result["error"]["message"]
                assert adapter.policy.predictions == 0
        finally:
            await server.close()

    asyncio.run(run())


def test_warmup_restores_seed_and_generated_history(adapter):
    def predict(value):
        actions = np.repeat(value.state[None], 30, axis=0)
        actions[:, 18:] += adapter.policy.rng.normal(0, 0.01, (30, 44)).astype(np.float32)
        actions[:, 18:] += len(adapter.policy.history) * 0.1
        adapter.policy.history.append(value.context.observation_id)
        return SimpleNamespace(actions=actions, action_hz=30, diagnostics_dict=lambda: {})

    adapter.policy.predict = predict
    first = observation()
    second = deepcopy(first)
    second["request_id"] = 1
    second["timestamp_ns"] += 1_000_000_000
    adapter.reset({"session_id": "session-test"})
    expected = [adapter.infer(first), adapter.infer(second)]
    adapter.invalidate()
    prepared = adapter.warmup()
    assert prepared["requests"] == 2
    assert prepared["image_sizes"] == {"ego": [352, 192], "left_wrist": [224, 128], "right_wrist": [224, 128]}
    assert adapter.session is None
    assert adapter.policy.seed == 42
    assert adapter.policy.history == []
    adapter.reset({"session_id": "session-test"})
    actual = [adapter.infer(first), adapter.infer(second)]
    for before, after in zip(expected, actual):
        for side in ("left", "right"):
            np.testing.assert_array_equal(before["action"]["hand_joint"][side], after["action"]["hand_joint"][side])
    assert adapter.policy.history == ["session-test:0", "session-test:1"]


def test_warmup_never_resets_an_owned_episode(adapter):
    adapter.reset({"session_id": "session-test"})
    with pytest.raises(RuntimeError, match="unowned"):
        adapter.warmup()
    assert adapter.session == "session-test"
    assert adapter.policy.predictions == 0


def test_startup_warmup_and_real_requests_share_one_worker(adapter, capsys):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    async def run():
        loop_thread = threading.get_ident()
        server = TestServer(create_app(adapter))
        await server.start_server()
        url = str(server.make_url("/infer")).replace("http:", "ws:")
        client = ServerClient(ServerConfig(url=url))
        try:
            assert adapter.session is None and adapter.policy.history == []
            assert adapter.policy.predictions == 2
            async with ClientSession() as http:
                response = await http.get(server.make_url("/healthz"))
                health = await response.json()
                assert health["ready"] is True
                assert health["warmup"]["requests"] == 2
                assert health["session_active"] is False
            await asyncio.to_thread(client.connect)
            await asyncio.to_thread(client.reset, "session-test")
            await asyncio.to_thread(client.infer, observation())
            assert adapter.policy.history == ["session-test:0"]
            assert len(set(adapter.policy.operation_threads)) == 1
            assert adapter.policy.operation_threads[0] != loop_thread
        finally:
            await asyncio.to_thread(client.close)
            await server.close()

    asyncio.run(run())
    output = capsys.readouterr().out
    assert output.index("warmup complete") < output.index("DFM deployment ready")


def test_failed_warmup_clears_episode_and_never_advertises_ready(adapter, capsys):
    from aiohttp.test_utils import TestServer

    adapter.policy.fail = True

    async def run():
        server = TestServer(create_app(adapter))
        try:
            with pytest.raises(RuntimeError, match="model failed"):
                await server.start_server()
        finally:
            await server.close()

    asyncio.run(run())
    assert adapter.session is None
    assert adapter.policy.history == []
    assert adapter.policy.seed == 42
    assert "DFM deployment ready" not in capsys.readouterr().out
