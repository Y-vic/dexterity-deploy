from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from sharpa_policy_v3_client.buffers import CameraFrame, ObservationBuffers
from sharpa_policy_v3_client.mock_server import MockPolicyServer, empty_metadata_format
from sharpa_policy_v3_client.observation import (
    ObservationBuilder,
    ObservationCapacityError,
)
from sharpa_policy_v3_client.session import PolicySessionRuntime
from sharpa_policy_v3_client.transport import PolicyV3Transport


class FakePolicyTransport(PolicyV3Transport):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:5500")
        self.mock = MockPolicyServer("127.0.0.1", 5500)
        self.is_connected = False
        self.metadata_calls = 0
        self.reset_calls: list[tuple[str, int]] = []
        self.observations: list[dict] = []
        self.next_metadata_format: dict | None = None
        self.health_calls = 0

    @property
    def connected(self) -> bool:
        return self.is_connected

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def metadata(self) -> dict:
        self.metadata_calls += 1
        result = self.mock.metadata()
        self.server_metadata = result
        self.active_metadata_format = deepcopy(result["metadata_format"])
        return result

    async def reset(self, session_id: str, request_id: int = 0) -> dict:
        self.reset_calls.append((session_id, request_id))
        self.active_metadata_format = deepcopy(self.mock.metadata_format)
        return {
            "schema": "sharpa_policy_reset.v1",
            "session_id": session_id,
            "request_id": request_id,
            "reset": True,
            "metadata_format": deepcopy(self.active_metadata_format),
        }

    async def connect(self) -> None:
        self.is_connected = True

    async def disconnect_infer(self) -> None:
        self.is_connected = False

    async def infer(self, observation: dict) -> dict:
        self.observations.append(observation)
        action = self.mock._action_for(observation)
        action["next_metadata_format"] = deepcopy(self.next_metadata_format)
        return action

    async def close(self) -> None:
        self.is_connected = False


class BlockingPolicyTransport(FakePolicyTransport):
    def __init__(self) -> None:
        super().__init__()
        self.infer_started = asyncio.Event()
        self.release_infer = asyncio.Event()

    async def infer(self, observation: dict) -> dict:
        self.infer_started.set()
        await self.release_infer.wait()
        return await super().infer(observation)


class BlockingStartPolicyTransport(FakePolicyTransport):
    def __init__(self) -> None:
        super().__init__()
        self.metadata_started = asyncio.Event()
        self.release_metadata = asyncio.Event()
        self.close_calls = 0

    async def metadata(self) -> dict:
        self.metadata_calls += 1
        self.metadata_started.set()
        await self.release_metadata.wait()
        result = self.mock.metadata()
        self.server_metadata = result
        self.active_metadata_format = deepcopy(result["metadata_format"])
        return result

    async def close(self) -> None:
        self.close_calls += 1
        await super().close()


def make_runtime(
    transport: FakePolicyTransport,
    buffers: ObservationBuffers | None = None,
) -> PolicySessionRuntime:
    return PolicySessionRuntime(
        transport,
        ObservationBuilder(buffers or ObservationBuffers()),
        session_id="episode-001",
        clock_ns=lambda: 123456789,
    )


def test_session_starts_at_request_zero_and_forwards_feedback() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = make_runtime(transport)
        await runtime.start()

        first = await runtime.infer_once()
        second = await runtime.infer_once(
            {
                "last_action_id": first.action_id,
                "executed_steps": first.execution.execute_length,
                "success": True,
            }
        )

        assert first.request_id == 0
        assert second.request_id == 1
        assert runtime.request_id == 2
        assert transport.reset_calls == [("episode-001", 0)]
        assert transport.health_calls == 1
        assert transport.observations[0]["timestamp_ns"] == 123456789
        assert transport.observations[1]["execution_feedback"] == {
            "last_action_id": first.action_id,
            "executed_steps": 2,
            "success": True,
        }

    asyncio.run(exercise())


def test_separated_runtime_accepts_observation_built_by_state_node() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = PolicySessionRuntime(
            transport,
            session_id="episode-001",
            clock_ns=lambda: 123456789,
        )
        await runtime.start()
        observation = ObservationBuilder(ObservationBuffers()).build(
            runtime.active_metadata_format,
            session_id="episode-001",
            request_id=0,
            timestamp_ns=123456789,
        )

        action = await runtime.infer_observation(observation)

        assert action.request_id == 0
        assert runtime.request_id == 1
        assert transport.observations == [observation]

    asyncio.run(exercise())


def test_separated_runtime_rejects_stale_metadata_format_id() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = PolicySessionRuntime(transport, session_id="episode-001")
        await runtime.start()
        observation = ObservationBuilder(ObservationBuffers()).build(
            runtime.active_metadata_format,
            session_id="episode-001",
            request_id=0,
            timestamp_ns=123456789,
        )
        observation["metadata_format_id"] = "stale-format"

        with pytest.raises(ValueError, match="metadata_format_id"):
            await runtime.infer_observation(observation)

        assert runtime.request_id == 0
        assert transport.observations == []

    asyncio.run(exercise())


def test_reset_rejects_reusing_the_active_session() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = make_runtime(transport)
        await runtime.start()

        with pytest.raises(ValueError, match="must differ"):
            await runtime.reset("episode-001")

        assert runtime.session_id == "episode-001"
        assert runtime.started
        assert transport.reset_calls == [("episode-001", 0)]

    asyncio.run(exercise())


def test_next_metadata_format_is_adopted_only_after_action_validation() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = make_runtime(transport)
        await runtime.start()
        next_format = empty_metadata_format()
        next_format["format_id"] = "mock-empty-v2"
        transport.next_metadata_format = next_format

        action = await runtime.infer_once()

        assert action.next_metadata_format["format_id"] == "mock-empty-v2"
        assert runtime.active_metadata_format["format_id"] == "mock-empty-v2"

    asyncio.run(exercise())


def test_next_metadata_format_configures_the_following_observation() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        buffers = ObservationBuffers(camera_frame_capacity=2)
        buffers.push_camera(
            "ego_cam",
            CameraFrame(
                timestamp_ns=100,
                encoding="jpeg",
                data=b"frame-100",
                valid=True,
            ),
        )
        runtime = make_runtime(transport, buffers)
        await runtime.start()
        next_format = empty_metadata_format()
        next_format["format_id"] = "with-current-ego"
        next_format["image"]["ego_cam"] = {
            "history_len": 0,
            "current": True,
        }
        transport.next_metadata_format = next_format

        await runtime.infer_once()
        transport.mock.metadata_format = deepcopy(next_format)
        await runtime.infer_once()

        first_observation, second_observation = transport.observations
        assert first_observation["metadata_format_id"] == "mock-empty-v1"
        assert first_observation["image"]["ego_cam"]["current"] is None
        assert second_observation["metadata_format_id"] == "with-current-ego"
        assert second_observation["image"]["ego_cam"]["current"] == {
            "timestamp_ns": 100,
            "encoding": "jpeg",
            "data": b"frame-100",
            "valid": True,
        }

    asyncio.run(exercise())


def test_rejects_unsatisfiable_next_metadata_format_before_adoption() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = make_runtime(transport)
        await runtime.start()
        impossible = empty_metadata_format()
        impossible["format_id"] = "too-much-history"
        impossible["image"]["ego_cam"] = {
            "history_len": 129,
            "current": False,
        }
        transport.next_metadata_format = impossible

        with pytest.raises(ObservationCapacityError, match="image.ego_cam"):
            await runtime.infer_once()

        assert runtime.active_metadata_format["format_id"] == "mock-empty-v1"
        assert runtime.request_id == 0

    asyncio.run(exercise())


def test_reconnect_fetches_metadata_and_starts_a_new_session() -> None:
    async def exercise() -> None:
        transport = FakePolicyTransport()
        runtime = make_runtime(transport)
        await runtime.start()
        original_session = runtime.session_id
        await runtime.infer_once()

        await runtime.reconnect()

        assert transport.metadata_calls == 2
        assert runtime.session_id != original_session
        assert runtime.request_id == 0
        assert runtime.last_action is None
        assert transport.reset_calls[-1] == (runtime.session_id, 0)
        assert runtime.started

    asyncio.run(exercise())


def test_reset_waits_for_inflight_inference_to_be_cancelled() -> None:
    async def exercise() -> None:
        transport = BlockingPolicyTransport()
        runtime = make_runtime(transport)
        await runtime.start()
        inference = asyncio.create_task(runtime.infer_once())
        await transport.infer_started.wait()
        reset = asyncio.create_task(runtime.reset("episode-002"))
        await asyncio.sleep(0)

        assert not reset.done()
        inference.cancel()
        try:
            await inference
        except asyncio.CancelledError:
            pass
        result = await reset

        assert result["session_id"] == "episode-002"
        assert runtime.request_id == 0
        assert runtime.started

    asyncio.run(exercise())


def test_reset_waits_for_start_to_finish() -> None:
    async def exercise() -> None:
        transport = BlockingStartPolicyTransport()
        runtime = make_runtime(transport)
        start = asyncio.create_task(runtime.start())
        await transport.metadata_started.wait()

        reset = asyncio.create_task(runtime.reset("episode-002"))
        await asyncio.sleep(0)

        assert not reset.done()
        assert transport.metadata_calls == 1
        assert transport.reset_calls == []

        transport.release_metadata.set()
        await start
        result = await reset

        assert result["session_id"] == "episode-002"
        assert transport.reset_calls == [
            ("episode-001", 0),
            ("episode-002", 0),
        ]
        assert runtime.started

    asyncio.run(exercise())


def test_close_waits_for_start_cancellation() -> None:
    async def exercise() -> None:
        transport = BlockingStartPolicyTransport()
        runtime = make_runtime(transport)
        start = asyncio.create_task(runtime.start())
        await transport.metadata_started.wait()

        close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)

        assert not close.done()
        assert transport.close_calls == 0

        start.cancel()
        try:
            await start
        except asyncio.CancelledError:
            pass
        await close

        assert transport.close_calls == 1
        assert not runtime.started

    asyncio.run(exercise())
