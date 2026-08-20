from __future__ import annotations

import asyncio
import importlib.util

import pytest

import numpy as np

from sharpa_policy_v3_client.buffers import (
    CameraFrame,
    DeformationFrame,
    ObservationBuffers,
    StateFrame,
    TauFrame,
    WrenchFrame,
)
from sharpa_policy_v3_client.mock_server import (
    MockPolicyServer,
    hardware_metadata_format,
)
from sharpa_policy_v3_client.observation import ObservationBuilder
from sharpa_policy_v3_client.serialization import packb, unpackb
from sharpa_policy_v3_client.session import PolicySessionRuntime
from sharpa_policy_v3_client.transport import PolicyV3Transport


requires_aiohttp = pytest.mark.skipif(
    importlib.util.find_spec("aiohttp") is None,
    reason="aiohttp is not installed",
)


@requires_aiohttp
def test_mock_server_session_feedback_and_reset_round_trip() -> None:
    async def exercise() -> None:
        from aiohttp import web

        server = MockPolicyServer("127.0.0.1", 0)
        runner = web.AppRunner(server.create_application())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        socket = site._server.sockets[0]
        server.port = int(socket.getsockname()[1])
        transport = PolicyV3Transport(
            f"http://127.0.0.1:{server.port}",
            heartbeat_s=None,
        )
        runtime = PolicySessionRuntime(
            transport,
            ObservationBuilder(ObservationBuffers()),
            session_id="episode-001",
        )
        try:
            await runtime.start()
            first = await runtime.infer_once()
            second = await runtime.infer_once(
                {
                    "last_action_id": first.action_id,
                    "executed_steps": first.execution.execute_length,
                    "success": True,
                }
            )
            reset = await runtime.reset("episode-002")
            after_reset = await runtime.infer_once()

            assert first.request_id == 0
            assert second.request_id == 1
            assert reset["session_id"] == "episode-002"
            assert after_reset.session_id == "episode-002"
            assert after_reset.request_id == 0
        finally:
            await runtime.close()
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_hardware_history_profile_round_trip_without_hardware() -> None:
    async def exercise() -> None:
        from aiohttp import web

        buffers = ObservationBuffers()
        pose = np.asarray(
            [0.25, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=np.float32,
        )
        for index in range(9):
            timestamp_ns = 1_000_000_000 + index * 33_333_333
            if index >= 6:
                buffers.push_camera(
                    "ego_cam",
                    CameraFrame(timestamp_ns, "jpeg", b"mock-jpeg", True),
                )
            if index >= 7:
                buffers.push_state(
                    StateFrame(
                        timestamp_ns=timestamp_ns,
                        left_joint=np.zeros(6, dtype=np.float32),
                        left_eef=pose,
                        left_eef_frame="robot_base",
                        right_joint=np.zeros(6, dtype=np.float32),
                        right_eef=pose,
                        right_eef_frame="robot_base",
                        left_hand_joint=np.zeros(22, dtype=np.float32),
                        right_hand_joint=np.zeros(22, dtype=np.float32),
                    )
                )
            buffers.push_tau(
                TauFrame(
                    timestamp_ns,
                    np.zeros(22, dtype=np.float32),
                    np.zeros(22, dtype=np.float32),
                    np.ones(22, dtype=np.bool_),
                    np.ones(22, dtype=np.bool_),
                )
            )
            buffers.push_wrench(
                WrenchFrame(
                    timestamp_ns,
                    np.zeros((5, 6), dtype=np.float32),
                    np.zeros((5, 6), dtype=np.float32),
                    np.ones(5, dtype=np.bool_),
                    np.ones(5, dtype=np.bool_),
                )
            )
        buffers.push_deformation(
            DeformationFrame(
                1_300_000_000,
                np.zeros((5, 240, 240), dtype=np.uint8),
                np.zeros((5, 240, 240), dtype=np.uint8),
                np.ones(5, dtype=np.bool_),
                np.ones(5, dtype=np.bool_),
            )
        )
        server = MockPolicyServer(
            "127.0.0.1",
            0,
            metadata_format=hardware_metadata_format(),
        )
        runner = web.AppRunner(server.create_application())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        server.port = int(site._server.sockets[0].getsockname()[1])
        runtime = PolicySessionRuntime(
            PolicyV3Transport(f"http://127.0.0.1:{server.port}", heartbeat_s=None),
            ObservationBuilder(buffers),
            session_id="hardware-smoke",
        )
        try:
            metadata = await runtime.start()
            action = await runtime.infer_once()
            assert metadata["metadata_format"]["image"]["ego_cam"]["history_len"] == 2
            assert metadata["metadata_format"]["sensor"]["tau"]["history_len"] == 8
            assert metadata["metadata_format"]["sensor"]["wrench"]["history_len"] == 8
            assert action.execution.execute_length == 2
            assert action.hand_joint.shape == (4, 44)
        finally:
            await runtime.close()
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_mock_server_rejects_invalid_reset_contract() -> None:
    async def exercise() -> None:
        from aiohttp import ClientSession, web

        server = MockPolicyServer("127.0.0.1", 0)
        runner = web.AppRunner(server.create_application())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        socket = site._server.sockets[0]
        port = int(socket.getsockname()[1])
        url = f"http://127.0.0.1:{port}/reset"
        try:
            async with ClientSession() as session:
                async with session.post(
                    url,
                    data=packb({"session_id": "episode", "request_id": 0}),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    assert response.status == 415
                    assert unpackb(await response.read())["error"]["code"] == (
                        "invalid_content_type"
                    )
                async with session.post(
                    url,
                    data=packb({"session_id": "episode", "request_id": False}),
                    headers={"Content-Type": "application/msgpack"},
                ) as response:
                    assert response.status == 400
                    assert unpackb(await response.read())["error"]["code"] == (
                        "invalid_reset"
                    )
                reset_body = packb(
                    {"session_id": "episode", "request_id": 0}
                )
                async with session.post(
                    url,
                    data=reset_body,
                    headers={"Content-Type": "application/msgpack"},
                ) as response:
                    assert response.status == 200
                async with session.post(
                    url,
                    data=reset_body,
                    headers={"Content-Type": "application/msgpack"},
                ) as response:
                    assert response.status == 400
                    error = unpackb(await response.read())
                    assert "must differ" in error["error"]["message"]
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_mock_server_enforces_single_client_and_request_sequence() -> None:
    async def exercise() -> None:
        from aiohttp import ClientSession, WSServerHandshakeError, web

        server = MockPolicyServer("127.0.0.1", 0)
        runner = web.AppRunner(server.create_application())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        socket = site._server.sockets[0]
        port = int(socket.getsockname()[1])
        base_url = f"http://127.0.0.1:{port}"
        try:
            async with ClientSession() as session:
                async with session.post(
                    f"{base_url}/reset",
                    data=packb({"session_id": "episode", "request_id": 0}),
                    headers={"Content-Type": "application/msgpack"},
                ) as response:
                    assert response.status == 200

                async def connect() -> object:
                    try:
                        return await session.ws_connect(f"{base_url}/infer")
                    except WSServerHandshakeError as error:
                        return error

                connections = await asyncio.gather(connect(), connect())
                websockets = [
                    connection
                    for connection in connections
                    if not isinstance(connection, WSServerHandshakeError)
                ]
                errors = [
                    connection
                    for connection in connections
                    if isinstance(connection, WSServerHandshakeError)
                ]
                assert len(websockets) == 1
                assert len(errors) == 1
                assert errors[0].status == 409
                websocket = websockets[0]

                observation = {
                    "schema": "sharpa_policy_observation.v3",
                    "metadata_format_id": "mock-empty-v1",
                    "session_id": "episode",
                    "request_id": 0,
                }
                await websocket.send_bytes(packb(observation))
                first_action = unpackb((await websocket.receive()).data)
                assert first_action["request_id"] == 0

                await websocket.send_bytes(b"\xc1")
                decode_error = unpackb((await websocket.receive()).data)
                assert decode_error["request_id"] is None

                await websocket.send_bytes(packb(observation))
                sequence_error = unpackb((await websocket.receive()).data)
                assert sequence_error["request_id"] == 0
                assert sequence_error["error"]["code"] == "invalid_observation"
                await websocket.close()
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
