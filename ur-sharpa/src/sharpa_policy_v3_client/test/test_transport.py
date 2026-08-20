from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pytest

from sharpa_policy_v3_client.metadata import (
    MetadataValidationError,
    validate_metadata_format,
    validate_reset_response,
    validate_server_metadata,
)
from sharpa_policy_v3_client.serialization import MAX_MESSAGE_SIZE, packb, unpackb
import sharpa_policy_v3_client.transport as transport_module
from sharpa_policy_v3_client.transport import (
    PolicyConcurrencyError,
    PolicyDependencyError,
    PolicyMessageTooLargeError,
    PolicyProtocolError,
    PolicyServerError,
    PolicyV3Transport,
    derive_policy_urls,
)


HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None
requires_aiohttp = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp is not installed")


def metadata_format(format_id: str = "format-001") -> dict:
    return {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": format_id,
        "image": {
            "ego_cam": {"history_len": 2, "current": True},
            "left_wrist_cam": {"history_len": 0, "current": False},
            "right_wrist_cam": {"history_len": 0, "current": False},
        },
        "state": {
            "history_len": 3,
            "current": True,
            "left_wrist": {"joint": True, "eef": False},
            "right_wrist": {"joint": True, "eef": False},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": {"history_len": 1, "current": True},
            "wrench": {"history_len": 0, "current": True},
            "deformation": {"history_len": 0, "current": False},
        },
    }


def server_metadata(*, max_message_size: int = MAX_MESSAGE_SIZE) -> dict:
    return {
        "schema": "sharpa_policy_server.v3",
        "policy_family": "mock",
        "checkpoint_id": "checkpoint-001",
        "checkpoint_path": "/models/checkpoint-001",
        "task_id": "task-001",
        "run_id": "run-001",
        "dataset_path": "/datasets/task-001",
        "prompt": "pick up the object",
        "transport": "websocket+binary_msgpack",
        "observation_schema": "sharpa_policy_observation.v3",
        "action_schema": "sharpa_policy_action.v4",
        "host": "127.0.0.1",
        "port": 5500,
        "infer_path": "/infer",
        "health_path": "/healthz",
        "metadata_path": "/metadata",
        "reset_path": "/reset",
        "max_message_size": max_message_size,
        "metadata_format": metadata_format(),
    }


def test_metadata_format_is_strict_and_returns_independent_copy():
    source = metadata_format()
    validated = validate_metadata_format(source)

    source["image"]["ego_cam"]["history_len"] = 99
    assert validated["image"]["ego_cam"]["history_len"] == 2

    invalid = metadata_format()
    invalid["state"]["history_len"] = True
    with pytest.raises(MetadataValidationError, match="must be an integer"):
        validate_metadata_format(invalid)

    invalid = metadata_format()
    invalid["unexpected"] = None
    with pytest.raises(MetadataValidationError, match="unknown fields"):
        validate_metadata_format(invalid)


def test_server_metadata_and_reset_response_are_correlated_strictly():
    validated = validate_server_metadata(server_metadata())
    assert validated["metadata_format"]["format_id"] == "format-001"

    too_large = server_metadata(max_message_size=MAX_MESSAGE_SIZE + 1)
    with pytest.raises(MetadataValidationError, match="64 MiB"):
        validate_server_metadata(too_large)

    reset = {
        "schema": "sharpa_policy_reset.v1",
        "session_id": "episode-001",
        "request_id": 0,
        "reset": True,
        "metadata_format": metadata_format("reset-format"),
    }
    result = validate_reset_response(
        reset,
        expected_session_id="episode-001",
        expected_request_id=0,
    )
    assert result["metadata_format"]["format_id"] == "reset-format"

    with pytest.raises(MetadataValidationError, match="session_id"):
        validate_reset_response(reset, expected_session_id="other")


@pytest.mark.parametrize(
    ("source", "base", "infer"),
    [
        (
            "http://127.0.0.1:5500",
            "http://127.0.0.1:5500",
            "ws://127.0.0.1:5500/infer",
        ),
        (
            "ws://robot.local:5500/infer",
            "http://robot.local:5500",
            "ws://robot.local:5500/infer",
        ),
        (
            "https://policy.example/metadata",
            "https://policy.example",
            "wss://policy.example/infer",
        ),
    ],
)
def test_url_derivation(source, base, infer):
    urls = derive_policy_urls(source)

    assert urls.base_url == base
    assert urls.metadata_url == f"{base}/metadata"
    assert urls.reset_url == f"{base}/reset"
    assert urls.infer_url == infer


def test_url_derivation_rejects_non_contract_paths_and_credentials():
    with pytest.raises(ValueError, match="fixed v3 endpoint"):
        derive_policy_urls("http://127.0.0.1:5500/api")
    with pytest.raises(ValueError, match="credentials"):
        derive_policy_urls("http://user:password@127.0.0.1:5500")


def test_aiohttp_is_loaded_only_when_network_is_used(monkeypatch):
    real_import = transport_module.importlib.import_module

    def import_without_aiohttp(name: str):
        if name == "aiohttp":
            raise ImportError("missing for test")
        return real_import(name)

    monkeypatch.setattr(transport_module.importlib, "import_module", import_without_aiohttp)
    transport = PolicyV3Transport("http://127.0.0.1:5500")

    with pytest.raises(PolicyDependencyError, match="aiohttp"):
        asyncio.run(transport.metadata())

    asyncio.run(transport.close())


async def _start_server(app):
    from aiohttp import web

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@requires_aiohttp
def test_http_reset_and_persistent_binary_inference_round_trip():
    async def exercise():
        from aiohttp import WSMsgType, web

        calls: dict[str, object] = {"observations": []}
        initial_metadata = server_metadata()
        reset_format = metadata_format("reset-format")

        async def health_handler(request):
            calls["health"] = True
            return web.Response(text="ok\n")

        async def metadata_handler(request):
            calls["metadata_accept"] = request.headers.get("Accept")
            return web.Response(
                body=packb(initial_metadata),
                content_type="application/msgpack",
            )

        async def reset_handler(request):
            calls["reset_content_type"] = request.content_type
            reset_request = unpackb(await request.read())
            calls["reset_request"] = reset_request
            return web.Response(
                body=packb(
                    {
                        "schema": "sharpa_policy_reset.v1",
                        "session_id": reset_request["session_id"],
                        "request_id": reset_request["request_id"],
                        "reset": True,
                        "metadata_format": reset_format,
                    }
                ),
                content_type="application/msgpack",
            )

        async def infer_handler(request):
            websocket = web.WebSocketResponse(compress=False)
            await websocket.prepare(request)
            async for message in websocket:
                assert message.type == WSMsgType.BINARY
                observation = unpackb(message.data)
                calls["observations"].append(observation)
                if observation.get("cause_error"):
                    await websocket.send_bytes(
                        packb(
                            {
                                "schema": "sharpa_policy_error.v1",
                                "request_id": observation["request_id"],
                                "error": {
                                    "code": "mock_error",
                                    "message": "requested by test",
                                    "retryable": True,
                                },
                            }
                        )
                    )
                else:
                    await websocket.send_bytes(
                        packb(
                            {
                                "schema": "sharpa_policy_action.v4",
                                "session_id": observation["session_id"],
                                "request_id": observation["request_id"],
                                "echo": observation["state"],
                            }
                        )
                    )
            return websocket

        app = web.Application()
        app.router.add_get("/healthz", health_handler)
        app.router.add_get("/metadata", metadata_handler)
        app.router.add_post("/reset", reset_handler)
        app.router.add_get("/infer", infer_handler)
        runner, base_url = await _start_server(app)
        transport = PolicyV3Transport(base_url, heartbeat_s=None)
        try:
            assert await transport.health()
            metadata = await transport.metadata()
            assert metadata["schema"] == "sharpa_policy_server.v3"
            reset = await transport.reset("episode-001")
            assert reset["metadata_format"]["format_id"] == "reset-format"
            await transport.connect()

            state = np.arange(4, dtype=np.float32)
            response = await transport.infer(
                {
                    "session_id": "episode-001",
                    "request_id": 0,
                    "state": state,
                }
            )
            np.testing.assert_array_equal(response["echo"], state)

            with pytest.raises(PolicyServerError) as error_info:
                await transport.infer(
                    {
                        "session_id": "episode-001",
                        "request_id": 1,
                        "state": state,
                        "cause_error": True,
                    }
                )
            assert error_info.value.code == "mock_error"
            assert error_info.value.request_id == 1
            assert error_info.value.retryable is True
            assert transport.connected

            await transport.infer(
                {
                    "session_id": "episode-001",
                    "request_id": 2,
                    "state": state,
                }
            )
            assert calls["metadata_accept"] == "application/msgpack"
            assert calls["health"] is True
            assert calls["reset_content_type"] == "application/msgpack"
            assert calls["reset_request"] == {
                "session_id": "episode-001",
                "request_id": 0,
            }
            assert len(calls["observations"]) == 3
        finally:
            await transport.close()
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_text_inference_response_is_rejected_and_socket_is_closed():
    async def exercise():
        from aiohttp import web

        async def metadata_handler(request):
            return web.Response(
                body=packb(server_metadata()),
                content_type="application/msgpack",
            )

        async def infer_handler(request):
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.receive()
            await websocket.send_str("not binary")
            return websocket

        app = web.Application()
        app.router.add_get("/metadata", metadata_handler)
        app.router.add_get("/infer", infer_handler)
        runner, base_url = await _start_server(app)
        transport = PolicyV3Transport(base_url, heartbeat_s=None)
        try:
            await transport.metadata()
            await transport.connect()
            with pytest.raises(PolicyProtocolError, match="binary msgpack"):
                await transport.infer({"request_id": 0})
            assert not transport.connected
        finally:
            await transport.close()
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_mismatched_error_request_id_is_rejected_and_socket_is_closed():
    async def exercise():
        from aiohttp import web

        async def metadata_handler(request):
            return web.Response(
                body=packb(server_metadata()),
                content_type="application/msgpack",
            )

        async def infer_handler(request):
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.receive()
            await websocket.send_bytes(
                packb(
                    {
                        "schema": "sharpa_policy_error.v1",
                        "request_id": 99,
                        "error": {
                            "code": "stale_error",
                            "message": "wrong request",
                            "retryable": True,
                        },
                    }
                )
            )
            return websocket

        app = web.Application()
        app.router.add_get("/metadata", metadata_handler)
        app.router.add_get("/infer", infer_handler)
        runner, base_url = await _start_server(app)
        transport = PolicyV3Transport(base_url, heartbeat_s=None)
        try:
            await transport.metadata()
            await transport.connect()
            with pytest.raises(PolicyProtocolError, match="does not match"):
                await transport.infer({"request_id": 0})
            assert not transport.connected
        finally:
            await transport.close()
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_second_concurrent_inference_is_rejected_not_queued():
    async def exercise():
        from aiohttp import web

        received = asyncio.Event()
        release = asyncio.Event()

        async def metadata_handler(request):
            return web.Response(
                body=packb(server_metadata()),
                content_type="application/msgpack",
            )

        async def infer_handler(request):
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.receive()
            received.set()
            await release.wait()
            await websocket.send_bytes(
                packb({"schema": "sharpa_policy_action.v4", "request_id": 0})
            )
            await websocket.close()
            return websocket

        app = web.Application()
        app.router.add_get("/metadata", metadata_handler)
        app.router.add_get("/infer", infer_handler)
        runner, base_url = await _start_server(app)
        transport = PolicyV3Transport(base_url, heartbeat_s=None)
        try:
            await transport.metadata()
            await transport.connect()
            first = asyncio.create_task(transport.infer({"request_id": 0}))
            await received.wait()
            with pytest.raises(PolicyConcurrencyError, match="already inflight"):
                await transport.infer({"request_id": 1})
            release.set()
            result = await first
            assert result["request_id"] == 0
        finally:
            await transport.close()
            await runner.cleanup()

    asyncio.run(exercise())


@requires_aiohttp
def test_negotiated_message_limit_rejects_oversized_outbound_observation():
    async def exercise():
        from aiohttp import web

        metadata_payload = server_metadata(max_message_size=1024)

        async def metadata_handler(request):
            return web.Response(
                body=packb(metadata_payload),
                content_type="application/msgpack",
            )

        async def infer_handler(request):
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            async for _ in websocket:
                pass
            return websocket

        app = web.Application()
        app.router.add_get("/metadata", metadata_handler)
        app.router.add_get("/infer", infer_handler)
        runner, base_url = await _start_server(app)
        transport = PolicyV3Transport(base_url, heartbeat_s=None)
        try:
            await transport.metadata()
            assert transport.effective_message_size == 1024
            await transport.connect()
            with pytest.raises(PolicyMessageTooLargeError):
                await transport.infer({"blob": b"x" * 2048})
            assert transport.connected
        finally:
            await transport.close()
            await runner.cleanup()

    asyncio.run(exercise())
