from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import time
from typing import Any

import numpy as np

from .serialization import MAX_MESSAGE_SIZE, packb, unpackb


def empty_metadata_format() -> dict[str, Any]:
    return {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": "mock-empty-v1",
        "image": {
            name: {"history_len": 0, "current": False}
            for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
        },
        "state": {
            "history_len": 0,
            "current": False,
            "left_wrist": {"joint": False, "eef": False},
            "right_wrist": {"joint": False, "eef": False},
            "hand_joint": {"left": False, "right": False},
        },
        "sensor": {
            name: {"history_len": 0, "current": False}
            for name in ("tau", "wrench", "deformation")
        },
    }


def hardware_metadata_format() -> dict[str, Any]:
    return {
        "schema": "sharpa_policy_metadata_format.v1",
        "format_id": "mock-hardware-30hz-v1",
        "image": {
            "ego_cam": {"history_len": 2, "current": True},
            "left_wrist_cam": {"history_len": 0, "current": False},
            "right_wrist_cam": {"history_len": 0, "current": False},
        },
        "state": {
            "history_len": 1,
            "current": True,
            "left_wrist": {"joint": True, "eef": True},
            "right_wrist": {"joint": True, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": {"history_len": 8, "current": True},
            "wrench": {"history_len": 8, "current": True},
            "deformation": {"history_len": 0, "current": True},
        },
    }


class MockPolicyServer:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        metadata_format: dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.metadata_format = deepcopy(metadata_format or empty_metadata_format())
        self._active_websocket = None
        self._connection_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._session_id: str | None = None
        self._next_request_id = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "sharpa_policy_server.v3",
            "policy_family": "mock",
            "checkpoint_id": "mock-checkpoint",
            "checkpoint_path": "mock://checkpoint",
            "task_id": "mock-task",
            "run_id": "mock-run",
            "dataset_path": "mock://dataset",
            "prompt": "",
            "transport": "websocket+binary_msgpack",
            "observation_schema": "sharpa_policy_observation.v3",
            "action_schema": "sharpa_policy_action.v4",
            "host": self.host,
            "port": self.port,
            "infer_path": "/infer",
            "health_path": "/healthz",
            "metadata_path": "/metadata",
            "reset_path": "/reset",
            "max_message_size": MAX_MESSAGE_SIZE,
            "metadata_format": self.metadata_format,
        }

    async def health(self, request: Any) -> Any:
        from aiohttp import web

        return web.Response(text="ok\n", content_type="text/plain")

    async def get_metadata(self, request: Any) -> Any:
        from aiohttp import web

        return web.Response(
            body=packb(self.metadata()),
            content_type="application/msgpack",
        )

    async def reset(self, request: Any) -> Any:
        from aiohttp import web

        if request.content_type != "application/msgpack":
            result = self._error(
                None,
                "invalid_content_type",
                "reset requires application/msgpack",
                False,
            )
            return web.Response(
                body=packb(result),
                status=415,
                content_type="application/msgpack",
            )
        body = await request.read()
        try:
            payload = unpackb(body)
            session_id = payload["session_id"]
            request_id = payload["request_id"]
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a non-empty string")
            if type(request_id) is not int or request_id != 0:
                raise ValueError("reset request_id must be 0")
            async with self._state_lock:
                active_websocket = self._active_websocket
                if active_websocket is not None and not active_websocket.closed:
                    raise ValueError("disconnect inference before reset")
                if self._session_id == session_id:
                    raise ValueError(
                        "new session_id must differ from the active session"
                    )
                self._session_id = session_id
                self._next_request_id = 0
                result = {
                    "schema": "sharpa_policy_reset.v1",
                    "session_id": session_id,
                    "request_id": request_id,
                    "reset": True,
                    "metadata_format": self.metadata_format,
                }
            status = 200
        except (KeyError, TypeError, ValueError) as exc:
            result = self._error(None, "invalid_reset", str(exc), False)
            status = 400
        return web.Response(
            body=packb(result),
            status=status,
            content_type="application/msgpack",
        )

    async def infer(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        async with self._connection_lock:
            if self._active_websocket is not None and not self._active_websocket.closed:
                return web.Response(status=409, text="a client is already connected\n")
            websocket = web.WebSocketResponse(max_msg_size=MAX_MESSAGE_SIZE)
            await websocket.prepare(request)
            self._active_websocket = websocket
        try:
            async for message in websocket:
                if message.type != WSMsgType.BINARY:
                    await websocket.close(code=1003, message=b"binary msgpack required")
                    break
                observation = None
                try:
                    observation = unpackb(message.data)
                    async with self._state_lock:
                        result = self._action_for(observation)
                except (KeyError, TypeError, ValueError) as exc:
                    request_id = None
                    if isinstance(observation, dict):
                        candidate = observation.get("request_id")
                        if type(candidate) is int and candidate >= 0:
                            request_id = candidate
                    result = self._error(
                        request_id,
                        "invalid_observation",
                        str(exc),
                        False,
                    )
                await websocket.send_bytes(packb(result))
        finally:
            if self._active_websocket is websocket:
                self._active_websocket = None
        return websocket

    def _action_for(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise ValueError("observation must be a map")
        if observation.get("schema") != "sharpa_policy_observation.v3":
            raise ValueError("unsupported observation schema")
        if observation.get("metadata_format_id") != self.metadata_format["format_id"]:
            raise ValueError("metadata_format_id mismatch")
        session_id = observation["session_id"]
        request_id = observation["request_id"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        if self._session_id is not None and session_id != self._session_id:
            raise ValueError("session_id does not match the active reset session")
        if request_id != self._next_request_id:
            raise ValueError(
                f"request_id must be {self._next_request_id} for the active session"
            )
        if self._session_id is None:
            self._session_id = session_id
        self._next_request_id += 1
        action_length = 4
        identity_rot6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        wrists = np.zeros((action_length, 9), dtype=np.float32)
        wrists[:, 3:] = identity_rot6d
        return {
            "schema": "sharpa_policy_action.v4",
            "session_id": session_id,
            "request_id": request_id,
            "action_id": f"mock-{session_id}-{request_id}",
            "revision": 0,
            "timestamp_ns": time.time_ns(),
            "execution": {
                "frequency_hz": 10.0,
                "action_length": action_length,
                "execute_start": 1,
                "execute_length": 2,
            },
            "action": {
                "left_wrist": {
                    "joint": None,
                    "eef": wrists.copy(),
                    "eef_def": "absolute",
                },
                "right_wrist": {
                    "joint": None,
                    "eef": wrists.copy(),
                    "eef_def": "absolute",
                },
                "hand_joint": {
                    "left": np.zeros((action_length, 22), dtype=np.float32),
                    "right": np.zeros((action_length, 22), dtype=np.float32),
                },
            },
            "auxiliary": {
                "video": {"ego": None, "left_wrist": None, "right_wrist": None},
                "tactile": {
                    "deformation": None,
                    "wrench": None,
                    "hand_tau": None,
                },
            },
            "diagnostics": {
                "policy_family": "mock",
                "checkpoint_id": "mock-checkpoint",
                "checkpoint_path": "mock://checkpoint",
                "inference_latency_ms": 0.0,
            },
            "next_metadata_format": None,
        }

    @staticmethod
    def _error(
        request_id: int | None,
        code: str,
        message: str,
        retryable: bool,
    ) -> dict[str, Any]:
        return {
            "schema": "sharpa_policy_error.v1",
            "request_id": request_id,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        }

    async def run(self) -> None:
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError(
                "mock server requires aiohttp; install python3-aiohttp"
            ) from exc
        application = self.create_application()
        runner = web.AppRunner(application)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(
            f"mock SharpA v3 server listening on http://{self.host}:{self.port}",
            flush=True,
        )
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    def create_application(self) -> Any:
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError(
                "mock server requires aiohttp; install python3-aiohttp"
            ) from exc
        application = web.Application(client_max_size=MAX_MESSAGE_SIZE)
        application.router.add_get("/healthz", self.health)
        application.router.add_get("/metadata", self.get_metadata)
        application.router.add_post("/reset", self.reset)
        application.router.add_get("/infer", self.infer)
        return application


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controller-free v3 mock server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--hardware-profile", action="store_true")
    arguments = parser.parse_args()
    if arguments.port <= 0 or arguments.port > 65535:
        parser.error("--port must be in [1, 65535]")
    try:
        asyncio.run(
            MockPolicyServer(
                arguments.host,
                arguments.port,
                metadata_format=(
                    hardware_metadata_format() if arguments.hardware_profile else None
                ),
            ).run()
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
