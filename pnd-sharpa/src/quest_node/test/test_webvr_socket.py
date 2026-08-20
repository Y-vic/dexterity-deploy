import asyncio
import json

import pytest


pytest.importorskip("rclpy")
pytest.importorskip("websockets")

from quest_node.quest_webvr import WebVRSocketServer  # noqa: E402


ACCESS_TOKEN = "quest-test-access-token-000001"
SECURE_HEADERS = {
    "Host": "10.10.20.127",
    "Origin": "https://10.10.20.127",
    "X-Forwarded-Proto": "https",
}


class FakeRequest:
    def __init__(self, headers):
        self.headers = headers


class FakeWebSocket:
    def __init__(self, messages, *, headers=None, receive_delay=0.0):
        self.request = FakeRequest(headers or SECURE_HEADERS)
        self.messages = list(messages)
        self.receive_delay = receive_delay
        self.sent = []
        self.closed = None

    async def recv(self):
        if self.receive_delay:
            await asyncio.sleep(self.receive_delay)
        return self.messages.pop(0)

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def close(self, *, code, reason):
        self.closed = (code, reason)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


def make_server(*, authentication_timeout=0.05):
    events = []
    samples = []
    errors = []
    server = WebVRSocketServer(
        "127.0.0.1",
        8442,
        access_token=ACCESS_TOKEN,
        authentication_timeout=authentication_timeout,
        on_connection=events.append,
        on_sample=lambda sample, received_at: samples.append((sample, received_at)),
        on_error=errors.append,
    )
    return server, events, samples, errors


def auth_message(token=ACCESS_TOKEN):
    return json.dumps({"type": "auth", "token": token})


def test_wrong_token_never_claims_headset_slot():
    server, events, samples, errors = make_server()
    websocket = FakeWebSocket([auth_message("wrong-token-value-000000")])

    asyncio.run(server._handle_client(websocket))

    assert events == []
    assert samples == []
    assert errors
    assert not server._clients
    assert websocket.sent[0]["error"] == "AUTH_FAILED"


def test_authentication_timeout_never_claims_headset_slot():
    server, events, _samples, errors = make_server(authentication_timeout=0.001)
    websocket = FakeWebSocket([auth_message()], receive_delay=0.01)

    asyncio.run(server._handle_client(websocket))

    assert events == []
    assert errors
    assert not server._clients
    assert websocket.sent[0]["error"] == "AUTH_TIMEOUT"


def test_valid_connection_claims_and_releases_slot_after_ack():
    server, events, _samples, errors = make_server()
    websocket = FakeWebSocket([auth_message()])

    asyncio.run(server._handle_client(websocket))

    assert events == [True, False]
    assert errors == []
    assert not server._clients
    assert websocket.sent == [{"type": "auth_ok"}]


def test_second_authenticated_client_does_not_disconnect_existing_headset():
    server, events, _samples, errors = make_server()
    existing_headset = object()
    server._clients.add(existing_headset)
    websocket = FakeWebSocket([auth_message()])

    asyncio.run(server._handle_client(websocket))

    assert events == []
    assert errors == []
    assert server._clients == {existing_headset}
    assert websocket.sent[0]["error"] == "VR_HEADSET_LIMIT"


def test_request_header_adapter_supports_new_and_legacy_websockets_api():
    new_api = FakeWebSocket([auth_message()])

    class LegacyWebSocket:
        request_headers = SECURE_HEADERS

    assert WebVRSocketServer._request_headers(new_api) is new_api.request.headers
    assert WebVRSocketServer._request_headers(LegacyWebSocket()) is SECURE_HEADERS


def test_calibration_event_is_sent_to_connected_headset():
    server, _events, _samples, _errors = make_server()
    websocket = FakeWebSocket([])
    server._clients.add(websocket)

    asyncio.run(
        server._broadcast_event(
            {
                "type": "calibration",
                "state": "calibrated",
                "robot_scale": 1.25,
            }
        )
    )

    assert websocket.sent == [
        {
            "type": "calibration",
            "state": "calibrated",
            "robot_scale": 1.25,
        }
    ]
