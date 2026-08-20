import json
from urllib.request import urlopen

from quest_node.quest_webvr import WebVRHTTPServer


ACCESS_TOKEN = "quest-test-access-token-000001"


def test_runtime_config_serves_uncached_access_token(tmp_path):
    (tmp_path / "index.html").write_text("Quest", encoding="utf-8")
    server = WebVRHTTPServer(
        "127.0.0.1",
        0,
        tmp_path,
        runtime_config={
            "accessToken": ACCESS_TOKEN,
            "iceServers": [
                {
                    "urls": ["turn:10.10.20.127:3478?transport=udp"],
                    "username": "quest-video",
                    "credential": "test-password",
                }
            ],
            "iceTransportPolicy": "relay",
        },
    )
    server.start()
    try:
        port = server._server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/runtime-config.json",
            timeout=2.0,
        ) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response) == {
                "accessToken": ACCESS_TOKEN,
                "iceServers": [
                    {
                        "urls": ["turn:10.10.20.127:3478?transport=udp"],
                        "username": "quest-video",
                        "credential": "test-password",
                    }
                ],
                "iceTransportPolicy": "relay",
            }
    finally:
        server.stop()


def test_static_web_assets_are_not_cached(tmp_path):
    (tmp_path / "quest_app.js").write_text("Quest", encoding="utf-8")
    server = WebVRHTTPServer(
        "127.0.0.1",
        0,
        tmp_path,
        runtime_config={"accessToken": ACCESS_TOKEN},
    )
    server.start()
    try:
        port = server._server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/quest_app.js",
            timeout=2.0,
        ) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == b"Quest"
    finally:
        server.stop()


def test_http_server_can_stop_before_start(tmp_path):
    server = WebVRHTTPServer(
        "127.0.0.1",
        0,
        tmp_path,
        runtime_config={"accessToken": ACCESS_TOKEN},
    )

    server.stop()
