import json
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dashboard_node.dashboard_node import (
    DashboardState, make_handler, recording_config_command,
)
from dashboard_node.model_manager import ModelManagerError


def idle_state():
    return {"control": {}, "recording": {
        "received_at": time.time(), "recording_root": "/data/old",
    }}


def test_directory_config_does_not_require_a_loaded_model(tmp_path):
    first = recording_config_command({"recording_root": str(tmp_path)}, idle_state())
    second = recording_config_command({"recording_root": str(tmp_path)}, idle_state())
    assert first["recording_root"] == str(tmp_path)
    assert first["id"] != second["id"]


@pytest.mark.parametrize("override", [
    {"control": {"ur_state": 4}}, {"control": {"sharpa_state": 4}},
    {"control": {"execution_start_pending": True}},
    {"recording": {"recording": True}}, {"recording": {"saving": True}},
    {"policy": {"rollout_recording": {"requested": True}}},
])
def test_directory_config_rejects_pending_start_and_capture(tmp_path, override):
    snapshot = idle_state()
    for key, value in override.items():
        snapshot[key] = {**snapshot.get(key, {}), **value}
    with pytest.raises(ModelManagerError, match="采集启动、录制或保存"):
        recording_config_command({"recording_root": str(tmp_path)}, snapshot)


def test_directory_config_rejects_disconnected_and_stale_status(tmp_path):
    for snapshot in ({}, {"recording": {"received_at": time.time()}},
                     {"control": {}, "recording": {"received_at": time.time() - 3}}):
        with pytest.raises(ModelManagerError, match="未连接|过期"):
            recording_config_command({"recording_root": str(tmp_path)}, snapshot)


def test_http_queues_directory_config_and_exposes_actual_recorder_status(tmp_path):
    dashboard = DashboardState()
    for key, value in idle_state().items():
        dashboard.update(key, value)
    commands = []
    handler = make_handler(dashboard, "http://127.0.0.1:1",
                           SimpleNamespace(snapshot=lambda: {}), lambda _: None,
                           publish_recording_config=commands.append)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(url + "/api/recording/config",
                          data=json.dumps({"recording_root": str(tmp_path)}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            result = json.load(response)
            assert response.status == 202 and result["queued"]
            assert result["command_id"] == commands[0]["id"]
        with urlopen(url + "/api/status", timeout=2) as response:
            assert json.load(response)["recording"]["recording_root"] == "/data/old"
        dashboard.update("recording", {"recording_root": str(tmp_path), "config_result": {
            "id": commands[0]["id"], "accepted": True,
        }})
        with urlopen(url + "/api/status", timeout=2) as response:
            recording = json.load(response)["recording"]
            assert recording["recording_root"] == str(tmp_path)
            assert recording["config_result"]["accepted"] is True
        request.data = b'{"recording_root":"relative/path"}'
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
        assert len(commands) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("control", [
    {"ur_state": 2, "sharpa_state": 1},
    {"execution_scope": "hands_only", "sharpa_state": 1},
    {"execution_dry_run": True}, {"execution_paused": True},
])
def test_directory_can_change_without_active_capture(tmp_path, control):
    snapshot = idle_state()
    snapshot["control"] = control
    assert recording_config_command({"recording_root": str(tmp_path)}, snapshot)["recording_root"] == str(tmp_path)
