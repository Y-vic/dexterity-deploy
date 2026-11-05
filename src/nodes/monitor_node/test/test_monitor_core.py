import threading
import time
import hashlib
import subprocess

import pytest

from monitor_node.monitor_node import (
    DEFAULT_REMOTE_DATA_PATH,
    HTML,
    MonitorController,
)


def test_status_heartbeat_expires():
    controller = MonitorController(status_timeout_s=0.01)
    controller.update_status({"ur_state": 2})
    controller.update_upstream("vive_status", {"valid": True})
    assert controller.snapshot()["status"]["online"] is True
    assert controller.snapshot()["upstream"]["vive_status"]["online"] is True
    time.sleep(0.02)
    assert controller.snapshot()["status"]["online"] is False
    assert controller.snapshot()["upstream"]["vive_status"]["online"] is False


def test_browser_script_keeps_newline_escape():
    assert r"join('\n')" in HTML
    assert "join('\n')" not in HTML
    assert "数据保存与上传" in HTML
    assert "Camera × 3" in HTML
    assert 'id="start-machine"' in HTML
    assert 'id="pause-machine"' in HTML
    assert 'data-action="pause"' in HTML
    assert "button.disabled = false" not in HTML
    assert "machineRunning" in HTML
    assert "/api/camera-stream/" in HTML
    assert "setInterval(refreshCameras" not in HTML
    assert 'id="remote-model-path"' in HTML
    assert "truncateMiddle" in HTML
    assert "不会自动重启" in HTML
    assert "restart_max_attempts" not in HTML
    assert 'data-action="mark_success"' in HTML
    assert 'data-action="mark_fail"' in HTML
    assert 'data-action="drop_sample"' in HTML
    assert 'id="saved-success"' in HTML
    assert 'id="saved-fail"' in HTML
    assert 'id="saved-undefined"' in HTML
    assert "recorder.saving" in HTML
    assert "正在保存…" in HTML
    assert "0.0 s" in HTML


def test_recording_counts_only_timestamp_named_samples(tmp_path):
    (tmp_path / "success").mkdir()
    (tmp_path / "fail").mkdir()
    (tmp_path / "success" / "2026-08-24_12-00-02.000001").mkdir()
    (tmp_path / "fail" / "2026-08-24_12-00-03.000001").mkdir()
    (tmp_path / "2026-08-24_12-00-04.000001.partial").mkdir()

    assert MonitorController(tmp_path).snapshot()["recording_counts"] == {
        "success": 1,
        "fail": 1,
        "undefined": 0,
    }


def test_default_remote_data_path_is_exposed_to_monitor():
    controller = MonitorController()
    assert controller.snapshot()["upload_path"] == DEFAULT_REMOTE_DATA_PATH
    assert DEFAULT_REMOTE_DATA_PATH.endswith("/ur-sharpa/raw_data")


@pytest.mark.parametrize("action", ["pause", "start_teleop_record"])
def test_monitor_forwards_control_action(action):
    commands = []
    controller = MonitorController(publish_command=commands.append)

    result = controller.action({"action": action})

    assert result["accepted"] is True
    assert commands == [action]


def test_monitor_forwards_selected_episode_with_reset_command():
    import json
    commands = []
    controller = MonitorController(publish_command=commands.append)
    target = {"mode": "episode", "root": "/media/yns/My Passport/data", "episode": "success/one"}
    assert controller.action({"action": "reset_both", "reset_target": target})["accepted"]
    assert json.loads(commands[0]) == {"action": "reset_both", "reset_target": target}


def test_monitor_forwards_sample_classification():
    labels = []
    controller = MonitorController(publish_label=labels.append)
    controller.update_upstream("record_status", {
        "validation":{"available":True,"items":[{}]*8}
    })
    result = controller.action({"action":"mark_fail"})
    assert result["accepted"] is True
    assert labels == ["fail"]


def test_monitor_forwards_drop_for_saved_sample():
    labels = []
    controller = MonitorController(publish_label=labels.append)
    controller.update_upstream("record_status", {
        "validation":{"available":True,"items":[{}]*8}
    })
    result = controller.action({"action":"drop_sample"})
    assert result["accepted"] is True
    assert labels == ["drop"]


def test_camera_status_is_per_view_and_repeated_source_does_not_refresh():
    controller = MonitorController(status_timeout_s=1.0)
    controller.update_cameras(
        {"ego": b"ego", "left_wrist": b"left", "right_wrist": b"right"},
        {"ego": 1, "left_wrist": 2, "right_wrist": 3},
    )
    first = controller.snapshot()
    assert all(first["camera_status"][name]["online"] for name in first["camera_status"])
    assert controller.camera_frame("ego") == (b"ego", 1)

    controller.update_cameras(
        {"ego": b"new", "left_wrist": b"new", "right_wrist": b"new"},
        {"ego": 1, "left_wrist": 2, "right_wrist": 3},
    )
    assert controller.cameras["ego"] == b"ego"


def test_camera_stream_waits_for_a_newer_source_frame():
    controller = MonitorController(status_timeout_s=1.0)
    controller.update_cameras({"ego": b"old"}, {"ego": 1})
    assert controller.wait_camera_frame("ego", 0) == (b"old", 1)

    result = []
    waiter = threading.Thread(
        target=lambda: result.append(controller.wait_camera_frame("ego", 1, timeout=1.0))
    )
    waiter.start()
    time.sleep(0.01)
    controller.update_cameras({"ego": b"new"}, {"ego": 2})
    waiter.join(timeout=1.0)
    assert result == [(b"new", 2)]


def test_upload_allows_only_one_background_worker(monkeypatch):
    controller = MonitorController()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def upload():
        calls.append(True)
        entered.set()
        release.wait(timeout=1.0)
        return "uploaded sample_0000"

    monkeypatch.setattr(controller, "_upload_latest", upload)
    assert controller.action({"action": "upload"})["accepted"] is True
    assert entered.wait(timeout=1.0)
    assert controller.action({"action": "upload"})["accepted"] is False
    release.set()
    deadline = time.monotonic() + 1.0
    while controller.snapshot()["uploading"] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert calls == [True]
    assert controller.snapshot()["message"] == "uploaded sample_0000"


def test_upload_rejects_directory_without_complete_marker(tmp_path):
    sample = tmp_path / "2026-08-25_12-34-56.123456"
    sample.mkdir()
    (sample / "manifest.json").write_text(
        '{"state":"complete","quality":{"ok":true}}', encoding="utf-8"
    )
    controller = MonitorController(tmp_path, upload_path="/remote")
    assert controller._upload_latest() == "upload failed: sample sanity check failed"


@pytest.mark.parametrize("category", ["", "success", "fail"])
def test_upload_preserves_sample_relative_category(tmp_path, monkeypatch, category):
    sample_name="2026-08-25_12-34-56.123456"
    parent=tmp_path if not category else tmp_path/category
    sample=parent/sample_name; sample.mkdir(parents=True)
    (sample/"manifest.json").write_text(
        '{"state":"complete","quality":{"ok":true}}', encoding="utf-8"
    )
    (sample/"COMPLETE").touch()
    payload=sample/"payload.bin"; payload.write_bytes(b"payload")
    digest=hashlib.sha256(payload.read_bytes()).hexdigest()
    (sample/"checksums.sha256").write_text(
        f"{digest}  payload.bin\n", encoding="utf-8"
    )
    calls=[]
    monkeypatch.setattr(subprocess,"run",lambda args,**kwargs:calls.append(args))
    controller=MonitorController(tmp_path,upload_host="host",upload_path="/remote/root")
    relative=f"{category}/{sample_name}" if category else sample_name
    assert controller._upload_latest() == f"uploaded {relative}"
    remote_parent=f"/remote/root/{category}" if category else "/remote/root"
    assert ["ssh","host",f"mkdir -p -- {remote_parent}"] in calls
    rsync=next(call for call in calls if call[0] == "rsync")
    assert rsync[-1] == f"host:{remote_parent}/{sample_name}.partial/"


def test_recording_path_cannot_change_during_sample():
    controller = MonitorController("/old")
    controller.update_status({"record_state": 1})
    result = controller.action({"action": "set_paths", "recording_root": "/new"})
    assert result["accepted"] is False
    assert controller.recording_root == "/old"


def test_directory_update_waits_for_recorder_confirmation(tmp_path):
    messages=[]
    controller=MonitorController(tmp_path, publish_config=messages.append)
    destination=tmp_path/'new'
    result=controller.action({'action':'set_paths','recording_root':str(destination)})
    assert result['command_id']==messages[0]['id']
    assert controller.recording_root==str(tmp_path)
    controller.update_upstream('record_status',{'recording_root':str(destination)})
    assert controller.recording_root==str(destination)


@pytest.mark.parametrize('action', ['mark_success', 'mark_fail', 'drop_sample'])
def test_classification_rejects_stale_sample_or_saving(action):
    labels=[]
    controller=MonitorController(publish_label=labels.append)
    controller.update_upstream('record_status',{'saved_path':'/new','validation':{'available':True}})
    assert not controller.action({'action':action,'saved_path':'/old'})['accepted']
    controller.update_upstream('record_status',{'saved_path':'/new','saving':True,'validation':{'available':True}})
    assert not controller.action({'action':action,'saved_path':'/new'})['accepted']
    controller.update_upstream('record_status',{'saved_path':'/new','recording':True,'validation':{'available':True}})
    assert not controller.action({'action':action,'saved_path':'/new'})['accepted']
    assert not labels
