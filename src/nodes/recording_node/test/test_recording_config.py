import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recording_node.recording_node import RecorderCore, RecordingNode
from ur_common.recording_config import normalize_recording_root, RecordingPreferences


@pytest.mark.parametrize("value", [None, "", " ", "/", "/tmp/..", "relative/path", 123, "/data/\0"])
def test_invalid_recording_root_is_rejected(value):
    with pytest.raises(ValueError):
        normalize_recording_root(value)


def test_root_normalization_supports_home_and_spaces():
    assert normalize_recording_root(" ~/my data/test ") == Path.home() / "my data/test"


def test_new_episode_uses_configured_root_and_cannot_change_while_active(tmp_path):
    core = RecorderCore(tmp_path / "old", record_tactile=False, rollout_recording=True)
    root = tmp_path / "new data" / "test"
    try:
        assert core.configure_root(str(root)) == root
        assert root.is_dir()
        assert core.status()["recording_root"] == str(root)
        assert core.start()
        assert core.writer.episode_dir.parent == root
        with pytest.raises(RuntimeError, match="不能修改"):
            core.configure_root(str(tmp_path / "other"))
        assert core.writer.root == root
        core.stop(False)
    finally:
        core.writer.close()


@pytest.mark.parametrize("flags", [{"saving": True}, {"requested": True}])
def test_directory_cannot_change_while_saving_or_awaiting_rollout(tmp_path, flags):
    core = RecorderCore(tmp_path / "old", record_tactile=False)
    try:
        with pytest.raises(RuntimeError):
            core.configure_root(str(tmp_path / "new"), **flags)
        assert core.writer.root == tmp_path / "old"
        assert not (tmp_path / "new").exists()
    finally:
        core.writer.close()


def test_directory_errors_leave_original_root_intact(tmp_path, monkeypatch):
    core = RecorderCore(tmp_path / "old", record_tactile=False)
    def denied(*args, **kwargs):
        raise PermissionError("Permission denied")
    monkeypatch.setattr(Path, "mkdir", denied)
    try:
        with pytest.raises(PermissionError):
            core.configure_root(str(tmp_path / "new"))
        assert core.writer.root == tmp_path / "old"
    finally:
        core.writer.close()


def test_config_callback_reports_acceptance_and_rejection_with_command_id(tmp_path):
    preferences = RecordingPreferences(tmp_path / 'config/recording.json')
    core = RecorderCore(tmp_path / "old", record_tactile=False, root_preferences=preferences)
    published = []
    node = SimpleNamespace(core=core, _saving=False, _auto_rollout=True,
                           rollout=SimpleNamespace(requested=""),
                           get_logger=lambda: SimpleNamespace(warning=lambda _: None),
                           _publish=lambda: published.append(True))
    try:
        config = {"id": "one", "recording_root": str(tmp_path / "new")}
        RecordingNode._config(node, SimpleNamespace(data=json.dumps(config)))
        assert node._record_config_result == {
            "id": "one", "accepted": True, "recording_root": str(tmp_path / "new"),
        }
        assert preferences.read('/fallback') == str(tmp_path / 'new')
        node.rollout.requested = "starting-episode"
        config = {"id": "two", "recording_root": str(tmp_path / "other")}
        RecordingNode._config(node, SimpleNamespace(data=json.dumps(config)))
        assert node._record_config_result["id"] == "two"
        assert node._record_config_result["accepted"] is False
        assert core.writer.root == tmp_path / "new"
        assert preferences.read('/fallback') == str(tmp_path / 'new')
        assert len(published) == 2
    finally:
        core.writer.close()


def test_applied_root_survives_recorder_restart(tmp_path):
    path = tmp_path / 'settings/recording.json'
    preferences = RecordingPreferences(path)
    default = tmp_path / 'default'
    assert preferences.read(default) == str(default)
    core = RecorderCore(default, record_tactile=False, root_preferences=preferences)
    selected = tmp_path / '采集 数据'
    try:
        core.configure_root(' '+str(selected)+' ')
    finally:
        core.writer.close()
    restarted = RecorderCore(RecordingPreferences(path).read(default), record_tactile=False)
    try:
        assert restarted.writer.root == selected
        assert not restarted.recording and not restarted.writer.active
        assert restarted.start()
        assert restarted.writer.episode_dir.parent == selected
        restarted.stop(False)
    finally:
        restarted.writer.close()


def test_failed_preference_write_preserves_previous_memory_and_active_root(tmp_path, monkeypatch):
    preferences = RecordingPreferences(tmp_path / 'config/recording.json')
    old = tmp_path / 'old'
    preferences.save(old)
    core = RecorderCore(old, record_tactile=False, root_preferences=preferences)
    def denied(*args):
        raise PermissionError('cannot save preferences')
    monkeypatch.setattr('ur_common.recording_config.os.replace', denied)
    try:
        with pytest.raises(PermissionError):
            core.configure_root(str(tmp_path / 'new'))
        assert core.writer.root == old
        assert preferences.read('/fallback') == str(old)
        assert list(preferences.path.parent.iterdir()) == [preferences.path]
    finally:
        core.writer.close()


@pytest.mark.parametrize('content', ['invalid json', '[]', '{}', '{"recording_root":"relative"}'])
def test_invalid_memory_is_reported_instead_of_silently_changing_destination(tmp_path, content):
    path = tmp_path / 'recording.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        RecordingPreferences(path).read('/fallback')


def test_capture_mode_changes_source_only_when_idle(tmp_path):
    from recording_node.recording_node import RecorderCore
    core = RecorderCore(tmp_path, record_tactile=False)
    assert core.hand_source == 2
    assert core.configure_capture_mode(True)
    assert core.hand_source == 3 and core.rollout_recording
    assert not core.configure_capture_mode(False, saving=True)
    core.recording=True
    assert not core.configure_capture_mode(False)
    core.recording=False
    assert core.configure_capture_mode(False)
    assert core.hand_source == 2 and not core.rollout_recording
