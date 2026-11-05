import json
from pathlib import Path
import re
import subprocess
import time

import pytest

from dashboard_node.model_manager import (
    ModelManagerError,
    RemoteModelManager,
    ServerProfile,
    load_server_profiles,
)


def catalog_payload(*, launchable=True):
    return {
        "models": ["groot_n17"],
        "tasks": [
            {
                "name": "ur_pick_place_push",
                "task_id": "ur_pick_place_push",
                "prompt": "pick",
                "dataset_path": "/remote/dataset",
                "compatible": True,
                "runs": [
                    {
                        "model": "groot_n17",
                        "id": "groot_n17_baseline",
                        "run_id": "full-run-id",
                        "checkpoint_path": "/remote/checkpoint",
                        "variant": "n17_baseline",
                        "stage": "posttrain",
                        "status": "complete",
                        "eligible": launchable,
                        "checkpoint_exists": launchable,
                        "launchable": launchable,
                        "disabled_reason": "checkpoint 不存在" if not launchable else "",
                    }
                ],
            }
        ]
    }


class FakeRunner:
    def __init__(self, payload):
        self.payload = payload
        self.commands = []
        self.in_use_ports = set()
        self.launched_port = None
        self.launch_count = 0

    def __call__(self, command, timeout_s):
        self.commands.append((command, timeout_s))
        remote = command[-1]
        if remote.startswith("python3 -c"):
            return subprocess.CompletedProcess(command, 0, json.dumps(self.payload), "")
        if remote.startswith("tmux has-session"):
            return subprocess.CompletedProcess(command, 1, "", "")
        if remote.startswith("set +e"):
            if self.launched_port is not None:
                self.in_use_ports.discard(self.launched_port)
                self.launched_port = None
            return subprocess.CompletedProcess(command, 0, "", "")
        if remote.startswith("(exec 3<>/dev/tcp"):
            port = int(re.search(r"/dev/tcp/127\.0\.0\.1/(\d+)", remote).group(1))
            return subprocess.CompletedProcess(
                command, 0 if port in self.in_use_ports else 1, "", ""
            )
        if remote.startswith("curl "):
            port = int(re.search(r"127\.0\.0\.1:(\d+)", remote).group(1))
            return subprocess.CompletedProcess(
                command, 0 if port in self.in_use_ports else 1, "", ""
            )
        if "tmux new-session" in remote:
            self.launched_port = 5500
            self.in_use_ports.add(self.launched_port)
            self.launch_count += 1
        if "tmux kill-session" in remote and self.launched_port is not None:
            self.in_use_ports.discard(self.launched_port)
            self.launched_port = None
        return subprocess.CompletedProcess(command, 0, "", "")


def wait_phase(manager, phase, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if manager.snapshot()["phase"] == phase:
            return
        time.sleep(0.02)
    raise AssertionError(f"model phase did not become {phase}: {manager.snapshot()}")


def test_catalog_and_start_use_only_catalog_launcher():
    runner = FakeRunner(catalog_payload())
    events = []
    manager = RemoteModelManager(remote_root="/remote",
        runner=runner,
        event_callback=events.append,
        startup_timeout_s=2.0,
    )

    state = manager.start("ur_pick_place_push", "full-run-id")
    assert state["phase"] == "starting"
    wait_phase(manager, "running")

    remote_commands = [command[-1] for command, _ in runner.commands]
    launch = next(command for command in remote_commands if "tmux new-session" in command)
    assert "scripts/deploy/groot_n17/launch.sh" in launch
    assert "ur_pick_place_push" in launch
    assert "full-run-id" in launch
    assert manager.snapshot()["phase"] == "running"
    assert manager.snapshot()["port"] == 5500
    assert "exec env" not in launch
    assert [event["phase"] for event in events] == [
        "starting",
        "loading",
        "running",
    ]
    assert manager.snapshot()["task_id"] == "ur_pick_place_push"
    assert manager.snapshot()["run_id"] == "full-run-id"
    assert manager.snapshot()["model"] == "groot_n17"


def test_start_rejects_unlaunchable_and_unknown_runs():
    manager = RemoteModelManager(remote_root="/remote", runner=FakeRunner(catalog_payload(launchable=False)))
    with pytest.raises(ModelManagerError, match="checkpoint"):
        manager.start("ur_pick_place_push", "full-run-id")

    manager = RemoteModelManager(remote_root="/remote", runner=FakeRunner(catalog_payload()))
    with pytest.raises(ModelManagerError, match="catalog"):
        manager.start("ur_pick_place_push", "not-a-run")


def test_server_switch_changes_both_lifecycle_and_policy_connection():
    runner = FakeRunner(catalog_payload())
    profiles = {
        "baai": ServerProfile(
            "baai", "BAAI", "BAAI", "/remote/baai", "/assets/baai",
            "/remote/baai/scripts/deploy/task_config",
        ),
        "ps": ServerProfile(
            "ps", "PS", "ps", "/remote/ps", "/assets/ps",
            "/remote/ps/scripts/deploy/task_config",
        ),
    }
    manager = RemoteModelManager(remote_root="/remote",
        profiles=profiles,
        default_server="baai",
        runner=runner,
        startup_timeout_s=2.0,
    )

    state = manager.select_server("ps")

    assert state["server_id"] == "ps"
    assert state["ssh_host"] == "ps"
    assert state["repo_root"] == "/remote/ps"
    assert state["server_url"] == "ws://127.0.0.1:5500/infer"
    manager.catalog()
    assert runner.commands[-1][0][-2] == "ps"


def test_checked_in_profiles_define_baai_and_ps_code_and_asset_roots():
    path = Path(__file__).resolve().parents[4] / "config" / "model_servers.json"

    profiles, default = load_server_profiles(path)

    assert default == "baai"
    assert set(profiles) == {'baai','4090'}
    assert profiles['baai'].repo_root == profiles['4090'].repo_root == ''
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(remote_root="/remote", profile_path=path, runner=runner)
    assert manager.catalog()['tasks'] == []
    manager.select_server('4090')
    assert manager.snapshot()['ssh_host'] == 'ps'
    assert manager.catalog()['tasks'] == []
    manager.stop()
    assert runner.commands == []


def test_busy_port_is_rejected_without_killing_unowned_process():
    runner = FakeRunner(catalog_payload())
    runner.in_use_ports.add(5500)
    manager = RemoteModelManager(remote_root="/remote", runner=runner, startup_timeout_s=2.0)
    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "fault")
    assert runner.launch_count == 0
    assert not any('kill' in command[-1] for command, _ in runner.commands)
    assert 5500 in runner.in_use_ports


def test_launch_does_not_patch_method_specific_configuration():
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(remote_root="/remote", runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    launch = next(
        command[-1] for command, _ in runner.commands if "tmux new-session" in command[-1]
    )
    assert "bash scripts/deploy/groot_n17/launch.sh ur_pick_place_push full-run-id" in launch
    assert "exec env" not in launch
    assert "sed -e" not in launch


def test_failed_launch_reports_fault_and_cleans_owned_session():
    class FailFirstRunner(FakeRunner):
        def __call__(self, command, timeout_s):
            result = super().__call__(command, timeout_s)
            if command[-1].startswith("tmux display-message") and self.launch_count == 1:
                return subprocess.CompletedProcess(command, 0, "1\n", "")
            return result

    runner = FailFirstRunner(catalog_payload())
    manager = RemoteModelManager(remote_root="/remote", runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "fault", timeout_s=4.0)

    assert runner.launch_count == 1
    assert manager.snapshot()["port"] == 5500


def test_loading_log_is_sanitized_and_kept_in_snapshot():
    class LogRunner(FakeRunner):
        def __call__(self, command, timeout_s):
            if command[-1].startswith("tmux capture-pane"):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "\x1b[31mLoading checkpoint\x1b[0m\r\nshard 3/3\n",
                    "",
                )
            return super().__call__(command, timeout_s)

    manager = RemoteModelManager(remote_root="/remote",
        runner=LogRunner(catalog_payload()), startup_timeout_s=2.0
    )

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    assert manager.snapshot()["log_tail"] == "Loading checkpoint\nshard 3/3"
    assert "\x1b" not in manager.snapshot()["log_tail"]


def test_stop_kills_managed_server_and_releases_fixed_port():
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(remote_root="/remote", runner=runner, startup_timeout_s=2.0)
    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    manager.stop()
    wait_phase(manager, "stopped")

    assert manager.snapshot()["port"] == 0
    cleanup = [command[-1] for command, _ in runner.commands if command[-1].startswith("tmux kill-session")]
    assert cleanup
    assert manager.session_name in cleanup[-1]
    assert not any("pgrep" in command[-1] for command, _ in runner.commands)


def test_unconfigured_default_does_not_contact_remote_host():
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(runner=runner)
    assert manager.snapshot()['configured'] is False
    assert manager.catalog()['tasks'] == []
    with pytest.raises(ModelManagerError, match='catalog'):
        manager.start('ur_pick_place_push', 'full-run-id')
    manager.close()
    assert runner.commands == []


def test_managers_use_distinct_owned_sessions():
    runner = FakeRunner(catalog_payload())
    first = RemoteModelManager(remote_root='/remote', runner=runner)
    second = RemoteModelManager(remote_root='/remote', runner=runner)
    assert first.session_name != second.session_name
    second.close()
    assert not runner.commands


def test_stop_during_launch_waits_then_cleans_only_owned_session():
    import threading

    entered = threading.Event()
    release = threading.Event()

    class SlowLaunchRunner(FakeRunner):
        def __call__(self, command, timeout_s):
            result = super().__call__(command, timeout_s)
            if 'tmux new-session' in command[-1]:
                entered.set()
                assert release.wait(2.0)
            return result

    runner = SlowLaunchRunner(catalog_payload())
    manager = RemoteModelManager(remote_root='/remote', runner=runner, startup_timeout_s=2.0)
    manager.start('ur_pick_place_push', 'full-run-id')
    assert entered.wait(1.0)
    assert manager.stop()['phase'] == 'stopping'
    with pytest.raises(ModelManagerError, match='停止'):
        manager.select_server('default')
    release.set()
    wait_phase(manager, 'stopped')
    assert runner.launched_port is None
    cleanup = [command[-1] for command, _ in runner.commands if 'kill-session' in command[-1]]
    assert len(cleanup) == 1
    assert manager.session_name in cleanup[0]
    assert not any('pgrep' in command[-1] for command, _ in runner.commands)
    manager.close()


def test_closed_manager_rejects_new_start_or_server_switch():
    manager = RemoteModelManager(remote_root='/remote', runner=FakeRunner(catalog_payload()))
    manager.close()
    with pytest.raises(ModelManagerError, match='关闭'):
        manager.start('ur_pick_place_push', 'full-run-id')
    with pytest.raises(ModelManagerError, match='关闭'):
        manager.select_server('default')


def test_delayed_notification_cannot_regress_model_status():
    events = []
    manager = RemoteModelManager(remote_root='/remote', runner=FakeRunner(catalog_payload()),
                                 event_callback=events.append, startup_timeout_s=2.0)
    starting = manager.start('ur_pick_place_push', 'full-run-id')
    wait_phase(manager, 'running')
    manager._notify(starting)
    assert events[-1]['phase'] == 'running'
    manager.stop()
    wait_phase(manager, 'stopped')
    manager._notify(starting)
    assert events[-1]['phase'] == 'stopped'
    manager.close()
