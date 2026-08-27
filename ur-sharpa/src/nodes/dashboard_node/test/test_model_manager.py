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
    manager = RemoteModelManager(
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
    manager = RemoteModelManager(runner=FakeRunner(catalog_payload(launchable=False)))
    with pytest.raises(ModelManagerError, match="checkpoint"):
        manager.start("ur_pick_place_push", "full-run-id")

    manager = RemoteModelManager(runner=FakeRunner(catalog_payload()))
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
    manager = RemoteModelManager(
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
    assert profiles["baai"].repo_root == "/share/project/yenaisheng/dreamzero"
    assert profiles["ps"].repo_root == "/home/ps/dexterity/repos/dexhand"
    assert profiles["ps"].asset_root == "/home/ps/dexterity/assets"
    assert profiles["ps"].ssh_host == "ps"


def test_start_always_reuses_fixed_default_port():
    runner = FakeRunner(catalog_payload())
    runner.in_use_ports.add(5500)
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    assert manager.snapshot()["port"] == 5500
    launch = next(
        command[-1] for command, _ in runner.commands if "tmux new-session" in command[-1]
    )
    assert "exec env" not in launch


def test_launch_does_not_patch_method_specific_configuration():
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    launch = next(
        command[-1] for command, _ in runner.commands if "tmux new-session" in command[-1]
    )
    assert "bash scripts/deploy/groot_n17/launch.sh ur_pick_place_push full-run-id" in launch
    assert "exec env" not in launch
    assert "sed -e" not in launch


def test_start_retries_once_on_the_same_fixed_port_after_process_failure():
    class FailFirstRunner(FakeRunner):
        def __call__(self, command, timeout_s):
            result = super().__call__(command, timeout_s)
            if command[-1].startswith("tmux display-message") and self.launch_count == 1:
                return subprocess.CompletedProcess(command, 0, "1\n", "")
            return result

    runner = FailFirstRunner(catalog_payload())
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running", timeout_s=4.0)

    assert runner.launch_count == 2
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

    manager = RemoteModelManager(
        runner=LogRunner(catalog_payload()), startup_timeout_s=2.0
    )

    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    assert manager.snapshot()["log_tail"] == "Loading checkpoint\nshard 3/3"
    assert "\x1b" not in manager.snapshot()["log_tail"]


def test_stop_kills_managed_server_and_releases_fixed_port():
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)
    manager.start("ur_pick_place_push", "full-run-id")
    wait_phase(manager, "running")

    manager.stop()
    wait_phase(manager, "stopped")

    assert manager.snapshot()["port"] == 0
    cleanup = [command[-1] for command, _ in runner.commands if command[-1].startswith("set +e")]
    assert cleanup
    assert "--port 5500" in cleanup[-1]
