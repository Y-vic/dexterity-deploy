import json
import re
import subprocess
import time

import pytest

from dashboard_node.model_manager import ModelManagerError, RemoteModelManager


def catalog_payload(*, launchable=True):
    return {
        "tasks": [
            {
                "name": "ur_pick_place_push",
                "task_id": "ur_pick_place_push",
                "prompt": "pick",
                "dataset_path": "/remote/dataset",
                "compatible": True,
                "runs": [
                    {
                        "family": "groot",
                        "launcher": "groot_n17",
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
                        "selected": True,
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
            port = int(re.search(r"DZ_DEPLOY_PORT=(\d+)", remote).group(1))
            self.launched_port = port
            self.in_use_ports.add(port)
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

    state = manager.start("ur_pick_place_push", "groot", "groot_n17_baseline")
    assert state["phase"] == "starting"
    wait_phase(manager, "running")

    remote_commands = [command[-1] for command, _ in runner.commands]
    launch = next(command for command in remote_commands if "tmux new-session" in command)
    assert "scripts/deploy/groot_n17/launch.sh" in launch
    assert "ur_pick_place_push" in launch
    assert "groot_n17_baseline" in launch
    assert manager.snapshot()["phase"] == "running"
    assert manager.snapshot()["port"] == 5500
    assert [event["phase"] for event in events] == [
        "starting",
        "loading",
        "running",
    ]
    assert manager.snapshot()["task_id"] == "ur_pick_place_push"
    assert manager.snapshot()["checkpoint_run_id"] == "full-run-id"


def test_start_rejects_unlaunchable_and_unknown_runs():
    manager = RemoteModelManager(runner=FakeRunner(catalog_payload(launchable=False)))
    with pytest.raises(ModelManagerError, match="checkpoint"):
        manager.start("ur_pick_place_push", "groot", "groot_n17_baseline")

    manager = RemoteModelManager(runner=FakeRunner(catalog_payload()))
    with pytest.raises(ModelManagerError, match="catalog"):
        manager.start("ur_pick_place_push", "groot", "not-a-run")


def test_start_always_reuses_fixed_default_port():
    runner = FakeRunner(catalog_payload())
    runner.in_use_ports.add(5500)
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "groot", "groot_n17_baseline")
    wait_phase(manager, "running")

    assert manager.snapshot()["port"] == 5500
    launch = next(
        command[-1] for command, _ in runner.commands if "tmux new-session" in command[-1]
    )
    assert "DZ_DEPLOY_PORT=5500" in launch


def test_groot_n17_family_uses_compatible_config_key_without_remote_edit():
    payload = catalog_payload()
    payload["tasks"][0]["runs"][0]["family"] = "groot_n17"
    runner = FakeRunner(payload)
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "groot_n17", "groot_n17_baseline")
    wait_phase(manager, "running")

    launch = next(
        command[-1] for command, _ in runner.commands if "tmux new-session" in command[-1]
    )
    assert "bash -c" in launch
    assert "DZ_DEPLOY_TASK=ur_pick_place_push" in launch
    assert "DZ_DEPLOY_RUN_ID=full-run-id" in launch
    assert "DZ_DEPLOY_PROMPT=pick" in launch
    assert "DZ_DEPLOY_CHECKPOINT=/remote/checkpoint" in launch
    assert "s|^dz_load_task_config groot .*|:|" in launch
    assert "ROOT=/share/project/yenaisheng/dreamzero" in launch


def test_start_retries_once_on_the_same_fixed_port_after_process_failure():
    class FailFirstRunner(FakeRunner):
        def __call__(self, command, timeout_s):
            result = super().__call__(command, timeout_s)
            if command[-1].startswith("tmux display-message") and self.launch_count == 1:
                return subprocess.CompletedProcess(command, 0, "1\n", "")
            return result

    runner = FailFirstRunner(catalog_payload())
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)

    manager.start("ur_pick_place_push", "groot", "groot_n17_baseline")
    wait_phase(manager, "running", timeout_s=4.0)

    assert runner.launch_count == 2
    assert manager.snapshot()["port"] == 5500


def test_stop_kills_managed_server_and_releases_fixed_port():
    runner = FakeRunner(catalog_payload())
    manager = RemoteModelManager(runner=runner, startup_timeout_s=2.0)
    manager.start("ur_pick_place_push", "groot", "groot_n17_baseline")
    wait_phase(manager, "running")

    manager.stop()
    wait_phase(manager, "stopped")

    assert manager.snapshot()["port"] == 0
    cleanup = [command[-1] for command, _ in runner.commands if command[-1].startswith("set +e")]
    assert cleanup
    assert "--port 5500" in cleanup[-1]
