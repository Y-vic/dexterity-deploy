"""Safe remote lifecycle management for the dashboard policy server."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Callable


REMOTE_ROOT = "/share/project/yenaisheng/sharpa-deploy"
TMUX_SESSION = "ur_sharpa_policy_server"
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
LOG_TAIL_LINES = 120
LOG_TAIL_CHARACTERS = 12000


class ModelManagerError(RuntimeError):
    """Raised when a requested model operation is unsafe or unavailable."""


@dataclass(frozen=True)
class ServerProfile:
    server_id: str
    label: str
    ssh_host: str
    repo_root: str
    asset_root: str
    task_config_root: str
    remote_host: str = "127.0.0.1"
    port: int = 5500
    env_file: str = ""

    @classmethod
    def from_mapping(cls, server_id: str, value: Any) -> "ServerProfile":
        if not SAFE_ID.fullmatch(server_id) or not isinstance(value, dict):
            raise ModelManagerError(f"无效的模型 server profile: {server_id!r}")
        repo_root = str(value.get("repo_root") or "").strip()
        ssh_host = str(value.get("ssh_host") or "").strip()
        if not repo_root.startswith("/") or not ssh_host:
            raise ModelManagerError(f"server profile {server_id!r} 缺少 ssh_host/repo_root")
        port = int(value.get("port", 5500))
        if not 1 <= port <= 65535:
            raise ModelManagerError(f"server profile {server_id!r} 端口无效")
        return cls(
            server_id=server_id,
            label=str(value.get("label") or server_id),
            ssh_host=ssh_host,
            repo_root=repo_root,
            asset_root=str(value.get("asset_root") or repo_root),
            task_config_root=str(
                value.get("task_config_root")
                or f"{repo_root}/scripts/deploy/task_config"
            ),
            remote_host=str(value.get("remote_host") or "127.0.0.1"),
            port=port,
            env_file=str(value.get("env_file") or ""),
        )

    def public(self) -> dict[str, Any]:
        return {
            "id": self.server_id,
            "label": self.label,
            "ssh_host": self.ssh_host,
            "repo_root": self.repo_root,
            "asset_root": self.asset_root,
            "remote_host": self.remote_host,
            "port": self.port,
        }


def load_server_profiles(path: str | Path) -> tuple[dict[str, ServerProfile], str]:
    source = Path(path).expanduser().absolute()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManagerError(f"无法读取模型 server 配置: {source}") from exc
    if payload.get("schema") != "ur_sharpa.model_servers.v1":
        raise ModelManagerError(f"模型 server 配置 schema 错误: {source}")
    raw_servers = payload.get("servers")
    if not isinstance(raw_servers, dict) or not raw_servers:
        raise ModelManagerError("模型 server 配置没有 servers")
    profiles = {
        server_id: ServerProfile.from_mapping(server_id, value)
        for server_id, value in raw_servers.items()
    }
    default = str(payload.get("default") or "")
    if default not in profiles:
        raise ModelManagerError(f"默认模型 server {default!r} 不存在")
    return profiles, default


def run_command(command: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


class RemoteModelManager:
    def __init__(
        self,
        *,
        ssh_host: str = "BAAI",
        remote_root: str = REMOTE_ROOT,
        session_name: str = TMUX_SESSION,
        runner: Callable[[list[str], float], subprocess.CompletedProcess[str]] = run_command,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        startup_timeout_s: float = 180.0,
        base_port: int = 5500,
        port_candidates: int = 16,
        profile_path: str | Path | None = None,
        profiles: dict[str, ServerProfile] | None = None,
        default_server: str | None = None,
    ) -> None:
        if profiles is None and profile_path is not None:
            profiles, configured_default = load_server_profiles(profile_path)
            default_server = default_server or configured_default
        if profiles is None:
            profiles = {
                "default": ServerProfile(
                    server_id="default",
                    label="Default",
                    ssh_host=ssh_host,
                    repo_root=remote_root,
                    asset_root=remote_root,
                    task_config_root=f"{remote_root}/scripts/deploy/task_config",
                    port=int(base_port),
                )
            }
            default_server = "default"
        if default_server not in profiles:
            raise ModelManagerError(f"模型 server {default_server!r} 不存在")
        self._profiles = dict(profiles)
        self._server_id = str(default_server)
        self.session_name = session_name
        self.runner = runner
        self.event_callback = event_callback
        self.startup_timeout_s = startup_timeout_s
        self.base_port = int(self._profiles[self._server_id].port)
        self.port_candidates = int(port_candidates)
        self._lock = threading.Lock()
        self._generation = 0
        self._cancel = threading.Event()
        self._state: dict[str, Any] = {
            "phase": "stopped",
            "generation": 0,
            "task": "",
            "task_id": "",
            "model": "",
            "run_id": "",
            "prompt": "",
            "port": 0,
            "detail": "",
            "log_tail": "",
            "updated_at": time.time(),
            **self._connection_state(self._profiles[self._server_id]),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    @staticmethod
    def _connection_state(profile: ServerProfile) -> dict[str, Any]:
        return {
            "server_id": profile.server_id,
            "server_label": profile.label,
            "ssh_host": profile.ssh_host,
            "repo_root": profile.repo_root,
            "asset_root": profile.asset_root,
            "remote_host": profile.remote_host,
            "server_url": f"ws://127.0.0.1:{profile.port}/infer",
        }

    def servers(self) -> dict[str, Any]:
        with self._lock:
            selected = self._server_id
        return {
            "selected": selected,
            "servers": [profile.public() for profile in self._profiles.values()],
        }

    def select_server(self, server_id: str) -> dict[str, Any]:
        with self._lock:
            if server_id not in self._profiles:
                raise ModelManagerError(f"未知模型 server: {server_id}")
            if self._state["phase"] not in {"stopped", "fault"}:
                raise ModelManagerError("请先停止当前模型再切换 server")
            self._server_id = server_id
            profile = self._profiles[server_id]
            self.base_port = profile.port
            self._generation += 1
            self._state.update(
                phase="stopped",
                generation=self._generation,
                task="",
                task_id="",
                model="",
                run_id="",
                prompt="",
                port=0,
                detail=f"已切换到 {profile.label}",
                log_tail="",
                updated_at=time.time(),
                **self._connection_state(profile),
            )
            state = deepcopy(self._state)
        self._notify(state)
        return state

    def _active_profile(self) -> ServerProfile:
        with self._lock:
            return self._profiles[self._server_id]

    def catalog(self) -> dict[str, Any]:
        script = """
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).absolute()
asset_root = Path(sys.argv[2]).absolute()
config_root = Path(sys.argv[3]).absolute()
models = sorted(
    path.parent.name
    for path in (root / "scripts/deploy").glob("*/launch.sh")
    if path.is_file() and path.parent.name != "task_config"
)
tasks = []
for path in sorted(config_root.glob("*/task.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    task_name = path.parent.name
    compatible = task_name.startswith("ur_")
    runs = []
    entries = data.get("runs") or []
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        model = str(entry.get("model") or "")
        launcher_exists = model in models
        checkpoint = str(entry.get("checkpoint") or "")
        checkpoint_path = str((asset_root / checkpoint).resolve()) if checkpoint else ""
        checkpoint_exists = bool(checkpoint_path) and Path(checkpoint_path).is_dir()
        eligible = bool((entry.get("deployment") or {}).get("eligible", False))
        status = str(entry.get("status") or "")
        reasons = []
        if not compatible:
            reasons.append("不是 UR task")
        if not model:
            reasons.append("run.model 缺失")
        if not eligible:
            reasons.append(str((entry.get("deployment") or {}).get("reason") or "未标记为可部署"))
        if status not in {"complete", "in_progress"}:
            reasons.append(f"run status={status or '<missing>'}")
        if not launcher_exists:
            reasons.append(f"缺少 scripts/deploy/{model}/launch.sh")
        if not checkpoint_exists:
            reasons.append("checkpoint 不存在")
        runs.append({
            "model": model,
            "id": str(entry.get("id") or ""),
            "run_id": str(entry.get("run_id") or ""),
            "checkpoint_path": checkpoint_path,
            "variant": str(entry.get("variant") or ""),
            "stage": str(entry.get("stage") or ""),
            "status": status,
            "eligible": eligible,
            "checkpoint_exists": checkpoint_exists,
            "launchable": not reasons,
            "disabled_reason": "；".join(reasons),
        })
    tasks.append({
        "name": task_name,
        "task_id": str((data.get("task") or {}).get("id") or task_name),
        "prompt": str((data.get("task") or {}).get("prompt") or ""),
        "dataset_path": str((asset_root / str((data.get("task") or {}).get("dataset") or "")).resolve()),
        "compatible": compatible,
        "runs": runs,
    })
print(json.dumps({"models": models, "tasks": tasks}, ensure_ascii=False, separators=(",", ":")))
"""
        profile = self._active_profile()
        remote_command = "python3 -c {} {} {} {}".format(
            shlex.quote(script),
            shlex.quote(profile.repo_root),
            shlex.quote(profile.asset_root),
            shlex.quote(profile.task_config_root),
        )
        result = self._ssh(remote_command, timeout_s=20.0)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ModelManagerError("远端 task catalog 不是有效 JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            raise ModelManagerError("远端 task catalog 格式错误")
        payload["server"] = profile.public()
        return payload

    def start(self, task: str, run_id: str) -> dict[str, Any]:
        selected = self._find_run(task, run_id)
        if not selected["run"]["launchable"]:
            raise ModelManagerError(selected["run"]["disabled_reason"] or "该模型不可部署")
        with self._lock:
            if self._state["phase"] in {
                "starting",
                "loading",
                "running",
                "stopping",
            }:
                raise ModelManagerError("请先停止当前模型")
            self._generation += 1
            generation = self._generation
            self._cancel = threading.Event()
            self._state.update(
                phase="starting",
                generation=generation,
                task=task,
                task_id=selected["task"]["task_id"],
                model=selected["run"]["model"],
                run_id=run_id,
                prompt=selected["task"]["prompt"],
                port=0,
                detail="正在创建远端模型进程",
                log_tail="",
                updated_at=time.time(),
            )
            state = deepcopy(self._state)
        self._notify(state)
        launch_run = deepcopy(selected["run"])
        launch_run.update(
            task=task,
            task_id=selected["task"]["task_id"],
            prompt=selected["task"]["prompt"],
            dataset_path=selected["task"]["dataset_path"],
        )
        threading.Thread(
            target=self._start_worker,
            args=(generation, launch_run, self._cancel),
            daemon=True,
            name="remote-model-start",
        ).start()
        return state

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._state["phase"] == "stopping":
                return deepcopy(self._state)
            self._generation += 1
            generation = self._generation
            self._cancel.set()
            self._state.update(
                phase="stopping",
                generation=generation,
                detail="正在停止远端模型",
                updated_at=time.time(),
            )
            state = deepcopy(self._state)
        self._notify(state)
        threading.Thread(
            target=self._stop_worker,
            args=(generation,),
            daemon=True,
            name="remote-model-stop",
        ).start()
        return state

    def _find_run(self, task: str, run_id: str) -> dict[str, Any]:
        catalog = self.catalog()
        for task_entry in catalog["tasks"]:
            if task_entry["name"] != task:
                continue
            for run in task_entry["runs"]:
                if run["run_id"] == run_id:
                    return {"task": task_entry, "run": run}
        raise ModelManagerError("所选 task / run 不在远端 catalog 中")

    def _start_worker(
        self,
        generation: int,
        run: dict[str, Any],
        cancel: threading.Event,
    ) -> None:
        try:
            failures = []
            for attempt in range(2):
                port = self.base_port
                try:
                    self._cleanup_remote_model(port)
                    self._launch_remote(run, port)
                    if cancel.is_set():
                        self._stop_remote_session()
                        return
                    self._transition(
                        generation,
                        "loading",
                        f"正在端口 {port} 加载 {run['run_id']} checkpoint",
                        port=port,
                    )
                    deadline = time.monotonic() + self.startup_timeout_s
                    while time.monotonic() < deadline:
                        if cancel.wait(1.0):
                            self._stop_remote_session()
                            return
                        log_tail = self._captured_log()
                        self._update_log(generation, log_tail)
                        if self._pane_dead():
                            raise ModelManagerError(
                                self._error_log(log_tail) or "远端模型进程已退出"
                            )
                        if self._healthy(port):
                            self._transition(
                                generation,
                                "running",
                                f"模型服务已在端口 {port} 就绪",
                                port=port,
                            )
                            return
                    raise ModelManagerError("等待远端模型健康检查超时")
                except Exception as exc:
                    failures.append(str(exc))
                    try:
                        self._cleanup_remote_model(port)
                    except Exception as cleanup_exc:
                        failures.append(f"清理失败：{cleanup_exc}")
                    if cancel.is_set():
                        return
                    if attempt == 0:
                        self._transition(
                            generation,
                            "starting",
                            f"首次启动失败，正在自动换端口重试：{exc}",
                            port=0,
                        )
                        continue
                    raise ModelManagerError(
                        "模型自动重试失败：" + "；".join(failures)
                    ) from exc
        except Exception as exc:
            if not cancel.is_set():
                self._transition(generation, "fault", str(exc))

    def _launch_remote(self, run: dict[str, Any], port: int) -> None:
        profile = self._active_profile()
        launch_script = f"scripts/deploy/{run['model']}/launch.sh"
        launcher_command = "bash {} {} {}".format(
            shlex.quote(launch_script),
            shlex.quote(run["task"]),
            shlex.quote(run["run_id"]),
        )
        setup = []
        if profile.env_file:
            setup.append(
                "set -a && . {} && set +a".format(shlex.quote(profile.env_file))
            )
        setup.extend(
            [
                "export SHARPA_DEPLOY_ASSET_ROOT={}".format(
                    shlex.quote(profile.asset_root)
                ),
                "export SHARPA_DEPLOY_TASK_CONFIG_ROOT={}".format(
                    shlex.quote(profile.task_config_root)
                ),
                "export SHARPA_DEPLOY_PORT={}".format(int(port)),
                "exec {}".format(launcher_command),
            ]
        )
        inner = " && ".join(setup)
        remote_command = " && ".join(
            [
                "tmux new-session -d -s {} -c {} {}".format(
                    shlex.quote(self.session_name),
                    shlex.quote(profile.repo_root),
                    shlex.quote(inner),
                ),
                "tmux set-option -t {} remain-on-exit on".format(
                    shlex.quote(self.session_name)
                ),
                "tmux set-option -t {} history-limit 5000".format(
                    shlex.quote(self.session_name)
                ),
            ]
        )
        self._ssh(remote_command, timeout_s=15.0)

    def _select_available_port(self, excluded: set[int]) -> int:
        return self.base_port

    def _port_in_use(self, port: int) -> bool:
        result = self._ssh_raw(
            f"(exec 3<>/dev/tcp/127.0.0.1/{int(port)}) >/dev/null 2>&1",
            timeout_s=8.0,
        )
        return result.returncode == 0

    def _stop_worker(self, generation: int) -> None:
        try:
            self._cleanup_remote_model(self.base_port)
            self._transition(generation, "stopped", "模型已停止", port=0)
        except Exception as exc:
            self._transition(generation, "fault", str(exc))

    def _cleanup_remote_model(self, port: int) -> None:
        remote_command = """
set +e
if tmux has-session -t {session} 2>/dev/null; then
  tmux send-keys -t {session}:0.0 C-c
  i=0
  while [ "$i" -lt 20 ]; do
    [ "$(tmux display-message -p -t {session}:0.0 '#{{pane_dead}}' 2>/dev/null)" = "1" ] && break
    sleep 0.25
    i=$((i + 1))
  done
  tmux kill-session -t {session} 2>/dev/null || true
fi
pattern='[d]exterity\\.deploy\\..*\\.server.*--port {port}([[:space:]]|$)'
pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
for pid in $pids; do
  kill -TERM "$pid" 2>/dev/null || true
done
i=0
while [ "$i" -lt 40 ]; do
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  [ -z "$pids" ] && break
  sleep 0.25
  i=$((i + 1))
done
pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
for pid in $pids; do
  kill -KILL "$pid" 2>/dev/null || true
done
i=0
while [ "$i" -lt 40 ]; do
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  port_open=0
  if (exec 3<>/dev/tcp/127.0.0.1/{port}) >/dev/null 2>&1; then
    port_open=1
  fi
  [ -z "$pids" ] && [ "$port_open" -eq 0 ] && exit 0
  sleep 0.25
  i=$((i + 1))
done
pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
[ -z "$pids" ] || exit 42
if (exec 3<>/dev/tcp/127.0.0.1/{port}) >/dev/null 2>&1; then
  exit 43
fi
exit 0
        """.format(session=shlex.quote(self.session_name), port=int(port)).lstrip()
        self._ssh(remote_command, timeout_s=12.0)

    def _stop_remote_session(self) -> None:
        self._cleanup_remote_model(self.base_port)

    def _session_exists(self) -> bool:
        result = self._ssh_raw(
            "tmux has-session -t {} 2>/dev/null".format(
                shlex.quote(self.session_name)
            ),
            timeout_s=8.0,
        )
        return result.returncode == 0

    def _pane_dead(self) -> bool:
        result = self._ssh_raw(
            "tmux display-message -p -t {}:0.0 '#{{pane_dead}}' 2>/dev/null".format(
                shlex.quote(self.session_name)
            ),
            timeout_s=8.0,
        )
        return result.returncode != 0 or result.stdout.strip() == "1"

    def _healthy(self, port: int) -> bool:
        result = self._ssh_raw(
            f"curl -fsS --max-time 2 http://127.0.0.1:{int(port)}/healthz >/dev/null",
            timeout_s=8.0,
        )
        return result.returncode == 0

    def _captured_log(self) -> str:
        result = self._ssh_raw(
            "tmux capture-pane -pJ -t {}:0.0 -S -160 2>/dev/null".format(
                shlex.quote(self.session_name)
            ),
            timeout_s=8.0,
        )
        clean = ANSI_ESCAPE.sub("", result.stdout.replace("\r\n", "\n").replace("\r", "\n"))
        lines = [
            CONTROL_CHARACTER.sub("", line).rstrip()
            for line in clean.splitlines()
        ]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines[-LOG_TAIL_LINES:])[-LOG_TAIL_CHARACTERS:]

    @staticmethod
    def _error_log(log_tail: str) -> str:
        lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
        return " | ".join(lines[-8:])[-1200:]

    def _update_log(self, generation: int, log_tail: str) -> None:
        if not log_tail:
            return
        with self._lock:
            if generation != self._generation or log_tail == self._state["log_tail"]:
                return
            self._state["log_tail"] = log_tail
            state = deepcopy(self._state)
        self._notify(state)

    def _transition(
        self,
        generation: int,
        phase: str,
        detail: str,
        **values: Any,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._state.update(
                phase=phase,
                detail=detail,
                updated_at=time.time(),
                **values,
            )
            state = deepcopy(self._state)
        self._notify(state)

    def _notify(self, state: dict[str, Any]) -> None:
        if self.event_callback is not None:
            try:
                self.event_callback(deepcopy(state))
            except Exception:
                pass

    def _ssh(self, remote_command: str, *, timeout_s: float) -> subprocess.CompletedProcess[str]:
        result = self._ssh_raw(remote_command, timeout_s=timeout_s)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ModelManagerError(detail or f"SSH 命令退出码 {result.returncode}")
        return result

    def _ssh_raw(
        self, remote_command: str, *, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        profile = self._active_profile()
        command = [
            "/usr/bin/ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForwardX11=no",
            "-o",
            "ConnectTimeout=8",
            profile.ssh_host,
            remote_command,
        ]
        try:
            return self.runner(command, timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ModelManagerError("SSH 操作超时") from exc
