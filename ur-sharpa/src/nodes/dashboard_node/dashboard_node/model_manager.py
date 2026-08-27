"""Safe remote lifecycle management for the dashboard policy server."""

from __future__ import annotations

from copy import deepcopy
import json
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Callable


REMOTE_ROOT = "/share/project/yenaisheng/sharpa-deploy"
TMUX_SESSION = "ur_sharpa_policy_server"
FAMILY_LAUNCHERS = {
    "cgp": "cgp",
    "gcc": "gcc",
    "groot": "groot_n17",
    "groot_n17": "groot_n17",
    "pace": "pace",
    "pi05": "pi05",
    "t_rex": "t_rex",
    "trex": "t_rex",
    "vitacformer": "vitacformer",
}
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LOG_TAIL_LINES = 120
LOG_TAIL_CHARACTERS = 12000


class ModelManagerError(RuntimeError):
    """Raised when a requested model operation is unsafe or unavailable."""


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
    ) -> None:
        self.ssh_host = ssh_host
        self.remote_root = remote_root
        self.session_name = session_name
        self.runner = runner
        self.event_callback = event_callback
        self.startup_timeout_s = startup_timeout_s
        self.base_port = int(base_port)
        self.port_candidates = int(port_candidates)
        self._lock = threading.Lock()
        self._generation = 0
        self._cancel = threading.Event()
        self._state: dict[str, Any] = {
            "phase": "stopped",
            "generation": 0,
            "task": "",
            "task_id": "",
            "family": "",
            "run_id": "",
            "checkpoint_run_id": "",
            "prompt": "",
            "port": 0,
            "detail": "",
            "log_tail": "",
            "updated_at": time.time(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def catalog(self) -> dict[str, Any]:
        launchers = json.dumps(FAMILY_LAUNCHERS, separators=(",", ":"))
        script = """
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).absolute()
launchers = json.loads(sys.argv[2])
tasks = []
for path in sorted((root / "scripts/deploy/task_config").glob("*/task.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    task_name = path.parent.name
    compatible = task_name.startswith("ur_")
    runs = []
    for family, entries in sorted((data.get("runs") or {}).items()):
        launcher = launchers.get(family, "")
        launcher_exists = bool(launcher) and (root / "scripts/deploy" / launcher / "launch.sh").is_file()
        for entry in entries or []:
            checkpoint = str(entry.get("checkpoint") or "")
            checkpoint_path = str((root / checkpoint).resolve()) if checkpoint else ""
            checkpoint_exists = bool(checkpoint_path) and Path(checkpoint_path).exists()
            eligible = bool((entry.get("deployment") or {}).get("eligible", False))
            reasons = []
            if not compatible:
                reasons.append("不是 UR task")
            if not eligible:
                reasons.append(str((entry.get("deployment") or {}).get("reason") or "未标记为可部署"))
            if not launcher_exists:
                reasons.append("缺少启动脚本")
            if not checkpoint_exists:
                reasons.append("checkpoint 不存在")
            runs.append({
                "family": family,
                "launcher": launcher,
                "id": str(entry.get("id") or ""),
                "run_id": str(entry.get("run_id") or ""),
                "checkpoint_path": checkpoint_path,
                "variant": str(entry.get("variant") or ""),
                "stage": str(entry.get("stage") or ""),
                "status": str(entry.get("status") or ""),
                "eligible": eligible,
                "checkpoint_exists": checkpoint_exists,
                "launchable": not reasons,
                "disabled_reason": "；".join(reasons),
                "selected": (data.get("selected") or {}).get(family) == entry.get("id"),
            })
    tasks.append({
        "name": task_name,
        "task_id": str((data.get("task") or {}).get("id") or task_name),
        "prompt": str((data.get("task") or {}).get("prompt") or ""),
        "dataset_path": str((root / str((data.get("task") or {}).get("dataset") or "")).resolve()),
        "compatible": compatible,
        "runs": runs,
    })
print(json.dumps({"tasks": tasks}, ensure_ascii=False, separators=(",", ":")))
"""
        remote_command = "python3 -c {} {} {}".format(
            shlex.quote(script),
            shlex.quote(self.remote_root),
            shlex.quote(launchers),
        )
        result = self._ssh(remote_command, timeout_s=20.0)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ModelManagerError("远端 task catalog 不是有效 JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            raise ModelManagerError("远端 task catalog 格式错误")
        return payload

    def start(self, task: str, family: str, run_id: str) -> dict[str, Any]:
        selected = self._find_run(task, family, run_id)
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
                family=family,
                run_id=run_id,
                checkpoint_run_id=selected["run"]["run_id"],
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

    def _find_run(self, task: str, family: str, run_id: str) -> dict[str, Any]:
        catalog = self.catalog()
        for task_entry in catalog["tasks"]:
            if task_entry["name"] != task:
                continue
            for run in task_entry["runs"]:
                if run["family"] == family and run["id"] == run_id:
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
                        f"正在端口 {port} 加载 {run['id']} checkpoint",
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
        launch_script = f"scripts/deploy/{run['launcher']}/launch.sh"
        launcher_command = "bash {} {} {}".format(
            shlex.quote(launch_script),
            shlex.quote(run["task"]),
            shlex.quote(run["id"]),
        )
        inner = "exec {}".format(launcher_command)
        remote_command = " && ".join(
            [
                "tmux new-session -d -s {} -c {} {}".format(
                    shlex.quote(self.session_name),
                    shlex.quote(self.remote_root),
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
        command = [
            "/usr/bin/ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForwardX11=no",
            "-o",
            "ConnectTimeout=8",
            self.ssh_host,
            remote_command,
        ]
        try:
            return self.runner(command, timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ModelManagerError("SSH 操作超时") from exc
