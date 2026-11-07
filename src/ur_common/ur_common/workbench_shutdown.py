"""Exit this workbench's ROS launch, including surviving child processes.

The detached watcher outlives the HTTP node. Linux process start times prevent
signals from reaching unrelated processes after a PID has been reused.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    started: int
    arguments: tuple[str, ...]


def read_process(pid: int) -> Process | None:
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        arguments = tuple(os.fsdecode(value) for value in
                          (directory / 'cmdline').read_bytes().split(b'\0') if value)
        return Process(pid, int(fields[1]), int(fields[19]), arguments)
    except (OSError, ValueError, IndexError):
        return None


def is_workbench_launch(process: Process) -> bool:
    args = process.arguments
    return any(
        Path(args[index]).name == 'ros2'
        and args[index + 1:index + 4] == ('launch', 'ur_launch', filename)
        for index in range(max(0, len(args) - 3))
        for filename in ('workbench.launch.py', 'deploy.launch.py', 'teleop.launch.py')
    )


def find_workbench_launch() -> Process:
    pid = os.getppid()
    while pid > 1:
        process = read_process(pid)
        if process is None:
            break
        if is_workbench_launch(process):
            return process
        pid = process.parent
    raise RuntimeError('未找到当前工作台的 ROS launch，不能退出其他进程')


def process_alive(process: Process) -> bool:
    current = read_process(process.pid)
    return current is not None and current.started == process.started


def collect_descendants(tracked: dict[int, Process]) -> None:
    processes = [read_process(int(path.name)) for path in Path('/proc').iterdir()
                 if path.name.isdigit()]
    # The watcher is itself a child of the dashboard, but must survive it.
    processes = [process for process in processes
                 if process is not None and process.pid != os.getpid()]
    changed = True
    while changed:
        changed = False
        for process in processes:
            parent = tracked.get(process.parent)
            if process.pid not in tracked and parent is not None and process_alive(parent):
                tracked[process.pid] = process
                changed = True


def signal_process(process: Process, value: int) -> None:
    if process_alive(process):
        try:
            os.kill(process.pid, value)
        except ProcessLookupError:
            pass


def stop_launch(process: Process, *, graceful_s: float = 12.0,
                terminate_s: float = 3.0) -> None:
    if not process_alive(process):
        return
    if not is_workbench_launch(process):
        raise RuntimeError('目标进程不是 UR 工作台 ROS launch')
    tracked = {process.pid: process}
    collect_descendants(tracked)
    signal_process(process, signal.SIGINT)
    for timeout, escalation in ((graceful_s, signal.SIGTERM),
                                (terminate_s, signal.SIGKILL), (2.0, None)):
        deadline = time.monotonic() + timeout
        while True:
            collect_descendants(tracked)
            remaining = [item for item in tracked.values() if process_alive(item)]
            if not remaining:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if escalation is not None:
            # Descendants first, so their parents can reap and finish cleanup.
            for item in reversed(remaining):
                signal_process(item, escalation)
    raise RuntimeError('ROS 退出后仍有子进程存活：' + ', '.join(str(item.pid) for item in remaining))


class WorkbenchShutdown:
    def __init__(self, prepare, log_path: Path):
        self.prepare = prepare
        self.log_path = log_path
        self._lock = threading.Lock()
        self._state = {'phase': 'idle', 'detail': ''}
        self._watcher = None
        self._launch = None

    def snapshot(self):
        with self._lock:
            if (self._state['phase'] == 'exiting' and self._watcher is not None
                    and self._watcher.poll() is not None and process_alive(self._launch)):
                self._state = {'phase': 'failed', 'detail': 'ROS 退出监控进程已结束，请检查退出日志后重试'}
            return dict(self._state)

    def active(self):
        return self.snapshot()['phase'] in {'stopping', 'exiting'}

    def request(self):
        with self._lock:
            if self._state['phase'] in {'stopping', 'exiting'}:
                return {'accepted': True, **self._state}
            launch = find_workbench_launch()
            self._launch = launch
            self._state = {'phase': 'stopping', 'detail': '正在停止执行并等待录制、模型服务收尾…'}
            result = {'accepted': True, **self._state}
            threading.Thread(target=self._run, args=(launch,), daemon=True,
                             name='workbench-shutdown').start()
            return result

    def _run(self, launch):
        try:
            self.prepare()
            if not process_alive(launch):
                raise RuntimeError('ROS launch 已退出，请刷新页面')
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('ab') as log:
                self._watcher = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()),
                     str(launch.pid), str(launch.started)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True, close_fds=True,
                )
            with self._lock:
                self._state = {'phase': 'exiting', 'detail': '正在退出 ROS 并释放设备…'}
        except Exception as exc:
            with self._lock:
                self._state = {'phase': 'failed', 'detail': str(exc)}


def main():
    pid, started = map(int, sys.argv[1:])
    process = read_process(pid)
    if process is None or process.started != started:
        return
    # Let the HTTP acknowledgement/status reach the browser before exiting.
    time.sleep(0.75)
    print(f'Exiting UR workbench ROS launch {pid}', flush=True)
    stop_launch(process)
    print('UR workbench ROS launch and all child processes exited', flush=True)


if __name__ == '__main__':
    main()
