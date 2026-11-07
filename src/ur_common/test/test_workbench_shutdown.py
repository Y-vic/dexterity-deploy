"""Shutdown checks use fake process tables and ordinary Python subprocesses only."""
import json
import signal
import subprocess
import sys
import time

import pytest

from ur_common import workbench_shutdown as shutdown


def launch_process(pid=100, parent=1, started=5):
    return shutdown.Process(pid, parent, started,
                            ('python3', '/opt/ros/humble/bin/ros2', 'launch',
                             'ur_launch', 'workbench.launch.py'))


def test_finds_only_own_workbench_ancestor(monkeypatch):
    launch = launch_process()
    processes = {12: shutdown.Process(12, 100, 9, ('wrapper',)), 100: launch}
    monkeypatch.setattr(shutdown.os, 'getppid', lambda: 12)
    monkeypatch.setattr(shutdown, 'read_process', processes.get)
    assert shutdown.find_workbench_launch() == launch
    processes[100] = shutdown.Process(100, 1, 5, ('ros2', 'launch', 'other', 'other.launch.py'))
    with pytest.raises(RuntimeError, match='未找到'):
        shutdown.find_workbench_launch()


def test_does_not_signal_reused_pid_or_unrelated_launch(monkeypatch):
    process = launch_process()
    monkeypatch.setattr(shutdown, 'read_process', lambda _: launch_process(started=6))
    monkeypatch.setattr(shutdown.os, 'kill', lambda *_: pytest.fail('unexpected signal'))
    shutdown.stop_launch(process, graceful_s=0)
    unrelated = shutdown.Process(100, 1, 6, ('python3', 'other_server.py'))
    monkeypatch.setattr(shutdown, 'read_process', lambda _: unrelated)
    with pytest.raises(RuntimeError, match='不是 UR 工作台'):
        shutdown.stop_launch(unrelated)


@pytest.mark.parametrize('ignore_interrupt', [False, True])
def test_real_python_process_tree_cleanup_preserves_other_processes(tmp_path, ignore_interrupt):
    # This synthetic launcher never imports ROS or accesses a device. Its child
    # ignores signals, including after the parent exits and it becomes orphaned.
    ready = tmp_path / 'ready.json'
    script = """
import json, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); '
    'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
signal.signal(signal.SIGINT, signal.SIG_IGN if sys.argv[2] == 'True' else lambda *_: sys.exit(0))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(.2)
with open(sys.argv[1], 'w') as stream: json.dump({'child': child.pid}, stream)
time.sleep(60)
"""
    root = subprocess.Popen([sys.executable, '-c', script, str(ready), str(ignore_interrupt),
                             'ros2', 'launch', 'ur_launch', 'workbench.launch.py'])
    outside = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    child = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert root.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        child = shutdown.read_process(json.loads(ready.read_text())['child'])
        process = shutdown.read_process(root.pid)
        assert child is not None and process is not None
        shutdown.stop_launch(process, graceful_s=.3, terminate_s=.3)
        root.wait(timeout=2)
        assert not shutdown.process_alive(child)
        assert outside.poll() is None
    finally:
        if child is not None:
            shutdown.signal_process(child, signal.SIGKILL)
        if root.poll() is None:
            root.kill()
        root.wait(timeout=2)
        outside.terminate()
        outside.wait(timeout=2)


def test_duplicate_shutdown_is_idempotent_and_prepare_failure_can_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(shutdown, 'find_workbench_launch', launch_process)
    calls = []
    class Thread:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def start(self):
            pass
    monkeypatch.setattr(shutdown.threading, 'Thread', Thread)
    def fail():
        raise RuntimeError('保存失败')
    control = shutdown.WorkbenchShutdown(fail, tmp_path / 'shutdown.log')
    assert control.request()['accepted']
    assert control.request()['accepted']
    assert len(calls) == 1 and control.active()
    control._run(launch_process())
    assert control.snapshot() == {'phase': 'failed', 'detail': '保存失败'}
    assert not control.active()
    assert control.request()['accepted']
    assert len(calls) == 2


def test_watcher_is_detached_after_prepare_and_reports_failure(monkeypatch, tmp_path):
    events = []
    process = launch_process()
    monkeypatch.setattr(shutdown, 'process_alive', lambda _: True)
    class Watcher:
        returncode = None
        def poll(self): return self.returncode
    watcher = Watcher()
    def spawn(arguments, **kwargs):
        events.append('watcher')
        assert arguments[-2:] == [str(process.pid), str(process.started)]
        assert kwargs['start_new_session'] and kwargs['close_fds']
        assert kwargs['stdin'] == subprocess.DEVNULL
        return watcher
    monkeypatch.setattr(shutdown.subprocess, 'Popen', spawn)
    control = shutdown.WorkbenchShutdown(lambda: events.append('prepare'), tmp_path / 'logs/shutdown.log')
    control._launch = process
    control._run(process)
    assert events == ['prepare', 'watcher']
    assert control.snapshot()['phase'] == 'exiting'
    watcher.returncode = 1
    assert control.snapshot()['phase'] == 'failed'
    assert not control.active()
