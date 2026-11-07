import io
import json

import pytest

from ur_common.workbench import WorkbenchHTTP, workbench_snapshot


def operator(**record):
    return {'status': {'online': True, 'active_source': 2},
            'upstream': {'record_status': {'online': True, **record}}}


def test_capabilities_follow_launch_and_hil_device_availability():
    assert workbench_snapshot(operator())['capabilities']['modes'] == ['teleop']
    deploy = {'execution_provider': 'hil', 'policy': {'hil': {'available': True}}}
    value = operator();value['status']['active_source'] = 3
    assert workbench_snapshot(value, deploy)['mode'] == 'hil'
    assert workbench_snapshot(value, deploy, switchable=True)['capabilities']['modes'] == ['teleop', 'policy', 'replay']


@pytest.mark.parametrize('record', [{'saving': True}, {'recording': True}, {'rollout_requested': True}, {'online': False}])
def test_no_mode_change_while_saving_or_recorder_offline(record):
    assert not workbench_snapshot(operator(**record), switchable=True)['mode_switch']['allowed']


class Handler:
    def __init__(self, path, method='GET', payload=None):
        self.path, self.command = path, method
        data = json.dumps(payload).encode() if payload is not None else b''
        self.headers = {'Content-Length': str(len(data))}
        self.rfile, self.wfile = io.BytesIO(data), io.BytesIO()
        self.code = None
    def send_response(self, code): self.code = code
    def send_header(self, *_): pass
    def end_headers(self): pass


def test_serves_shared_assets_and_queues_operator_commands():
    calls=[]
    http=WorkbenchHTTP(lambda: {'mode':'teleop'}, lambda payload: calls.append(payload) or {'accepted':True}, None)
    for path in ('/', '/workbench.js', '/workbench-state.js', '/workbench.css', '/workbench-reference.js'):
        request=Handler(path);assert http.handle(request);assert request.code==200;assert request.wfile.getvalue()
    request=Handler('/api/workbench/action','POST',{'action':'start_record'})
    assert http.handle(request) and request.code==200
    assert calls==[{'action':'start_record'}]
    request=Handler('/api/workbench/action','POST',[])
    http.handle(request);assert request.code==400;assert len(calls)==1


def test_shutdown_route_requires_explicit_action_and_blocks_other_commands():
    calls = []
    class Shutdown:
        def snapshot(self): return {'phase': 'stopping'}
        def active(self): return bool(calls)
        def request(self):
            calls.append('shutdown')
            return {'accepted': True, **self.snapshot()}
    http = WorkbenchHTTP(lambda: {}, lambda _: pytest.fail('robot command'), None,
                         shutdown=Shutdown())
    for payload in ({}, {'action': 'stop'}, {'action': 'killed', 'pid': 1}, []):
        request = Handler('/api/workbench/shutdown', 'POST', payload)
        http.handle(request)
        assert request.code == 400
    assert calls == []
    request = Handler('/api/workbench/shutdown', 'POST', {'action': 'killed'})
    http.handle(request)
    assert request.code == 202
    assert json.loads(request.wfile.getvalue())['accepted']
    request = Handler('/api/workbench/action', 'POST', {'action': 'run_ur'})
    http.handle(request)
    assert request.code == 400
    request = Handler('/api/workbench/status')
    http.handle(request)
    assert json.loads(request.wfile.getvalue())['shutdown']['phase'] == 'stopping'


def test_shutdown_unavailable_in_offline_or_unsupported_entry():
    http = WorkbenchHTTP(lambda: {}, lambda _: {}, None)
    request = Handler('/api/workbench/shutdown', 'POST', {'action': 'killed'})
    http.handle(request)
    assert request.code == 400


def test_preferences_survive_server_restart_without_robot_commands(tmp_path):
    from ur_common.workbench_preferences import WorkbenchPreferences
    path = tmp_path / "config" / "workbench.json"
    calls = []
    http = WorkbenchHTTP(lambda: {}, calls.append, None, WorkbenchPreferences(path))
    value = {"reference_root": "/data/参考", "reference_episode": "success/one", "reset_mode": "episode"}
    request = Handler('/api/workbench/preferences', 'POST', value)
    http.handle(request)
    assert request.code == 200
    restarted = WorkbenchHTTP(lambda: {}, calls.append, None, WorkbenchPreferences(path))
    request = Handler('/api/workbench/preferences')
    restarted.handle(request)
    assert json.loads(request.wfile.getvalue()) == value
    assert calls == []
    assert path.stat().st_mode & 0o777 == 0o600
    for invalid in ({"action": "reset_both"}, {"reset_mode": "bad"}, {"reference_root": []}):
        request = Handler('/api/workbench/preferences', 'POST', invalid)
        restarted.handle(request)
        assert request.code == 400
    assert WorkbenchPreferences(path).read() == value


def test_host_directory_picker_lists_only_directories_and_handles_bad_paths(tmp_path):
    from urllib.parse import urlencode
    from ur_common.workbench_preferences import WorkbenchPreferences
    (tmp_path / "new folder").mkdir()
    (tmp_path / "sample.txt").write_text("not a directory")
    (tmp_path / ".private").mkdir()
    http = WorkbenchHTTP(lambda: {}, lambda _: {}, None, WorkbenchPreferences(tmp_path / "prefs.json"))
    request = Handler('/api/workbench/directories?' + urlencode({"path": str(tmp_path)}))
    http.handle(request)
    result = json.loads(request.wfile.getvalue())
    assert request.code == 200
    assert result["directories"] == [{"name": "new folder", "path": str(tmp_path / "new folder")}]
    assert result["parent"] == str(tmp_path.parent)
    for path in ('relative', str(tmp_path / 'missing'), str(tmp_path / 'sample.txt')):
        request = Handler('/api/workbench/directories?' + urlencode({"path": path}))
        http.handle(request)
        assert request.code == 400


def test_directory_picker_creates_one_folder_and_never_overwrites(tmp_path):
    http = WorkbenchHTTP(lambda: {}, lambda _: pytest.fail('not a robot command'), None)
    request = Handler('/api/workbench/directories', 'POST',
                      {'parent': str(tmp_path), 'name': '参考数据 01'})
    assert http.handle(request) and request.code == 200
    created = tmp_path / '参考数据 01'
    assert created.is_dir()
    assert json.loads(request.wfile.getvalue())['path'] == str(created)
    sentinel = created / 'keep.txt'
    sentinel.write_text('keep')
    request = Handler('/api/workbench/directories', 'POST',
                      {'parent': str(tmp_path), 'name': '参考数据 01'})
    http.handle(request)
    assert request.code == 400
    assert '已存在' in json.loads(request.wfile.getvalue())['error']
    assert sentinel.read_text() == 'keep'


@pytest.mark.parametrize('parent,name', [
    ('relative', 'one'), ('', 'one'), (None, 'one'),
    ('PARENT', ''), ('PARENT', '.'), ('PARENT', '..'), ('PARENT', '../outside'),
    ('PARENT', '/absolute'), ('PARENT', 'nested/child'), ('PARENT', 'bad\\name'),
    ('PARENT', 'bad\0name'), ('PARENT', []),
])
def test_directory_creation_rejects_invalid_parent_or_child(tmp_path, parent, name):
    http = WorkbenchHTTP(lambda: {}, lambda _: pytest.fail('not a robot command'), None)
    request = Handler('/api/workbench/directories', 'POST',
                      {'parent': str(tmp_path) if parent == 'PARENT' else parent, 'name': name})
    http.handle(request)
    assert request.code == 400
    assert list(tmp_path.iterdir()) == []
