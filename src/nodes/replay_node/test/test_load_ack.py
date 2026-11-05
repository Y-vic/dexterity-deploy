"""Replay failures must reach the UI as acknowledgements for the exact request."""
import json
from types import SimpleNamespace
import pytest

from replay_node import replay_node as replay


class FakeNode:
    def __init__(self, name):
        self.params, self.output = {}, []
    def declare_parameter(self, name, value):
        self.params[name] = value
    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])
    def create_publisher(self, *args, **kw):
        return SimpleNamespace(publish=self.output.append)
    def create_subscription(self, *args, **kw):
        pass
    def create_timer(self, *args, **kw):
        pass
    def get_logger(self):
        return SimpleNamespace(error=lambda text: None)
    def destroy_node(self):
        pass


def test_failed_load_reports_request_id_path_and_reason_then_accepts_unload(tmp_path, monkeypatch):
    import rclpy
    import rclpy.node
    from std_msgs.msg import String
    (tmp_path/'COMPLETE').touch()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='ur_sharpa_sample.v1',state='complete',quality=dict(ok=False))))
    (tmp_path/'schema.json').write_text(json.dumps(dict(schema='ur_sharpa_sample.v1')))
    monkeypatch.setattr(rclpy, 'init', lambda **kw: None)
    monkeypatch.setattr(rclpy, 'ok', lambda: False)
    monkeypatch.setattr(rclpy.node, 'Node', FakeNode)
    def check(node):
        assert node._state == 'idle' and not node._error
        for identity in ('first', 'retry'):
            node._command(String(data=json.dumps(dict(id=identity,action='load',sample_dir=str(tmp_path)))))
            status=json.loads(node.output[-1].data)
            assert status['state']=='fault'
            assert status['requested_sample_dir']==str(tmp_path)
            result=status['command_result']
            assert result['id']==identity and not result['accepted']
            assert 'quality.ok is false' in result['reason']
        node._command(String(data=json.dumps(dict(id='unload',action='unload'))))
        result=json.loads(node.output[-1].data)['command_result']
        assert result['id']=='unload' and result['accepted']
    monkeypatch.setattr(rclpy, 'spin', check)
    replay.main()
