import pytest

from dashboard_node.dashboard_node import learning_command


def state(mode='remote'):
    return dict(execution_provider='hil', model=dict(model=mode,instance='model-1'),
                policy=dict(model_identity_ok=True,hil=dict(enabled=True)))


def test_commands_bind_to_the_loaded_model_and_get_unique_ids():
    first=learning_command({'action':'takeover'},state())
    second=learning_command({'action':'takeover'},state())
    assert first['instance']=='model-1' and first['id']!=second['id']
    assert first['issued_at']>0


@pytest.mark.parametrize('action',['start_episode','success','handover','retry_uploads'])
def test_remote_policy_cannot_issue_rl_commands(action):
    with pytest.raises(ValueError):
        learning_command({'action':action},state('remote'))


def test_remote_hil_is_supported_but_replay_and_unloaded_models_are_rejected():
    assert learning_command({'action':'takeover'},state('remote'))['action']=='takeover'
    snapshot=state();snapshot['execution_provider']='replay'
    with pytest.raises(ValueError):
        learning_command({'action':'takeover'},snapshot)
    snapshot=state();snapshot['policy']['model_identity_ok']=False
    with pytest.raises(ValueError):
        learning_command({'action':'takeover'},snapshot)


@pytest.mark.parametrize('action', ['takeover', 'release', 'failure'])
def test_policy_rejects_hil_actions_even_with_hil_hardware_enabled(action):
    snapshot = state()
    snapshot['execution_provider'] = 'policy'
    with pytest.raises(ValueError, match='HIL'):
        learning_command({'action': action}, snapshot)
    snapshot['execution_provider'] = 'hil'
    assert learning_command({'action': action}, snapshot)['action'] == action


def test_http_queues_command_and_serves_learning_script():
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from types import SimpleNamespace
    from urllib.request import Request, urlopen
    from dashboard_node.dashboard_node import make_handler, DashboardState
    current = state()
    dashboard = DashboardState()
    for key,value in current.items():
        dashboard.update(key,value)
    commands = []
    handler = make_handler(dashboard,'http://127.0.0.1:1',
        SimpleNamespace(snapshot=lambda:current['model']),lambda _:None,
        publish_learning_command=commands.append)
    server = ThreadingHTTPServer(('127.0.0.1',0),handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        url=f'http://127.0.0.1:{server.server_port}'
        with urlopen(Request(url+'/api/learning',data=b'{"action":"takeover"}',
                     headers={'Content-Type':'application/json'}),timeout=2) as response:
            result=json.load(response)
            assert response.status==202 and result['queued']
            assert result['command_id']==commands[0]['id']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
