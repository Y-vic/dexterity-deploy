import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import Request,urlopen
from dashboard_node.dashboard_node import DashboardState,make_handler


def test_replay_queue_response_identifies_the_forwarded_command():
    commands=[]
    handler=make_handler(DashboardState(),'http://127.0.0.1:1',SimpleNamespace(snapshot=lambda:{}),commands.append)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        request=Request(f'http://127.0.0.1:{server.server_port}/api/replay',
            data=json.dumps(dict(action='load',sample_dir='/example',execution_steps=40,frequency_hz=30.,loop=False,dry_run=True)).encode(),
            headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=2) as response:
            result=json.load(response)
            assert response.status==202 and result['queued']
            assert result['command_id'] and result['command_id']==commands[0]['id']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
