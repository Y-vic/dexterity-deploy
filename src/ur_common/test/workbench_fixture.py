"""Offline HTTP fixture for browser checks. Never imports ROS or controls hardware."""
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from PIL import Image, ImageDraw
from ur_common.workbench import WorkbenchHTTP
from ur_common.workbench_preferences import WorkbenchPreferences
from ur_common.recording_config import RecordingPreferences

storage = tempfile.TemporaryDirectory(prefix='workbench-fixture-')
root = Path(storage.name)
(root / 'dataset' / 'success' / 'fixture').mkdir(parents=True)
(root / 'dataset' / 'success' / 'second').mkdir()
(root / 'captures').mkdir()
state = {
    'mode':'teleop', 'control':{'online':True, 'ur_state':1, 'sharpa_state':3},
    'recording':{'online':True,'recording_root':str(root / 'captures')},
    'devices':{
        'ur_status':{'online':True,'valid':True,'normal_mode':True,'age_s':.03},
        'sharpa_status':{'online':True,'left_valid':True,'right_valid':True,'control_mode':'POSITION','age_s':.02},
        'vive_status':{'online':True,'valid':True,'valid_arms':{'left':True,'right':True},'calibrated':True,'age_s':.04},
        'glove_status':{'online':True,'valid':True,'left_valid':True,'right_valid':True,'left_robot_hz':25,'right_robot_hz':25,'age_s':.03}},
    'ur':{'valid':True}, 'sharpa':{'left_valid':True,'right_valid':True},
    'model':{'phase':'stopped','server_id':'baai','server_label':'BAAI','ssh_host':'BAAI','configured':True}, 'policy':{'hil':{'available':True,'enabled':True},'model_identity_ok':True},
    'capabilities':{'modes':['teleop','policy','replay'],'deployment':True,'switchable':True},
    'mode_switch':{'allowed':True},'operator':{},
}
commands=[]

def save_episode():
    record = state['recording']
    if not record.get('recording'): return
    episode = record.get('episode_id', 'offline-preview')
    record.update(recording=False, saving=True, rollout_requested=False)
    def finish():
        if record.get('episode_id') == episode and not record.get('recording'):
            record.update(saving=False, saved_path=record['recording_root']+'/'+episode,
                          validation={'available':True,'ok':True,'items':[]})
    threading.Timer(.9, finish).start()


def action(payload):
    commands.append(payload)
    name=payload.get('action','')
    if name in ('enable_policy_recording', 'disable_policy_recording'):
        if state.get('reject_policy_recording'):
            return {'accepted': False, 'gate_reason': '保存选项测试拒绝'}
        state['policy_recording'] = name == 'enable_policy_recording'
    if name.startswith('select_'): state['mode']=name[7:]
    if name in ('start_record', 'start_teleop_record'): state['recording'].update(recording=True,episode_id='offline-preview')
    if name=='stop_record': save_episode()
    if name=='takeover': state['policy']['hil']['active']=True
    if name=='release': state['policy']['hil']['active']=False
    if name in ('run_ur','start_teleop_record','start_deploy','continue_deploy'):
        state['control'].update(ur_state=2,sharpa_state=1,execution_paused=False)
        if state['mode']=='hil' or state['mode']=='policy' and state.get('policy_recording'):
            state['recording'].update(recording=True, saving=False, episode_id='policy-episode', validation=None)
    if name in ('mark_success', 'mark_fail', 'drop_sample'):
        record=state['recording']
        if record.get('saving') or record.get('recording') or payload.get('saved_path')!=record.get('saved_path'):
            return {'accepted':False,'message':'样本已变化或保存尚未完成'}
        if name == 'drop_sample':
            record.update(saved_path='', validation=None)
        else:
            record['saved_path']=record['recording_root']+'/'+name.removeprefix('mark_')+'/'+Path(record['saved_path']).name
    if name in ('stop','stop_deploy'):
        state['control'].update(ur_state=0,sharpa_state=0,execution_dry_run=False,execution_paused=False)
        state['policy']['hil']['active']=False
        save_episode()
    if name=='reset_both':
        state['control'].update(ur_state=3,sharpa_state=2,execution_dry_run=False,execution_paused=False)
    if name=='pause_deploy' or (name=='pause' and state['mode']=='replay'):
        state['control']['execution_paused']=True
    if name=='preview_deploy': state['control']['execution_dry_run']=True
    time.sleep(.08)
    return {'accepted':True}

def jpeg(reference=False):
    image = Image.new('RGB', (640, 360), '#a4afba' if reference else '#8398a8')
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 140, 585, 330), fill='#cad1d7')
    draw.ellipse((285 if reference else 300, 150, 355 if reference else 370, 220), fill='#c88c60')
    draw.line((80, 280, 190, 200, 270, 245), fill='#597080', width=25)
    draw.line((565, 280, 455, 200, 385, 245), fill='#597080', width=25)
    draw.text((20,20), 'OFFLINE FIXTURE / REFERENCE' if reference else 'OFFLINE FIXTURE / LIVE', fill='#243545')
    data = io.BytesIO(); image.save(data, format='JPEG'); return data.getvalue()
LIVE, REFERENCE = jpeg(), jpeg(True)

class Preview:
    def camera_frame(self, name): return (LIVE, str(time.time_ns()))

def snapshot():
    if state.get('status_error'):
        raise RuntimeError('Offline fixture: status connection unavailable')
    c, r = state['control'], state['recording']
    state['mode_switch'] = {'allowed': not (c.get('ur_state') in (2,3,4) or c.get('sharpa_state') in (1,2,4) or c.get('execution_dry_run') or c.get('execution_paused') or r.get('recording') or r.get('saving')), 'reason': '停止执行并等待保存后可以切换'}
    return state

class Shutdown:
    def snapshot(self): return state.get('shutdown', {'phase': 'idle', 'detail': ''})
    def active(self): return self.snapshot()['phase'] in {'stopping', 'exiting'}
    def request(self):
        if state.get('reject_shutdown'):
            raise RuntimeError('退出测试拒绝')
        if not self.active():
            commands.append({'action': 'killed'})
            state['shutdown'] = {'phase': 'stopping', 'detail': '正在保存并退出 ROS…'}
            threading.Timer(.4, lambda: state.update(shutdown={'phase': 'exiting', 'detail': '正在退出 ROS…'})).start()
            threading.Timer(1.2, lambda: state.update(status_error=True)).start()
        return {'accepted': True, **self.snapshot()}

http=WorkbenchHTTP(snapshot,action,Preview(),WorkbenchPreferences(root/'prefs.json'), shutdown=Shutdown())
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_): pass
    def send(self,value):
        data=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if http.handle(self):return
        url=urlsplit(self.path); query=parse_qs(url.query)
        if url.path=='/test/commands':self.send(commands)
        elif url.path=='/test/paths':self.send({'root':str(root),'dataset':str(root/'dataset')})
        elif url.path=='/api/servers':self.send({'selected':state['model']['server_id'],'servers':[{'id':'baai','label':'BAAI'},{'id':'ur','label':'UR (4090D)'},{'id':'4090','label':'PS (4090)'}]})
        elif url.path=='/api/models':self.send({'server':{'id':query.get('server_id',[state['model']['server_id']])[0]},'configured':True,'tasks':[{'name':'offline-demo','label':'离线任务','prompt':'默认离线任务文本','runs':[{'model':'remote','run_id':'fixture','id':'fixture','prompt':'This checkpoint has its own instruction.','checkpoint_path':'/offline/checkpoint','launchable':True}]}]})
        elif url.path=='/api/reference/config':self.send({'root':str(root/'dataset')})
        elif url.path=='/api/reference/episodes':
            dataset = Path(query['root'][0])
            names = sorted(p.name for p in (dataset/'success').iterdir()) if (dataset/'success').is_dir() else []
            self.send({'root':str(dataset), 'episodes':[{'id':'success/'+name,'name':name,'category':'success'} for name in names]})
        elif url.path=='/api/reference/metadata':
            if query['episode'][0]=='success/fixture':time.sleep(.25)
            self.send({'root':query['root'][0],'episode':query['episode'][0],'frames':[0,1,2],'seconds':[0,.1,.2],'duration_s':.2})
        elif url.path=='/api/reference/frame.jpg':
            time.sleep(.12)
            self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(REFERENCE)));self.end_headers();self.wfile.write(REFERENCE)
        else:self.send_error(404)
    def do_POST(self):
        global http
        if http.handle(self):return
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        payload = json.loads(body) if body else {}
        if self.path=='/test/state':
            for key,value in payload.items():
                if isinstance(value,dict):state.setdefault(key,{}).update(value)
                else:state[key]=value
            self.send({'ok':True});return
        if self.path=='/test/restart':
            http=WorkbenchHTTP(snapshot,action,Preview(),WorkbenchPreferences(root/'prefs.json'), shutdown=Shutdown())
            state['recording']['recording_root'] = RecordingPreferences(root/'recording.json').read(str(root/'captures'))
            self.send({'ok':True});return
        if urlsplit(self.path).path=='/api/reference/close':self.send({'ok':True});return
        if self.path=='/api/server/select':
            selected=payload['server_id']
            state['model'].update(server_id=selected,server_label={'baai':'BAAI','ur':'UR (4090D)','4090':'PS (4090)'}[selected],ssh_host={'baai':'BAAI','ur':'','4090':'ps'}[selected],local=selected=='ur',prompt='',task='',task_id='',run_id='')
        if self.path=='/api/model/start':
            state['model'].update(phase='running',model='remote',task=payload['task'],task_id=payload['task'],run_id=payload['run_id'],prompt='This checkpoint has its own instruction.')
        if self.path=='/api/model/stop':state['model']['phase']='stopped'
        if self.path=='/api/replay':state['replay']={**payload,'state':'ready','row_count':120}
        if self.path=='/api/replay/preview/start':state['control']['execution_dry_run']=True
        if self.path=='/api/replay/live/start':state['control'].update(ur_state=2,sharpa_state=1)
        if self.path=='/api/recording/config':
            commands.append(payload)
            time.sleep(.35)
            RecordingPreferences(root/'recording.json').save(payload['recording_root'])
            state['recording'].update(recording_root=payload['recording_root'], config_result={'id':'config-1','accepted':True})
            self.send({'accepted':True,'command_id':'config-1'});return
        self.send(action(payload))

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    print(server.server_address[1],flush=True)
    server.serve_forever()
