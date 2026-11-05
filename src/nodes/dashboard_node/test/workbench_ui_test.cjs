const {test}=require('node:test');
const assert=require('node:assert/strict');
const {derive}=require('../../../ur_common/ur_common/assets/workbench/workbench-state.js');
function state(mode='teleop') {return {mode,devices:{vive_status:{online:true,valid:true},glove_status:{online:true,valid:true}},control:{online:true,ur_state:1,sharpa_state:3},recording:{online:true},ur:{valid:true},sharpa:{left_valid:true,right_valid:true},model:{phase:'running'},policy:{model_identity_ok:true,hil:{enabled:true}},capabilities:{deployment:true}};}
test('collection separates machine and recorder lifecycle',()=>{
 const s=state();assert.equal(derive(s).allowed.start,true);assert.equal(derive(s).allowed.record,true);
 s.recording.recording=true;s.control.ur_state=2;s.control.sharpa_state=1;
 const v=derive(s);assert.equal(v.allowed.finish,true);assert.equal(v.allowed.configure,false);assert.equal(v.allowed.record,false);assert.equal(v.allowed.stop,true);
});
test('disconnect, pending commands and saving disable new execution',()=>{
 const s=state('hil');assert.equal(derive(s).allowed.start,true);
 s.recording.saving=true;assert.equal(derive(s).allowed.start,false);assert.equal(derive(s).allowed.label,false);
 s.recording.saving=false;s.control.online=false;assert.equal(derive(s).allowed.start,false);assert.equal(derive(s).allowed.record,false);
 s.control.online=true;assert.equal(derive(s,true).allowed.start,false);
 s.control.ur_state=2;s.control.sharpa_state=1;assert.equal(derive(s,true).allowed.stop,true);
});
test('HIL handoff follows controller ownership',()=>{
 const s=state('hil');s.control.ur_state=2;s.control.sharpa_state=1;
 assert.equal(derive(s).allowed.takeover,true);assert.equal(derive(s).allowed.release,false);assert.equal(derive(s).allowed.start_episode,undefined);
 s.policy.hil.active=true;assert.equal(derive(s).allowed.takeover,false);assert.equal(derive(s).allowed.release,true);
 s.policy.hil.return_pending=true;assert.equal(derive(s).allowed.release,false);
 s.mode='policy';assert.equal(derive(s).allowed.takeover,false);
});
test('HIL return transition remains visible and may be interrupted by takeover',()=>{
 const s=state('hil');s.control.ur_state=2;s.control.sharpa_state=1;
 s.policy.hil.transition={active:true,progress:.5,duration_s:1};
 assert.equal(derive(s).owner,'交还策略 · 平滑过渡');
 assert.equal(derive(s).allowed.takeover,true);
 assert.equal(derive(s).allowed.release,false);
 assert.equal(derive(s).allowed.stop,true);
});
test('Replay completion requires stop before restart, paused Replay resumes, preview never records',()=>{
 const s=state('replay');s.replay={state:'complete'};assert.equal(derive(s).allowed.start,false);assert.equal(derive(s).allowed.stop,true);assert.match(derive(s).reason,/停止/);
 s.replay.state='ready';assert.equal(derive(s).allowed.start,true);
 s.replay.state='paused';assert.equal(derive(s).allowed.resume,true);
 s.control.execution_dry_run=true;assert.equal(derive(s).allowed.start,false);assert.equal(derive(s).allowed.record,false);assert.equal(derive(s).allowed.stop,true);
});

test('server selection locks during loading, execution and unloading',()=>{
 const s=state('policy');s.model={phase:'stopped'};
 assert.equal(derive(s).allowed.serverConfigure,true);
 for(const phase of ['starting','loading','running','stopping']){
  s.model.phase=phase;
  assert.equal(derive(s).allowed.serverConfigure,false);
  assert.equal(derive(s).allowed.modelLoad,false);
 }
 s.model.phase='stopped';s.control.execution_paused=true;
 assert.equal(derive(s).allowed.serverConfigure,false);
 s.control.execution_paused=false;s.control.online=false;
 assert.equal(derive(s).allowed.serverConfigure,false);
});

test('teleop cannot start when a glove or tracker is stale',()=>{
 const s=state();s.devices.vive_status.online=false;assert.equal(derive(s).allowed.start,false);assert.match(derive(s).reason,/Vive/);
});

test('paused reset is available while background saving continues',()=>{
 const s=state('hil');s.control.ur_state=2;s.control.sharpa_state=1;
 assert.equal(derive(s).allowed.reset,false);
 s.control.execution_paused=true;s.recording.saving=true;
 assert.equal(derive(s).allowed.reset,true);assert.equal(derive(s).allowed.resume,false);
 assert.equal(derive(s).allowed.configure,false);
 s.recording.saving=false;assert.equal(derive(s).allowed.configure,true);
 s.control.online=false;assert.equal(derive(s).allowed.reset,false);
});
test('recording directory gates follow capture and pending start, not robot movement',()=>{
 const s=state();s.control.ur_state=2;s.control.sharpa_state=1;
 assert.equal(derive(s).allowed.configure,true);
 s.recording.rollout_requested=true;assert.equal(derive(s).allowed.configure,false);
 s.recording.rollout_requested=false;s.control.execution_start_pending=true;
 assert.equal(derive(s).allowed.configure,false);
});

test('stopped machine may reset during saving, but new capture and directory changes stay blocked',()=>{
 const s=state();s.recording.saving=true;
 assert.equal(derive(s).allowed.reset,true);
 for(const action of ['start','record','configure','label'])assert.equal(derive(s).allowed[action],false);
 s.recording.recording=true;assert.equal(derive(s).allowed.reset,false);
 s.recording.recording=false;s.recording.online=false;assert.equal(derive(s).allowed.reset,false);
});

test('policy recording preference is independent of recorder availability and model loading',()=>{
 const s=state('policy');s.model.phase='stopped';s.recording.online=false;s.recording.saving=true;
 assert.equal(derive(s).allowed.policyRecord,true);
 s.control.ur_state=3;s.control.sharpa_state=2;
 assert.equal(derive(s).allowed.policyRecord,true,'reset may prepare next execution preference');
 for(const mode of ['teleop','hil','replay']) {s.mode=mode;assert.equal(derive(s).allowed.policyRecord,false);}
 s.mode='policy';s.control.online=false;assert.equal(derive(s).allowed.policyRecord,false);
 s.control.online=true;assert.equal(derive(s,true).allowed.policyRecord,false);
});
test('policy recording cannot change during running, paused, preview or pending start',()=>{
 for(const control of [{ur_state:2},{sharpa_state:1},{execution_paused:true},{execution_dry_run:true},{execution_start_pending:true}]){
  const s=state('policy');Object.assign(s.control,control);
  assert.equal(derive(s).allowed.policyRecord,false);
 }
});

test('device cards distinguish heartbeat freshness, partial inputs and hardware protection',()=>{
 const {deviceCards}=require('../../../ur_common/ur_common/assets/workbench/workbench-state.js');
 const s={devices:{ur_status:{online:true,valid:true,normal_mode:false},sharpa_status:{online:true,left_valid:true,right_valid:false},glove_status:{online:false,valid:true,left_valid:true,right_valid:true},vive_status:{online:true,valid:true,valid_arms:{left:true,right:false}}}};
 const cards=deviceCards(s);
 assert.equal(cards.find(d=>d.key==='ur').tone,'error');
 assert.equal(cards.find(d=>d.key==='sharpa').status,'部分有效');
 assert.equal(cards.find(d=>d.key==='glove').tone,'offline');
 assert.equal(cards.find(d=>d.key==='vive').status,'部分有效');
 s.devices.ur_status={online:true,valid:true,normal_mode:true,age_s:4};
 assert.equal(deviceCards(s)[0].tone,'offline');
 assert.ok(deviceCards(s,{},false).every(d=>d.tone==='offline'));
 const camera={ego:{online:true,updatedAt:1000,width:640,height:360}};
 assert.equal(deviceCards(s,camera,true,1500).find(d=>d.key==='ego').tone,'ok');
 assert.equal(deviceCards(s,camera,true,4000).find(d=>d.key==='ego').tone,'offline');
});

test('combined capture requires both start and recording readiness',()=>{
 const s=state();assert.equal(derive(s).allowed.startRecord,true);
 assert.equal(derive(s,true).allowed.startRecord,false);
 for(const mode of ['policy','hil','replay'])assert.equal(derive(state(mode)).allowed.startRecord,false);
 for(const changes of [
  {control:{online:false}}, {control:{ur_state:0}}, {control:{ur_state:2,sharpa_state:1}},
  {control:{execution_start_pending:true}}, {recording:{online:false}},
  {recording:{recording:true}}, {recording:{saving:true}}, {recording:{rollout_requested:true}},
  {devices:{vive_status:{online:false}}}, {devices:{glove_status:{online:false}}},
 ]){
  const next=state();for(const [key,value] of Object.entries(changes))Object.assign(next[key],value);
  assert.equal(derive(next).allowed.startRecord,false,JSON.stringify(changes));
 }
});

test('Replay stop resets its buffer, while reset cancellation still uses the operator', async()=>{
 const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
 const source=fs.readFileSync(path.join(__dirname,'../../../ur_common/ur_common/assets/workbench/workbench.js'),'utf8');
 const handler=source.slice(source.indexOf("  $('stop').onclick = async () => {"),source.indexOf("  $('reset').onclick"));
 const calls=[],stop={};
 const context={snapshot:state('replay'),resetEpoch:0,WorkbenchState:{derive},$:()=>stop,
   command:async fn=>{await fn();return true},
   post:async(url,payload)=>calls.push([url,payload.action]),
   operator:async action=>calls.push(['operator',action]),feedback:()=>{}};
 vm.createContext(context);vm.runInContext(handler,context);
 await stop.onclick();assert.deepEqual(calls.pop(),['/api/replay','stop']);
 context.snapshot.control.ur_state=3;
 await stop.onclick();assert.deepEqual(calls.pop(),['operator','stop']);
 context.snapshot=state('teleop');
 await stop.onclick();assert.deepEqual(calls.pop(),['operator','stop']);
});
