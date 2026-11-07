/* Execute the real page script against a small DOM/network fixture; no browser or hardware. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assets=path.resolve(__dirname,'../../../ur_common/ur_common/assets/workbench');
const html=fs.readFileSync(path.join(assets,'index.html'),'utf8');
const script=fs.readFileSync(path.join(assets,'workbench.js'),'utf8');
const {derive,deviceCards}=require(path.join(assets,'workbench-state.js'));
const flush=async()=>{for(let i=0;i<4;i++)await new Promise(resolve=>setImmediate(resolve));};
function fixture({configured=false, rejectSwitch=false, firstCatalog=null, catalogs=null, mode='policy', replay=null, reference=null, replayReply={queued:true,command_id:'load-new'}}={}) {
  class Element {
    constructor(tag='div',attrs={}) {
      this.tagName=tag;this.attrs=attrs;this.dataset={};this.children=[];this.value=attrs.value||'';
      this.textContent='';this.hidden=false;this.disabled=false;this.open=false;
      this.classList={toggle(){}};this.style={};
      for(const [name,value] of Object.entries(attrs))if(name.startsWith('data-'))this.dataset[name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;
    }
    get options(){return this.children;}
    append(...children){this.children.push(...children);}
    add(option){this.append(option);if(this.children.length===1)this.value=option.value;}
    replaceChildren(...children){this.children=children;if(this.tagName==='select')this.value='';}
    setAttribute(name,value){this.attrs[name]=value;}
    addEventListener(){}
    showModal(){this.open=true;}
    close(){this.open=false;}
  }
  const elements=[],ids=new Map();
  for(const tag of html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*)>/g)) {
    const attrs=Object.fromEntries([...tag[2].matchAll(/([a-z][a-z-]*)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
    const element=new Element(tag[1],attrs);elements.push(element);if(attrs.id)ids.set(attrs.id,element);
  }
  ids.get('resetMode').value='bias';ids.get('replayScope').value='full';
  const descendants=()=>{const all=[];const walk=e=>{all.push(e);e.children.forEach(walk);};elements.forEach(walk);return all;};
  const querySelectorAll=selector=>descendants().filter(e=>selector.split(',').some(part=>{
    const m=part.trim().match(/^\[([^=\]]+)(?:="?([^"\]]+)"?)?\]$/);if(!m)return false;
    const value=m[1].startsWith('data-')?e.dataset[m[1].slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]:e.attrs[m[1]];
    return value!==undefined&&(m[2]===undefined||value===m[2]);
  }));
  const state={mode,replay,control:{online:true,ur_state:1,sharpa_state:3},recording:{online:true},
    ur:{valid:true},sharpa:{left_valid:true,right_valid:true},policy:{model_identity_ok:true},
    capabilities:{modes:['teleop','policy','replay'],deployment:true,switchable:true},mode_switch:{allowed:true},
    model:{phase:'stopped',server_id:'baai',server_label:'BAAI',ssh_host:'BAAI',configured}};
  const requests=[],scheduled=[],pollDelays=[];let catalogRequests=0,clockOffset=0;
  const context={console,AbortSignal,URL,URLSearchParams,Date:class extends Date{static now(){return Date.now()+clockOffset;}},Map,JSON,Event,performance,
    document:{hidden:true,getElementById:id=>ids.get(id)||null,querySelectorAll,querySelector:s=>querySelectorAll(s)[0],createElement:tag=>new Element(tag)},
    window:{addEventListener(){}},WorkbenchState:{derive,deviceCards},
    WorkbenchReference:class{initializing=false;selection=reference;init(){}},
    Option:class extends Element{constructor(label,value){super('option');this.textContent=label;this.value=value;}},
    setTimeout(callback,delayMs){scheduled.push(callback);if(callback.name==='poll')pollDelays.push(delayMs);return scheduled.length;},
    fetch:async(url,options={})=>{
      requests.push({url,payload:options.body?JSON.parse(options.body):null});
      let body;
      if(url==='/api/workbench/status')body=state;
      else if(url==='/api/servers')body={selected:state.model.server_id,servers:[{id:'baai',label:'BAAI'},{id:'4090',label:'4090'}]};
      else if(url.startsWith('/api/models?')) {
        if(++catalogRequests===1&&firstCatalog)return {ok:true,json:()=>firstCatalog};
        const serverId=new URL(url,'http://fixture').searchParams.get('server_id');
        body={server:{id:serverId},configured,tasks:catalogs?.[serverId] || (configured?[{name:'task',label:'Duck task',prompt:'Test',runs:[{id:'run',run_id:'immutable-training-run',label:'101 real + 500 sim',model:'remote',prompt:'Checkpoint instruction',checkpoint_path:'/offline/checkpoint',launchable:true}]}]:[])};
      } else if(url==='/api/server/select') {
        if(rejectSwitch)body={accepted:false,error:'模型启动中，暂不可切换'};
        else {const id=JSON.parse(options.body).server_id;Object.assign(state.model,{server_id:id,server_label:id==='baai'?'BAAI':'4090',ssh_host:id==='baai'?'BAAI':'ps'});body={accepted:true};}
      } else if(url==='/api/model/start') {state.model.phase='running';body={accepted:true};}
      else if(url==='/api/replay')body=replayReply;
      else throw Error('Unexpected request '+url);
      return {ok:true,json:async()=>structuredClone(body)};
    },
  };
  vm.runInNewContext(script,context,{filename:'workbench.js'});
  return {ids,state,requests,pollDelays,querySelectorAll,chooseTask:name=>{ids.get('modelTask').value=name;ids.get('modelTask').onchange();},advanceTime:ms=>{clockOffset+=ms;},refresh:async()=>{await scheduled.find(callback=>callback.name==='poll')();await flush();},switchServer:async id=>{const select=ids.get('serverSelect');select.value=id;await select.onchange({target:select});await flush();}};
}

test('page initializes with all DOM references present and only three primary modes',async()=>{
  const page=fixture();await flush();
  assert.deepEqual(page.querySelectorAll('[data-mode]').map(e=>e.dataset.mode),['teleop','policy','replay']);
  for(const id of ['rtcEnabled','executionHorizon','syncSkipSteps','rltControls','rltStatus'])assert.equal(page.ids.has(id),false);
  assert.deepEqual(page.ids.get('serverSelect').options.map(e=>e.value),['baai','4090']);
  assert.equal(page.ids.get('modelList').children.length,0);
  assert.match(page.ids.get('taskPrompt').textContent,/尚未配置模型目录/);
  assert.equal(page.ids.get('policyRequestLatency').textContent,'—');
  assert.equal(page.ids.get('policyAverageLatency').textContent,'—');
  assert.equal(page.ids.get('policyChunkProgress').textContent,'—');
  assert.equal(page.ids.get('policyInferenceCount').textContent,'0 次有效请求');
});

test('Policy shows the confirmed UR recording root while directory edits remain pending',async()=>{
  const page=fixture();await flush();
  page.state.recording.recording_root='/media/yns/T9/captures';
  await page.refresh();
  assert.equal(page.ids.get('policyRecordingOptions').hidden,false);
  assert.equal(page.ids.get('policyRecordPath').textContent,'保存至 UR：/media/yns/T9/captures');
  page.ids.get('recordRoot').value='/home/yns/data/new captures';
  page.ids.get('recordRoot').oninput();
  page.state.recording.recording=true;
  await page.refresh();await page.refresh();
  assert.equal(page.ids.get('recordRoot').value,'/home/yns/data/new captures');
  assert.equal(page.ids.get('policyRecordPath').textContent,'保存至 UR：/media/yns/T9/captures · 新目录待应用');
  assert.equal(page.ids.get('applyRoot').disabled,true);
  assert.equal(page.requests.some(r=>r.url==='/api/recording/config'),false);
  page.state.model.server_id='4090';await page.refresh();
  assert.match(page.ids.get('policyRecordPath').textContent,/^保存至 UR：/,'storage remains on UR for remote inference');
  page.state.mode='teleop';await page.refresh();
  assert.equal(page.ids.get('policyRecordingOptions').hidden,true);
});

function runningPolicy(page, extra={}) {
  Object.assign(page.state.model,{phase:'running',generation:4});
  Object.assign(page.state.control,{online:true,ur_state:2,sharpa_state:1,execution_paused:false});
  Object.assign(page.state.policy,{model_identity_ok:true,model_generation:4,session_id:'episode-1',error:'',
    inference_count:2,last_request_id:6,latency_s:1.234,average_latency_s:.987,...extra});
}

test('request timing uses backend pending elapsed and loaded-model average without accumulating on polls',async()=>{
  const page=fixture();await flush();
  runningPolicy(page,{inference_pending:true,pending_request_id:7,pending_session_id:'episode-1',pending_elapsed_s:2.345});
  await page.refresh();
  assert.equal(page.ids.get('policyRequestLatency').textContent,'2.35 s');
  assert.equal(page.ids.get('policyRequestLabel').textContent,'等待 Request #7');
  assert.equal(page.ids.get('policyAverageLatency').textContent,'0.99 s');
  assert.equal(page.ids.get('policyInferenceCount').textContent,'2 次有效请求');
  assert.equal(page.pollDelays.at(-1),200);
  page.advanceTime(600000);await page.refresh();
  assert.equal(page.ids.get('policyRequestLatency').textContent,'2.35 s','browser wall clock does not invent RPC elapsed');
  Object.assign(page.state.policy,{inference_pending:false,inference_count:3,last_request_id:7,latency_s:2.56,average_latency_s:1.511});
  await page.refresh();await page.refresh();
  assert.equal(page.ids.get('policyRequestLatency').textContent,'2.56 s');
  assert.equal(page.ids.get('policyRequestLabel').textContent,'最近完成 Request #7');
  assert.equal(page.ids.get('policyAverageLatency').textContent,'1.51 s');
  assert.equal(page.ids.get('policyInferenceCount').textContent,'3 次有效请求');
  assert.equal(page.pollDelays.at(-1),500);
});

test('chunk progress displays real dispatch count and actual execution slice without a fixed horizon',async()=>{
  const page=fixture();await flush();runningPolicy(page);
  page.state.actor_latency={execution_progress:{active:true,provider:'policy',session_id:'episode-1',request_id:6,
    action_length:30,execute_start:0,execute_length:30,dispatched_steps:7,acknowledged_steps:4}};
  await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'7 / 30 步');
  Object.assign(page.state.actor_latency.execution_progress,{action_length:40,execute_length:40,dispatched_steps:17});
  await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'17 / 40 步');
  Object.assign(page.state.actor_latency.execution_progress,{action_length:30,execute_start:5,execute_length:15,dispatched_steps:3,executed_steps:8,acknowledged_steps:1});
  page.advanceTime(100000);await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'3 / 15 步','slice-local dispatch, not original index, ACK or wall time');
  assert.match(page.ids.get('policyChunkLabel').textContent,/从第 6 步起，完整 30 步/);
  assert.match(page.ids.get('policyChunkProgress').title,/\[5, 20\)/);
  page.state.control.execution_paused=true;page.state.actor_latency.execution_progress.paused=true;
  await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'3 / 15 步');
  assert.match(page.ids.get('policyChunkLabel').textContent,/已暂停/);
  assert.equal(page.pollDelays.at(-1),500);
});

test('old chunk activity is cleared on stop, failure, stale session, model changes and loss of state',async()=>{
  const page=fixture();await flush();runningPolicy(page);
  const progress={active:true,provider:'policy',session_id:'episode-1',request_id:6,
    action_length:30,execute_start:0,execute_length:30,dispatched_steps:7};
  page.state.actor_latency={execution_progress:progress};await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'7 / 30 步');
  progress.active=false;await page.refresh();assert.equal(page.ids.get('policyChunkProgress').textContent,'—');
  progress.active=true;progress.session_id='old-episode';await page.refresh();assert.equal(page.ids.get('policyChunkProgress').textContent,'—');
  progress.session_id='episode-1';page.state.policy.error='model failed';await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'—');assert.equal(page.ids.get('policyChunkLabel').textContent,'执行已停止');
  page.state.policy.error='';page.state.model.generation=5;await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'—');assert.equal(page.ids.get('policyInferenceCount').textContent,'0 次有效请求');
  page.state.model.generation=4;Object.assign(page.state.control,{ur_state:1,sharpa_state:3});await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'—');assert.equal(page.ids.get('policyChunkLabel').textContent,'未执行');
  assert.equal(page.ids.get('policyAverageLatency').textContent,'0.99 s','stopping retains loaded-model statistics');
  Object.assign(page.state.control,{online:false,ur_state:2,sharpa_state:1});await page.refresh();
  assert.equal(page.ids.get('policyChunkProgress').textContent,'—');assert.equal(page.ids.get('policyChunkLabel').textContent,'等待状态连接');
  page.state.model.phase='stopped';await page.refresh();assert.equal(page.ids.get('policyRequestLatency').textContent,'—');
  assert.equal(page.ids.get('policyInferenceCount').textContent,'0 次有效请求');
});

test('remote selection posts server identity and empty catalogs remain usable',async()=>{
  const page=fixture();await flush();await page.switchServer('4090');
  assert.deepEqual(page.requests.find(r=>r.url==='/api/server/select').payload,{server_id:'4090'});
  assert.equal(page.ids.get('serverSelect').value,'4090');
  assert.match(page.ids.get('serverHint').textContent,/4090.*ps.*尚未配置模型目录/);
  assert.equal(page.ids.get('serverSelect').disabled,false);
  assert.equal(page.ids.get('modelList').children.length,0);
});

test('model start sends the established remote task/run contract without local runtime overrides',async()=>{
  const page=fixture({configured:true});await flush();page.chooseTask('task');
  const button=page.querySelectorAll('[data-model-load]')[0];assert.equal(button.disabled,false);
  await button.onclick();await flush();
  assert.deepEqual(page.requests.find(r=>r.url==='/api/model/start').payload,{server_id:'baai',task:'task',run_id:'immutable-training-run',prompt:'Checkpoint instruction',checkpoint_path:'/offline/checkpoint'});
  assert.equal(page.ids.get('modelList').children[0].children[0].children[0].textContent,'101 real + 500 sim');
  assert.equal(page.ids.get('serverSelect').disabled,true);
});

test('rejected remote selection restores confirmed server and reports the reason',async()=>{
  const page=fixture({rejectSwitch:true});await flush();await page.switchServer('4090');
  assert.equal(page.ids.get('serverSelect').value,'baai');
  assert.match(page.ids.get('feedback').textContent,/暂不可切换/);
});

test('a late previous-server catalog cannot restore stale model buttons',async()=>{
  let complete;const firstCatalog=new Promise(resolve=>{complete=resolve;});
  const page=fixture({firstCatalog});await flush();await page.switchServer('4090');
  complete({tasks:[{name:'old-server',runs:[{id:'stale-run',model:'old',launchable:true}]}]});await flush();
  assert.equal(page.ids.get('serverSelect').value,'4090');
  assert.equal(page.querySelectorAll('[data-model-load]').length,0);
});


const replayReference={root:'/data',episode:'success/example'};
const replayFault={state:'fault',sample_dir:'',requested_sample_dir:'/data/success/invalid',error:'sample manifest quality.ok is false',command_result:{id:'load-old',action:'load',accepted:false,reason:'sample manifest quality.ok is false'}};
test('Replay fault and attempted sample stay visible after repeated state polling',async()=>{
  const page=fixture({mode:'replay',replay:replayFault,reference:replayReference});await flush();
  assert.match(page.ids.get('replayStatus').textContent,/失败/);
  assert.equal(page.ids.get('replayError').hidden,false);
  assert.match(page.ids.get('replayError').textContent,/quality.ok is false/);
  assert.equal(page.ids.get('replaySample').textContent,'/data/success/invalid');
  await page.refresh();
  assert.match(page.ids.get('replayError').textContent,/quality.ok is false/);
  assert.equal(page.ids.get('start').disabled,true);
});

test('Replay queued load waits for its own completion and ignores a previous fault',async()=>{
  const page=fixture({mode:'replay',replay:replayFault,reference:replayReference});await flush();
  await page.ids.get('loadReplay').onclick();await flush();
  assert.match(page.ids.get('replayStatus').textContent,/正在加载/);
  assert.equal(page.ids.get('replayError').hidden,true,'old fault is not the outcome of this request');
  assert.equal(page.ids.get('loadReplay').disabled,true);
  assert.equal(page.ids.get('start').disabled,true);
  assert.equal(page.ids.get('replaySample').textContent,'/data/success/example');
  await page.refresh();
  assert.match(page.ids.get('replayStatus').textContent,/正在加载/);
  page.state.replay={state:'fault',sample_dir:'',requested_sample_dir:'/data/success/example',error:'arm trajectory contains invalid rows: 1297',command_result:{id:'load-new',action:'load',accepted:false,reason:'arm trajectory contains invalid rows: 1297'}};
  await page.refresh();
  assert.match(page.ids.get('feedback').textContent,/Replay 加载失败.*1297/);
  assert.match(page.ids.get('replayError').textContent,/1297/);
  assert.equal(page.ids.get('replayError').hidden,false);
  assert.equal(page.ids.get('loadReplay').disabled,false,'failure allows an explicit corrected retry');
  assert.equal(page.ids.get('start').disabled,true);
});

test('Replay successful confirmation enables execution without starting hardware',async()=>{
  const page=fixture({mode:'replay',replay:replayFault,reference:replayReference});await flush();
  await page.ids.get('loadReplay').onclick();await flush();
  page.state.replay={state:'ready',error:'',sample_dir:'/data/success/example',execution_scope:'full',frequency_hz:30,execution_steps:40,row_count:123,command_result:{id:'load-new',action:'load',accepted:true,reason:''}};
  await page.refresh();
  assert.match(page.ids.get('feedback').textContent,/样本已加载/);
  assert.match(page.ids.get('replayStatus').textContent,/已就绪/);
  assert.equal(page.ids.get('replayProgress').textContent,'0 / 123 步');
  assert.equal(page.ids.get('replayError').hidden,true);
  assert.equal(page.ids.get('start').disabled,false);
  assert.equal(page.requests.filter(request=>request.payload?.action==='load').length,1);
  assert.equal(page.requests.some(request=>request.url==='/api/replay/live/start'),false);
});


test('Replay acknowledgment timeout stays distinct from old fault and accepts a late current result',async()=>{
  const page=fixture({mode:'replay',replay:replayFault,reference:replayReference});await flush();
  await page.ids.get('loadReplay').onclick();await flush();
  page.advanceTime(61000);await page.refresh();
  assert.match(page.ids.get('replayStatus').textContent,/尚未确认/);
  assert.match(page.ids.get('replayError').textContent,/本次加载确认/);
  assert.doesNotMatch(page.ids.get('replayError').textContent,/quality.ok/);
  assert.equal(page.ids.get('loadReplay').disabled,false);
  assert.equal(page.ids.get('start').disabled,true);
  page.state.replay={state:'ready',error:'',sample_dir:'/data/success/example',execution_scope:'full',frequency_hz:30,execution_steps:40,row_count:123,command_result:{id:'load-new',action:'load',accepted:true,reason:''}};
  await page.refresh();
  assert.equal(page.ids.get('replayError').hidden,true);
  assert.equal(page.ids.get('start').disabled,false);
  assert.match(page.ids.get('feedback').textContent,/样本已加载/);
});


test('machine then task scopes checkpoint prompts without carrying task selection across machines',async()=>{
  const run=(id,prompt)=>({id,run_id:id,model:'dfm',label:id,prompt,launchable:true});
  const task=(name,prompt,runs)=>({name,label:name,prompt,runs});
  const catalogs={baai:[task('duck','Pick ducks',[run('duck-small','Pick yellow ducks'),run('duck-large','Pick ducks gently')]),task('button','Press button',[run('button-run','Press the blue button')])],
    '4090':[task('duck','Different machine task',[run('ur-run','UR machine instruction')])]};
  const page=fixture({configured:true,catalogs});await flush();
  assert.equal(page.ids.get('modelTask').value,'');
  assert.equal(page.ids.get('modelList').children.length,0);
  page.chooseTask('duck');
  assert.match(page.ids.get('modelList').children[0].children[0].children[2].textContent,/Pick yellow ducks/);
  assert.match(page.ids.get('modelList').children[1].children[0].children[2].textContent,/Pick ducks gently/);
  const oldButton=page.querySelectorAll('[data-model-load]')[0];
  page.chooseTask('button');
  assert.match(page.ids.get('taskPrompt').textContent,/Press button/);
  assert.match(page.ids.get('modelList').children[0].children[0].children[2].textContent,/blue button/);
  await oldButton.onclick();
  assert.equal(page.requests.some(r=>r.url==='/api/model/start'),false);
  await page.switchServer('4090');
  assert.equal(page.ids.get('modelTask').value,'');
  assert.equal(page.ids.get('modelList').children.length,0);
  page.chooseTask('duck');
  assert.match(page.ids.get('modelList').children[0].children[0].children[2].textContent,/UR machine instruction/);
  assert.doesNotMatch(page.ids.get('taskPrompt').textContent,/Pick ducks|Press button/);
});

test('catalog with a mismatched machine is rejected before creating model options',async()=>{
  const firstCatalog=Promise.resolve({server:{id:'4090'},configured:true,tasks:[{name:'wrong',prompt:'wrong',runs:[]}]});
  const page=fixture({configured:true,firstCatalog});await flush();
  assert.equal(page.ids.get('modelTask').options.length,0);
  assert.match(page.ids.get('feedback').textContent,/推理机器已变化/);
});


test('unavailable checkpoints explain their reason and missing effective prompts stay disabled',async()=>{
  const catalogs={baai:[{name:'task',prompt:'Default instruction',runs:[
    {id:'missing',run_id:'missing',model:'dfm',prompt:'',launchable:true},
    {id:'not-ready',run_id:'not-ready',model:'dfm',prompt:'Use this instruction',launchable:false,disabled_reason:'checkpoint 不存在'}]}]};
  const page=fixture({configured:true,catalogs});await flush();page.chooseTask('task');
  const buttons=page.querySelectorAll('[data-model-load]');
  assert.ok(buttons.every(button=>button.disabled));
  assert.match(page.ids.get('modelList').children[0].children[0].children[2].textContent,/未配置/);
  assert.equal(page.ids.get('modelList').children[1].children[0].children[1].textContent,'checkpoint 不存在');
});
