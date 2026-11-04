/* Optional real-browser smoke test. Requires local Chrome and system Python only.
 * PYTHONPATH must include src/ur_common. All HTTP operations use an offline fixture.
 */
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const {once}=require('node:events');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const directory=await fs.mkdtemp(path.join(os.tmpdir(),'workbench-browser-'));
 const fixture=spawn('/usr/bin/python3',[path.resolve(__dirname,'../../../ur_common/test/workbench_fixture.py')],{stdio:['ignore','pipe','inherit']});
 let chrome,ws;
 try{
  const [chunk]=await once(fixture.stdout,'data');const origin='http://127.0.0.1:'+chunk.toString().trim();
  chrome=spawn(process.env.CHROME_BIN||'/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+directory,'about:blank'],{stdio:'ignore'});
  let port;
  for(let i=0;i<100;i++){try{port=(await fs.readFile(path.join(directory,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await delay(100);}}
  assert.ok(port,'Chrome started');
  const targets=await (await fetch('http://127.0.0.1:'+port+'/json')).json();
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await once(ws,'open');
  let id=0;const waiting=new Map(),errors=[];
  ws.addEventListener('message',e=>{const msg=JSON.parse(e.data);if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails);if(msg.id){const promise=waiting.get(msg.id);waiting.delete(msg.id);if(msg.error)promise.reject(msg.error);else promise.resolve(msg.result);}});
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;waiting.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
  const evaluate=async expression=>{const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});assert.ok(!result.exceptionDetails,JSON.stringify(result.exceptionDetails));return result.result.value;};
  const waitFor=async expression=>{for(let i=0;i<80;i++){if(await evaluate(expression))return;await delay(100);}throw Error('Timed out: '+expression);};
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const patch=body=>fetch(origin+'/test/state',{method:'POST',body:JSON.stringify(body)});
  const commands=async()=> (await fetch(origin+'/test/commands')).json();
  const set=async(id,value,event='change')=>evaluate(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(value)};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event(${JSON.stringify(event)}))`);
  const paths=await(await fetch(origin+'/test/paths')).json();
  const idle={online:true,ur_state:1,sharpa_state:3,execution_paused:false,execution_dry_run:false};
  await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});await send('Page.navigate',{url:origin});
  await waitFor("document.getElementById('connection')?.textContent==='控制器在线' && !document.getElementById('referenceRoot').disabled");
  assert.equal(await evaluate("document.querySelectorAll('[data-mode]').length"),3);
  assert.equal(await evaluate("document.querySelectorAll('[data-page],aside,#deviceList,#preview').length"),0);
  assert.equal(await evaluate("document.getElementById('stop').parentElement.className"),'execution-buttons');
  await waitFor("document.querySelectorAll('.device-card').length===8");
  assert.equal(await evaluate("document.querySelector('[data-device=ur]').dataset.tone"),'ok');
  await patch({devices:{ur_status:{online:false,valid:true,normal_mode:true,age_s:4},sharpa_status:{online:true,left_valid:true,right_valid:false},glove_status:{online:true,valid:false,error:'AE01 SDK exited'},vive_status:{online:true,valid:true,valid_arms:{left:true,right:false}}}});
  await waitFor("document.querySelector('[data-device=ur]').dataset.tone==='offline' && document.querySelector('[data-device=glove]').dataset.tone==='error'");
  assert.equal(await evaluate("document.querySelector('[data-device=sharpa]').dataset.tone"),'warn');
  assert.equal(await evaluate("document.querySelector('[data-device=vive]').dataset.tone"),'warn');
  await patch({devices:{ur_status:{online:true,valid:true,normal_mode:true,age_s:.03},sharpa_status:{online:true,left_valid:true,right_valid:true,control_mode:'POSITION',age_s:.02},glove_status:{online:true,valid:true,left_valid:true,right_valid:true,left_robot_hz:25,right_robot_hz:25,age_s:.03},vive_status:{online:true,valid:true,valid_arms:{left:true,right:true},calibrated:true,age_s:.04}}});
  await waitFor("!document.getElementById('startRecord').disabled");
  await click('#startRecord');await click('#startRecord');
  await waitFor("document.getElementById('captureStatus').textContent==='正在录制' && document.getElementById('stage').textContent==='执行中'");
  assert.equal((await commands()).filter(c=>c.action==='start_teleop_record').length,1,'combined capture submits only once');
  await click('#stop');await waitFor("document.getElementById('captureStatus').textContent==='已保存'");
  await patch({control:idle});await waitFor("!document.getElementById('record').disabled");
  await click('#record');await click('#record');await waitFor("document.getElementById('captureStatus').textContent==='正在录制'");
  assert.equal((await commands()).filter(c=>c.action==='start_record').length,1,'double click does not duplicate recording');
  // Inputs accept typed/pasted paths without opening a dialog or being overwritten by polling.
  await click('#recordRoot');assert.equal(await evaluate("document.getElementById('directoryDialog').open"),false);
  assert.equal(await evaluate("document.getElementById('recordRoot').readOnly"),false);
  await set('recordRoot',paths.root+'/pasted captures','input');await delay(650);
  assert.equal(await evaluate("document.getElementById('recordRoot').value"),paths.root+'/pasted captures');
  assert.equal(await evaluate("document.getElementById('applyRoot').disabled"),true);
  // Browse an existing parent, create and enter a new folder, then choose it while recording.
  await click('#chooseDirectory');await waitFor("document.getElementById('directoryDialog').open");
  await set('directoryPath',paths.root,'input');await click('#browseDirectory');
  await waitFor("!document.getElementById('newDirectory').disabled");
  await click('#newDirectory');await set('newDirectoryName','new captures');await click('#createDirectory');
  await waitFor(`document.getElementById('directoryPath').value===${JSON.stringify(paths.root+'/new captures')} && !document.getElementById('newDirectory').disabled`);
  const listing=await(await fetch(origin+'/api/workbench/directories?'+new URLSearchParams({path:paths.root}))).json();
  assert.ok(listing.directories.some(d=>d.name==='new captures'),'new folder exists on host filesystem');
  await click('#useDirectory');
  assert.equal((await commands()).filter(c=>c.recording_root).length,0,'choosing during recording only creates a draft');
  await click('#finish');await waitFor("!document.getElementById('applyRoot').disabled");await click('#applyRoot');
  // A newer edit must survive confirmation of an older in-flight directory request.
  await set('recordRoot',paths.root+'/latest pasted path','input');
  await waitFor("document.getElementById('recordRootActual').textContent.startsWith('待应用；当前：') && !document.getElementById('applyRoot').disabled");
  assert.equal(await evaluate("document.getElementById('recordRoot').value"),paths.root+'/latest pasted path');
  await click('#applyRoot');await waitFor(`document.getElementById('recordRootActual').textContent==='当前：'+${JSON.stringify(paths.root+'/latest pasted path')}`);
  await waitFor("document.getElementById('feedback').textContent.includes('保存目录已应用并记忆')");
  // Reference uses the same filesystem picker and can create a folder independently of recording.
  await click('#chooseReferenceDirectory');await waitFor("document.getElementById('directoryTitle').textContent==='选择参考数据集目录' && !document.getElementById('newDirectory').disabled");
  await click('#newDirectory');await set('newDirectoryName','新参考数据集');await click('#createDirectory');
  await waitFor(`document.getElementById('directoryPath').value===${JSON.stringify(paths.dataset+'/新参考数据集')} && !document.getElementById('newDirectory').disabled`);
  await click('#useDirectory');await waitFor("document.getElementById('referenceEpisode').options.length===1 && !document.getElementById('referenceRoot').disabled");
  assert.equal(await evaluate("document.getElementById('referenceRoot').value"),paths.dataset+'/新参考数据集');
  assert.equal(await evaluate("document.getElementById('recordRoot').value"),paths.root+'/latest pasted path','reference choice does not change recording destination');
  // Paste a dataset path directly, then browse and choose the existing dataset again.
  await set('referenceRoot',paths.dataset,'input');await click('#scanReference');
  await waitFor("document.getElementById('referenceEpisode').options.length===3");
  await click('#chooseReferenceDirectory');await waitFor("!document.getElementById('directoryParent').disabled");await click('#directoryParent');
  await waitFor(`document.getElementById('directoryPath').value===${JSON.stringify(paths.root)} && !document.getElementById('newDirectory').disabled`);
  await evaluate("[...document.querySelectorAll('#directoryList button')].find(b=>b.textContent==='dataset').click()");
  await waitFor(`document.getElementById('directoryPath').value===${JSON.stringify(paths.dataset)} && !document.getElementById('newDirectory').disabled`);
  await click('#useDirectory');
  // One selected episode supplies reference frames, reset and replay.
  await waitFor("document.getElementById('referenceEpisode').options.length===3");
  await set('referenceEpisode','success/fixture');await delay(30);await set('referenceEpisode','success/second');
  await waitFor("!document.getElementById('referenceImage').hidden && document.getElementById('referenceStatus').textContent.includes('已记忆')");
  assert.equal(await evaluate("document.getElementById('referenceBadge').textContent.includes('second')"),true,'late metadata cannot replace latest selection');
  await click('#nextFrame');await waitFor("document.getElementById('referenceFrame').textContent.includes('帧 1')");
  await set('referenceAlpha','35','input');assert.equal(await evaluate("document.getElementById('referenceImage').style.opacity"),'0.35');
  await click('#reset');await waitFor("document.getElementById('stage').textContent==='复位中'");
  assert.deepEqual((await commands()).find(c=>c.action==='reset_both').reset_target,{mode:'episode',root:paths.dataset,episode:'success/second'});
  await patch({control:idle});await waitFor("!document.querySelector('[data-mode=policy]').disabled");
  const count=(await commands()).length;
  await fetch(origin+'/test/restart',{method:'POST',body:'{}'});await send('Page.reload');
  await waitFor("document.getElementById('referenceEpisode')?.value==='success/second' && !document.getElementById('referenceRoot').disabled");
  assert.equal(await evaluate("document.getElementById('resetMode').value"),'episode');
  assert.equal(await evaluate("document.getElementById('recordRoot').value"),paths.root+'/latest pasted path','restart restores the applied recording directory');
  assert.equal((await commands()).length,count,'server restart and reload restore selection without execution');
  await click('[data-mode=policy]');await waitFor("document.getElementById('modeLabel').textContent==='推理'");
  // A render during command submission must not reverse the user's checkbox intent.
  await waitFor("!document.getElementById('policyRecord').disabled");
  const recordingCommands=async()=> (await commands()).filter(c=>['enable_policy_recording','disable_policy_recording'].includes(c.action));
  await click('#policyRecord');await waitFor("document.getElementById('policyRecord').checked && !document.getElementById('policyRecord').disabled");
  assert.equal((await recordingCommands()).at(-1).action,'enable_policy_recording');
  await delay(650);assert.equal(await evaluate("document.getElementById('policyRecord').checked"),true,'enabled value survives status polling');
  await click('#policyRecord');await waitFor("!document.getElementById('policyRecord').checked && !document.getElementById('policyRecord').disabled");
  assert.equal((await recordingCommands()).at(-1).action,'disable_policy_recording');
  // Setting the preference does not require a loaded model or an idle online recorder.
  await patch({recording:{online:false,saving:true}});await waitFor("document.getElementById('captureStatus').textContent==='数采未连接'");
  assert.equal(await evaluate("document.getElementById('policyRecord').disabled"),false);
  await click('#policyRecord');await waitFor("document.getElementById('policyRecord').checked && !document.getElementById('policyRecord').disabled");
  await patch({recording:{online:true,saving:false},reject_policy_recording:true});
  await click('#policyRecord');await waitFor("document.getElementById('feedback').textContent==='保存选项测试拒绝' && !document.getElementById('policyRecord').disabled");
  assert.equal(await evaluate("document.getElementById('policyRecord').checked"),true,'rejection restores confirmed server value');
  await patch({reject_policy_recording:false});
  assert.equal(await evaluate("document.querySelectorAll('#rtcEnabled,#executionHorizon,#syncSkipSteps,#rltControls,[data-mode=hil]').length"),0);
  await waitFor("!document.getElementById('serverSelect').disabled");
  await set('serverSelect','4090');
  await waitFor("document.getElementById('serverHint').textContent.includes('4090')");
  assert.equal((await commands()).find(c=>c.server_id==='4090').server_id,'4090');
  await click('#chooseModel');await waitFor("document.querySelector('[data-model-load]')&&!document.querySelector('[data-model-load]').disabled");await click('[data-model-load]');
  await waitFor("document.getElementById('currentModel').textContent.includes('fixture') && !document.getElementById('modelDialog').open");
  assert.deepEqual((await commands()).find(c=>c.run_id==='fixture'),{task:'offline-demo',run_id:'fixture'});
  assert.equal(await evaluate("document.getElementById('serverSelect').disabled"),true);
  await send('Page.reload');
  await waitFor("document.getElementById('connection')?.textContent==='控制器在线' && document.getElementById('serverSelect').disabled");
  await waitFor("document.getElementById('serverSelect').value==='4090'");
  await click('#start');await waitFor("document.getElementById('stage').textContent==='执行中'");
  assert.equal(await evaluate("document.getElementById('policyRecord').disabled"),true,'running execution keeps capture setting locked');
  assert.equal(await evaluate("document.querySelector('[data-label=mark_success]').disabled"),true);
  await click('#stop');await waitFor("document.getElementById('captureStatus').textContent==='正在保存'");
  assert.equal(await evaluate("document.querySelector('[data-label=mark_success]').disabled"),true,'classification waits for save');
  assert.equal(await evaluate("document.querySelector('[data-label=drop_sample]').disabled"),true,'discard waits for save');
  await waitFor("!document.querySelector('[data-label=mark_success]').disabled && document.getElementById('stage').textContent==='已停止'");
  const saved=await evaluate("document.getElementById('savedPath').textContent");
  await patch({recording:{capture_notice:'启停边界 17 帧缺少动作，已保留无效标记',last_error:''}});
  await waitFor("document.getElementById('recordingHint').textContent.includes('启停边界 17 帧')");
  assert.equal(await evaluate("document.getElementById('recordError').textContent"),'');
  await patch({recording:{last_error:'SharpA hand action has no complete successful SDK command'}});
  await waitFor("document.getElementById('recordError').textContent.includes('no complete successful SDK command')");
  await patch({recording:{capture_notice:'',last_error:''}});
  await click('[data-label=mark_success]');await waitFor("document.getElementById('sampleCategory').textContent==='已标记成功'");
  assert.equal((await commands()).filter(c=>c.action==='mark_success').at(-1).saved_path,saved,'classify precisely the saved policy episode');
  await click('[data-label=mark_fail]');await waitFor("document.getElementById('sampleCategory').textContent==='已标记失败'");
  assert.equal(await evaluate("document.getElementById('savedPath').textContent.includes('/fail/policy-episode')"),true);
  const discardedPath=await evaluate("document.getElementById('savedPath').textContent");
  assert.equal(await evaluate("document.querySelector('.sample-outcome [data-label=drop_sample]').textContent"),'丢弃数据');
  await click('[data-label=drop_sample]');
  await waitFor("document.getElementById('savedPath').textContent==='尚无已保存样本' && document.querySelector('[data-label=drop_sample]').disabled");
  assert.equal((await commands()).filter(c=>c.action==='drop_sample').at(-1).saved_path,discardedPath,'discard targets the currently displayed classified episode');
  assert.equal(await evaluate("document.querySelector('[data-label=mark_success]').disabled && document.querySelector('[data-label=mark_fail]').disabled"),true);

  await click('#start');await waitFor("document.getElementById('stage').textContent==='执行中'");
  await click('#pause');await waitFor("document.getElementById('stage').textContent==='已暂停'");
  await patch({recording:{saving:true,recording:true}});await delay(650);let before=(await commands()).filter(c=>c.action==='reset_both').length;
  await click('#reset');await waitFor("document.getElementById('reset').textContent.includes('等待采样')");await delay(350);
  assert.equal((await commands()).filter(c=>c.action==='reset_both').length,before,'reset waits for final sampling');
  await patch({recording:{recording:false}});await waitFor("document.getElementById('stage').textContent==='复位中'");
  assert.equal((await commands()).filter(c=>c.action==='reset_both').length,before+1);
  assert.equal(await evaluate("document.getElementById('captureStatus').textContent"),'正在保存','reset starts before background save finishes');
  // A stopped machine can also reset while saving, without enabling a new episode.
  await patch({control:idle});await waitFor("!document.getElementById('reset').disabled");
  assert.equal(await evaluate("document.getElementById('start').disabled && document.getElementById('applyRoot').disabled"),true);
  await click('#reset');await waitFor("document.getElementById('stage').textContent==='复位中'");
  assert.equal((await commands()).filter(c=>c.action==='reset_both').length,before+2);
  assert.equal(await evaluate("document.getElementById('captureStatus').textContent"),'正在保存');
  // An urgent stop cancels a reset still waiting for sampling to close.
  await patch({control:{...idle,ur_state:2,sharpa_state:1,execution_paused:true},recording:{recording:true,saving:true,rollout_requested:true}});
  await waitFor("document.getElementById('stage').textContent==='已暂停' && !document.getElementById('reset').disabled");
  await click('#reset');await waitFor("document.getElementById('reset').textContent.includes('等待采样')");await click('#stop');
  await waitFor("document.getElementById('stage').textContent==='保存中'");await patch({recording:{saving:false,rollout_requested:false}});await delay(750);
  assert.equal((await commands()).filter(c=>c.action==='reset_both').length,before+2,'stop cancels deferred reset');
  await waitFor("!document.querySelector('[data-mode=replay]').disabled");await click('[data-mode=replay]');await waitFor("document.getElementById('modeLabel').textContent==='Replay'");
  await click('#loadReplay');await waitFor("!document.getElementById('start').disabled");
  assert.equal((await commands()).find(c=>c.action==='load').sample_dir,paths.dataset+'/success/second');
  await click('#start');await waitFor("document.getElementById('stage').textContent==='执行中'");await click('#stop');await waitFor("document.getElementById('stage').textContent==='已停止'");
  // Protection appears once per event and retains diagnostics in its dialog.
  await patch({policy:{protection:{active:true,code:'tracking_error',title:'跟踪保护',message:'离线保护示例',timestamp_ns:1}}});
  await waitFor("document.getElementById('protectionDialog').open");await click('[data-close=protectionDialog]');await delay(1100);
  assert.equal(await evaluate("document.getElementById('protectionDialog').open"),false);
  await patch({policy:{protection:{active:true,code:'tracking_error',title:'跟踪保护',timestamp_ns:2}}});await waitFor("document.getElementById('protectionDialog').open");await click('[data-close=protectionDialog]');
  await patch({policy:{protection:{active:false}}});
  await patch({mode:'teleop',control:idle});
  await waitFor("!document.getElementById('startRecord').disabled");
  // Check real hit targets and sibling rectangles, not just CSS declarations.
  for(const width of [1440,1280,1024,390]){
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width===390});await delay(200);
    assert.equal(await evaluate(`document.documentElement.scrollWidth <= ${width}`),true,'no horizontal overflow at '+width);
    assert.equal(await evaluate("document.getElementById('startRecord').textContent"),'开始采集并录制');
    assert.equal(await evaluate(`(()=>{const ids=['startRecord','record','finish'],rects=ids.map(id=>document.getElementById(id).getBoundingClientRect());return rects.every((a,i)=>a.width>0&&rects.slice(i+1).every(b=>a.right<=b.left+.1||b.right<=a.left+.1||a.bottom<=b.top+.1||b.bottom<=a.top+.1));})()`),true,'capture buttons do not overlap at '+width);
    assert.equal(await evaluate("document.querySelector('.reference-selection').getBoundingClientRect().top >= document.querySelector('.ego-panel').getBoundingClientRect().bottom"),true,'reference selection is below Ego');
    assert.equal(await evaluate("document.querySelector('.device-section').getBoundingClientRect().top >= document.querySelector('.results').getBoundingClientRect().bottom"),true,'devices sit at page bottom');
    assert.equal(await evaluate(`(()=>{const b=document.getElementById('stop'),r=b.getBoundingClientRect();return r.bottom<innerHeight&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===b;})()`),true,'stop button visible and unobstructed');
    assert.equal(await evaluate(`(()=>{const b=document.getElementById('reset'),r=b.getBoundingClientRect();return r.top>=0&&r.bottom<innerHeight&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===b&&!b.closest('details');})()`),true,'reset visible and unobstructed at '+width);
    assert.equal(await evaluate(`(()=>{const rects=[...document.querySelectorAll('.reset-row > *, .execution-buttons > *')].filter(e=>!e.hidden).map(e=>e.getBoundingClientRect());return rects.every((a,i)=>rects.slice(i+1).every(b=>a.right<=b.left+.1||b.right<=a.left+.1||a.bottom<=b.top+.1||b.bottom<=a.top+.1));})()`),true,'machine controls do not overlap at '+width);
    if(width===1440||width===390){const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await fs.writeFile('/tmp/ur-sharpa-workbench-'+(width===1440?'desktop':'mobile')+'.png',Buffer.from(shot.data,'base64'));}
  }
  await patch({control:{online:false}});await waitFor("document.getElementById('start').disabled && document.getElementById('connection').textContent==='连接已断开'");
  assert.equal(await evaluate("document.querySelector('[data-device=ego]').dataset.tone"),'ok','camera feedback remains independent of the controller heartbeat');
  await patch({status_error:true});
  await waitFor("[...document.querySelectorAll('.device-card')].every(d=>d.dataset.tone==='offline')");
  assert.equal(await evaluate("[...document.querySelectorAll('.device-card')].every(d=>d.dataset.tone==='offline')"),true,'stale cached device values are not shown as healthy after disconnect');
  assert.deepEqual(errors,[],'no browser script exceptions');
  console.log('Browser smoke passed: four modes, model modal, editable paths, shared directory picker, folder creation, shared reference, restart persistence, paused reset/save/stop, inference stop/save/classification, device freshness, protection modal and four responsive layouts.');

 }finally{if(ws)ws.close();if(chrome){chrome.kill();await once(chrome,'exit').catch(()=>{});}fixture.kill();await fs.rm(directory,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
