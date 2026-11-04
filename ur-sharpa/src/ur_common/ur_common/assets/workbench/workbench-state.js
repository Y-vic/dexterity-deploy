/* Pure presentation state, shared with the offline UI tests. ROS owns all gates. */
(function(root) {
  const names = {teleop:'数采', policy:'推理', hil:'HIL', replay:'Replay'};
  function derive(s, busy=false) {
    const c=s.control||{}, r=s.recording||{}, p=s.policy||{}, m=s.model||{}, h=p.hil||{};
    const mode=s.mode||'teleop', online=c.online===true;
    const running=c.ur_state===2||c.sharpa_state===1, starting=c.ur_state===4||c.sharpa_state===4;
    const resetting=c.ur_state===3||c.sharpa_state===2, preview=!!c.execution_dry_run, paused=!!c.execution_paused;
    const executing=running||starting||preview||paused;
    const locked=busy||!online, saving=!!r.saving, recording=!!r.recording||c.record_state===1;
    const capturePending=!!r.rollout_requested||!!p.rollout_recording?.requested;
    const idle=!executing&&!resetting&&!recording&&!saving&&!capturePending;
    const ur=s.ur||s.devices?.ur_status||{}, hand=s.sharpa||s.devices?.sharpa_status||{};
    const hardware=ur.valid===true&&(hand.left_valid===true&&hand.right_valid===true);
    const modelReady=m.phase==='running'&&p.model_identity_ok===true;
    const teleopReady=s.devices?.vive_status?.online&&s.devices?.vive_status?.valid&&s.devices?.glove_status?.online&&s.devices?.glove_status?.valid;
    const sourceReady=mode==='teleop'?(c.ur_state===1&&c.sharpa_state===3&&teleopReady):mode==='replay'?['ready','paused'].includes(s.replay?.state):modelReady;
    const active=online&&!busy&&!p.error&&!paused&&(running||preview);
    let stage=!online?'连接中':resetting?'复位中':starting?'启动准备中':paused?'已暂停':preview?'预览中':running?'执行中':saving?'保存中':'已停止';
    const allowed={
      start:!locked&&!executing&&!resetting&&!saving&&sourceReady&&(hardware||(mode==='replay'&&s.replay?.execution_scope==='hands_only'&&hand.left_valid&&hand.right_valid)),
      preview:!locked&&idle&&mode!=='teleop'&&sourceReady,
      pause:!locked&&(running||preview)&&!paused,
      resume:!locked&&(paused||(mode==='replay'&&s.replay?.state==='paused'))&&!saving,
      stop:online&&(executing||recording||resetting||(mode==='replay'&&['paused','complete','fault'].includes(s.replay?.state))),
      reset:!locked&&hardware&&!resetting&&!starting&&(paused||(!executing&&!recording&&!capturePending&&(!saving||r.online))),
      record:!locked&&mode==='teleop'&&r.online&&!recording&&!saving,
      finish:!locked&&mode==='teleop'&&recording,
      policyRecord:!locked&&mode==='policy'&&s.capabilities?.deployment&&!executing&&!c.execution_start_pending,
      configure:!locked&&r.online&&!starting&&!c.execution_start_pending&&!recording&&!saving&&!capturePending,
      label:!locked&&r.online&&!recording&&!saving&&!!r.validation?.available,
      takeover:active&&mode==='hil'&&h.enabled&&!h.active,
      release:active&&mode==='hil'&&h.active&&!h.return_pending,
      failure:active&&mode==='hil'&&!preview&&!!p.rollout_recording?.recording,
      modelLoad:!locked&&idle&&s.capabilities?.deployment&&!['starting','loading','running','stopping'].includes(m.phase),
      serverConfigure:!locked&&idle&&s.capabilities?.deployment&&!['starting','loading','running','stopping'].includes(m.phase),
      modelUnload:!locked&&idle&&['starting','loading','running','fault'].includes(m.phase),
    };
    allowed.startRecord=allowed.start&&allowed.record&&!capturePending&&!c.execution_start_pending;
    const owner=mode==='teleop'?'人工遥操':mode==='hil'&&h.active?(h.return_pending?'人工 · 等待策略':'人工接管'):mode==='hil'&&h.transition?.active?'交还策略 · 平滑过渡':mode==='replay'?'轨迹回放':'策略';
    const capture=!r.online?'数采未连接':saving?'正在保存':recording?'正在录制':r.last_error?'采集异常':r.saved_path?'已保存':'未录制';
    const reason=!online?'等待控制器连接':busy?'正在提交操作':saving?'等待本条数据保存完成':resetting?'等待复位完成':executing?'请先停止当前执行':!hardware?'检查机械臂与双手状态':!sourceReady?(mode==='teleop'?(!teleopReady?'检查 Vive 与 AE01 输入状态':'请先复位机器人'):mode==='replay'?(s.replay?.state==='complete'?'回放已完成，请先停止以清空进度':'请先加载有效样本'):'请先加载模型并等待就绪'):c.gate_reason||'准备就绪';
    return {mode,online,idle,executing,paused,running,starting,resetting,preview,stage,owner,capture,reason,allowed:Object.fromEntries(Object.entries(allowed).map(([k,v])=>[k,!!v])),names};
  }
  function deviceCards(s, cameras={}, connected=true, now=Date.now()) {
    const devices=s.devices||{}, c=s.control||{}, r=s.recording||{};
    const fresh=d=>connected&&d.online===true&&(d.age_s==null||Number(d.age_s)<=2);
    const valid=value=>value===true?'有效':value===false?'无效':'未报告';
    const age=d=>Number.isFinite(d.age_s)?`状态更新 ${d.age_s<1?Math.round(d.age_s*1000)+' ms':d.age_s.toFixed(1)+' s'} 前`:'等待状态心跳';
    function card(key,name,kind,d,healthy,rows,detail='',error='',partial=false,busy='') {
      const online=fresh(d);
      return {key,name,kind,online,
        tone:!online?'offline':error?'error':busy?'busy':healthy?'ok':'warn',
        status:!online?'离线':error?'异常':busy|| (healthy?'正常':partial?'部分有效':'未就绪'),
        rows:rows.map(([label,value])=>[label,online?value:'—']),
        detail:!online?(connected?'未收到新鲜状态，请检查连接':'页面连接已断开'):error||detail||age(d)};
    }
    const ur=devices.ur_status||{}, hand=devices.sharpa_status||{}, glove=devices.glove_status||{}, vive=devices.vive_status||{};
    const states={ur:{0:'已停止',1:'已复位',2:'运行中',3:'复位中',4:'启动准备'},hand:{0:'已停止',1:'运行中',2:'复位中',3:'已复位',4:'启动准备'}};
    const list=[
      card('ur','UR5','双机械臂',ur,ur.valid===true&&ur.normal_mode===true,
        [['双臂反馈',valid(ur.valid)],['机械臂状态',states.ur[c.ur_state]||'未报告']],
        ur.normal_mode===true?'Normal 模式 · '+age(ur):'运行模式未就绪',
        ur.normal_mode===false?(ur.message||ur.status||'机械臂不在 Normal 模式'):''),
      card('sharpa','SharpA','双灵巧手',hand,hand.left_valid===true&&hand.right_valid===true,
        [['左手 / 右手',valid(hand.left_valid)+' / '+valid(hand.right_valid)],['控制模式',hand.control_mode||'未报告']],
        (states.hand[c.sharpa_state]||'状态未报告')+' · '+age(hand),hand.control_error||'',hand.left_valid===true||hand.right_valid===true),
      card('glove','AE01','动作捕捉手套',glove,glove.valid===true&&glove.left_valid!==false&&glove.right_valid!==false,
        [['左手 / 右手',valid(glove.left_valid)+' / '+valid(glove.right_valid)],['ROBOT 帧率 · 左 / 右',`${glove.left_robot_hz??'—'} / ${glove.right_robot_hz??'—'} Hz`]],
        age(glove),glove.error||'',glove.left_valid===true||glove.right_valid===true),
      card('vive','Vive','空间定位',vive,vive.valid===true&&vive.valid_arms?.left!==false&&vive.valid_arms?.right!==false,
        [['左 / 右 Tracker',valid(vive.valid_arms?.left)+' / '+valid(vive.valid_arms?.right)],['标定',vive.calibrated===true?'已完成':vive.calibrated===false?'待标定':'未报告']],
        age(vive),vive.error||'',vive.valid_arms?.left===true||vive.valid_arms?.right===true),
    ];
    for (const [key,name] of [['ego','Ego'],['left_wrist','左腕相机'],['right_wrist','右腕相机']]) {
      const camera=cameras[key]||{};
      const d={online:camera.online===true,age_s:camera.updatedAt?(now-camera.updatedAt)/1000:null};
      list.push(card(key,name,'相机画面',d,true,[['图像',camera.width&&camera.height?`${camera.width} × ${camera.height}`:'未报告'],['画面更新',age(d).replace('状态更新 ','')]],'实时画面有效'));
    }
    list.push(card('recorder','数采服务','Episode 录制与保存',r,true,
      [['采样帧',String(r.frame_count??'—')],['实际采样率',Number.isFinite(r.achieved_rate_hz)?r.achieved_rate_hz.toFixed(1)+' Hz':'—']],
      r.saving?'正在封装视频与写入文件':r.recording?'正在采样':r.saved_path?'最近样本已写入磁盘':'就绪，等待下一条采集',r.last_error||r.rollout_error||'',false,r.saving?'保存中':r.recording?'录制中':''));
    return list;
  }
  const api={derive,names,deviceCards};

  if(typeof module!=='undefined')module.exports=api;else root.WorkbenchState=api;
})(globalThis);
