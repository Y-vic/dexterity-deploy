'use strict';
(() => {
  const $ = id => document.getElementById(id), all = selector => [...document.querySelectorAll(selector)];
  let snapshot = {}, busy = false, connected = false, rootEdited = false, catalog = null;
  let pending = null, pollFlight = null, validationKey = '', resetEpoch = 0, queuedReset = false;
  let protectionKey = '', protection = null, protectionDiagnostics = '', directoryVersion = 0;
  let directoryTarget = 'recording', directoryParent = '', creatingDirectory = false;
  const images = new Map(), stamps = new Map(), cameraHealth = {};
  let deviceKey = '', catalogVersion = 0, catalogServerId = '', replayRequest = null;
  let shutdownRequested = false, shutdownPhase = '';
  const shuttingDown = () => shutdownRequested || ['stopping','exiting','closed'].includes(shutdownPhase);
  function feedback(text, error=false) {$('feedback').textContent = text; $('feedback').classList.toggle('error', error);}
  async function json(url, options={}) {
    const response = await fetch(url, {cache:'no-store', signal:AbortSignal.timeout(25000), ...options});
    const body = await response.json();
    if (!response.ok || body.accepted === false) throw new Error(body.gate_reason || body.error || body.message || `请求失败 (${response.status})`);
    return body;
  }
  const post = (url, payload) => json(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const action = (name, extra={}) => post('/api/action', {action:name, ...extra});
  const operator = (name, extra={}) => post('/api/workbench/action', {action:name, ...extra});
  function view() {return WorkbenchState.derive(snapshot, busy || !!pending || shuttingDown());}
  function referenceLocked() {return shuttingDown() || busy || !!pending || reference.initializing || (view().executing && !view().paused) || view().resetting;}
  const reference = new WorkbenchReference({json, post, changed:render, locked:referenceLocked});
  function referencePath() {const s = reference.selection; return s ? s.root.replace(/\/$/, '') + '/' + s.episode : '';}
  function replayMatches() {
    const r = snapshot.replay || {};
    return !!referencePath() && r.sample_dir === referencePath() &&
      r.execution_scope === $('replayScope').value && Number(r.frequency_hz) === Number($('replayHz').value) && Number(r.execution_steps) === Number($('replaySteps').value);
  }
  function replayConfig(dry=false) {
    if (!reference.selection) throw new Error('请在参考数据集中选择 episode。');
    return {sample_dir:referencePath(), execution_steps:Number($('replaySteps').value), frequency_hz:Number($('replayHz').value), execution_scope:$('replayScope').value, joint_source:'measured', loop:false, dry_run:dry};
  }
  function resetTarget() {
    if ($('resetMode').value === 'bias') return {mode:'bias'};
    if (!reference.selection) throw new Error('请先加载参考数据集并选择 episode。');
    return {mode:'episode', ...reference.selection};
  }
  async function command(operation, {urgent=false, ack=null, ackValue=null}={}) {
    if (shuttingDown()) return false;
    if (!urgent && (busy || pending)) return false;
    if (!connected) {feedback('连接已断开，请等待状态恢复。', true); return false;}
    if (!urgent) busy = true;
    render(); feedback('正在提交操作…');
    try {
      const result = await operation();
      if (ack && result.command_id) {
        pending = {id:result.command_id, type:ack, value:ackValue, at:Date.now()};
        if (ack === 'replay_load') replayRequest = {id:result.command_id, path:ackValue, timedOut:false};
      }
      feedback(ack === 'replay_load' ? '已提交 Replay 加载，正在检查样本；尚未开始回放。' : '请求已接受，等待控制器状态确认。'); return true;
    } catch (error) {feedback(error.message, true); return false;}
    finally {if (!urgent) busy = false; await refresh();}
  }
  async function selectMode(mode) {
    if (mode === snapshot.mode) return;
    if (!snapshot.capabilities?.modes.includes(mode) || !snapshot.mode_switch?.allowed) return;
    await command(() => mode === 'teleop' ? operator('select_teleop') : action('select_' + mode));
  }
  async function resetRobot() {
    const epoch = ++resetEpoch;
    await command(async () => {
      const target = resetTarget();
      if (snapshot.control?.execution_paused) {
        queuedReset = true; render();
        const deadline = Date.now() + 30000;
        try {
          // Wait for the recorder to close sampling, including final feedback.
          // Video/data saving then runs alongside reset; stop cancels the wait.
          while (true) {
            if (epoch !== resetEpoch) throw new Error('复位等待已取消。');
            await refresh();
            if (!connected || !snapshot.control?.online || !snapshot.control?.execution_paused) throw new Error('暂停状态已变化，复位已取消。');
            const r = snapshot.recording || {}, p = snapshot.policy || {};
            if (r.online && r.recording === false && !r.rollout_requested && !p.rollout_recording?.requested) break;
            if (Date.now() > deadline) throw new Error('采样尚未结束，未发起复位。请稍后再试。');
            feedback('等待当前样本采样结束后复位…');
            await new Promise(resolve => setTimeout(resolve, 150));
          }
        } finally {queuedReset = false;}
      }
      if (epoch !== resetEpoch) throw new Error('复位已取消。');
      return operator('reset_both', {reset_target:target});
    });
  }
  function showProtection() {
    if (!protection) return;
    $('protectionTitle').textContent = protection.title || '执行保护';
    $('protectionDetail').textContent = (protection.details || []).join('\n') || protection.message || '控制器触发保护，执行已停止。';
    $('protectionMeta').textContent = [protection.code, protection.source, protection.timestamp].filter(Boolean).join(' · ');
    $('protectionRaw').textContent = protectionDiagnostics;
    if (!$('protectionDialog').open) $('protectionDialog').showModal();
  }
  function renderPolicyMetrics(v, p, m) {
    const seconds = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toFixed(2) + ' s' : '—';
    const currentModel = m.phase === 'running' && p.model_identity_ok === true &&
      !(Number.isInteger(m.generation) && Number.isInteger(p.model_generation) && m.generation !== p.model_generation);
    const count = currentModel && Number.isInteger(p.inference_count) && p.inference_count > 0 ? p.inference_count : 0;
    const waiting = currentModel && connected && v.online && !p.error && p.phase !== 'fault' && !v.paused &&
      (v.running || v.preview) && p.inference_pending === true && Number.isInteger(p.pending_request_id) && p.pending_request_id >= 0 &&
      typeof p.pending_session_id === 'string' && p.pending_session_id === p.session_id;
    $('policyRequestLatency').textContent = waiting ? seconds(p.pending_elapsed_s) : count ? seconds(p.latency_s) : '—';
    $('policyRequestLabel').textContent = waiting ? `等待 Request #${p.pending_request_id}` : count ? `最近完成 Request #${p.last_request_id}` : '尚无请求';
    $('policyAverageLatency').textContent = count ? seconds(p.average_latency_s) : '—';
    $('policyInferenceCount').textContent = `${count} 次有效请求`;
    const progress = snapshot.actor_latency?.execution_progress || {};
    const length = progress.execute_length, start = progress.execute_start, done = progress.dispatched_steps;
    const validSlice = Number.isInteger(progress.request_id) && progress.request_id >= 0 &&
      Number.isInteger(length) && length > 0 && Number.isInteger(done) && done >= 0 && done <= length &&
      Number.isInteger(start) && start >= 0 && Number.isInteger(progress.action_length) && start + length <= progress.action_length;
    const active = currentModel && connected && v.online && !p.error && p.phase !== 'fault' && !progress.error && progress.active === true &&
      ['policy','hil'].includes(progress.provider) && typeof progress.session_id === 'string' && progress.session_id === p.session_id &&
      (v.running || v.preview || v.paused) && validSlice;
    $('policyChunkProgress').textContent = active ? `${done} / ${length} 步` : '—';
    $('policyChunkLabel').textContent = active ? `${v.paused || progress.paused ? '已暂停 · ' : ''}Request #${progress.request_id}` +
      (start || length !== progress.action_length ? ` · 从第 ${start + 1} 步起，完整 ${progress.action_length} 步` : '') :
      !connected || !v.online ? '等待状态连接' : p.error || p.phase === 'fault' || progress.error ? '执行已停止' : !currentModel ? '等待模型就绪' : v.running || v.preview ? '等待下一 Chunk' : '未执行';
    $('policyChunkProgress').title = active ? `已下发当前执行切片 ${done} 步；执行索引 [${start}, ${start + length})，完整动作 ${progress.action_length} 步。下发及 ACK 不表示物理动作已完成。` : '';
  }
  function render() {
    const v = view(), s = snapshot, r = s.recording || {}, m = s.model || {}, p = s.policy || {};
    $('connection').textContent = connected && v.online ? '控制器在线' : '连接已断开';
    $('connection').classList.toggle('online', connected && v.online);
    $('killed').disabled = !connected || !s.shutdown || shuttingDown();
    $('killed').textContent = shuttingDown() ? '退出中…' : 'killed';
    all('main > section').forEach(section => section.inert = shuttingDown());
    $('ownership').textContent = v.owner; $('captureStatus').textContent = v.capture; $('stage').textContent = v.stage;
    $('modeLabel').textContent = v.names[v.mode];
    $('modeReason').textContent = s.capabilities?.switchable ? s.mode_switch?.allowed ? '' : s.mode_switch?.reason : '当前启动入口仅提供部分模式';
    all('[data-mode]').forEach(e => {
      const available = s.capabilities?.modes.includes(e.dataset.mode);
      e.classList.toggle('selected', e.dataset.mode === v.mode); e.setAttribute('aria-pressed', String(e.dataset.mode === v.mode));
      e.disabled = shuttingDown() || busy || !!pending || !s.mode_switch?.allowed || !available;
      e.title = !available ? '当前入口不提供此模式' : s.mode_switch?.reason || '';
    });
    all('[data-modes]').forEach(e => e.hidden = !e.dataset.modes.split(' ').includes(v.mode));
    for (const id of ['start','pause','resume','stop','reset','startRecord','record','finish']) {
      $(id).disabled = !v.allowed[id]; $(id).title = v.allowed[id] ? '' : v.reason;
    }
    if (queuedReset) $('stop').disabled = false;
    const selected = reference.selection;
    if ($('resetMode').value === 'episode' && !selected) {$('reset').disabled = true; $('reset').title = '请先选择参考 episode';}
    $('reset').textContent = queuedReset ? '等待采样结束…' : v.resetting ? '复位中…' : '机器复位';
    $('resetMode').disabled = busy || !!pending || v.resetting || (v.executing && !v.paused) || reference.initializing;
    $('saveBias').disabled = !v.allowed.reset || v.executing;
    $('resetHint').textContent = ($('resetMode').value === 'bias' ? '目标：当前 Bias 位置。' : selected ? '目标：' + selected.episode + ' 的初始实测姿态。' : '请在下方选择参考 episode。') + (v.paused ? ' 采样收尾结束后即可复位，后台保存可同时进行。' : r.saving ? ' episode 正在后台保存，可同时复位。' : '');
    $('start').textContent = v.mode === 'teleop' ? '启用遥操' : '开始执行';
    $('resume').hidden = v.mode === 'teleop';
    $('stop').textContent = r.recording || r.rollout_requested || p.rollout_recording?.requested ? '停止并保存' : '停止';
    $('readiness').textContent = v.executing ? `${v.stage} · ${v.owner}` : v.reason;
    $('policyRecordLabel').hidden = v.mode !== 'policy'; $('policyRecord').checked = !!s.policy_recording;
    $('policyRecord').disabled = !v.allowed.policyRecord;
    $('policyRecordLabel').title = v.allowed.policyRecord ? '正式执行时采集，预览不录制' : !v.online ? '等待控制器连接' : busy || pending ? '正在提交操作' : '请先停止执行再修改保存选项';
    $('currentModel').textContent = m.run_id ? `${m.server_label || m.server_id} / ${m.task_id || m.task} / ${m.run_id}` : '尚未加载模型';
    $('currentPrompt').textContent = m.prompt ? '当前任务文本：' + m.prompt : ''; $('currentPrompt').hidden = !m.prompt;
    if (catalogServerId && m.server_id && catalogServerId !== m.server_id) clearModelCatalog();
    $('modelMachine').textContent = '1 · 推理机器：' + (m.server_label || '等待选择');
    $('serverSelect').disabled = !v.allowed.serverConfigure;
    if (m.server_id && [...$('serverSelect').options].some(option => option.value === m.server_id)) $('serverSelect').value = m.server_id;
    $('serverHint').textContent = m.server_label ? `${m.server_label}${m.local ? ' · 本机推理' : m.ssh_host ? ' · ' + m.ssh_host : ''}${m.configured === false ? ' · 尚未配置模型目录' : ''}` : '等待服务器配置';
    const phase = ({stopped:'未加载', starting:'启动中', loading:'加载与预热中', running:p.pending_interface_update ? '更新观测接口' : p.model_identity_ok ? '模型就绪' : '确认模型与观测接口', stopping:'卸载中', fault:'加载失败'})[m.phase] || '';
    $('sourcePhase').textContent = ['policy','hil'].includes(v.mode) ? phase : '';
    $('modelPhase').textContent = phase + (m.detail ? ' · ' + m.detail : '');
    $('modelLog').textContent = m.log_tail || '等待模型输出'; $('unloadModel').disabled = !v.allowed.modelUnload;
    $('chooseModel').disabled = !s.capabilities?.deployment;
    $('refreshModels').disabled = !s.capabilities?.deployment || busy;
    $('modelTask').disabled = !catalog || busy;
    all('[data-model-load]').forEach(e => e.disabled = !v.allowed.modelLoad || e.dataset.launchable !== 'true');
    renderPolicyMetrics(v, p, m);
    const root = r.recording_root || s.operator?.recording_root;
    if (root && !rootEdited) $('recordRoot').value = root;
    $('recordRootActual').textContent = (rootEdited ? '待应用；当前：' : '当前：') + (root || '等待数采节点');
    $('policyRecordPath').textContent = '保存至 UR：' + (root || '等待数采节点确认') + (rootEdited ? ' · 新目录待应用' : '');
    $('policyRecordPath').title = root ? `${root}/<episode>/raw/policy_inference.jsonl；相机和实测状态保存在同一 episode。` : '';
    $('applyRoot').disabled = !v.allowed.configure || !rootEdited;
    $('recordingHint').textContent = v.mode === 'hil' ? '正式执行自动采集；暂停或停止自动保存本条。' : v.mode === 'policy' ? s.policy_recording ? '已开启采集：停止自动保存，保存后可标记成功 / 失败。' : '勾选「保存推理数据」后，停止会自动保存 episode。' : v.mode === 'replay' ? 'Replay 使用参考 episode，不覆盖原始样本。' : '可随时浏览目录；录制或保存结束后应用。';
    $('directoryApplyHint').textContent = directoryTarget === 'reference' ? '选择后加载参考 episode 列表。' : v.allowed.configure ? '选择后应用为数据保存目录。' : '可先选择，录制 / 保存完成后再应用。';
    if(r.capture_notice)$('recordingHint').textContent += ' '+r.capture_notice;
    $('replaySelection').textContent = selected ? selected.episode : '选择下方参考 episode';
    const replay = s.replay || {}, loading = !!replayRequest && !replayRequest.timedOut;
    const replayError = replayRequest ? (replayRequest.timedOut ? '尚未收到本次加载确认，请检查 Replay 节点状态；可重新加载。' : '') :
      replay.error || (replay.command_result?.accepted === false ? replay.command_result.reason : '') || '';
    const replayLabels = {idle:'尚未加载样本', ready:'样本已就绪，可以开始回放', playing:'正在回放', aligning:'正在对齐初始姿态', paused:'回放已暂停', stopping:'正在停止回放', complete:'回放已完成', fault:'Replay 加载或执行失败'};
    $('replayStatus').textContent = loading ? '正在加载并检查样本，请等待…' : replayRequest?.timedOut ? '加载尚未确认' : replayLabels[replay.state] || '等待 Replay 节点状态';
    const replayPath = replayRequest?.path || (replay.error ? replay.requested_sample_dir : replay.sample_dir) || replay.command_result?.sample_dir || '';
    $('replaySample').textContent = replayPath; $('replaySample').hidden = !replayPath;
    $('replayError').textContent = replayError; $('replayError').hidden = !replayError;
    $('replayProgress').textContent = !replayRequest && replay.row_count ? `${replay.next_row || 0} / ${replay.row_count} 步` : '';
    $('loadReplay').textContent = loading ? '加载中…' : '加载参考 episode';
    const replayLocked = busy || !!pending || !v.online || v.executing || r.saving || r.recording;
    for (const id of ['replayScope','replayHz','replaySteps','loadReplay','unloadReplay']) $(id).disabled = replayLocked;
    $('loadReplay').disabled ||= !selected;
    if (v.mode === 'replay' && (replayRequest || !replayMatches())) {
      for (const id of ['start','resume']) {$(id).disabled = true; $(id).title = replayRequest ? '等待本次样本加载确认' : replayError || (v.paused ? '请先停止回放，再加载当前参考 episode 与回放参数' : '请先加载当前参考 episode 与回放参数');}
    }
    $('referenceRoot').disabled = $('scanReference').disabled = $('chooseReferenceDirectory').disabled = referenceLocked();
    $('referenceEpisode').disabled = referenceLocked() || $('referenceEpisode').options.length <= 1;
    $('episode').textContent = r.episode_id || '尚无记录';
    $('duration').textContent = r.episode_id ? `${Number(r.duration_s || 0).toFixed(1)} s` : '—';
    $('frames').textContent = r.episode_id ? `${r.frame_count || 0} 帧 · ${Number(r.achieved_rate_hz || 0).toFixed(1)} Hz` : '—';
    all('[data-learning]').forEach(e => e.disabled = !v.allowed[e.dataset.learning]);
    const transition=p.hil?.transition;
    $('hilTransition').textContent = transition?.active ? `交还策略：${transition.duration_s.toFixed(1)} 秒平滑过渡 · ${Math.round(transition.progress*100)}%` : '交还时重新推理，并平滑过渡至模型动作。';
    const event = p.protection || {};
    if (event.active) {
      const key = JSON.stringify([event.code, event.title, event.message, event.details, event.source, event.timestamp, event.timestamp_ns]);
      if (key !== protectionKey) {
        protectionKey = key; protection = event;
        protectionDiagnostics = JSON.stringify({protection, control:s.control, devices:s.devices, policy:s.policy}, null, 2);
        showProtection();
      }
    } else protectionKey = '';
    $('showProtection').hidden = !protection;
    $('savedPath').textContent = r.saved_path || '尚无已保存样本';
    $('quality').textContent = r.saving ? '保存中' : r.validation?.available ? r.validation.ok ? '质量检查通过' : '质量检查需关注' : '等待保存校验';
    $('recordError').textContent = r.last_error || r.rollout_error || '';
    all('[data-label]').forEach(e => e.disabled = !v.allowed.label || (e.dataset.label === 'upload' && (s.operator?.uploading || !r.validation?.ok)));
    $('uploadStatus').textContent = s.operator?.uploading ? '上传中…' : s.operator?.message || '';
    const parts=(r.saved_path||'').split('/').filter(Boolean), category=['success','fail'].includes(parts.at(-2))?parts.at(-2):'';
    const complete=r.validation?.available&&!!r.saved_path&&!r.recording&&!r.saving;
    $('sampleCategory').textContent = r.saving?'保存中':r.recording?'录制中':complete?({success:'已标记成功',fail:'已标记失败'})[category]||'待标记':'等待保存';
    $('sampleCategory').dataset.category = complete?category:'';
    $('classificationHint').textContent = r.saving?'写盘完成后可标记结果或丢弃本条数据。':r.recording?'停止并保存后，为本条 episode 选择结果。':complete?`最近样本：${parts.at(-1)} · 丢弃会永久删除本条数据。`:'保存完成后选择成功、失败或丢弃。';
    all('[data-label="mark_success"], [data-label="mark_fail"]').forEach(e=>e.setAttribute('aria-pressed',String(!!complete&&e.dataset.label==='mark_'+category)));
    renderDevices();
    const key = JSON.stringify(r.validation || {});
    if (key !== validationKey) {
      validationKey = key; $('validation').replaceChildren();
      for (const item of r.validation?.items || []) {
        const tr = document.createElement('tr');
        for (const value of [item.label || item.name || item.key, item.valid_frames ?? item.frames ?? '—', item.hz ?? item.rate_hz ?? '—', item.skipped ? '未启用' : item.ok ? '通过' : (item.details || []).join(' · ') || item.detail || '需检查']) {
          const td = document.createElement('td'); td.textContent = value; tr.append(td);
        }
        $('validation').append(tr);
      }
    }
  }
  function renderDevices() {
    const cards=WorkbenchState.deviceCards(snapshot,cameraHealth,connected), key=JSON.stringify(cards);
    if(key===deviceKey)return;
    deviceKey=key;
    const healthy=cards.filter(d=>['ok','busy'].includes(d.tone)).length, attention=cards.filter(d=>['error','warn'].includes(d.tone)).length;
    $('deviceSummary').textContent = `${healthy} / ${cards.length} 项正常${attention?' · '+attention+' 项需关注':''}`;
    $('deviceCards').replaceChildren();
    for(const d of cards){
      const card=document.createElement('article');card.className='device-card';card.dataset.device=d.key;card.dataset.tone=d.tone;
      const heading=document.createElement('div');heading.className='device-heading';
      const name=document.createElement('h3');name.textContent=d.name;
      const badge=document.createElement('span');badge.className='device-badge';badge.textContent=d.status;heading.append(name,badge);
      const kind=document.createElement('p');kind.className='device-kind';kind.textContent=d.kind;
      const metrics=document.createElement('dl');metrics.className='device-metrics';
      for(const [label,value] of d.rows){const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;row.append(dt,dd);metrics.append(row);}
      const detail=document.createElement('p');detail.className='device-detail';detail.textContent=d.detail;
      card.append(heading,kind,metrics,detail);$('deviceCards').append(card);
    }
  }
  function checkAck() {
    const replayResult = snapshot.replay?.command_result;
    if (replayRequest && replayResult?.id === replayRequest.id && replayResult.action === 'load') {
      feedback(replayResult.accepted ? 'Replay 样本已加载；点击「开始执行」进行回放。' : 'Replay 加载失败：' + (replayResult.reason || snapshot.replay.error || '请检查样本'), !replayResult.accepted);
      if (pending?.id === replayRequest.id) pending = null;
      replayRequest = null;
    }
    if (!pending) return;
    if (pending.type === 'replay_load') {
      if (Date.now() - pending.at > 60000) {
        replayRequest.timedOut = true;
        feedback('尚未收到 Replay 加载确认，请检查状态；不会自动重试。', true);
        pending = null;
      }
      return;
    }
    const result = pending.type === 'learning' ? snapshot.policy?.learning_command : snapshot.recording?.config_result;
    if (result?.id === pending.id) {
      feedback(result.accepted ? pending.type === 'root' ? '保存目录已应用并记忆，程序重启后自动恢复。' : '控制器已确认操作生效。' : result.reason || '操作已拒绝', !result.accepted);
      if (result.accepted && pending.type === 'root' && $('recordRoot').value.trim() === pending.value) rootEdited = false;
      pending = null;
    } else if (Date.now() - pending.at > 10000) {feedback('尚未收到控制器确认，请检查状态；不会自动重试。', true); pending = null;}
  }
  function refresh() {
    if (shutdownPhase === 'closed') return Promise.resolve();
    if (pollFlight) return pollFlight;
    pollFlight = (async () => {
      try {
        snapshot = await json('/api/workbench/status', {signal:AbortSignal.timeout(2000)}); connected = true;
        if (snapshot.shutdown) {
          shutdownPhase = snapshot.shutdown.phase;
          if (shutdownPhase === 'failed') shutdownRequested = false;
        }
        checkAck(); render();
        if (shuttingDown() || shutdownPhase === 'failed') feedback(snapshot.shutdown?.detail || '正在退出 ROS…', shutdownPhase === 'failed');
      }
      catch (error) {
        connected = false; snapshot = {...snapshot, control:{...snapshot.control, online:false}, mode_switch:{allowed:false, reason:'等待控制器连接'}};
        if (shutdownPhase === 'exiting') shutdownPhase = 'closed';
        render();
        feedback(shutdownPhase === 'closed' ? 'ROS 正在退出，网页服务已关闭。重新启动 ROS 后刷新页面。' : error.message, shutdownPhase !== 'closed');
      }
      finally {pollFlight = null;}
    })(); return pollFlight;
  }
  async function poll() {
    await refresh();
    if (shutdownPhase === 'closed') return;
    const v = view(), p = snapshot.policy || {};
    const active = connected && v.online && !v.paused && ['policy','hil'].includes(v.mode) && (v.running || v.preview) &&
      snapshot.model?.phase === 'running' && !p.error && p.phase !== 'fault' &&
      (p.inference_pending || snapshot.actor_latency?.execution_progress?.active);
    setTimeout(poll, active ? 200 : 500);
  }
  async function cameraLoop(name) {
    if (['exiting','closed'].includes(shutdownPhase)) return;
    const image = document.querySelector(`[data-camera="${name}"]`), empty = document.querySelector(`[data-camera-empty="${name}"]`), status = document.querySelector(`[data-camera-state="${name}"]`);
    if (!document.hidden) {
      let url;
      try {
        const response = await fetch('/api/workbench/camera/' + name, {cache:'no-store', signal:AbortSignal.timeout(2500)});
        if (!response.ok) throw new Error('等待相机连接');
        const stamp = response.headers.get('X-Source-Timestamp-Ns'), previous = stamps.get(name);
        if (previous?.stamp === stamp && performance.now() - previous.time > 2000) throw new Error('实时画面已过期');
        if (!previous || previous.stamp !== stamp) stamps.set(name, {stamp, time:performance.now(), wallTime:Date.now()});
        url = URL.createObjectURL(await response.blob()); const decoded = new Image(); decoded.src = url; await decoded.decode();
        const old = images.get(name); image.src = url; images.set(name, url); url = null; if (old) URL.revokeObjectURL(old);
        image.hidden = false; empty.hidden = true; status.textContent = '实时画面';
        cameraHealth[name]={online:true,width:decoded.naturalWidth,height:decoded.naturalHeight,updatedAt:stamps.get(name).wallTime};
        if (name === 'ego') reference.live([decoded.naturalWidth, decoded.naturalHeight]);
      } catch (_) {
        image.hidden = true; empty.hidden = false; empty.textContent = '画面未连接或已过期'; status.textContent = '无有效画面';
        cameraHealth[name]={...cameraHealth[name],online:false};
        if (name === 'ego') reference.live(null);
      } finally {if (url) URL.revokeObjectURL(url);}
    }
    if (!['exiting','closed'].includes(shutdownPhase)) setTimeout(() => cameraLoop(name), 200);
  }
  function clearModelCatalog() {
    catalogVersion++; catalog = null; catalogServerId = '';
    $('modelTask').replaceChildren(); $('modelList').replaceChildren();
    $('taskPrompt').textContent = '请选择当前机器的任务。';
  }
  async function loadModels() {
    const version = ++catalogVersion;
    const previousServer = catalogServerId, selectedTask = $('modelTask').value;
    catalog = null; $('modelTask').replaceChildren(); renderModels();
    try {
      const servers = await json('/api/servers');
      if (version !== catalogVersion) return;
      const serverId = servers.selected;
      catalogServerId = serverId;
      $('serverSelect').replaceChildren();
      for (const server of servers.servers) $('serverSelect').add(new Option(server.label, server.id));
      $('serverSelect').value = serverId;
      const models = await json('/api/models?' + new URLSearchParams({server_id:serverId}));
      if (version !== catalogVersion || (snapshot.model?.server_id && snapshot.model.server_id !== serverId)) return;
      if (models.server?.id !== serverId) throw new Error('推理机器已变化，请刷新清单。');
      catalog = models; $('modelTask').replaceChildren();
      $('modelTask').add(new Option('请选择任务', ''));
      for (const task of catalog.tasks || []) $('modelTask').add(new Option(task.label || task.task_id || task.name, task.name));
      if (previousServer === serverId && [...$('modelTask').options].some(o => o.value === selectedTask)) $('modelTask').value = selectedTask;
      renderModels();
    } catch (error) {if (version === catalogVersion) feedback(error.message, true);}
  }
  function renderModels() {
    const task = catalog?.tasks?.find(t => t.name === $('modelTask').value);
    $('taskPrompt').textContent = task ? task.prompt ? '任务默认文本：' + task.prompt : '任务未设置默认文本，请查看各 checkpoint 的任务文本。' :
      catalog?.configured === false ? '尚未配置模型目录。配置所选机器后刷新清单。' : catalog?.tasks?.length ? '请选择任务，再加载该任务下的 checkpoint。' : '模型清单为空。';
    $('modelList').replaceChildren();
    const version = catalogVersion, serverId = catalogServerId;
    for (const run of task?.runs || []) {
      const item = document.createElement('article'); item.className = 'model-item';
      const copy = document.createElement('div'), name = document.createElement('strong'), detail = document.createElement('p'), prompt = document.createElement('p'), button = document.createElement('button');
      name.textContent = run.label || `${run.model} / ${run.id || run.run_id}`;
      detail.className = 'muted'; detail.textContent = run.launchable ? `${run.variant || run.stage || '模型可用'}` : run.disabled_reason || '模型不可加载';
      const effectivePrompt = run.prompt ?? task.prompt ?? '';
      prompt.className = 'model-prompt'; prompt.textContent = '任务文本：' + (effectivePrompt || '未配置');
      button.textContent = '加载 checkpoint'; button.dataset.modelLoad = run.id || run.run_id; button.dataset.launchable = String(!!run.launchable && !!effectivePrompt.trim());
      button.onclick = async () => {
        if (version !== catalogVersion || serverId !== snapshot.model?.server_id || $('modelTask').value !== task.name) {
          feedback('机器或任务已变化，请重新选择 checkpoint。', true); return;
        }
        const payload = {server_id:serverId, task:task.name, run_id:run.run_id, prompt:effectivePrompt, checkpoint_path:run.checkpoint_path};
        if (await command(() => post('/api/model/start', payload))) $('modelDialog').close();
      };
      copy.append(name, detail, prompt); item.append(copy, button); $('modelList').append(item);
    }
    render();
  }
  function clearDirectoryParent() {
    directoryParent = ''; $('directoryParent').disabled = $('newDirectory').disabled = true;
    $('createDirectoryForm').hidden = true;
  }
  async function browseDirectory(path) {
    const version = ++directoryVersion;
    clearDirectoryParent();
    $('directoryPath').value = path; $('directoryError').textContent = '正在读取目录…'; $('directoryList').replaceChildren();
    try {
      const listing = await json('/api/workbench/directories?' + new URLSearchParams({path}));
      if (version !== directoryVersion) return;
      directoryParent = listing.path; $('directoryParent').disabled = $('newDirectory').disabled = false;
      $('directoryPath').value = listing.path; $('directoryError').textContent = listing.truncated ? '仅显示前 500 个子目录，也可直接输入路径。' : '';
      $('directoryParent').onclick = () => browseDirectory(listing.parent);
      $('directoryLocations').replaceChildren();
      for (const location of listing.locations) {const button = document.createElement('button'); button.textContent = location; button.onclick = () => browseDirectory(location); $('directoryLocations').append(button);}
      for (const child of listing.directories) {const button = document.createElement('button'); button.textContent = child.name; button.onclick = () => browseDirectory(child.path); $('directoryList').append(button);}
      if (!listing.directories.length) $('directoryList').textContent = '此目录没有子目录。';
    } catch (error) {if (version === directoryVersion) $('directoryError').textContent = error.message + '；可打开已有父目录，再新建文件夹。';}
  }
  function openDirectory(target) {
    directoryTarget = target;
    $('directoryTitle').textContent = target === 'reference' ? '选择参考数据集目录' : '选择数据保存目录';
    $('directoryLocations').replaceChildren();
    for (const [label, path] of [['主目录', ''], ['/media', '/media'], ['/mnt', '/mnt'], ['/tmp', '/tmp'], ['/', '/']]) {
      const button = document.createElement('button'); button.textContent = label;
      button.onclick = () => browseDirectory(path); $('directoryLocations').append(button);
    }
    $('directoryDialog').showModal(); render();
    browseDirectory($(target === 'reference' ? 'referenceRoot' : 'recordRoot').value || '');
  }
  async function applyRoot() {
    if (!view().allowed.configure) {feedback('请等待当前采集启动、录制或保存完成，再应用目录。'); return;}
    const requested = $('recordRoot').value.trim();
    await command(async () => {
      const result = await (snapshot.capabilities?.deployment ? post('/api/recording/config', {recording_root:requested}) : operator('set_paths', {recording_root:requested}));
      if (!result.command_id && $('recordRoot').value.trim() === requested) rootEdited = false;
      return result;
    }, {ack:'root', ackValue:requested});
  }
  all('[data-close]').forEach(e => e.onclick = () => $(e.dataset.close).close());
  $('killed').onclick = async () => {
    if (shuttingDown() || !connected) return;
    shutdownRequested = true; resetEpoch++; pending = null;
    all('dialog[open]').forEach(dialog => dialog.close());
    render(); feedback('正在停止执行并准备退出 ROS…');
    try {
      const result = await post('/api/workbench/shutdown', {action:'killed'});
      shutdownPhase = result.phase;
      feedback(result.detail || '正在退出 ROS…');
    } catch (error) {
      shutdownRequested = false; feedback(error.message, true);
    }
    await refresh();
  };
  all('[data-mode]').forEach(e => e.onclick = () => selectMode(e.dataset.mode));
  $('start').onclick = () => command(() => snapshot.mode === 'teleop' ? operator('run_ur') : snapshot.mode === 'replay' ? post('/api/replay/live/start', replayConfig(false)) : action('start_deploy'));
  $('pause').onclick = () => command(() => snapshot.mode === 'teleop' ? operator('pause') : snapshot.mode === 'replay' ? post('/api/replay', {action:'pause'}) : action('pause_deploy'));
  $('resume').onclick = () => command(() => snapshot.mode === 'replay' ? post(snapshot.replay?.dry_run ? '/api/replay/preview/start' : '/api/replay/live/start', replayConfig(!!snapshot.replay?.dry_run)) : action('continue_deploy'));
  // Stop cancels a queued reset even while the normal command lane is busy.
  $('stop').onclick = async () => {
    resetEpoch++;
    const savingExpected=snapshot.recording?.recording||snapshot.recording?.rollout_requested||snapshot.policy?.rollout_recording?.requested;
    if(await command(() => snapshot.mode === 'replay' && !WorkbenchState.derive(snapshot).resetting ? post('/api/replay', {action:'stop'}) : operator('stop'), {urgent:true})) {
      if(savingExpected)feedback('已请求停止，当前 episode 将自动保存；请等待保存结果。');
    }
  };
  $('reset').onclick = resetRobot;
  $('startRecord').onclick = () => command(() => operator('start_teleop_record'));
  $('record').onclick = () => command(() => operator('start_record')); $('finish').onclick = () => command(() => operator('stop_record'));
  $('policyRecord').onchange = event => {
    // command() renders the confirmed snapshot before invoking its callback.
    // Capture the click now so that render cannot reverse the requested value.
    const enabled = event.currentTarget.checked;
    command(() => action(enabled ? 'enable_policy_recording' : 'disable_policy_recording'));
  };
  $('loadReplay').onclick = () => command(() => post('/api/replay', {action:'load', ...replayConfig(false)}), {ack:'replay_load', ackValue:referencePath()});
  $('unloadReplay').onclick = async () => {if (await command(() => post('/api/replay', {action:'unload'}))) {replayRequest = null; render();}};
  for (const id of ['replayScope','replayHz','replaySteps']) $(id).onchange = render;
  $('recordRoot').oninput = () => {rootEdited = true; render();};
  $('recordRoot').onkeydown = e => {if (e.key === 'Enter') applyRoot();};
  $('chooseDirectory').onclick = () => openDirectory('recording');
  $('policyRecordDirectory').onclick = () => openDirectory('recording');
  $('chooseReferenceDirectory').onclick = () => openDirectory('reference');
  $('directoryDialog').addEventListener('close', () => {directoryVersion++; clearDirectoryParent();});
  $('browseDirectory').onclick = () => browseDirectory($('directoryPath').value.trim());
  $('directoryPath').oninput = () => {directoryVersion++; clearDirectoryParent(); $('directoryList').replaceChildren();};
  $('directoryPath').onkeydown = e => {if (e.key === 'Enter') browseDirectory($('directoryPath').value.trim());};
  $('newDirectory').onclick = () => {
    if (!directoryParent) return;
    $('createDirectoryForm').hidden = false; $('newDirectoryName').value = ''; $('newDirectoryName').focus();
  };
  $('cancelCreateDirectory').onclick = () => {$('createDirectoryForm').hidden = true;};
  $('createDirectoryForm').onsubmit = async e => {
    e.preventDefault();
    if (creatingDirectory || !directoryParent) return;
    const parent = directoryParent, version = directoryVersion, name = $('newDirectoryName').value.trim();
    if (!name || ['.', '..'].includes(name) || /[/\\\0]/.test(name)) {
      $('directoryError').textContent = '请输入单个文件夹名称，不能包含路径分隔符或 ..'; return;
    }
    creatingDirectory = true; $('createDirectory').disabled = true;
    try {
      const result = await post('/api/workbench/directories', {parent, name});
      if (version === directoryVersion && $('directoryDialog').open) {
        await browseDirectory(result.path);
        $('directoryError').textContent = '已创建文件夹：' + name;
      }
    } catch (error) {if (version === directoryVersion) $('directoryError').textContent = error.message;}
    finally {creatingDirectory = false; $('createDirectory').disabled = false;}
  };
  $('useDirectory').onclick = async () => {
    const path = $('directoryPath').value.trim();
    if (!path.startsWith('/') || path.includes('\0')) {$('directoryError').textContent = '请输入机器人主机上的绝对路径。'; return;}
    if (directoryTarget === 'reference') {
      if (referenceLocked()) {$('directoryError').textContent = '当前执行中，请暂停或停止后更换参考数据集。'; return;}
      $('referenceRoot').value = path;
      $('referenceRoot').dispatchEvent(new Event('input'));
      $('directoryDialog').close(); await reference.scan();
    } else {
      $('recordRoot').value = path; rootEdited = true; $('directoryDialog').close();
      if (view().allowed.configure) await applyRoot();
      else {feedback('已选择目录；当前录制或保存结束后，点击「应用目录」。'); render();}
    }
  };
  $('applyRoot').onclick = applyRoot;
  all('[data-learning]').forEach(e => e.onclick = () => command(() => post('/api/learning', {action:e.dataset.learning}), {ack:'learning'}));
  all('[data-label]').forEach(e => e.onclick = () => {
    const saved_path = snapshot.recording?.saved_path; if (!saved_path) return;
    command(() => operator(e.dataset.label, {saved_path}));
  });
  $('saveBias').onclick = () => command(() => operator('save_bias'));
  $('serverSelect').onchange = async event => {
    const id = event.target.value;
    clearModelCatalog(); renderModels();
    if (await command(() => post('/api/server/select', {server_id:id}))) await loadModels();
  };
  loadModels();
  $('chooseModel').onclick = () => {$('modelDialog').showModal(); loadModels();};
  $('refreshModels').onclick = loadModels; $('modelTask').onchange = renderModels;
  $('unloadModel').onclick = () => command(() => post('/api/model/stop', {}));
  $('showProtection').onclick = showProtection;
  window.addEventListener('pagehide', () => {resetEpoch++; for (const url of images.values()) URL.revokeObjectURL(url);});
  render(); poll(); reference.init(); ['ego','left_wrist','right_wrist'].forEach(cameraLoop);
})();
