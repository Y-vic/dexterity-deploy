/* Shared, display-only scene comparison. No robot or recorder commands. */
class EgoReferenceView extends HTMLElement {
  connectedCallback() {
    if (this.ready) return;
    this.ready = true;
    this.attachShadow({mode: 'open'});
    this.shadowRoot.innerHTML = `
      <style>
        :host {display:block;margin:18px 0;color:#e5edf4;font:14px/1.5 system-ui,sans-serif}
        * {box-sizing:border-box} .panel {padding:18px;border:1px solid #31505d;border-radius:12px;background:#0b1820}
        h2 {font-size:18px;margin:0} .row {display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:12px 0}
        label {display:flex;align-items:center;gap:8px} .root {flex:1;min-width:220px;white-space:nowrap} .root input {width:100%;min-width:0}
        input,select,button {font:inherit;color:inherit;background:#132733;border:1px solid #416171;border-radius:6px;padding:7px}
        button {cursor:pointer} button:disabled,input:disabled,select:disabled {opacity:.5;cursor:default}
        input[type=range] {padding:0;accent-color:#5cdbc1} #episode {flex:1;min-width:220px;max-width:100%}
        #time {flex:1;min-width:140px} .stage {position:relative;width:100%;aspect-ratio:16/9;background:#02070c;overflow:hidden}
        .stage img {position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
        [hidden] {display:none!important} .notice {position:absolute;inset:0;display:grid;place-items:center;background:#07121de8;color:#ffd499;text-align:center;padding:24px;z-index:3}
        .hint {font-size:12px;color:#afc4d0;overflow-wrap:anywhere} #error,#alignment {color:#ffd499;min-height:18px}
        #liveStatus {margin-left:auto} .view {max-width:1100px;margin:auto}
        .panel:fullscreen {overflow:auto;border-radius:0;width:100%;height:100%}
        .panel:fullscreen .view {max-width:min(100%,150vh)}
        :focus-visible {outline:2px solid #5cdbc1;outline-offset:2px}
      </style>
      <section class="panel" aria-label="Ego 场景对照">
        <div class="row"><h2>Ego 场景对照</h2><label><input id="feature" type="checkbox" role="switch">开启</label><span id="liveStatus" role="status">已关闭</span><button id="fullscreen" hidden>全屏</button></div>
        <div id="content" hidden>
        <div class="row"><label class="root">参考数据集<input id="root" aria-label="参考数据集绝对路径" placeholder="/home/yns/data/0916_ppp"></label><button id="scan">加载目录</button></div>
        <div class="row"><select id="category" aria-label="样本分类"><option value="all">全部分类</option><option value="success">Success</option><option value="fail">Fail</option><option value="root">未分类</option></select><select id="episode" aria-label="参考样本" disabled><option value="">请选择样本</option></select></div>
        <div class="view"><div class="stage"><img id="live" alt="当前 Ego 实时画面" hidden><img id="reference" alt="历史参考帧" hidden><div class="notice" id="offline">等待实时 Ego 画面</div></div></div>
        <div class="row"><label><input type="checkbox" id="enabled" checked>叠加参考帧</label><label>参考透明度<input id="alpha" type="range" min="0" max="100" value="50"><output id="alphaValue">50%</output></label></div>
        <div class="row"><button id="previous" disabled>上一帧</button><input id="time" aria-label="参考采集时间" type="range" min="0" max="0" step="0.001" value="0" disabled><button id="next" disabled>下一帧</button></div>
        <div class="hint" id="selection">尚未选择参考样本</div><div class="hint" id="alignment"></div><div class="hint" id="error" role="status"></div>
        <div class="hint">历史帧固定，当前画面持续更新。背景整体重影时，请先检查相机位置和视角。</div>
        </div>
      </section>`;
    this.$ = id => this.shadowRoot.getElementById(id);
    this.active = false;
    this.activation = 0;
    this.generation = 0;
    this.frameVersion = 0;
    this.index = 0;
    this.episodes = [];
    this.saved = {};
    try {this.saved = JSON.parse(localStorage.getItem('ego-reference-selection') || '{}') || {};} catch (_) {}
    this.$('alpha').value = this.saved.alpha ?? 50;
    this.$('enabled').checked = this.saved.enabled ?? true;
    this.$('feature').onchange = () => this.setActive(this.$('feature').checked);
    this.$('scan').onclick = () => this.scan();
    this.$('root').onkeydown = event => {if (event.key === 'Enter') this.scan();};
    this.$('category').onchange = () => this.renderEpisodes();
    this.$('episode').onchange = () => this.selectEpisode();
    this.$('previous').onclick = () => this.seek(this.index - 1);
    this.$('next').onclick = () => this.seek(this.index + 1);
    this.$('time').oninput = () => {
      if (!this.meta) return;
      const seconds = Number(this.$('time').value), times = this.meta.seconds;
      let low = 0, high = times.length - 1;
      while (low < high) {const mid = (low + high) >> 1; if (times[mid] < seconds) low = mid + 1; else high = mid;}
      if (low && seconds - times[low - 1] <= times[low] - seconds) low--;
      this.seek(low, 150);
    };
    for (const id of ['alpha', 'enabled']) this.$(id).oninput = () => {this.paint(); this.save();};
    this.$('fullscreen').onclick = async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await this.shadowRoot.querySelector('.panel').requestFullscreen();
      } catch (error) {this.$('error').textContent = `无法进入全屏：${error.message}`;}
    };
    this.paint();

  }

  async json(url) {
    const response = await fetch(url, {cache:'no-store', signal:this.abort.signal});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  async initialize() {
    const activation = this.activation;
    try {
      const config = await this.json('/api/reference/config');
      if (!this.active || activation !== this.activation) return;
      this.$('root').value = this.saved.root || config.root;
      await this.scan(this.saved.episode);
    } catch (error) {if (this.active && activation === this.activation) this.$('error').textContent = error.message;}
  }

  clearReference() {
    this.meta = null;
    this.frameVersion++;
    clearTimeout(this.seekTimer);
    this.$('reference').hidden = true;
    this.hasReference = false;
    for (const id of ['previous', 'next', 'time']) this.$(id).disabled = true;
    this.$('selection').textContent = '尚未选择参考样本';
    this.$('alignment').textContent = '';
  }

  async scan(preferred = '') {
    const generation = ++this.generation;
    this.clearReference();
    this.episodes = [];
    this.$('episode').replaceChildren(new Option('请选择参考样本', ''));
    this.$('episode').disabled = true;
    this.$('error').textContent = '正在读取样本目录…';
    try {
      const result = await this.json('/api/reference/episodes?' + new URLSearchParams({root:this.$('root').value.trim()}));
      if (generation !== this.generation) return;
      this.rootPath = result.root;
      this.$('root').value = result.root;
      this.episodes = result.episodes;
      this.$('error').textContent = result.episodes.length ? '' : '目录内没有已完成且带 Ego 视频和索引的样本';
      this.renderEpisodes(preferred);
    } catch (error) {if (generation === this.generation) this.$('error').textContent = error.message;}
  }

  renderEpisodes(preferred = '') {
    ++this.generation;
    this.clearReference();
    const select = this.$('episode'), category = this.$('category').value;
    select.replaceChildren(new Option('请选择参考样本', ''));
    for (const item of this.episodes) {
      if (category === 'all' || item.category === category) select.add(new Option(`[${item.category}] ${item.name}`, item.id));
    }
    select.disabled = select.options.length <= 1;
    if (preferred && [...select.options].some(option => option.value === preferred)) {
      select.value = preferred;
      this.selectEpisode(true);
    }
  }

  query(extra = {}) {
    return new URLSearchParams({root:this.rootPath, episode:this.$('episode').value, client:this.client, ...extra});
  }

  async selectEpisode(restore = false) {
    const generation = ++this.generation;
    this.clearReference();
    if (!this.$('episode').value) return;
    this.$('error').textContent = '正在读取时间索引…';
    try {
      const meta = await this.json('/api/reference/metadata?' + this.query());
      if (generation !== this.generation) return;
      this.meta = meta;
      this.$('time').max = meta.duration_s;
      this.$('time').disabled = false;
      const savedIndex = restore ? meta.frames.indexOf(this.saved.frame) : 0;
      this.seek(Math.max(0, savedIndex));
    } catch (error) {if (generation === this.generation) this.$('error').textContent = error.message;}
  }

  seek(index, delay = 0) {
    if (!this.active || !this.meta) return;
    this.index = Math.max(0, Math.min(index, this.meta.frames.length - 1));
    this.$('time').value = this.meta.seconds[this.index];
    this.$('previous').disabled = this.index === 0;
    this.$('next').disabled = this.index === this.meta.frames.length - 1;
    this.hasReference = false;
    this.paint();
    this.$('selection').textContent = `${this.meta.episode} · 正在加载帧 ${this.meta.frames[this.index]} · ${this.meta.seconds[this.index].toFixed(3)} 秒`;
    this.$('error').textContent = '';
    this.frameVersion++;
    clearTimeout(this.seekTimer);
    this.seekTimer = setTimeout(() => this.loadFrame(), delay);
  }

  async loadFrame() {
    if (!this.active || !this.meta) return;
    // One outstanding decode per browser. New seeks replace the pending target.
    if (this.loadingFrame) {this.pendingFrame = true; return;}
    this.loadingFrame = true;
    const version = this.frameVersion, generation = this.generation;
    const meta = this.meta, index = this.index;
    let url;
    try {
      const response = await fetch('/api/reference/frame.jpg?' + this.query({frame:meta.frames[index]}), {cache:'no-store', signal:this.abort.signal});
      if (!response.ok) throw new Error((await response.json()).error || '参考帧读取失败');
      url = URL.createObjectURL(await response.blob());
      const decoded = new Image(); decoded.src = url; await decoded.decode();
      if (!this.active || version !== this.frameVersion || generation !== this.generation) return;
      this.$('reference').src = url;
      if (this.referenceURL) URL.revokeObjectURL(this.referenceURL);
      this.referenceURL = url; url = null;
      this.referenceSize = [decoded.naturalWidth, decoded.naturalHeight];
      this.hasReference = true;
      this.$('selection').textContent = `${meta.episode} · 帧 ${meta.frames[index]} · ${meta.seconds[index].toFixed(3)} 秒 · ${decoded.naturalWidth}×${decoded.naturalHeight}`;
      this.$('error').textContent = '';
      this.paint(); this.save();
    } catch (error) {
      if (version === this.frameVersion && generation === this.generation) this.$('error').textContent = error.message;
    } finally {
      if (url) URL.revokeObjectURL(url);
      this.loadingFrame = false;
      if (this.pendingFrame) {this.pendingFrame = false; if (this.meta) this.loadFrame();}
    }
  }

  paint() {
    const alpha = Number(this.$('alpha').value);
    this.$('alphaValue').textContent = `${alpha}%`;
    this.$('reference').style.opacity = alpha / 100;
    const a = this.liveSize, b = this.referenceSize;
    const mismatch = this.hasReference && a && b && Math.abs(a[0] / a[1] - b[0] / b[1]) > 0.01;
    this.$('alignment').textContent = mismatch ? '实时与参考画面宽高比不同，已暂停叠加；请检查相机分辨率和裁切设置。' : '';
    this.$('reference').hidden = !this.hasReference || !this.$('enabled').checked || Boolean(mismatch);
  }

  save() {
    const state = {root:this.rootPath, episode:this.$('episode').value,
      frame:this.meta?.frames[this.index], alpha:Number(this.$('alpha').value), enabled:this.$('enabled').checked};
    try {localStorage.setItem('ego-reference-selection', JSON.stringify(state));} catch (_) {}
  }

  async pollLive() {
    if (!this.isConnected || !this.active) return;
    const activation = this.activation;
    let url;
    try {
      const response = await fetch('/api/reference/live.jpg', {cache:'no-store', signal:AbortSignal.any([this.abort.signal, AbortSignal.timeout(2000)])});
      if (!response.ok) throw new Error('实时 Ego 画面已过期或未连接');
      const stamp = response.headers.get('X-Source-Timestamp-Ns');
      if (stamp === this.liveStamp) {
        if (performance.now() - this.liveReceived > 2000) throw new Error('实时 Ego 画面已过期');
        return;
      }
      url = URL.createObjectURL(await response.blob());
      const decoded = new Image(); decoded.src = url; await decoded.decode();
      if (!this.active || activation !== this.activation) return;
      this.$('live').src = url;
      if (this.liveURL) URL.revokeObjectURL(this.liveURL);
      this.liveURL = url; url = null;
      this.liveStamp = stamp; this.liveReceived = performance.now();
      this.liveSize = [decoded.naturalWidth, decoded.naturalHeight];
      this.shadowRoot.querySelector('.stage').style.aspectRatio = `${decoded.naturalWidth}/${decoded.naturalHeight}`;
      this.$('live').hidden = false;
      this.$('offline').hidden = true;
      this.$('liveStatus').textContent = `实时在线 · ${decoded.naturalWidth}×${decoded.naturalHeight}`;
      this.paint();
    } catch (error) {
      if (!this.active || activation !== this.activation) return;
      this.$('offline').hidden = false;
      this.$('offline').textContent = error.name === 'TimeoutError' ? '实时 Ego 画面请求超时' : error.message;
      this.$('liveStatus').textContent = '实时画面不可用';
    } finally {
      if (url) URL.revokeObjectURL(url);
      if (this.isConnected && this.active && activation === this.activation) this.liveTimer = setTimeout(() => this.pollLive(), 100);
    }
  }

  async setActive(active) {
    if (active === this.active) return;
    this.active = active;
    const activation = ++this.activation;
    this.$('feature').checked = active;
    this.$('content').hidden = !active;
    this.$('fullscreen').hidden = !active;
    if (active) {
      await this.closing;
      if (!this.active || activation !== this.activation) return;
      this.abort = new AbortController();
      this.client = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      try {this.saved = JSON.parse(localStorage.getItem('ego-reference-selection') || '{}') || {};} catch (_) {}
      this.liveStamp = null;
      this.$('liveStatus').textContent = '等待实时画面';
      this.$('offline').hidden = false;
      this.$('offline').textContent = '等待实时 Ego 画面';
      this.pollLive(); this.initialize();
    } else {
      this.abort?.abort();
      clearTimeout(this.liveTimer); clearTimeout(this.seekTimer);
      this.generation++; this.frameVersion++;
      this.pendingFrame = false;
      this.clearReference(); this.episodes = [];
      for (const id of ['live', 'reference']) {this.$(id).removeAttribute('src'); this.$(id).hidden = true;}
      for (const key of ['liveURL', 'referenceURL']) {if (this[key]) URL.revokeObjectURL(this[key]); this[key] = null;}
      this.$('episode').replaceChildren(new Option('请选择样本', ''));
      this.$('liveStatus').textContent = '已关闭';
      this.closing = fetch('/api/reference/close?' + new URLSearchParams({client:this.client || ''}),
        {method:'POST', keepalive:true}).catch(() => {});
    }
  }

  disconnectedCallback() {
    this.setActive(false);
  }
}
customElements.define('ego-reference-view', EgoReferenceView);
