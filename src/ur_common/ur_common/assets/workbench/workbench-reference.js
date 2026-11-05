'use strict';
/* One selection owns the overlay, reset target and replay source. Restoring it
 * reads metadata only; it never sends an execution or model command. */
class WorkbenchReference {
  constructor({json, post, changed, locked}) {
    Object.assign(this, {json, post, changed, locked});
    this.$ = id => document.getElementById(id);
    this.client = 'workbench-' + crypto.randomUUID();
    this.generation = 0; this.version = 0; this.root = ''; this.episode = '';
    this.persist = Promise.resolve(); this.initializing = true;
    this.$('referenceRoot').oninput = () => {
      this.invalidate(); this.root = ''; this.episode = '';
      this.$('referenceEpisode').replaceChildren(new Option('请先加载目录', ''));
      this.$('referenceEpisode').disabled = true;
      this.$('referenceStatus').textContent = '目录已修改，请加载目录并选择 episode。'; this.changed();
    };
    this.$('scanReference').onclick = () => this.scan();
    this.$('referenceRoot').onkeydown = e => {if (e.key === 'Enter') this.scan();};
    this.$('referenceEpisode').onchange = () => this.select(false);
    this.$('resetMode').onchange = () => {this.save(); this.changed();};
    this.$('referenceEnabled').onchange = () => this.paint();
    this.$('referenceAlpha').oninput = () => this.paint();
    this.$('referenceTime').oninput = () => {
      if (!this.meta) return;
      const seconds = Number(this.$('referenceTime').value), times = this.meta.seconds;
      let low = 0, high = times.length - 1;
      while (low < high) {const mid = (low + high) >> 1; if (times[mid] < seconds) low = mid + 1; else high = mid;}
      if (low && seconds - times[low - 1] <= times[low] - seconds) low--;
      this.seek(low);
    };
    this.$('previousFrame').onclick = () => this.seek(this.index - 1);
    this.$('nextFrame').onclick = () => this.seek(this.index + 1);
    window.addEventListener('pagehide', () => {
      this.invalidate();
      fetch('/api/reference/close?client=' + encodeURIComponent(this.client), {method:'POST', keepalive:true}).catch(() => {});
    });
  }
  get selection() {return this.root && this.episode ? {root:this.root, episode:this.episode} : null;}
  query(extra={}) {return new URLSearchParams({...this.selection, client:this.client, ...extra});}
  invalidate() {
    this.generation++; this.version++; this.meta = null; this.hasFrame = false;
    if (this.abort) this.abort.abort();
    if (this.url) {URL.revokeObjectURL(this.url); this.url = null;}
    this.$('referenceImage').removeAttribute('src');
    this.$('referenceTime').disabled = this.$('previousFrame').disabled = this.$('nextFrame').disabled = true;
    this.$('referenceFrame').textContent = '未选择参考帧'; this.$('referenceError').textContent = '';
    this.paint(); this.changed();
  }
  async init() {
    try {
      let saved = await this.json('/api/workbench/preferences');
      if (!saved.reference_root) {
        try {const old = JSON.parse(localStorage.getItem('ego-reference-selection') || '{}');
          if (old.root) saved = {...saved, reference_root:old.root, reference_episode:old.episode || ''};
        } catch (_) {}
      }
      if (!saved.reference_root) saved.reference_root = (await this.json('/api/reference/config')).root || '';
      this.$('referenceRoot').value = saved.reference_root;
      this.$('resetMode').value = saved.reset_mode || 'bias';
      if (saved.reference_root) await this.scan(saved.reference_episode || '', true);
    } catch (error) {this.$('referenceStatus').textContent = '读取已记忆的参考失败：' + error.message;}
    finally {this.initializing = false; this.changed();}
  }
  save() {
    const value = {reference_root:this.root, reference_episode:this.episode, reset_mode:this.$('resetMode').value};
    // Serialize writes so a slower older request cannot overwrite a newer choice.
    this.persist = this.persist.catch(() => {}).then(() => this.post('/api/workbench/preferences', value));
    this.persist.then(() => {
      if (this.root === value.reference_root && this.episode === value.reference_episode)
        this.$('referenceStatus').textContent = this.episode ? '已记忆 · Ego 对照 / 初始姿态复位 / Replay 共用此 episode。拖动历史帧不改变初始复位位置。' : '目录已记忆，请选择 episode。';
    }).catch(error => {this.$('referenceStatus').textContent = '记忆选择失败：' + error.message;});
  }
  async scan(preferred='', restore=false) {
    if (this.locked() && !this.initializing) return;
    const root = this.$('referenceRoot').value.trim();
    this.invalidate(); this.root = ''; this.episode = '';
    const generation = this.generation;
    this.$('referenceEpisode').replaceChildren(new Option('请选择 episode', ''));
    this.$('referenceEpisode').disabled = true;
    this.$('referenceStatus').textContent = '正在读取参考数据集…';
    try {
      const result = await this.json('/api/reference/episodes?' + new URLSearchParams({root}));
      if (generation !== this.generation) return;
      this.root = result.root; this.$('referenceRoot').value = result.root;
      for (const item of result.episodes) this.$('referenceEpisode').add(new Option(item.id, item.id));
      this.$('referenceEpisode').disabled = !result.episodes.length;
      if (preferred && result.episodes.some(item => item.id === preferred)) {
        this.$('referenceEpisode').value = preferred; await this.select(restore);
      } else {
        this.$('referenceStatus').textContent = preferred ? '已记忆的 episode 当前不存在，请重新选择；不会自动替换样本。' : `${result.episodes.length} 条 episode，请选择参考。`;
        if (!restore) this.save();
      }
    } catch (error) {if (generation === this.generation) this.$('referenceStatus').textContent = error.message;}
    this.changed();
  }
  async select(restore) {
    this.invalidate(); this.episode = this.$('referenceEpisode').value;
    if (!restore && this.episode) this.$('resetMode').value = 'episode';
    this.save(); this.changed();
    if (!this.selection) return;
    const generation = this.generation;
    try {
      const meta = await this.json('/api/reference/metadata?' + this.query());
      if (generation !== this.generation) return;
      if (!meta.frames?.length || meta.frames.length !== meta.seconds?.length) throw new Error('样本没有有效 Ego 时间索引');
      this.meta = meta; this.$('referenceTime').max = meta.duration_s;
      this.$('referenceTime').disabled = false; this.seek(0);
    } catch (error) {if (generation === this.generation) this.$('referenceError').textContent = '历史画面不可用：' + error.message;}
  }
  seek(index) {
    if (!this.meta) return;
    this.index = Math.max(0, Math.min(index, this.meta.frames.length - 1));
    this.$('referenceTime').value = this.meta.seconds[this.index];
    this.$('previousFrame').disabled = this.index === 0;
    this.$('nextFrame').disabled = this.index === this.meta.frames.length - 1;
    this.hasFrame = false; this.version++; this.paint();
    this.$('referenceFrame').textContent = '正在读取历史帧…';
    clearTimeout(this.seekTimer); this.seekTimer = setTimeout(() => this.loadFrame(), 70);
  }
  async loadFrame() {
    if (!this.meta) return;
    if (this.loading) {this.pendingFrame = true; return;}
    this.loading = true; this.abort = new AbortController();
    const generation = this.generation, version = this.version, meta = this.meta, index = this.index;
    let url;
    try {
      const response = await fetch('/api/reference/frame.jpg?' + this.query({frame:meta.frames[index]}), {cache:'no-store', signal:AbortSignal.any([this.abort.signal, AbortSignal.timeout(25000)])});
      if (!response.ok) throw new Error('参考帧读取失败');
      url = URL.createObjectURL(await response.blob());
      const decoded = new Image(); decoded.src = url; await decoded.decode();
      if (generation !== this.generation || version !== this.version) return;
      if (this.url) URL.revokeObjectURL(this.url);
      this.url = url; url = null; this.$('referenceImage').src = this.url;
      this.size = [decoded.naturalWidth, decoded.naturalHeight]; this.hasFrame = true;
      this.$('referenceFrame').textContent = `帧 ${meta.frames[index]} · ${Number(meta.seconds[index]).toFixed(2)} s`;
      this.$('referenceError').textContent = ''; this.paint();
    } catch (error) {if (generation === this.generation && version === this.version) this.$('referenceError').textContent = error.message;}
    finally {
      if (url) URL.revokeObjectURL(url);
      this.loading = false;
      if (this.pendingFrame) {this.pendingFrame = false; this.loadFrame();}
    }
  }
  live(size) {this.liveSize = size; this.paint();}
  paint() {
    const mismatch = this.hasFrame && this.liveSize && this.size && Math.abs(this.liveSize[0]/this.liveSize[1] - this.size[0]/this.size[1]) > .01;
    const visible = this.hasFrame && this.$('referenceEnabled').checked && !mismatch;
    this.$('referenceAlphaValue').textContent = this.$('referenceAlpha').value + '%';
    this.$('referenceImage').style.opacity = this.liveSize ? Number(this.$('referenceAlpha').value)/100 : 1;
    this.$('referenceImage').hidden = !visible; this.$('referenceBadge').hidden = !visible && !mismatch;
    this.$('referenceBadge').textContent = mismatch ? '画面宽高比不一致，已暂停叠加' : this.liveSize ? '历史参考叠加 · ' + this.episode : '仅历史参考 · 实时相机未连接';
  }
}
