"use strict";

const cameras = ["ego", "left_wrist", "right_wrist"];
const videos = Object.fromEntries(cameras.map((name) => [name, document.getElementById(`video-${name}`)]));
const ui = {
  episodePath: document.getElementById("episodePath"),
  playButton: document.getElementById("playButton"),
  backFrame: document.getElementById("backFrame"),
  forwardFrame: document.getElementById("forwardFrame"),
  speed: document.getElementById("speed"),
  position: document.getElementById("position"),
  scrubber: document.getElementById("scrubber"),
  timeline: document.getElementById("timeline"),
  markStart: document.getElementById("markStart"),
  markEnd: document.getElementById("markEnd"),
  cancelPending: document.getElementById("cancelPending"),
  saveButton: document.getElementById("saveButton"),
  status: document.getElementById("status"),
  summary: document.getElementById("summary"),
  emptyRanges: document.getElementById("emptyRanges"),
  rangesTable: document.getElementById("rangesTable"),
  rangesBody: document.getElementById("rangesBody"),
};

const state = {
  info: null,
  row: 0,
  ranges: [],
  pendingStart: null,
  playing: false,
  lastTickMs: null,
  playTimeS: 0,
  lastHardSyncMs: 0,
  dirty: false,
};

function setStatus(message, isError = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", isError);
}

function timeAtRow(row) {
  return state.info.timeline_elapsed_s[Math.max(0, Math.min(row, state.info.num_frames - 1))];
}

function rowAtTime(seconds) {
  const times = state.info.timeline_elapsed_s;
  let low = 0;
  let high = times.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (times[mid] < seconds) low = mid + 1;
    else high = mid;
  }
  if (low > 0 && Math.abs(times[low - 1] - seconds) <= Math.abs(times[low] - seconds)) return low - 1;
  return low;
}

function cameraTime(camera, row) {
  const meta = state.info.cameras[camera];
  return meta.timeline_to_video_frame[row] / meta.fps;
}

function syncVideos(force = false) {
  if (!state.info) return;
  for (const camera of cameras) {
    const video = videos[camera];
    const target = cameraTime(camera, state.row);
    const rate = (video.duration && state.info.duration_s)
      ? (video.duration / state.info.duration_s) * Number(ui.speed.value)
      : Number(ui.speed.value);
    video.playbackRate = Math.max(0.1, Math.min(rate, 8));
    if (force || !Number.isFinite(video.currentTime) || Math.abs(video.currentTime - target) > 0.12) {
      video.currentTime = Math.max(0, target);
    }
  }
}

function seekRow(row, force = true) {
  if (!state.info) return;
  state.row = Math.max(0, Math.min(Math.round(row), state.info.num_frames - 1));
  state.playTimeS = timeAtRow(state.row);
  ui.scrubber.value = String(state.row);
  ui.position.textContent = `帧 ${state.row} / ${state.info.num_frames - 1} · ${state.playTimeS.toFixed(3)} s`;
  syncVideos(force);
  drawTimeline();
}

async function togglePlay() {
  if (!state.info) return;
  state.playing = !state.playing;
  state.lastTickMs = null;
  ui.playButton.textContent = state.playing ? "❚❚ 暂停" : "▶ 播放";
  if (state.playing) {
    syncVideos(true);
    await Promise.all(cameras.map((camera) => videos[camera].play().catch(() => undefined)));
    requestAnimationFrame(playLoop);
  } else {
    cameras.forEach((camera) => videos[camera].pause());
    syncVideos(true);
  }
}

function playLoop(nowMs) {
  if (!state.playing) return;
  if (state.lastTickMs !== null) {
    state.playTimeS += ((nowMs - state.lastTickMs) / 1000) * Number(ui.speed.value);
  }
  state.lastTickMs = nowMs;
  if (state.playTimeS >= state.info.duration_s) {
    state.playTimeS = state.info.duration_s;
    seekRow(state.info.num_frames - 1, true);
    togglePlay();
    return;
  }
  state.row = rowAtTime(state.playTimeS);
  ui.scrubber.value = String(state.row);
  ui.position.textContent = `帧 ${state.row} / ${state.info.num_frames - 1} · ${state.playTimeS.toFixed(3)} s`;
  if (nowMs - state.lastHardSyncMs > 350) {
    syncVideos(false);
    state.lastHardSyncMs = nowMs;
  }
  drawTimeline();
  requestAnimationFrame(playLoop);
}

function normalizeRanges(ranges) {
  const sorted = ranges
    .map((range) => ({
      start_frame: Number(range.start_frame),
      end_frame_exclusive: Number(range.end_frame_exclusive),
    }))
    .filter((range) => range.start_frame < range.end_frame_exclusive)
    .sort((a, b) => a.start_frame - b.start_frame);
  const merged = [];
  for (const range of sorted) {
    const previous = merged[merged.length - 1];
    if (previous && range.start_frame <= previous.end_frame_exclusive) {
      previous.end_frame_exclusive = Math.max(previous.end_frame_exclusive, range.end_frame_exclusive);
    } else {
      merged.push({ ...range });
    }
  }
  return merged;
}

function markStart() {
  state.pendingStart = state.row;
  if (state.playing) togglePlay();
  setStatus(`错误动作起点：帧 ${state.row}。移动到有效恢复开始位置后按 ]。`);
  drawTimeline();
}

function markEnd() {
  if (state.pendingStart === null) {
    setStatus("请先按 [ 标记错误动作开始。", true);
    return;
  }
  if (state.row <= state.pendingStart) {
    setStatus("恢复开始必须位于错误动作开始之后。", true);
    return;
  }
  state.ranges = normalizeRanges([
    ...state.ranges,
    { start_frame: state.pendingStart, end_frame_exclusive: state.row },
  ]);
  state.pendingStart = null;
  state.dirty = true;
  renderRanges();
  setStatus("红色区间已加入，尚未保存。按 Ctrl+S 保存。");
}

function cancelPending() {
  state.pendingStart = null;
  setStatus("已取消当前起点。");
  drawTimeline();
}

function removeRange(index) {
  state.ranges.splice(index, 1);
  state.dirty = true;
  renderRanges();
  setStatus("区间已删除，尚未保存。");
}

function renderRanges() {
  ui.rangesBody.innerHTML = "";
  let masked = 0;
  state.ranges.forEach((range, index) => {
    masked += range.end_frame_exclusive - range.start_frame;
    const startS = timeAtRow(range.start_frame);
    const endS = range.end_frame_exclusive < state.info.num_frames
      ? timeAtRow(range.end_frame_exclusive)
      : state.info.duration_s + 1 / state.info.sample_rate_hz;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${index + 1}</td>
      <td><code>${range.start_frame}</code> · ${startS.toFixed(3)} s</td>
      <td><code>${range.end_frame_exclusive}</code> · ${endS.toFixed(3)} s</td>
      <td>${range.end_frame_exclusive - range.start_frame} 帧 · ${(endS - startS).toFixed(3)} s</td>
      <td><button class="delete-range" data-index="${index}">删除</button></td>`;
    ui.rangesBody.appendChild(row);
  });
  ui.emptyRanges.hidden = state.ranges.length > 0;
  ui.rangesTable.hidden = state.ranges.length === 0;
  ui.summary.textContent = `${masked} / ${state.info.num_frames} 帧被屏蔽`;
  document.querySelectorAll(".delete-range").forEach((button) => {
    button.addEventListener("click", () => removeRange(Number(button.dataset.index)));
  });
  drawTimeline();
}

function drawTimeline() {
  if (!state.info) return;
  const canvas = ui.timeline;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#239b69";
  ctx.fillRect(0, 0, width, height);
  for (const range of state.ranges) {
    const x = (range.start_frame / state.info.num_frames) * width;
    const w = ((range.end_frame_exclusive - range.start_frame) / state.info.num_frames) * width;
    ctx.fillStyle = "#f05259";
    ctx.fillRect(x, 0, Math.max(2, w), height);
  }
  if (state.pendingStart !== null) {
    const from = Math.min(state.pendingStart, state.row);
    const to = Math.max(state.pendingStart, state.row);
    ctx.fillStyle = "rgba(255, 183, 77, .7)";
    ctx.fillRect((from / state.info.num_frames) * width, 0, Math.max(2, ((to - from) / state.info.num_frames) * width), height);
  }
  const cursorX = ((state.row + 0.5) / state.info.num_frames) * width;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(Math.max(0, cursorX - ratio), 0, Math.max(2, 2 * ratio), height);
}

async function saveAnnotations() {
  try {
    const response = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ranges: state.ranges }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.ranges = normalizeRanges(payload.ranges);
    state.dirty = false;
    renderRanges();
    setStatus(`已保存：${payload.trainable_frames} 帧参与训练，${payload.masked_frames} 帧被屏蔽。`);
  } catch (error) {
    setStatus(`保存失败：${error.message}`, true);
  }
}

function bindEvents() {
  ui.playButton.addEventListener("click", togglePlay);
  ui.backFrame.addEventListener("click", () => seekRow(state.row - 1));
  ui.forwardFrame.addEventListener("click", () => seekRow(state.row + 1));
  ui.scrubber.addEventListener("input", () => {
    if (state.playing) togglePlay();
    seekRow(Number(ui.scrubber.value));
  });
  ui.speed.addEventListener("change", () => syncVideos(true));
  ui.markStart.addEventListener("click", markStart);
  ui.markEnd.addEventListener("click", markEnd);
  ui.cancelPending.addEventListener("click", cancelPending);
  ui.saveButton.addEventListener("click", saveAnnotations);
  ui.timeline.addEventListener("click", (event) => {
    if (state.playing) togglePlay();
    const rect = ui.timeline.getBoundingClientRect();
    seekRow(((event.clientX - rect.left) / rect.width) * (state.info.num_frames - 1));
  });
  window.addEventListener("resize", drawTimeline);
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (event.code === "Space") {
      event.preventDefault();
      togglePlay();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      if (state.playing) togglePlay();
      seekRow(state.row - (event.shiftKey ? Math.round(state.info.sample_rate_hz) : 1));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      if (state.playing) togglePlay();
      seekRow(state.row + (event.shiftKey ? Math.round(state.info.sample_rate_hz) : 1));
    } else if (event.key === "[") {
      event.preventDefault();
      markStart();
    } else if (event.key === "]") {
      event.preventDefault();
      markEnd();
    } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotations();
    } else if (event.key === "Escape") {
      cancelPending();
    }
  });
}

async function initialize() {
  try {
    const [episodeResponse, annotationsResponse] = await Promise.all([
      fetch("/api/episode"),
      fetch("/api/annotations"),
    ]);
    if (!episodeResponse.ok) throw new Error(`episode HTTP ${episodeResponse.status}`);
    state.info = await episodeResponse.json();
    const annotations = annotationsResponse.ok ? await annotationsResponse.json() : { ranges: [] };
    state.ranges = normalizeRanges(annotations.ranges || []);
    ui.episodePath.textContent = state.info.episode_path;
    ui.scrubber.max = String(state.info.num_frames - 1);
    for (const camera of cameras) {
      videos[camera].addEventListener("loadeddata", () => {
        videos[camera].currentTime = Math.max(0.001, cameraTime(camera, state.row));
        const primingPlay = videos[camera].play();
        if (primingPlay) {
          primingPlay.then(() => window.setTimeout(() => {
            if (!state.playing) videos[camera].pause();
          }, 80)).catch(() => undefined);
        }
      }, { once: true });
      videos[camera].src = state.info.cameras[camera].video_url;
      videos[camera].load();
    }
    bindEvents();
    renderRanges();
    seekRow(0, true);
    setStatus("已加载。按 [ 标错误动作开始，按 ] 标有效恢复开始。");
  } catch (error) {
    setStatus(`加载失败：${error.message}`, true);
  }
}

initialize();
