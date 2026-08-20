import GstWebRTCAPI from './gstwebrtc-api-3.0.0.esm.js';

const video = document.getElementById('egoviewVideo');
const hud = document.getElementById('hud');

const state = {
  ready: false,
  streamRunning: false,
  session: null,
  listener: null,
  channel: null,
  producerId: '',
  reconnecting: false,
  frameCount: 0,
  decodedFrames: 0,
  fps: 0,
  lastFpsAt: performance.now(),
  lastFrameAt: 0,
  lastReconnectAt: 0,
  bitrateKbps: 0,
  lastBytes: 0,
  lastStatsAt: 0,
  status: null,
  error: '',
  phase: 'boot',
  signalingUrl: '',
  connectionReady: false,
  channelId: '',
  connectAttempt: 0,
};

const protocol = window.location.protocol.startsWith('https') ? 'wss' : 'ws';

function setPhase(phase) {
  state.phase = phase;
  renderHud();
}

function buildWebRtcConfig() {
  return {
    iceServers: [
      { urls: [`stun:${window.location.hostname}:3478`] },
      { urls: ['stun:stun.l.google.com:19302'] },
    ],
    bundlePolicy: 'max-bundle',
  };
}

function setError(error) {
  state.error = error ? String(error.message || error) : '';
  renderHud();
}

function zedStatus() {
  return state.status || {};
}

function renderHud() {
  const status = zedStatus();
  const webrtc = status.webrtc || {};
  const metricsError = state.error || status.last_error || '';
  hud.textContent =
    `fps: ${state.fps.toFixed(1)}\n` +
    `decoded: ${state.decodedFrames}\n` +
    `resolution: ${video.videoWidth || 0}x${video.videoHeight || 0}\n` +
    `bitrate_kbps: ${state.bitrateKbps.toFixed(0)}\n` +
    `stream: ${state.streamRunning ? 'running' : 'starting'}\n` +
    `phase: ${state.phase}\n` +
    `pipeline: ${status.pipeline_alive ? 'alive' : 'offline'}\n` +
    `encoder: ${status.encoder || '-'}\n` +
    `producer_count: ${Number(webrtc.producer_count || 0)}\n` +
    `signal: ${state.signalingUrl || '-'}\n` +
    `latency_ms: ${Number(webrtc.check_latency_ms || 0).toFixed(1)}\n` +
    `deploy: ${status.deploy_output || '-'}\n` +
    `storage: ${status.storage_output || 'disabled'}\n` +
    `error: ${metricsError}`;
}

function publishMetrics() {
  const payload = {
    fps: state.fps,
    decoded_frames: state.decodedFrames,
    width: video.videoWidth || 0,
    height: video.videoHeight || 0,
    bitrate_kbps: state.bitrateKbps,
    stream_running: state.streamRunning,
    phase: state.phase,
    signaling_url: state.signalingUrl,
    producer_id: state.producerId,
    connection_ready: state.connectionReady,
    channel_id: state.channelId,
    session_active: !!state.session,
    error: state.error,
  };
  fetch('/egoview_metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => {});
}

async function refreshStatus() {
  try {
    const response = await fetch('/zed_status', { cache: 'no-store' });
    state.status = await response.json();
    renderHud();
  } catch (error) {
    setError(error);
  }
}

function signalingCandidates() {
  const urls = [];
  const status = zedStatus();
  const output = String(status.webrtc_output || '');
  const match = output.match(/^webrtc:\/\/([^:/]+)(?::([0-9]+))?/);
  if (!window.location.protocol.startsWith('https') && match) {
    const host = match[1];
    const port = match[2] || '8443';
    urls.push(`ws://${host}:${port}`);
  }
  urls.push(`${protocol}://${window.location.host}/webrtc`);
  return [...new Set(urls)];
}

function trackVideoFrames() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const onFrame = (_now, metadata = {}) => {
      state.frameCount += 1;
      state.decodedFrames = Number(metadata.presentedFrames || state.decodedFrames + 1);
      state.lastFrameAt = performance.now();
      const now = performance.now();
      const elapsed = now - state.lastFpsAt;
      if (elapsed >= 1000) {
        state.fps = state.frameCount * 1000 / elapsed;
        state.frameCount = 0;
        state.lastFpsAt = now;
        renderHud();
      }
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    setInterval(() => {
      if (!video.paused && video.readyState >= 2) {
        state.fps = 30;
      }
      renderHud();
    }, 1000);
  }
}

function resetFrameCounters() {
  state.frameCount = 0;
  state.fps = 0;
  state.decodedFrames = 0;
  state.lastFpsAt = performance.now();
  state.lastFrameAt = 0;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function attachSession(session) {
  session.addEventListener('streamsChanged', () => onSessionStream(session));
  session.addEventListener('error', (event) => {
    setPhase('session_error');
    setError(event.error || event.message);
  });
  session.addEventListener('closed', () => {
    if (state.session === session) {
      state.session = null;
    }
    if (!state.streamRunning) {
      setPhase('session_closed');
    }
  });
  session.addEventListener('rtcPeerConnectionChanged', () => {
    setPhase('peer_connection_created');
  });
}

function startProducerSession(producer) {
  if (!producer || !producer.id || state.session) {
    return;
  }
  state.producerId = producer.id;
  setPhase('start_session');
  const session = state.session = state.channel
    ? state.channel.createConsumerSession(producer.id)
    : null;
  if (!session) {
    state.session = null;
    setPhase('session_create_failed');
    setError(`cannot create consumer session for ${producer.id}`);
    return;
  }
  attachSession(session);
  session.connect();
}

async function waitForProducer() {
  for (let i = 0; i < 60; i += 1) {
    await refreshStatus();
    const status = zedStatus();
    const webrtc = status.webrtc || {};
    const producerCount = Number(webrtc.producer_count || 0);
    if (webrtc.signal_connected && producerCount > 0) {
      return true;
    }
    await sleep(500);
  }
  throw new Error('ZED WebRTC producer timed out');
}

function onSessionStream(session) {
  if (!session.streams.length) {
    return;
  }
  state.streamRunning = true;
  state.reconnecting = false;
  setPhase('streaming');
  setError('');
  resetFrameCounters();
  video.srcObject = session.streams[0];
  video.play().catch(setError);
}

function connectWebRtc(candidates = signalingCandidates()) {
  const urls = candidates.length ? candidates : [`${protocol}://${window.location.host}/webrtc`];
  const url = urls[0];
  state.connectAttempt += 1;
  const attempt = state.connectAttempt;
  state.signalingUrl = url;
  state.connectionReady = false;
  state.channelId = '';
  setPhase('connect_signal');
  const channel = state.channel = new GstWebRTCAPI({
    meta: { name: `Egoview-${Date.now()}` },
    signalingServerUrl: url,
    reconnectionTimeout: 1000,
    webrtcConfig: buildWebRtcConfig(),
  });

  channel.registerConnectionListener({
    connected: (channelId) => {
      if (attempt !== state.connectAttempt) {
        return;
      }
      state.connectionReady = true;
      state.channelId = channelId || '';
      setPhase('signal_connected');
      for (const producer of channel.getAvailableProducers()) {
        startProducerSession(producer);
      }
    },
    disconnected: () => {
      if (attempt !== state.connectAttempt) {
        return;
      }
      state.connectionReady = false;
      state.channelId = '';
      if (!state.streamRunning) {
        setPhase('signal_disconnected');
      }
    },
  });

  state.listener = {
    producerAdded: (producer) => {
      startProducerSession(producer);
    },
    producerRemoved: () => {
      closeCurrentSession();
    },
  };

  channel.registerPeerListener(state.listener);
  for (const producer of channel.getAvailableProducers()) {
    startProducerSession(producer);
  }

  setTimeout(() => {
    if (attempt !== state.connectAttempt || state.streamRunning) {
      return;
    }
    if (urls.length > 1) {
      setError(`no video via ${url}, trying fallback`);
      closeCurrentSession();
      connectWebRtc(urls.slice(1));
    }
  }, 7000);
}

function closeCurrentSession() {
  if (state.session) {
    const session = state.session;
    state.session = null;
    try {
      session.close();
    } catch (error) {
      // Closing a stale WebRTC session is best-effort.
    }
  }
  state.streamRunning = false;
  video.srcObject = null;
}

function reconnectStream(reason) {
  const now = performance.now();
  if (state.reconnecting || now - state.lastReconnectAt < 3000) {
    return;
  }
  state.reconnecting = true;
  state.lastReconnectAt = now;
  setError(reason);
  closeCurrentSession();
  setTimeout(() => {
    if (state.reconnecting && !state.streamRunning) {
      state.reconnecting = false;
    }
  }, 5000);
  if (state.channel && state.producerId) {
    startProducerSession({ id: state.producerId });
    if (state.session) {
      return;
    }
  }
  if (state.channel && state.channel.connectChannel) {
    state.channel.connectChannel();
  }
}

async function updateStats() {
  try {
    const pc = state.session
      ? (
        state.session.rtcPeerConnection
        || state.session._rtcPeerConnection
        || state.session._pc
        || state.session.peerConnection
        || state.session.pc
      )
      : null;
    if (!pc || !pc.getStats) {
      return;
    }
    const stats = await pc.getStats();
    stats.forEach((report) => {
      const isVideoRtp = report.type === 'inbound-rtp'
        && !report.isRemote
        && (report.kind === 'video'
          || report.mediaType === 'video'
          || String(report.id || '').includes('Video'));
      if (isVideoRtp) {
        const now = report.timestamp;
        const bytes = report.bytesReceived || 0;
        if (state.lastStatsAt && now > state.lastStatsAt) {
          state.bitrateKbps = (bytes - state.lastBytes) * 8 / (now - state.lastStatsAt);
        }
        state.lastStatsAt = now;
        state.lastBytes = bytes;
        state.decodedFrames = Number(
          report.framesDecoded
          || report.framesReceived
          || state.decodedFrames
        );
      }
    });
  } catch (error) {
    // Stats are best-effort.
  } finally {
    renderHud();
  }
}

function watchStreamHealth() {
  if (!state.streamRunning || state.reconnecting) {
    return;
  }
  const now = performance.now();
  const stalledMs = state.lastFrameAt ? now - state.lastFrameAt : now - state.lastFpsAt;
  if (video.readyState >= 2 && stalledMs > 3500) {
    reconnectStream(`video frame stalled ${Math.round(stalledMs)}ms`);
  }
}

trackVideoFrames();
setInterval(updateStats, 1000);
setInterval(renderHud, 1000);
setInterval(publishMetrics, 1000);
setInterval(refreshStatus, 1000);
setInterval(watchStreamHealth, 1000);

window.addEventListener('error', (event) => {
  setError(event.error || event.message);
});
window.addEventListener('unhandledrejection', (event) => {
  setError(event.reason || 'unhandled promise rejection');
});

refreshStatus()
  .then(() => connectWebRtc())
  .catch((error) => {
    setError(error);
    connectWebRtc();
  });
