export class GstVideoClient {
  constructor(video, onState = () => {}, rtcConfig = {}, onDiagnostic = () => {}) {
    this.video = video;
    this.onState = onState;
    this.enabled = false;
    this.api = null;
    this.session = null;
    this.listener = null;
    this.loading = null;
    this.producerId = '';
    this.retryTimer = null;
    this.retryDelay = 500;
    this.rtcConfig = rtcConfig;
    this.onDiagnostic = onDiagnostic;
    this.boundPeerConnection = null;
    this.disconnectedTimer = null;
  }

  async setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    if (!this.enabled) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
      window.clearTimeout(this.disconnectedTimer);
      this.disconnectedTimer = null;
      this.closeSession();
      this.video.classList.remove('visible');
      this.onState('hidden');
      return;
    }

    this.onState('connecting');
    try {
      await this.ensureApi();
      this.connectAvailableProducer();
    } catch (error) {
      this.onState('error', error);
      this.scheduleReconnect();
      throw error;
    }
  }

  async ensureApi() {
    if (this.api) {
      return;
    }
    if (this.loading) {
      await this.loading;
      return;
    }

    this.onDiagnostic('video_import_started');
    this.loading = import('/gstwebrtc-api-3.0.0.esm.js').then(({ default: GstWebRTCAPI }) => {
      this.onDiagnostic('video_import_complete');
      const socketProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      this.onDiagnostic('video_api_constructing');
      this.api = new GstWebRTCAPI({
        meta: { name: `Quest-WebVR-${Date.now()}` },
        signalingServerUrl: `${socketProtocol}://${window.location.host}/webrtc`,
        reconnectionTimeout: 1000,
        webrtcConfig: {
          iceServers: this.rtcConfig.iceServers || [],
          iceTransportPolicy: this.rtcConfig.iceTransportPolicy || 'all',
          bundlePolicy: 'max-bundle',
        },
      });
      this.onDiagnostic('video_api_constructed');
      this.listener = {
        producerAdded: (producer) => {
          window.clearTimeout(this.retryTimer);
          this.retryTimer = null;
          this.startProducer(producer);
        },
        producerRemoved: (producer) => {
          if (this.session && (!producer || producer.id === this.producerId)) {
            this.closeSession();
            if (this.enabled) {
              this.onState('waiting');
              this.scheduleReconnect();
            }
          }
        },
      };
      this.api.registerPeerListener(this.listener);
      this.api.registerConnectionListener({
        connected: () => {
          this.onDiagnostic('video_signal_connected');
          this.onState(this.enabled ? 'waiting' : 'hidden');
          this.connectAvailableProducer();
        },
        disconnected: () => {
          this.onDiagnostic('video_signal_disconnected');
          this.closeSession();
          if (this.enabled) {
            this.onState('waiting');
            this.scheduleReconnect();
          }
        },
      });
    }).catch((error) => {
      this.onDiagnostic('video_import_failed', error);
      throw error;
    }).finally(() => {
      this.loading = null;
    });

    await this.loading;
  }

  connectAvailableProducer() {
    if (!this.enabled || !this.api || this.session) {
      return;
    }
    const [producer] = this.api.getAvailableProducers();
    if (producer) {
      this.startProducer(producer);
    } else {
      this.onState('waiting');
      this.scheduleReconnect();
    }
  }

  startProducer(producer) {
    if (!this.enabled || this.session || !producer?.id) {
      return;
    }
    const session = this.api.createConsumerSession(producer.id);
    if (!session) {
      this.onState('waiting');
      this.scheduleReconnect();
      return;
    }

    this.producerId = producer.id;
    this.onDiagnostic('video_producer_found', producer.id);
    this.session = session;
    window.clearTimeout(this.retryTimer);
    this.retryTimer = null;
    session.addEventListener('streamsChanged', () => this.attachStream(session));
    session.addEventListener('rtcPeerConnectionChanged', () => {
      window.queueMicrotask(() => this.watchPeerConnection(session));
    });
    session.addEventListener('error', (event) => {
      if (this.session !== session) {
        return;
      }
      this.onState('error', event.error || event.message);
      this.onDiagnostic('video_session_error', event.error || event.message);
      this.closeSession();
      this.scheduleReconnect();
    });
    session.addEventListener('closed', () => {
      if (this.session === session) {
        this.session = null;
        this.producerId = '';
        this.video.pause();
        this.video.srcObject = null;
        this.video.classList.remove('visible');
        if (this.enabled) {
          this.onState('waiting');
          this.scheduleReconnect();
        }
      }
    });
    this.onState('connecting');
    this.onDiagnostic('video_session_connecting', producer.id);
    session.connect();
    window.queueMicrotask(() => this.watchPeerConnection(session));
  }

  attachStream(session) {
    if (!this.enabled || session !== this.session || !session.streams.length) {
      return;
    }
    this.video.srcObject = session.streams[0];
    this.video.play().then(() => {
      if (session !== this.session || !this.enabled) {
        return;
      }
      this.video.classList.add('visible');
      this.retryDelay = 500;
      this.onState('streaming');
      this.onDiagnostic('video_streaming', `${this.video.videoWidth}x${this.video.videoHeight}`);
    }).catch((error) => {
      if (session !== this.session) {
        return;
      }
      this.onState('error', error);
      this.onDiagnostic('video_play_failed', error);
      this.closeSession();
      this.scheduleReconnect();
    });
  }

  scheduleReconnect() {
    if (!this.enabled || this.retryTimer) {
      return;
    }
    this.retryTimer = window.setTimeout(async () => {
      this.retryTimer = null;
      try {
        await this.ensureApi();
        this.connectAvailableProducer();
      } catch (error) {
        this.onState('error', error);
      }
      if (this.enabled && !this.session) {
        this.retryDelay = Math.min(this.retryDelay * 2, 5000);
        this.scheduleReconnect();
      }
    }, this.retryDelay);
  }

  peerConnection(session) {
    return session.rtcPeerConnection
      || session._rtcPeerConnection
      || session._pc
      || session.peerConnection
      || session.pc
      || null;
  }

  watchPeerConnection(session) {
    const connection = this.peerConnection(session);
    if (!connection || connection === this.boundPeerConnection) {
      return;
    }
    this.boundPeerConnection = connection;
    connection.addEventListener('iceconnectionstatechange', () => {
      if (session !== this.session || connection !== this.boundPeerConnection) {
        return;
      }
      const state = connection.iceConnectionState;
      if (state === 'connected' || state === 'completed') {
        window.clearTimeout(this.disconnectedTimer);
        this.disconnectedTimer = null;
        this.onState('media-connected');
      } else if (state === 'disconnected') {
        window.clearTimeout(this.disconnectedTimer);
        this.disconnectedTimer = window.setTimeout(() => {
          if (session === this.session && connection.iceConnectionState === 'disconnected') {
            this.failSession('ZED 媒体中继已断开');
          }
        }, 3000);
      } else if (state === 'failed' || state === 'closed') {
        this.failSession('ZED 媒体中继连接失败');
      }
    });
  }

  failSession(message) {
    this.onState('error', new Error(message));
    this.closeSession();
    this.scheduleReconnect();
  }

  closeSession() {
    const session = this.session;
    this.session = null;
    this.producerId = '';
    this.boundPeerConnection = null;
    window.clearTimeout(this.disconnectedTimer);
    this.disconnectedTimer = null;
    if (session) {
      try {
        session.close();
      } catch (_error) {}
    }
    this.video.pause();
    this.video.srcObject = null;
    this.video.classList.remove('visible');
  }
}
