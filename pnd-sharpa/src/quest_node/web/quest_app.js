import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';

import { buildQuestPacket } from './protocol.js';
import { GstVideoClient } from './video_client.js';

const FATAL_SOCKET_ERRORS = new Set(['AUTH_FAILED', 'ORIGIN_REJECTED']);
const VIDEO_DISTANCE = 3.5;
const VIDEO_WIDTH = 4.2;
const LEFT_EYE_LAYER = 1;
const RIGHT_EYE_LAYER = 2;

window.questReport?.('quest_module_loaded');

function accessTokenFromFragment() {
  const parameters = new URLSearchParams(window.location.hash.slice(1));
  return (parameters.get('token') || '').trim();
}

async function loadRuntimeConfig() {
  const response = await fetch('./runtime-config.json', {
    cache: 'no-store',
    credentials: 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`Quest 配置加载失败 (${response.status})`);
  }
  const config = await response.json();
  const fragmentToken = accessTokenFromFragment();
  return {
    accessToken: fragmentToken || String(config.accessToken || '').trim(),
    iceServers: Array.isArray(config.iceServers) ? config.iceServers : [],
    iceTransportPolicy: config.iceTransportPolicy === 'relay' ? 'relay' : 'all',
    videoEnabled: config.videoEnabled !== false,
    videoLayout: config.videoLayout === 'top-bottom' ? 'top-bottom' : 'mono',
    videoSwapEyes: Boolean(config.videoSwapEyes),
  };
}

class TrackingSocket {
  constructor(accessToken, onState, onEvent) {
    this.accessToken = accessToken;
    this.onState = onState;
    this.onEvent = onEvent;
    this.active = false;
    this.authenticated = false;
    this.fatalError = false;
    this.socket = null;
    this.retryTimer = null;
    this.retryDelay = 250;
  }

  setActive(active) {
    this.active = Boolean(active);
    if (this.active) {
      this.fatalError = false;
      this.connect();
    } else {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
      if (this.socket) {
        this.socket.onclose = null;
        this.socket.close(1000, 'WebXR session ended');
        this.socket = null;
      }
      this.authenticated = false;
      this.fatalError = false;
      this.onState('disconnected');
    }
  }

  connect() {
    if (!this.active || this.socket || this.fatalError) {
      return;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${window.location.host}/vrwebsocket`);
    this.socket = socket;
    this.onState('connecting');

    socket.onopen = () => {
      if (socket !== this.socket) {
        return;
      }
      this.authenticated = false;
      this.onState('authenticating');
      socket.send(JSON.stringify({ type: 'auth', token: this.accessToken }));
    };
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'auth_ok') {
          this.authenticated = true;
          this.retryDelay = 250;
          this.onState('connected');
          return;
        }
        if (message.type === 'calibration') {
          this.onEvent(message);
          return;
        }
        if (message.error) {
          if (FATAL_SOCKET_ERRORS.has(message.error)) {
            this.fatalError = true;
          }
          this.onState('error', new Error(message.message || message.error));
          socket.close(1008, 'Server rejected tracking connection');
        }
      } catch (_error) {}
    };
    socket.onerror = () => {
      if (socket === this.socket) {
        this.onState('error', new Error('Tracking WebSocket 连接失败'));
      }
    };
    socket.onclose = () => {
      if (socket !== this.socket) {
        return;
      }
      this.socket = null;
      this.authenticated = false;
      if (!this.fatalError) {
        this.onState('disconnected');
      }
      if (this.active && !this.fatalError) {
        this.retryTimer = window.setTimeout(() => this.connect(), this.retryDelay);
        this.retryDelay = Math.min(this.retryDelay * 2, 3000);
      }
    };
  }

  send(packet) {
    const socket = this.socket;
    if (
      !socket
      || !this.authenticated
      || socket.readyState !== WebSocket.OPEN
      || socket.bufferedAmount > 0
    ) {
      return false;
    }
    socket.send(JSON.stringify(packet));
    return true;
  }

  diagnostics() {
    return {
      active: this.active,
      authenticated: this.authenticated,
      readyState: this.socket?.readyState ?? -1,
      bufferedAmount: this.socket?.bufferedAmount ?? -1,
    };
  }
}

class QuestScene {
  constructor(canvas, video) {
    this.video = video;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x03080c);
    this.camera = new THREE.PerspectiveCamera(70, 1, 0.05, 100);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: false,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.xr.enabled = true;
    this.renderer.xr.setReferenceSpaceType('local-floor');
    this.renderer.setAnimationLoop((time, frame) => this.onFrame?.(time, frame));

    this.videoTexture = new THREE.VideoTexture(video);
    this.videoTexture.colorSpace = THREE.SRGBColorSpace;
    this.videoTexture.minFilter = THREE.LinearFilter;
    this.videoTexture.magFilter = THREE.LinearFilter;
    this.videoTexture.generateMipmaps = false;
    this.videoLayout = 'mono';
    this.videoPlanes = [];
    this.configureVideo('mono', false);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 2.5));
    this.addInputModels();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  addInputModels() {
    const controllerFactory = new XRControllerModelFactory();
    controllerFactory.setPath('./profiles');
    const handFactory = new XRHandModelFactory();
    handFactory.setPath('./profiles/generic-hand/');

    for (let index = 0; index < 2; index += 1) {
      const grip = this.renderer.xr.getControllerGrip(index);
      grip.add(controllerFactory.createControllerModel(grip));
      this.scene.add(grip);

      const hand = this.renderer.xr.getHand(index);
      hand.add(handFactory.createHandModel(hand, 'mesh'));
      this.scene.add(hand);
    }
  }

  configureVideo(layout, swapEyes) {
    for (const plane of this.videoPlanes) {
      this.scene.remove(plane);
      plane.geometry.dispose();
      plane.material.dispose();
    }
    this.videoLayout = layout;
    this.videoPlanes = layout === 'top-bottom'
      ? [
        this.createVideoPlane(swapEyes ? 'bottom' : 'top', LEFT_EYE_LAYER),
        this.createVideoPlane(swapEyes ? 'top' : 'bottom', RIGHT_EYE_LAYER),
      ]
      : [this.createVideoPlane('full', 0)];
    for (const plane of this.videoPlanes) {
      this.scene.add(plane);
    }
  }

  createVideoPlane(region, layer) {
    const geometry = new THREE.PlaneGeometry(1, 1);
    if (region !== 'full') {
      const uv = geometry.getAttribute('uv');
      const offset = region === 'top' ? 0.5 : 0;
      for (let index = 0; index < uv.count; index += 1) {
        uv.setY(index, offset + uv.getY(index) * 0.5);
      }
      uv.needsUpdate = true;
    }
    const plane = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ map: this.videoTexture, toneMapped: false }),
    );
    plane.layers.set(layer);
    plane.frustumCulled = false;
    plane.renderOrder = -1;
    return plane;
  }

  resize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  }

  async setSession(session) {
    await this.renderer.xr.setSession(session);
  }

  referenceSpace() {
    return this.renderer.xr.getReferenceSpace();
  }

  render(viewerPose) {
    this.updateVideoPlane(viewerPose);
    this.renderer.render(this.scene, this.camera);
  }

  updateVideoPlane(viewerPose) {
    const eyeHeight = this.videoLayout === 'top-bottom'
      ? this.video.videoHeight / 2
      : this.video.videoHeight;
    const aspect = this.video.videoWidth && eyeHeight
      ? this.video.videoWidth / eyeHeight
      : 16 / 9;
    const transform = viewerPose.transform;
    for (const plane of this.videoPlanes) {
      plane.position.set(
        transform.position.x,
        transform.position.y,
        transform.position.z,
      );
      plane.quaternion.set(
        transform.orientation.x,
        transform.orientation.y,
        transform.orientation.z,
        transform.orientation.w,
      );
      plane.translateZ(-VIDEO_DISTANCE);
      plane.scale.set(VIDEO_WIDTH, VIDEO_WIDTH / aspect, 1);
      plane.visible = this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
    }
  }
}

class QuestApp {
  constructor() {
    this.enterButton = document.getElementById('enter-xr');
    this.xrStatus = document.getElementById('xr-status');
    this.trackingStatus = document.getElementById('tracking-status');
    this.calibrationStatus = document.getElementById('calibration-status');
    this.videoStatus = document.getElementById('video-status');
    this.errorMessage = document.getElementById('error-message');
    this.video = document.getElementById('robot-video');
    this.accessToken = '';
    this.scene = new QuestScene(document.getElementById('xr-canvas'), this.video);
    this.scene.onFrame = (time, frame) => this.onXRFrame(time, frame);
    this.trackingSocket = null;
    this.videoClient = null;
    this.session = null;
    this.xrMode = null;
    this.lastTrackingDiagnosticAt = 0;
    this.sentTrackingFrames = 0;
    this.calibratePressed = false;
    this.decalibratePressed = false;
  }

  async initialize() {
    window.questReport?.('initialize_started');
    this.enterButton.addEventListener('click', () => this.toggleXR());
    if (!window.isSecureContext) {
      window.questReport?.('initialize_failed', 'insecure_context');
      this.setError('WebXR 需要 HTTPS 安全连接');
      this.xrStatus.textContent = '需要 HTTPS';
      return;
    }

    let runtimeConfig;
    try {
      runtimeConfig = await loadRuntimeConfig();
    } catch (error) {
      window.questReport?.('initialize_failed', `runtime_config: ${error}`);
      this.setError(error);
      this.xrStatus.textContent = '配置失败';
      return;
    }
    this.accessToken = runtimeConfig.accessToken;
    this.scene.configureVideo(runtimeConfig.videoLayout, runtimeConfig.videoSwapEyes);
    window.questReport?.(
      'video_layout',
      `${runtimeConfig.videoLayout},swap=${runtimeConfig.videoSwapEyes}`,
    );
    if (!this.accessToken) {
      window.questReport?.('initialize_failed', 'missing_access_token');
      this.setError('Quest 配置缺少访问令牌');
      this.xrStatus.textContent = '配置失败';
      return;
    }

    if (runtimeConfig.videoEnabled) {
      this.videoClient = new GstVideoClient(
        this.video,
        (state, error) => this.onVideoState(state, error),
        runtimeConfig,
        (phase, detail) => window.questReport?.(phase, detail),
      );
      this.videoClient.setEnabled(true).catch((error) => this.setError(error));
    } else {
      this.videoStatus.textContent = '已关闭';
      window.questReport?.('video_disabled');
    }
    this.trackingSocket = new TrackingSocket(
      this.accessToken,
      (state, error) => this.onTrackingState(state, error),
      (event) => this.onTrackingEvent(event),
    );

    if (!navigator.xr) {
      window.questReport?.('initialize_failed', 'navigator.xr unavailable');
      this.setError('当前浏览器不支持 WebXR');
      this.xrStatus.textContent = '不支持';
      return;
    }
    try {
      if (await navigator.xr.isSessionSupported('immersive-vr')) {
        this.xrMode = 'immersive-vr';
      }
    } catch (error) {
      window.questReport?.('webxr_support_check_failed', error);
    }
    if (!this.xrMode) {
      window.questReport?.('initialize_failed', 'immersive-vr unavailable');
      this.setError('没有可用的沉浸式 VR 模式');
      this.xrStatus.textContent = '不支持';
      return;
    }

    this.xrStatus.textContent = 'VR 就绪';
    this.enterButton.textContent = '进入 Quest WebXR';
    this.enterButton.disabled = false;
    window.questReport?.('initialize_complete');
  }

  async toggleXR() {
    if (this.session) {
      await this.session.end();
      return;
    }
    this.enterButton.disabled = true;
    this.setError('');
    try {
      const session = await navigator.xr.requestSession(this.xrMode, {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['bounded-floor', 'hand-tracking', 'dom-overlay'],
        domOverlay: { root: document.body },
      });
      this.session = session;
      session.addEventListener('end', () => this.onSessionEnd(session), { once: true });
      await this.scene.setSession(session);
      this.trackingSocket.setActive(true);
      this.xrStatus.textContent = '运行中';
      this.enterButton.textContent = '退出 WebXR';
      this.enterButton.disabled = false;
      document.body.classList.add('xr-active');
      window.questReport?.('xr_started');
    } catch (error) {
      window.questReport?.('xr_start_failed', error);
      const failedSession = this.session;
      this.session = null;
      if (failedSession) {
        try {
          await failedSession.end();
        } catch (_endError) {}
      }
      this.setError(error);
      this.xrStatus.textContent = '启动失败';
      this.enterButton.textContent = '重试 WebXR';
      this.enterButton.disabled = false;
    }
  }

  onSessionEnd(session) {
    if (session !== this.session) {
      return;
    }
    this.session = null;
    this.trackingSocket.setActive(false);
    this.xrStatus.textContent = 'VR 就绪';
    this.enterButton.textContent = '进入 Quest WebXR';
    this.enterButton.disabled = false;
    document.body.classList.remove('xr-active');
  }

  onXRFrame(_time, frame) {
    if (!frame || frame.session !== this.session) {
      return;
    }
    const referenceSpace = this.scene.referenceSpace();
    if (!referenceSpace) {
      return;
    }
    try {
      const viewerPose = frame.getViewerPose(referenceSpace);
      if (!viewerPose) {
        this.trackingStatus.textContent = '头显跟踪缺失';
        return;
      }
      const controllers = this.controllerData(frame, referenceSpace);
      this.observeCalibrationButtons(viewerPose, controllers);
      let trackingSent = false;
      if (
        !viewerPose.emulatedPosition
        && controllers.left?.pose
        && controllers.right?.pose
      ) {
        trackingSent = this.trackingSocket.send(buildQuestPacket({
          timestamp: Date.now(),
          head: viewerPose.transform,
          leftHand: controllers.left.pose.transform,
          rightHand: controllers.right.pose.transform,
          leftGamepad: controllers.left.source.gamepad,
          rightGamepad: controllers.right.source.gamepad,
        }));
        if (trackingSent) {
          this.sentTrackingFrames += 1;
          this.trackingStatus.textContent = '跟踪中';
        }
      } else if (viewerPose.emulatedPosition) {
        this.trackingStatus.textContent = '头显位置无效';
      } else if (!controllers.left?.pose && !controllers.right?.pose) {
        this.trackingStatus.textContent = '双手柄跟踪缺失';
      } else if (!controllers.left?.pose) {
        this.trackingStatus.textContent = '左手柄跟踪缺失';
      } else if (!controllers.right?.pose) {
        this.trackingStatus.textContent = '右手柄跟踪缺失';
      }
      this.reportTrackingDiagnostic(viewerPose, controllers, trackingSent);
      this.scene.render(viewerPose);
    } catch (error) {
      this.setError(error);
    }
  }

  reportTrackingDiagnostic(viewerPose, controllers, trackingSent) {
    const now = performance.now();
    if (now - this.lastTrackingDiagnosticAt < 2000) {
      return;
    }
    this.lastTrackingDiagnosticAt = now;
    const socket = this.trackingSocket.diagnostics();
    const sourceSummary = [...this.session.inputSources].map((source) => ({
      hand: source.handedness || 'none',
      mode: source.targetRayMode,
      grip: Boolean(source.gripSpace),
      handTracking: Boolean(source.hand),
      gamepad: Boolean(source.gamepad),
    }));
    window.questReport?.('xr_tracking', JSON.stringify({
      sent: trackingSent,
      sentFrames: this.sentTrackingFrames,
      viewerEmulated: viewerPose.emulatedPosition,
      leftPose: Boolean(controllers.left?.pose),
      rightPose: Boolean(controllers.right?.pose),
      sources: sourceSummary,
      socket,
    }));
  }

  controllerData(frame, referenceSpace) {
    const controllers = { left: null, right: null };
    for (const source of frame.session.inputSources) {
      if (source.handedness !== 'left' && source.handedness !== 'right') {
        continue;
      }
      const pose = this.inputPose(frame, source, referenceSpace);
      controllers[source.handedness] = {
        source,
        pose: pose && !pose.emulatedPosition ? pose : null,
      };
    }
    return controllers;
  }

  observeCalibrationButtons(viewerPose, controllers) {
    const rightSource = controllers.right?.source;
    const calibratePressed = this.buttonPressed(rightSource, 4);
    const decalibratePressed = this.buttonPressed(rightSource, 5);
    if (calibratePressed && !this.calibratePressed) {
      if (viewerPose.emulatedPosition || !controllers.left?.pose || !controllers.right?.pose) {
        this.calibrationStatus.textContent = '等待双手柄跟踪';
        this.pulseController('right', 0.35, 350);
      } else {
        this.calibrationStatus.textContent = '校准请求中';
      }
    }
    if (decalibratePressed && !this.decalibratePressed) {
      this.calibrationStatus.textContent = '取消请求中';
    }
    this.calibratePressed = calibratePressed;
    this.decalibratePressed = decalibratePressed;
  }

  buttonPressed(source, index) {
    const button = source?.gamepad?.buttons?.[index];
    return Boolean(button?.pressed || Number(button?.value) >= 0.5);
  }

  pulseController(handedness, intensity, duration) {
    const source = [...(this.session?.inputSources || [])]
      .find((inputSource) => inputSource.handedness === handedness);
    const gamepad = source?.gamepad;
    try {
      const haptic = gamepad?.hapticActuators?.[0];
      if (haptic?.pulse) {
        Promise.resolve(haptic.pulse(intensity, duration)).catch(() => {});
        return;
      }
      const actuator = gamepad?.vibrationActuator;
      if (actuator?.playEffect) {
        Promise.resolve(actuator.playEffect('dual-rumble', {
          duration,
          strongMagnitude: intensity,
          weakMagnitude: intensity,
        })).catch(() => {});
      }
    } catch (_error) {}
  }

  inputPose(frame, source, referenceSpace) {
    if (source.gripSpace) {
      return frame.getPose(source.gripSpace, referenceSpace);
    }
    const wristSpace = source.hand?.get('wrist');
    return wristSpace ? frame.getJointPose(wristSpace, referenceSpace) : null;
  }

  onTrackingState(state, error) {
    const labels = {
      connected: '已连接',
      connecting: '连接中',
      authenticating: '鉴权中',
      disconnected: '未连接',
      error: '异常',
    };
    this.trackingStatus.textContent = labels[state] || state;
    if (error) {
      this.setError(error);
    }
  }

  onTrackingEvent(event) {
    if (event.state === 'calibrated') {
      this.calibrationStatus.textContent = '已校准';
      this.setError('');
      this.pulseController('right', 1.0, 220);
      return;
    }
    if (event.state === 'already_calibrated') {
      this.calibrationStatus.textContent = '已校准';
      this.pulseController('right', 0.55, 100);
      return;
    }
    if (event.state === 'rejected') {
      this.calibrationStatus.textContent = '校准失败';
      this.setError(event.message || '校准姿态无效');
      this.pulseController('right', 0.35, 450);
      return;
    }
    if (event.state === 'reset') {
      this.calibrationStatus.textContent = '需要重新校准';
      if (event.reason === 'tracking_stale') {
        this.trackingStatus.textContent = '跟踪已暂停';
        this.setError('请唤醒并移动两个手柄，然后重新按 A');
      }
      this.pulseController('right', 0.45, 280);
    }
  }

  onVideoState(state, error) {
    window.questReport?.(`video_${state}`, error || '');
    const labels = {
      connecting: '信令连接中',
      waiting: '等待 ZED',
      'media-connected': '媒体已连接',
      streaming: '播放中',
      error: '异常',
    };
    this.videoStatus.textContent = labels[state] || state;
    if (error) {
      this.setError(error);
    } else if (state === 'streaming') {
      this.setError('');
    }
  }

  setError(error) {
    this.errorMessage.textContent = error ? String(error.message || error) : '';
  }
}

new QuestApp().initialize();
