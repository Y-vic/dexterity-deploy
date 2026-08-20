"""Remote NX helpers for the ZED node."""

from __future__ import annotations

import os
import pwd
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass


APP_NAME = "pnd-gst-webrtc"
DEFAULT_JETSON_WEBRTC_ROOT = "/home/pnd-humanoid/Documents/pnd_teleoperation/external"
DEFAULT_MONITOR_STREAM_HOST = "10.10.20.127"
DEFAULT_MONITOR_STREAM_PORT = 5600
DEFAULT_INFERENCE_STREAM_HOST = "10.10.20.110"
DEFAULT_INFERENCE_STREAM_PORT = 5601
DEFAULT_QUEST_STREAM_BIND_HOST = "0.0.0.0"
DEFAULT_QUEST_STREAM_PORT = 5602
RTP_PAYLOAD_TYPE = 96
VIDEO_LAYOUT_MONO = "mono"
VIDEO_LAYOUT_TOP_BOTTOM = "top-bottom"
VIDEO_LAYOUTS = frozenset({VIDEO_LAYOUT_MONO, VIDEO_LAYOUT_TOP_BOTTOM})


@dataclass(frozen=True)
class RemoteConfig:
    host: str = "10.10.20.126"
    user: str = "pnd-humanoid"
    webrtc_port: int = 8443
    root: str = DEFAULT_JETSON_WEBRTC_ROOT
    width: int = 1280
    height: int = 720
    fps: int = 15
    bitrate: int = 2500000
    video_layout: str = VIDEO_LAYOUT_MONO
    monitor_stream_enabled: bool = True
    monitor_stream_host: str = DEFAULT_MONITOR_STREAM_HOST
    monitor_stream_port: int = DEFAULT_MONITOR_STREAM_PORT
    inference_stream_enabled: bool = True
    inference_stream_host: str = DEFAULT_INFERENCE_STREAM_HOST
    inference_stream_port: int = DEFAULT_INFERENCE_STREAM_PORT
    quest_stream_enabled: bool = False
    quest_stream_bind_host: str = DEFAULT_QUEST_STREAM_BIND_HOST
    quest_stream_port: int = DEFAULT_QUEST_STREAM_PORT

    @property
    def app_dir(self) -> str:
        return f"{self.root}/{APP_NAME}"


class RemoteProcess:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.last_returncode: int | None = None

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    try:
                        self.proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
            self.last_returncode = self.proc.poll()
        finally:
            self.proc = None

    def returncode(self) -> int | None:
        if self.proc is None:
            return self.last_returncode
        return self.proc.poll()


def ssh_key_path() -> str | None:
    candidates: list[str] = []
    env_key = os.environ.get("JETSON_SSH_KEY")
    if env_key:
        candidates.append(os.path.expanduser(env_key))
    operator_home = os.environ.get("PND_TELEOP_USER_HOME")
    if operator_home:
        candidates.append(os.path.join(operator_home, ".ssh", "jetson_ed25519"))
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        try:
            candidates.append(
                os.path.join(pwd.getpwnam(sudo_user).pw_dir, ".ssh", "jetson_ed25519")
            )
        except KeyError:
            pass
    candidates.append("/home/pnd-humanoid/.ssh/jetson_ed25519")
    candidates.append(os.path.expanduser("~/.ssh/jetson_ed25519"))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def known_hosts_path() -> str | None:
    candidates: list[str] = []
    operator_home = os.environ.get("PND_TELEOP_USER_HOME")
    if operator_home:
        candidates.append(os.path.join(operator_home, ".ssh", "known_hosts"))
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        try:
            candidates.append(
                os.path.join(pwd.getpwnam(sudo_user).pw_dir, ".ssh", "known_hosts")
            )
        except KeyError:
            pass
    candidates.append("/home/pnd-humanoid/.ssh/known_hosts")
    candidates.append(os.path.expanduser("~/.ssh/known_hosts"))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _ssh_base(config: RemoteConfig) -> list[str]:
    key = ssh_key_path()
    if not key:
        raise RuntimeError("Jetson SSH key is not configured")
    command = ["ssh", "-i", key]
    known_hosts = known_hosts_path()
    if known_hosts:
        command.extend(["-o", f"UserKnownHostsFile={known_hosts}"])
    command.extend(
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            f"{config.user}@{config.host}",
        ]
    )
    return command


def ssh_exec(
    config: RemoteConfig, command: str, timeout: float = 8.0
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_ssh_base(config), command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _remote_logger(pipe, prefix: str = "[ZED-NX]") -> None:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    try:
        for line_bytes in iter(pipe.readline, b""):
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace")
            line = ansi_escape.sub("", line).replace("\r", "").rstrip("\n")
            if line:
                print(f"{prefix} {line}", flush=True)
    finally:
        pipe.close()


def _env_prefix(app_dir: str, webrtc_port: int) -> str:
    return (
        f"export LD_LIBRARY_PATH={app_dir}/thirdparty/lib/custom:"
        f"{app_dir}/thirdparty/lib/aarch64-linux-gnu:"
        f"{app_dir}/thirdparty/lib:$LD_LIBRARY_PATH; "
        f"export GST_PLUGIN_PATH={app_dir}/thirdparty/lib/custom:"
        f"{app_dir}/thirdparty/lib/aarch64-linux-gnu/gstreamer-1.0:"
        "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:$GST_PLUGIN_PATH; "
        f"export GST_REGISTRY_1_0=/tmp/pnd-gst-webrtc-registry-{int(webrtc_port)}.bin; "
    )


def _camera_resolution(width: int, height: int) -> int:
    mapping = {
        (2208, 1242): 0,
        (1920, 1080): 1,
        (1920, 1200): 2,
        (1280, 720): 3,
        (960, 600): 4,
        (672, 376): 5,
    }
    try:
        return mapping[(int(width), int(height))]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ZED RGB size {width}x{height}; "
            "supported sizes are 672x376, 960x600, 1280x720, 1920x1080, "
            "1920x1200, 2208x1242"
        ) from exc


def _pipeline_args(config: RemoteConfig) -> list[str]:
    width = int(config.width)
    height = int(config.height)
    fps = int(config.fps)
    bitrate = int(config.bitrate)
    if config.video_layout not in VIDEO_LAYOUTS:
        raise ValueError(
            f"unsupported ZED video layout {config.video_layout!r}; "
            f"expected one of {sorted(VIDEO_LAYOUTS)}"
        )
    stream_type = 2 if config.video_layout == VIDEO_LAYOUT_TOP_BOTTOM else 0
    stream_height = height * 2 if stream_type == 2 else height
    resolution = _camera_resolution(width, height)
    idr_interval = max(1, fps)
    args = [
        f"{config.app_dir}/thirdparty/bin/gst-launch-1.0",
        "-e",
        "zedsrc",
        f"camera-resolution={resolution}",
        f"camera-fps={fps}",
        f"stream-type={stream_type}",
        "depth-mode=0",
        "camera-image-flip=1",
        "do-timestamp=true",
        "!",
        f"video/x-raw,format=BGRA,width={width},height={stream_height},framerate={fps}/1",
        "!",
        "queue",
        "max-size-buffers=2",
        "leaky=downstream",
        "!",
        "videoconvert",
        "!",
        f"video/x-raw,format=BGRx,width={width},height={stream_height},framerate={fps}/1",
        "!",
        "nvvidconv",
        "!",
        f"video/x-raw(memory:NVMM),format=NV12,width={width},height={stream_height},framerate={fps}/1",
        "!",
        "nvv4l2h264enc",
        f"bitrate={bitrate}",
        "control-rate=1",
        f"iframeinterval={idr_interval}",
        f"idrinterval={idr_interval}",
        "insert-sps-pps=true",
        "insert-aud=true",
        "maxperf-enable=true",
        "preset-level=1",
        "!",
        "h264parse",
        "config-interval=-1",
        "!",
        "video/x-h264,stream-format=byte-stream,alignment=au",
        "!",
        "tee",
        "name=t",
        "t.",
        "!",
        "queue",
        "max-size-buffers=2",
        "leaky=downstream",
        "!",
        "webrtcsink",
        "run-signalling-server=true",
        f"signalling-server-port={int(config.webrtc_port)}",
        "video-caps=video/x-h264",
        "enable-mitigation-modes=none",
        f"start-bitrate={bitrate}",
        f"max-bitrate={bitrate}",
    ]
    if config.monitor_stream_enabled:
        args.extend(
            _udp_branch_args(
                config.monitor_stream_host,
                config.monitor_stream_port,
            )
        )
    if config.inference_stream_enabled:
        args.extend(
            _rtp_branch_args(
                config.inference_stream_host,
                config.inference_stream_port,
            )
        )
    if config.quest_stream_enabled:
        args.extend(
            _tcp_server_branch_args(
                config.quest_stream_bind_host,
                config.quest_stream_port,
            )
        )
    return args


def _udp_branch_args(host: str, port: int) -> list[str]:
    return [
        "t.",
        "!",
        "queue",
        "max-size-buffers=2",
        "leaky=downstream",
        "!",
        "rtph264pay",
        "config-interval=1",
        f"pt={RTP_PAYLOAD_TYPE}",
        "!",
        "udpsink",
        f"host={host}",
        f"port={int(port)}",
        "sync=false",
        "async=false",
    ]


def _rtp_branch_args(host: str, port: int) -> list[str]:
    return _udp_branch_args(host, port)


def _tcp_server_branch_args(host: str, port: int) -> list[str]:
    return [
        "t.",
        "!",
        "queue",
        "max-size-buffers=2",
        "leaky=downstream",
        "!",
        "mpegtsmux",
        "alignment=7",
        "!",
        "tcpserversink",
        f"host={host}",
        f"port={int(port)}",
        "sync=false",
        "async=false",
        "buffers-max=1024",
        "buffers-soft-max=256",
        "recover-policy=latest",
        "sync-method=latest",
    ]


def hardware_command(config: RemoteConfig) -> str:
    app_dir = config.app_dir
    return (
        f"cd {shlex.quote(app_dir)} && "
        f"{_env_prefix(app_dir, config.webrtc_port)} "
        f"exec {' '.join(shlex.quote(arg) for arg in _pipeline_args(config))}"
    )


def check_hardware(config: RemoteConfig) -> None:
    app_dir = config.app_dir
    plugins = [
        "zedsrc",
        "videoconvert",
        "nvvidconv",
        "nvv4l2h264enc",
        "h264parse",
        "tee",
        "queue",
        "webrtcsink",
    ]
    if config.monitor_stream_enabled:
        plugins.extend(["rtph264pay", "udpsink"])
    if config.inference_stream_enabled:
        plugins.extend(["rtph264pay", "udpsink"])
    if config.quest_stream_enabled:
        plugins.extend(["mpegtsmux", "tcpserversink"])
    inspect_checks = " ".join(
        f"{app_dir}/thirdparty/bin/gst-inspect-1.0 {shlex.quote(plugin)} >/dev/null &&"
        for plugin in plugins
    )
    check = (
        f"test -x {app_dir}/thirdparty/bin/gst-launch-1.0 || exit 66; "
        f"test -x {app_dir}/thirdparty/bin/gst-inspect-1.0 || exit 66; "
        f"{_env_prefix(app_dir, config.webrtc_port)} "
        f"{inspect_checks} true"
    )
    result = ssh_exec(config, check, timeout=12)
    if result.returncode != 0:
        raise RuntimeError(
            f"NX ZED hardware prerequisites failed: stdout={result.stdout} stderr={result.stderr}"
        )


def stop_video_processes(config: RemoteConfig) -> subprocess.CompletedProcess:
    python_code = f"""
import os
import signal
import subprocess
import time

PORT_TOKEN = "signalling-server-port={int(config.webrtc_port)}"

def matching_pids():
    result = subprocess.run(
        ["pgrep", "-af", "gst-launch-1.0"],
        text=True,
        capture_output=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        pid_text, _, command = line.partition(" ")
        if "pgrep -af" in command:
            continue
        stale_webrtc_sink_match = "gst-launch-1.0" in command and "webrtcsink" in command and PORT_TOKEN in command
        if not stale_webrtc_sink_match:
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            pass
    return pids

for pid in matching_pids():
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(0.5)
for pid in matching_pids():
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
"""
    ssh_cmd = [*_ssh_base(config), "python3"]
    return subprocess.run(
        ssh_cmd,
        input=python_code,
        text=True,
        capture_output=True,
        timeout=10,
    )


def start_remote(config: RemoteConfig, command: str, owner: RemoteProcess) -> None:
    owner.close()
    owner.last_returncode = None
    proc = subprocess.Popen(
        [*_ssh_base(config), command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
    )
    owner.proc = proc
    if proc.stdout is not None:
        threading.Thread(
            target=_remote_logger, args=(proc.stdout,), daemon=True
        ).start()
