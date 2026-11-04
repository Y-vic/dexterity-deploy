"""Raw BGR to persistent NVENC HEVC subprocess; no JPEG intermediate."""
import math
import os
import select
import subprocess
import time


def nvenc_command(ffmpeg, width, height, fps, bitrate_mbps, output, output_format="mp4"):
    if (min(width, height) <= 0 or width % 2 or height % 2
            or not math.isfinite(fps) or fps <= 0
            or not math.isfinite(bitrate_mbps) or bitrate_mbps <= 0):
        raise ValueError("HEVC requires positive even dimensions, FPS and bitrate")
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-filter_threads", "1", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps:g}", "-i", "pipe:0",
        "-map", "0:v:0", "-an", "-c:v", "hevc_nvenc", "-preset", "p4",
        "-tune", "ll", "-pix_fmt", "yuv420p", "-g", "2", "-bf", "0",
        "-rc-lookahead", "0", "-no-scenecut", "1", "-forced-idr", "1",
        "-force_key_frames", "expr:gte(n,n_forced*2)",
        "-rc", "vbr", "-cq", "18", "-b:v", f"{bitrate_mbps:g}M",
        "-maxrate:v", f"{bitrate_mbps*2:g}M", "-bufsize:v", f"{bitrate_mbps*2:g}M",
        "-tag:v", "hvc1", "-vsync", "0",
        "-f", output_format, str(output),
    ]


def check_nvenc(ffmpeg):
    """Fail clearly at node startup; never silently fall back to CPU encoding."""
    command = nvenc_command(ffmpeg, 256, 256, 30, 12, "-", "null")
    try:
        result = subprocess.run(command, input=bytes(256*256*3*2),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"NVENC HEVC startup check failed: {exc}") from exc
    if result.returncode:
        raise RuntimeError("NVENC HEVC unavailable: " + result.stderr.decode(errors="replace")[-4000:])


def write_frame(process, payload, timeout=5.0):
    """Bound pipe backpressure; caller owns this stream and kills on failure."""
    descriptor = process.stdin.fileno()
    view = memoryview(payload)
    deadline = time.monotonic() + timeout
    while view:
        if process.poll() is not None:
            raise RuntimeError(f"NVENC exited with code {process.returncode}")
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("NVENC raw frame pipe blocked for 5 seconds")
        if not select.select([], [descriptor], [], min(remaining, 0.1))[1]:
            continue
        try:
            count = os.write(descriptor, view[:65536])
        except BlockingIOError:
            continue
        if count <= 0:
            raise BrokenPipeError("NVENC raw frame pipe closed")
        view = view[count:]
