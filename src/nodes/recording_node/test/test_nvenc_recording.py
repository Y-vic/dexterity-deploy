"""RUN_NVENC_TESTS=1 exercises the real GPU, using synthetic pixels only."""
import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from recording_node.video_encoding import nvenc_command, check_nvenc, write_frame
from recording_node.recording_node import RecorderCore
from ur_common.raw_camera import pack_bgr, RAW_ENCODING
from ur_common.camera_transport import SharedCameraReader, SharedCameraWriter, CAMERA_NAMES
from recording_node.episode_writer import EpisodeWriterCompat


def test_video_decode_errors_fail_validation_even_when_probe_exits_zero(tmp_path, monkeypatch):
    import recording_node.episode_writer as module
    monkeypatch.setattr(module.shutil, "which", lambda _: "ffprobe")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout='{"streams":[{"codec_name":"hevc","nb_read_frames":"3"}]}',
        stderr="[hevc] Could not find ref with POC 2"))
    writer = EpisodeWriterCompat(tmp_path, 30)
    try:
        with pytest.raises(RuntimeError, match="video decode errors"):
            writer._probe_camera_video(tmp_path/"corrupt.mp4")
    finally:
        writer.close()


def test_nvenc_command_has_no_jpeg_intermediate_and_fixed_gop():
    args = nvenc_command("ffmpeg", 1280, 720, 30, 40, "out.mp4")
    for name, value in (("-g", "2"), ("-bf", "0"), ("-rc-lookahead", "0"),
                        ("-b:v", "40M"), ("-maxrate:v", "80M"), ("-c:v", "hevc_nvenc")):
        assert args[args.index(name)+1] == value
    assert "mjpeg" not in args and "rawvideo" in args
    for value in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            nvenc_command("ffmpeg", 1280, 720, 30, value, "out.mp4")


def test_raw_pipe_handles_short_writes(monkeypatch):
    import recording_node.video_encoding as module
    output = bytearray()
    monkeypatch.setattr(module.select, "select", lambda r, w, e, t: ([], w, []))
    def short_write(fd, data):
        count = min(3, len(data))
        output.extend(data[:count])
        return count
    monkeypatch.setattr(module.os, "write", short_write)
    process = SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: 9), poll=lambda: None)
    write_frame(process, b"0123456789")
    assert output == b"0123456789"


def test_raw_pipe_timeout_is_reported(monkeypatch):
    import recording_node.video_encoding as module
    ticks = iter([0.0, 6.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    process = SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: 9), poll=lambda: None)
    with pytest.raises(TimeoutError):
        write_frame(process, b"123")


@pytest.mark.skipif(os.environ.get("RUN_NVENC_TESTS") != "1", reason="requires NVIDIA GPU")
def test_raw_ring_to_nvenc_two_episodes_gop_color_and_frame_count(tmp_path, monkeypatch):
    import recording_node.recording_node as module
    from test_recording_core import msgset
    monkeypatch.setattr(module, "_fk", lambda q: ([0.0]*15, [0.0]*15))
    check_nvenc("ffmpeg")
    core = RecorderCore(tmp_path / "episodes", record_tactile=False, video_backend="hevc_nvenc")
    writer = SharedCameraWriter("nvenc_test", 1280*720*3+20, tmp_path / "shm", slot_count=4)
    reader = SharedCameraReader("nvenc_test", tmp_path / "shm", encoding=RAW_ENCODING)
    try:
        for episode in range(2):
            assert core.start()
            start = core.writer._started_unix_ns + 1_000_000
            for i in range(36):
                stamp = start + i*33_333_333
                for name in CAMERA_NAMES:
                    h, w = (720, 1280) if name == "ego" else (270, 480)
                    image = np.zeros((h, w, 3), np.uint8)
                    image[:] = (10+i, 60, 180)  # BGR; strong red catches channel swaps.
                    writer.write(name, stamp, pack_bgr(image))
                for frame in reader.read_pending():
                    core.update_camera(frame, received=stamp/1e9)
                for key, msg in msgset().items():
                    if key not in ("camera", "wrench", "deform"):
                        msg.timestamp_ns = stamp
                        if key == "hand":
                            msg.q_cmd_timestamp_ns = stamp-1000
                        core.update(key, msg, received=stamp/1e9)
                assert core.sample(now=stamp/1e9), core.last_error
            # No waiting here: stop must drain already accepted frames itself.
            saved = core.stop(True)
            assert saved is not None, core.writer.last_validation
            for name in CAMERA_NAMES:
                directory = saved / "cameras" / name
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                        "-show_frames", "-show_entries", "frame=key_frame,pict_type",
                                        "-of", "json", str(directory / "rgb.mp4")],
                                       capture_output=True, text=True, check=True)
                frames = json.loads(probe.stdout)["frames"]
                assert len(frames) == 36
                assert [i for i, f in enumerate(frames) if f["key_frame"]] == list(range(0, 36, 2))
                assert all(f["pict_type"] != "B" for f in frames)
                meta = json.loads((directory / "rgb_meta.json").read_text())
                assert meta["video_codec"] == "hevc" and meta["hardware_encode"]
                assert meta["bitrate_mbps"] == (30 if name == "ego" else 12)
                assert np.load(directory / "index.npz")["source_timestamp_ns"].tolist() == [start+i*33_333_333 for i in range(36)]
                pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(directory / "rgb.mp4"),
                                                  "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
                h, w = (720, 1280) if name == "ego" else (270, 480)
                decoded = np.frombuffer(pixels, np.uint8).reshape(36, h, w, 3)
                for i in (0, 17, 35):
                    assert np.max(np.abs(decoded[i].mean(axis=(0, 1)) - [10+i, 60, 180])) < 6
            assert all(v["written"] == v["accepted"] == 36 for v in core.writer.video_status()["cameras"].values())
    finally:
        reader.close()
        writer.close()
        core.writer.close()
