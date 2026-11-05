import io
import json
import shutil
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import numpy as np
import pytest

from ur_common.camera_preview import CameraPreview
from ur_common.reference_dataset import ReferenceDataset
from ur_common.reference_http import ReferenceHTTP


def episode(root, name="success/sample", valid=(True, False, True, True)):
    path = root / name
    camera = path / "cameras/ego"
    camera.mkdir(parents=True)
    (path / "COMPLETE").touch()
    (camera / "rgb.mp4").write_bytes(b"placeholder")
    np.savez(camera / "index.npz", frame_index=np.arange(4),
             source_timestamp_ns=np.array([1_000_000_000, 1_100_000_000, 1_700_000_000, 3_000_000_000]),
             timeline_row=np.array([0, 0, 1, 5]), valid=np.array(valid))
    return path


def test_catalog_categories_and_incomplete_samples(tmp_path):
    for name in ("sample", "success/sample", "fail/sample", "unfinished.partial", "broken", "recording"):
        path = episode(tmp_path, name)
        if name == "broken":
            (path / "ERROR").touch()
        if name == "recording":
            (path / "COMPLETE").unlink()
    items = ReferenceDataset().episodes(str(tmp_path))["episodes"]
    assert {(v["id"], v["category"]) for v in items} == {
        ("sample", "root"), ("success/sample", "success"), ("fail/sample", "fail")}


def test_uses_source_time_and_video_frame_not_timeline_or_nominal_fps(tmp_path):
    episode(tmp_path)
    metadata = ReferenceDataset().metadata(str(tmp_path), "success/sample")
    assert metadata["frames"] == [0, 2, 3]
    assert metadata["seconds"] == [0.0, 0.7, 2.0]
    assert metadata["frame_count"] == 3


def test_rejects_escape_including_video_symlink(tmp_path):
    root = tmp_path / "data"
    path = episode(root)
    dataset = ReferenceDataset()
    for relative in ("../outside", str(path)):
        with pytest.raises(ValueError):
            dataset.metadata(str(root), relative)
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"outside")
    video = path / "cameras/ego/rgb.mp4"
    video.unlink()
    video.symlink_to(target)
    with pytest.raises(ValueError, match="超出"):
        dataset.metadata(str(root), "success/sample")
    assert dataset.episodes(str(root))["episodes"] == []


def test_corrupt_index_and_invalid_frame(tmp_path):
    path = episode(tmp_path)
    dataset = ReferenceDataset()
    for frame in (-1, 1, 4):
        with pytest.raises(ValueError, match="无效"):
            dataset.frame(str(tmp_path), "success/sample", frame)
    np.savez(path / "cameras/ego/index.npz", frame_index=[0, 1], valid=[True, True],
             source_timestamp_ns=[10, 5])
    with pytest.raises(ValueError, match="时间戳"):
        dataset.metadata(str(tmp_path), "success/sample")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for video integration")
def test_decodes_exact_frame_and_caches_it(tmp_path, monkeypatch):
    from PIL import Image
    path = episode(tmp_path)
    colors = [(240, 0, 0), (0, 240, 0), (0, 0, 240), (240, 240, 240)]
    raw = b"".join(np.full((36, 64, 3), color, dtype=np.uint8).tobytes() for color in colors)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "64x36", "-framerate", "30", "-i", "pipe:0", "-threads", "1",
                    "-c:v", "mjpeg", str(path / "cameras/ego/rgb.mp4")], input=raw, check=True)
    dataset = ReferenceDataset()
    data = dataset.frame(str(tmp_path), "success/sample", 2)
    image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    assert image[:, :, 2].mean() > 200
    assert image[:, :, :2].mean() < 30
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cached frame decoded twice"))
    assert dataset.frame(str(tmp_path), "success/sample", 2) == data


def test_duplicate_camera_timestamp_does_not_keep_stale_preview_online():
    preview = CameraPreview()
    preview.update_cameras({"ego": b"jpeg"}, {"ego": 123}, received=time.monotonic() - 3)
    preview.update_cameras({"ego": b"jpeg"}, {"ego": 123})
    assert preview.camera_frame("ego") is None
    preview.update_cameras({"ego": b"new jpeg"}, {"ego": 124})
    assert preview.camera_frame("ego") == (b"new jpeg", 124)


def test_camera_message_keeps_preview_separate_and_ignores_raw():
    preview = CameraPreview()
    message = SimpleNamespace(encoding="jpeg", ego=b"ego", ego_valid=True, ego_timestamp_ns=7,
                              left_wrist=b"", left_wrist_valid=False, left_wrist_timestamp_ns=0,
                              right_wrist=b"", right_wrist_valid=False, right_wrist_timestamp_ns=0)
    preview.update_message(message)
    assert preview.camera_frame("ego") == (b"ego", 7)
    assert message.ego == b"ego"
    message.encoding = "bgr8"
    message.ego_timestamp_ns = 8
    preview.update_message(message)
    assert preview.camera_frame("ego")[1] == 7


class Request:
    def __init__(self, path):
        self.path = path
        self.wfile = io.BytesIO()
        self.headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass


def test_http_offline_catalog_and_bad_frame(tmp_path):
    episode(tmp_path)
    preview = CameraPreview()
    api = ReferenceHTTP(preview, str(tmp_path))
    request = Request('/api/reference/live.jpg')
    assert api.handle(request)
    assert request.status == 503
    preview.update_cameras({"ego": b"jpeg"}, {"ego": 23})
    request = Request('/api/reference/live.jpg')
    api.handle(request)
    assert request.wfile.getvalue() == b"jpeg"
    assert request.headers["X-Source-Timestamp-Ns"] == "23"
    request = Request('/api/reference/episodes')
    api.handle(request)
    assert json.loads(request.wfile.getvalue())["episodes"][0]["id"] == 'success/sample'
    request = Request('/api/reference/frame.jpg?' + urlencode({"episode": "success/sample", "frame": 1}))
    api.handle(request)
    assert request.status == 400
    assert not api.handle(Request('/api/action'))


def test_cancel_running_decode_kills_process_and_releases_slot(tmp_path, monkeypatch):
    import threading
    episode(tmp_path)
    dataset = ReferenceDataset()
    cancel = threading.Event()

    class Process:
        returncode = None
        killed = False
        def communicate(self, timeout=None):
            if self.killed:
                return b'', b''
            cancel.set()
            raise subprocess.TimeoutExpired('ffmpeg', timeout)
        def poll(self):
            return self.returncode
        def kill(self):
            self.killed = True
            self.returncode = -9

    process = Process()
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: process)
    with pytest.raises(RuntimeError, match='取消'):
        dataset.frame(str(tmp_path), 'success/sample', 0, cancel)
    assert process.killed
    assert not dataset._frames
    assert dataset._decode_slots.acquire(blocking=False)
    assert dataset._decode_slots.acquire(blocking=False)
    dataset._decode_slots.release()
    dataset._decode_slots.release()


def test_close_cancels_only_this_viewers_jobs_and_clears_cache(tmp_path):
    import threading
    api = ReferenceHTTP(CameraPreview(), str(tmp_path))
    own, other = threading.Event(), threading.Event()
    api._jobs = {'own': {own}, 'other': {other}}
    api.dataset._frames['cached'] = b'jpeg'
    request = Request('/api/reference/close?client=own')
    request.command = 'POST'
    assert api.handle(request)
    assert own.is_set() and not other.is_set()
    assert not api.dataset._frames


def test_deploy_preview_is_read_on_demand_and_releases_transport(monkeypatch):
    from ur_common.camera_preview import OnDemandCameraPreview
    from ur_common import camera_transport
    calls = []
    class Reader:
        def __init__(self, prefix): calls.append(('open', prefix))
        def read(self): calls.append('read'); return None
        def close(self): calls.append('close')
    monkeypatch.setattr(camera_transport, 'SharedCameraReader', Reader)
    preview = OnDemandCameraPreview('test')
    preview.accept_source(object())
    assert calls == [] and preview._source is None
    assert preview.camera_frame('ego') is None
    assert calls == [('open', 'test'), 'read', 'close']
    preview.release()
    preview.accept_source(object())
    assert preview._source is None and not preview.cameras
