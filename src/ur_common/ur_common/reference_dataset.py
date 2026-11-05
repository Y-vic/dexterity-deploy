"""Read-only ego reference frames from recorded episodes.

Frame numbers refer to decoded video frames, not the recorder's timeline rows.
FFmpeg runs only in HTTP workers; no ROS callback performs media decoding.
"""

from collections import OrderedDict
from pathlib import Path
import subprocess
import threading
import time

import numpy as np


class ReferenceDataset:
    def __init__(self, default_root="/home/yns/data/0916_ppp"):
        self.default_root = str(default_root)
        self._lock = threading.Lock()
        self._decode_slots = threading.BoundedSemaphore(2)
        self._metadata = OrderedDict()
        self._frames = OrderedDict()

    def root(self, value=None):
        root = Path(value or self.default_root).expanduser()
        if not root.is_absolute():
            raise ValueError("请填写数据集的绝对路径")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("数据集路径不是目录")
        return root

    @staticmethod
    def _inside(root, path):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("样本或文件路径超出数据集目录")
        return resolved

    def _episode(self, root, episode):
        relative = Path(episode)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("无效的样本路径")
        path = self._inside(root, root / relative)
        if path.name.endswith(".partial") or not (path / "COMPLETE").is_file() or (path / "ERROR").exists():
            raise ValueError("请选择已完成的样本")
        return path

    def episodes(self, root=None):
        root = self.root(root)
        episodes = []
        for category in ("", "success", "fail"):
            parent = root / category
            if not parent.is_dir():
                continue
            for path in sorted(parent.iterdir(), reverse=True):
                try:
                    episode = path.relative_to(root).as_posix()
                    path = self._episode(root, episode)
                    self._inside(root, path / "cameras/ego/rgb.mp4")
                    self._inside(root, path / "cameras/ego/index.npz")
                except (OSError, ValueError):
                    continue
                episodes.append({"id": episode, "name": path.name, "category": category or "root"})
        return {"root": str(root), "episodes": episodes}

    @staticmethod
    def _signature(paths):
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)

    def _load(self, root, episode):
        root = self.root(root)
        path = self._episode(root, episode)
        video, index = [self._inside(root, path / "cameras/ego" / name)
                        for name in ("rgb.mp4", "index.npz")]
        key = self._signature((video, index))
        with self._lock:
            cached = self._metadata.get(key)
            if cached is not None:
                self._metadata.move_to_end(key)
                return video, key, cached
        with np.load(index, allow_pickle=False) as values:
            frames = values["frame_index"]
            stamps = values["source_timestamp_ns"]
            valid = values["valid"]
            if (frames.ndim != 1 or frames.size > 1_000_000 or stamps.shape != frames.shape
                    or valid.shape != frames.shape or frames.dtype.kind not in "iu"
                    or stamps.dtype.kind not in "iu" or valid.dtype.kind != "b"):
                raise ValueError("相机时间索引格式无效")
            keep = valid & (frames >= 0) & (stamps > 0)
            frames, stamps = frames[keep], stamps[keep]
            if not frames.size or np.any(np.diff(frames.astype(np.int64)) <= 0):
                raise ValueError("相机索引没有有效帧或帧号不递增")
            if np.any(np.diff(stamps.astype(np.int64)) < 0):
                raise ValueError("相机源时间戳不递增")
            seconds = (stamps.astype(np.int64) - int(stamps[0])) / 1e9
            metadata = {"episode": episode, "frames": frames.tolist(),
                        "seconds": seconds.tolist(), "frame_count": len(frames),
                        "duration_s": float(seconds[-1])}
        with self._lock:
            self._metadata[key] = metadata
            while len(self._metadata) > 8:
                self._metadata.popitem(last=False)
        return video, key, metadata

    def metadata(self, root, episode):
        return self._load(root, episode)[2]

    def clear_cache(self):
        with self._lock:
            self._metadata.clear()
            self._frames.clear()

    def frame(self, root, episode, frame, cancel=None):
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise RuntimeError("参考帧读取已取消")
        video, key, metadata = self._load(root, episode)
        if frame not in metadata["frames"]:
            raise ValueError("参考帧不存在或无效")
        cache_key = key, frame
        with self._lock:
            if cache_key in self._frames:
                self._frames.move_to_end(cache_key)
                return self._frames[cache_key]
        if not self._decode_slots.acquire(blocking=False):
            raise RuntimeError("正在读取参考帧，请稍后重试")
        try:
            # Exact decode-order selection also works with long-GOP HEVC. PTS or
            # nominal FPS seeking would misidentify frames in older recordings.
            process = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(video),
                 "-map", "0:v:0", "-an", "-vf", f"select=eq(n\\,{frame})",
                 "-vsync", "0", "-frames:v", "1", "-threads", "1", "-q:v", "2",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 20
            try:
                while True:
                    if cancel.is_set():
                        raise RuntimeError("参考帧读取已取消")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("参考帧读取超时，请选择更靠前的帧")
                    try:
                        data, _ = process.communicate(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
            if cancel.is_set():
                raise RuntimeError("参考帧读取已取消")
            if process.returncode or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                raise ValueError("无法解码参考帧，请检查视频完整性与 FFmpeg 编解码支持")
            if len(data) > 16 * 1024 * 1024:
                raise ValueError("参考帧尺寸过大")
        except FileNotFoundError as exc:
            raise RuntimeError("未安装 ffmpeg，无法读取历史视频") from exc
        finally:
            self._decode_slots.release()
        with self._lock:
            if cancel.is_set():
                raise RuntimeError("参考帧读取已取消")
            self._frames[cache_key] = data
            while len(self._frames) > 16 or sum(map(len, self._frames.values())) > 32 * 1024 * 1024:
                self._frames.popitem(last=False)
        return data
