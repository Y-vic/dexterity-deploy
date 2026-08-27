#!/usr/bin/env python3
"""Small, dependency-light train-mask annotation server."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


DEFAULT_EPISODE = Path(
    "/media/yns/T9/0824_yns_unscrew_cap/2026-08-24_20-45-49.080158"
)
CAMERAS = ("ego", "left_wrist", "right_wrist")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_ANNOTATIONS_DIR = Path(__file__).resolve().parent / "annotations"
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(tmp, path)


def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right)


class EpisodeData:
    def __init__(
        self,
        episode_path: str | Path,
        annotations_dir: str | Path = DEFAULT_ANNOTATIONS_DIR,
        *,
        make_browser_proxies: bool = True,
    ) -> None:
        self.path = Path(episode_path).expanduser().resolve()
        self.annotations_dir = Path(annotations_dir).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"episode directory does not exist: {self.path}")

        clock_path = self.path / "timeline" / "clock.npz"
        if not clock_path.is_file():
            raise FileNotFoundError(f"missing timeline: {clock_path}")
        with np.load(clock_path, allow_pickle=False) as clock:
            self.elapsed_s = np.asarray(clock["elapsed_ns"], dtype=np.float64) / 1e9
            self.sample_rate_hz = float(np.asarray(clock["sample_rate_hz"]).item())
        if self.elapsed_s.ndim != 1 or len(self.elapsed_s) == 0:
            raise ValueError("timeline elapsed_ns must be a non-empty one-dimensional array")

        self.camera_meta: dict[str, dict[str, Any]] = {}
        self.source_videos: dict[str, Path] = {}
        for camera in CAMERAS:
            camera_dir = self.path / "cameras" / camera
            video_path = camera_dir / "rgb.mp4"
            index_path = camera_dir / "index.npz"
            if not video_path.is_file() or not index_path.is_file():
                raise FileNotFoundError(f"missing video or index for camera {camera}")
            meta_path = camera_dir / "rgb_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            fps = float(meta.get("nominal_fps", self.sample_rate_hz))
            with np.load(index_path, allow_pickle=False) as index:
                camera_elapsed = np.asarray(index["elapsed_ns"], dtype=np.float64) / 1e9
                frame_index = np.asarray(index["frame_index"], dtype=np.int64)
            if len(camera_elapsed) == 0 or len(camera_elapsed) != len(frame_index):
                raise ValueError(f"invalid camera index for {camera}")
            nearest = _nearest_indices(camera_elapsed, self.elapsed_s)
            self.camera_meta[camera] = {
                "fps": fps,
                "frame_count": int(len(frame_index)),
                "timeline_to_video_frame": frame_index[nearest].astype(int).tolist(),
                "source_codec": str(
                    meta.get("video_codec")
                    or meta.get("probe", {}).get("codec_name")
                    or "unknown"
                ),
            }
            self.source_videos[camera] = video_path

        self.served_videos = dict(self.source_videos)
        if make_browser_proxies:
            self._prepare_browser_proxies()

        digest = hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()[:8]
        self.episode_id = f"{self.path.name}__{digest}"
        self.annotation_json = self.annotations_dir / f"{self.episode_id}.json"
        self.mask_npy = self.annotations_dir / f"{self.episode_id}.train_mask.npy"

    @property
    def length(self) -> int:
        return int(len(self.elapsed_s))

    def _prepare_browser_proxies(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("warning: ffmpeg not found; serving original videos")
            return

        jobs: list[tuple[str, Path, Path]] = []
        for camera, source in self.source_videos.items():
            codec = self.camera_meta[camera]["source_codec"].lower()
            if codec in {"h264", "avc1"}:
                continue
            stat = source.stat()
            cache_key = hashlib.sha1(
                f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
            ).hexdigest()[:16]
            cache_dir = Path(tempfile.gettempdir()) / "train-mask-annotator" / cache_key
            target = cache_dir / f"{camera}.mp4"
            jobs.append((camera, source, target))

        def transcode(job: tuple[str, Path, Path]) -> tuple[str, Path | None]:
            camera, source, target = job
            if target.is_file() and target.stat().st_size > 0:
                return camera, target
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.stem + ".tmp.mp4")
            try:
                subprocess.run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(source),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ],
                    check=True,
                )
                os.replace(temporary, target)
                return camera, target
            except (OSError, subprocess.CalledProcessError) as error:
                temporary.unlink(missing_ok=True)
                print(f"warning: failed to create {camera} browser proxy: {error}")
                return camera, None

        if jobs:
            print(f"preparing {len(jobs)} browser-compatible video proxies...")
            with ThreadPoolExecutor(max_workers=min(3, len(jobs))) as executor:
                for camera, target in executor.map(transcode, jobs):
                    if target is not None:
                        self.served_videos[camera] = target

    def metadata(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_path": str(self.path),
            "sample_rate_hz": self.sample_rate_hz,
            "num_frames": self.length,
            "duration_s": float(self.elapsed_s[-1]),
            "timeline_elapsed_s": np.round(self.elapsed_s, 6).tolist(),
            "cameras": {
                camera: {
                    **self.camera_meta[camera],
                    "video_url": f"/video/{camera}.mp4",
                }
                for camera in CAMERAS
            },
            "annotation_json": str(self.annotation_json),
            "mask_npy": str(self.mask_npy),
        }

    def load_annotations(self) -> dict[str, Any]:
        if not self.annotation_json.is_file():
            return {"ranges": []}
        payload = json.loads(self.annotation_json.read_text(encoding="utf-8"))
        return {"ranges": payload.get("ranges", [])}

    def save_annotations(self, raw_ranges: Any) -> dict[str, Any]:
        if not isinstance(raw_ranges, list):
            raise ValueError("ranges must be a list")
        normalized: list[tuple[int, int]] = []
        for item in raw_ranges:
            if not isinstance(item, dict):
                raise ValueError("each range must be an object")
            start = item.get("start_frame")
            end = item.get("end_frame_exclusive")
            if isinstance(start, bool) or isinstance(end, bool):
                raise ValueError("range boundaries must be integers")
            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError) as error:
                raise ValueError("range boundaries must be integers") from error
            if not 0 <= start < end <= self.length:
                raise ValueError(
                    f"range [{start}, {end}) is outside timeline [0, {self.length})"
                )
            normalized.append((start, end))

        merged: list[list[int]] = []
        for start, end in sorted(normalized):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        train_mask = np.ones(self.length, dtype=bool)
        ranges: list[dict[str, Any]] = []
        for start, end in merged:
            train_mask[start:end] = False
            end_time = (
                float(self.elapsed_s[end])
                if end < self.length
                else float(self.elapsed_s[-1] + 1.0 / self.sample_rate_hz)
            )
            ranges.append(
                {
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "start_time_s": round(float(self.elapsed_s[start]), 6),
                    "end_time_s": round(end_time, 6),
                }
            )

        payload = {
            "schema": "train_mask_annotation.v1",
            "episode_id": self.episode_id,
            "episode_path": str(self.path),
            "sample_rate_hz": self.sample_rate_hz,
            "num_frames": self.length,
            "ranges": ranges,
            "masked_frames": int((~train_mask).sum()),
            "trainable_frames": int(train_mask.sum()),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.annotation_json, payload)
        _atomic_npy(self.mask_npy, train_mask)
        return payload


class AnnotationHandler(BaseHTTPRequestHandler):
    episode: EpisodeData

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/episode":
            self._send_json(self.episode.metadata())
            return
        if path == "/api/annotations":
            self._send_json(self.episode.load_annotations())
            return
        if path.startswith("/video/") and path.endswith(".mp4"):
            camera = path.removeprefix("/video/").removesuffix(".mp4")
            video = self.episode.served_videos.get(camera)
            if video is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(video, allow_range=True)
            return
        static_name = "index.html" if path == "/" else path.removeprefix("/static/")
        if static_name in {"index.html", "app.js", "styles.css"}:
            self._send_file(STATIC_DIR / static_name)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/video/") and path.endswith(".mp4"):
            camera = path.removeprefix("/video/").removesuffix(".mp4")
            video = self.episode.served_videos.get(camera)
            if video is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(video, allow_range=True, head_only=True)
            return
        static_name = "index.html" if path == "/" else path.removeprefix("/static/")
        if static_name in {"index.html", "app.js", "styles.css"}:
            self._send_file(STATIC_DIR / static_name, head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/annotations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid request body length")
            payload = json.loads(self.rfile.read(length))
            result = self.episode.save_annotations(payload.get("ranges"))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(
        self,
        path: Path,
        *,
        allow_range: bool = False,
        head_only: bool = False,
    ) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range") if allow_range else None
        if range_header:
            match = RANGE_RE.match(range_header.strip())
            if match is None:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            elif right:
                suffix = min(int(right), size)
                start = size - suffix
            if start < 0 or start >= size or end < start:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")


def create_server(host: str, port: int, episode: EpisodeData) -> ThreadingHTTPServer:
    class BoundHandler(AnnotationHandler):
        pass

    BoundHandler.episode = episode
    return ThreadingHTTPServer((host, port), BoundHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser-proxy",
        action="store_true",
        help="serve the source MJPEG MP4 files without making cached H.264 proxies",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = EpisodeData(
        args.episode,
        args.annotations_dir,
        make_browser_proxies=not args.no_browser_proxy,
    )
    server = create_server(args.host, args.port, episode)
    print(f"episode: {episode.path}")
    print(f"frames: {episode.length}, duration: {episode.elapsed_s[-1]:.3f}s")
    print(f"open: http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
