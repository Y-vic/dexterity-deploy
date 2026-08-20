"""Browser-facing EgoView HTTP server for the ZED node."""

from __future__ import annotations

import contextlib
import json
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


DEFAULT_BROWSER_UI_HOST = "127.0.0.1"
DEFAULT_BROWSER_UI_PORT = 12100
MAX_METRICS_BODY_BYTES = 8192


def _default_metrics() -> dict[str, Any]:
    return {
        "updated_at": 0.0,
        "fps": 0.0,
        "decoded_frames": 0,
        "width": 0,
        "height": 0,
        "bitrate_kbps": 0.0,
        "stream_running": False,
        "phase": "boot",
        "signaling_url": "",
        "producer_id": "",
        "connection_ready": False,
        "channel_id": "",
        "session_active": False,
        "recording": False,
        "recording_s": 0,
        "error": "",
    }


@dataclass
class BrowserUiState:
    metrics: dict[str, Any] = field(default_factory=_default_metrics)
    last_error: str = ""


class _HttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BrowserUiServer:
    """Serve EgoView assets, ZED status JSON, and browser metrics on localhost."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        web_root: str,
        status_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.host = host
        self.port = port
        self.web_root = self._resolve_web_root(web_root)
        self.status_provider = status_provider
        self.state = BrowserUiState()
        self.lock = threading.Lock()
        self.server = _HttpServer((self.host, self.port), self._handler_class())
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="zed_egoview_http",
            daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.server.shutdown()
        with contextlib.suppress(Exception):
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def status_payload(self) -> dict[str, Any]:
        with self.lock:
            metrics = dict(self.state.metrics)
            last_error = self.state.last_error
        return {
            "enabled": True,
            "running": self.thread is not None and self.thread.is_alive(),
            "host": self.host,
            "port": self.port,
            "web_root": str(self.web_root),
            "egoview_url": "/egoview",
            "internal_url": f"http://{self.host}:{self.port}/egoview",
            "nginx_path": "/egoview",
            "status_paths": ["/status", "/zed_status"],
            "metrics_path": "/egoview_metrics",
            "metrics": metrics,
            "last_error": last_error,
        }

    @staticmethod
    def _resolve_web_root(value: str) -> Path:
        candidates: list[Path] = []
        if value.strip():
            candidates.append(Path(value).expanduser())
        try:
            candidates.append(Path(get_package_share_directory("zed_node")) / "web")
        except PackageNotFoundError:
            pass
        candidates.append(Path(__file__).resolve().parents[1] / "web")
        candidates.append(Path.cwd() / "src" / "zed_node" / "web")
        for candidate in candidates:
            if (
                (candidate / "egoview.html").is_file()
                and (candidate / "egoview.js").is_file()
                and (candidate / "gstwebrtc-api-3.0.0.esm.js").is_file()
            ):
                return candidate
        raise FileNotFoundError(
            "Cannot locate ZED EgoView web root with egoview.html, egoview.js, "
            "and gstwebrtc-api-3.0.0.esm.js"
        )

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt: str, *_args: Any) -> None:
                pass

            def end_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                super().end_headers()

            def do_OPTIONS(self) -> None:
                self.send_response(200)
                self.end_headers()

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in {"/status", "/zed_status"}:
                    self._send_json(owner.status_provider())
                    return
                if path == "/egoview_metrics":
                    with owner.lock:
                        self._send_json(dict(owner.state.metrics))
                    return
                self._serve_static(path)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path != "/egoview_metrics":
                    self.send_error(404, "Not found")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(min(length, MAX_METRICS_BODY_BYTES))
                    data = json.loads(raw.decode("utf-8"))
                    metrics = {
                        "updated_at": time.time(),
                        "fps": float(data.get("fps") or 0.0),
                        "decoded_frames": int(data.get("decoded_frames") or 0),
                        "width": int(data.get("width") or 0),
                        "height": int(data.get("height") or 0),
                        "bitrate_kbps": float(data.get("bitrate_kbps") or 0.0),
                        "stream_running": bool(data.get("stream_running")),
                        "phase": str(data.get("phase") or ""),
                        "signaling_url": str(data.get("signaling_url") or ""),
                        "producer_id": str(data.get("producer_id") or ""),
                        "connection_ready": bool(data.get("connection_ready")),
                        "channel_id": str(data.get("channel_id") or ""),
                        "session_active": bool(data.get("session_active")),
                        "recording": bool(data.get("recording")),
                        "recording_s": int(data.get("recording_s") or 0),
                        "error": str(data.get("error") or ""),
                    }
                    with owner.lock:
                        owner.state.metrics.update(metrics)
                        owner.state.last_error = ""
                        response = dict(owner.state.metrics)
                    self._send_json(response)
                except Exception as exc:
                    with owner.lock:
                        owner.state.last_error = str(exc)
                    self.send_error(400, "Invalid metrics")

            def _serve_static(self, path: str) -> None:
                route = {
                    "/": "egoview.html",
                    "/egoview": "egoview.html",
                    "/egoview/": "egoview.html",
                    "/egoview.html": "egoview.html",
                    "/egoview.js": "egoview.js",
                    "/egoview.css": "egoview.css",
                    "/gstwebrtc-api-3.0.0.esm.js": "gstwebrtc-api-3.0.0.esm.js",
                    "/favicon.ico": "favicon.ico",
                }.get(path)
                if route is None:
                    self.send_error(404, "Not found")
                    return
                file_path = owner.web_root / route
                try:
                    content = file_path.read_bytes()
                except OSError:
                    self.send_error(404, "Not found")
                    return
                content_type = mimetypes.guess_type(str(file_path))[0]
                if file_path.suffix == ".js":
                    content_type = "application/javascript"
                elif file_path.suffix == ".css":
                    content_type = "text/css"
                self.send_response(200)
                self.send_header("Content-Type", content_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(content)

            def _send_json(self, data: dict[str, Any]) -> None:
                content = json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode(
                    "utf-8"
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(content)

        return Handler
