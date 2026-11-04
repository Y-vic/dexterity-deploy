"""HTTP adapter shared by the teleop and deployment dashboards."""

import json
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zipfile import BadZipFile

from .reference_dataset import ReferenceDataset


class ReferenceHTTP:
    def __init__(self, preview, root="/home/yns/data/0916_ppp"):
        self.preview = preview
        self.dataset = ReferenceDataset(root)
        self._jobs = {}
        self._jobs_lock = threading.Lock()

    @staticmethod
    def _send(handler, body, content_type="application/json", code=200, headers=None):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode()
        try:
            handler.send_response(code)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(body)))
            handler.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                handler.send_header(key, str(value))
            handler.end_headers()
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The viewer may abort a request when its switch is closed.

    def handle(self, handler):
        url = urlsplit(handler.path)
        if not url.path.startswith("/api/reference/"):
            return False
        query = parse_qs(url.query)
        get = lambda name, default="": query.get(name, [default])[0]
        try:
            if url.path == "/api/reference/close" and getattr(handler, "command", "GET") == "POST":
                with self._jobs_lock:
                    for cancel in self._jobs.get(get("client"), ()):
                        cancel.set()
                self.dataset.clear_cache()
                if hasattr(self.preview, "release"):
                    self.preview.release()
                self._send(handler, {"closed": True})
            elif getattr(handler, "command", "GET") != "GET":
                self._send(handler, {"error": "Method not allowed"}, code=405)
            elif url.path == "/api/reference/view.js":
                self._send(handler, Path(__file__).with_name("assets").joinpath("ego_reference.js").read_bytes(),
                           "text/javascript; charset=utf-8")
            elif url.path == "/api/reference/config":
                self._send(handler, {"root": self.dataset.default_root})
            elif url.path == "/api/reference/live.jpg":
                frame = self.preview.camera_frame("ego")
                if frame is None:
                    self._send(handler, {"error": "实时 Ego 画面已过期或未连接"}, code=503)
                else:
                    self._send(handler, frame[0], "image/jpeg",
                               headers={"X-Source-Timestamp-Ns": frame[1]})
            elif url.path == "/api/reference/episodes":
                self._send(handler, self.dataset.episodes(get("root")))
            elif url.path == "/api/reference/metadata":
                self._send(handler, self.dataset.metadata(get("root"), get("episode")))
            elif url.path == "/api/reference/frame.jpg":
                client, cancel = get("client"), threading.Event()
                with self._jobs_lock:
                    self._jobs.setdefault(client, set()).add(cancel)
                try:
                    data = self.dataset.frame(get("root"), get("episode"), int(get("frame", "0")), cancel)
                    self._send(handler, data, "image/jpeg")
                finally:
                    with self._jobs_lock:
                        self._jobs[client].discard(cancel)
                        if not self._jobs[client]:
                            del self._jobs[client]
            else:
                self._send(handler, {"error": "接口不存在"}, code=404)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            self._send(handler, {"error": str(exc)}, code=400)
        except RuntimeError as exc:
            self._send(handler, {"error": str(exc)}, code=503)
        return True
