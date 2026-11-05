"""Shared operator UI and read model for Monitor and Dashboard.

Commands still pass through the existing ROS owners. This adapter does not
publish hardware commands or infer success from an HTTP acknowledgement.
"""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .reference_http import ReferenceHTTP
from .workbench_preferences import WorkbenchPreferences, directory_listing, create_directory


ASSETS = Path(__file__).with_name("assets") / "workbench"


def workbench_snapshot(operator, deployment=None, *, switchable=False):
    deployment = deployment or {}
    control = operator.get("status") or {}
    upstream = operator.get("upstream") or {}
    recording = upstream.get("record_status") or {}
    mode = "teleop" if control.get("active_source", 2) == 2 else deployment.get("execution_provider", "policy")
    busy = (control.get("ur_state") in (2, 3, 4)
            or control.get("sharpa_state") in (1, 2, 4)
            or control.get("execution_dry_run") or control.get("execution_paused")
            or recording.get("recording") or recording.get("saving")
            or recording.get("rollout_requested"))
    connected = bool(control.get("online"))
    can_switch = connected and not busy and bool(recording.get("online"))
    modes = ["teleop", "policy", "replay"] if switchable else (
        ["policy", "replay"] if deployment else ["teleop"])
    hil_available = bool((deployment.get("policy", {}).get("hil") or {}).get("available"))
    if "hil" in modes and not hil_available:
        modes.remove("hil")
    return {
        **deployment, "mode": mode, "control": control, "recording": recording,
        "operator": operator, "devices": upstream,
        "capabilities": {"modes": modes, "deployment": bool(deployment),
                         "switchable": switchable},
        "mode_switch": {"allowed": can_switch,
                        "reason": "" if can_switch else "请等待控制器和数采在线，并停止执行、录制和保存"},
    }


class WorkbenchHTTP:
    def __init__(self, snapshot, action, preview, preferences=None):
        self.snapshot, self.action, self.preview = snapshot, action, preview
        self.preferences = preferences or WorkbenchPreferences()

    def handle(self, handler):
        url = urlsplit(handler.path)
        path = url.path
        assets = {"/": "index.html", "/index.html": "index.html",
                  "/workbench.css": "workbench.css", "/workbench.js": "workbench.js",
                  "/workbench-state.js": "workbench-state.js",
                  "/workbench-reference.js": "workbench-reference.js"}
        if path not in assets and not path.startswith("/api/workbench/"):
            return False
        send = lambda body, **kw: ReferenceHTTP._send(handler, body, **kw)
        try:
            if handler.command == "GET" and path in assets:
                name = assets[path]
                kind = "text/html" if name.endswith(".html") else "text/css" if name.endswith(".css") else "text/javascript"
                send((ASSETS / name).read_bytes(), content_type=kind + "; charset=utf-8")
            elif handler.command == "GET" and path == "/api/workbench/status":
                send(self.snapshot())
            elif handler.command == "GET" and path == "/api/workbench/preferences":
                send(self.preferences.read())
            elif handler.command == "GET" and path == "/api/workbench/directories":
                send(directory_listing(parse_qs(url.query).get("path", [""])[0]))
            elif handler.command == "GET" and path.startswith("/api/workbench/camera/"):
                name = path.rsplit("/", 1)[-1]
                if name not in {"ego", "left_wrist", "right_wrist"}:
                    raise ValueError("未知相机")
                frame = self.preview.camera_frame(name)
                if frame is None:
                    send({"error": "画面未连接或已过期"}, code=503)
                else:
                    send(frame[0], content_type="image/jpeg", headers={"X-Source-Timestamp-Ns": frame[1]})
            elif handler.command == "POST" and path in {"/api/workbench/action", "/api/workbench/preferences", "/api/workbench/directories"}:
                length = int(handler.headers.get("Content-Length", 0))
                if not 0 < length <= 8192:
                    raise ValueError("请求大小无效")
                payload = json.loads(handler.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须是对象")
                if path.endswith("/directories"):
                    send(create_directory(payload.get("parent"), payload.get("name")))
                elif path.endswith("/preferences"):
                    send(self.preferences.update(payload))
                else:
                    send(self.action(payload))
            else:
                send({"error": "接口不存在"}, code=404)
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            send({"accepted": False, "error": str(exc)}, code=400)
        return True
