"""Persistent display preferences and server-side directory navigation.

Restoring a selection never starts motion, loads a replay or changes recording.
"""
import json
import os
from pathlib import Path
import tempfile
import threading


class WorkbenchPreferences:
    DEFAULTS = {"reference_root": "", "reference_episode": "", "reset_mode": "bias"}

    def __init__(self, path=None):
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        self.path = Path(path) if path is not None else base / "ur_sharpa" / "workbench.json"
        self._lock = threading.RLock()

    def read(self):
        with self._lock:
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return dict(self.DEFAULTS)
            if not isinstance(value, dict):
                raise ValueError("工作台偏好文件格式无效")
            return {key: value.get(key, default) for key, default in self.DEFAULTS.items()}

    def update(self, value):
        if not isinstance(value, dict) or set(value) - self.DEFAULTS.keys():
            raise ValueError("不支持的工作台偏好")
        for key, item in value.items():
            if not isinstance(item, str) or len(item) > 4096 or "\0" in item:
                raise ValueError("参考路径或样本格式无效")
            if key == "reset_mode" and item not in ("bias", "episode"):
                raise ValueError("复位目标无效")
        with self._lock:
            result = {**self.read(), **value}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".workbench-", delete=False) as stream:
                    name = stream.name
                    json.dump(result, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)
            return result


def directory_listing(value):
    if not isinstance(value, str) or "\0" in value:
        raise ValueError("目录路径无效")
    root = Path(value.strip()).expanduser() if value.strip() else Path.home()
    if not root.is_absolute():
        raise ValueError("请输入机器人主机上的绝对路径")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("所选路径不是目录")
    directories = []
    for child in root.iterdir():
        try:
            if child.is_dir() and not child.name.startswith("."):
                directories.append({"name": child.name, "path": str(child)})
        except OSError:
            continue
    directories.sort(key=lambda item: item["name"].casefold())
    return {"path": str(root), "parent": str(root.parent), "directories": directories[:500],
            "truncated": len(directories) > 500,
            "locations": [str(Path.home()), "/media", "/mnt", "/tmp", "/"]}


def create_directory(parent, name):
    """Create exactly one new child in an existing host directory; never overwrite."""
    if not isinstance(parent, str) or not parent.strip() or "\0" in parent:
        raise ValueError("请先打开父目录")
    if (not isinstance(name, str) or not name.strip() or name.strip() in {".", ".."}
            or any(char in name for char in ("/", "\\", "\0"))):
        raise ValueError("请输入单个文件夹名称，不能包含路径分隔符或 ..")
    root = Path(parent.strip()).expanduser()
    if not root.is_absolute():
        raise ValueError("请输入机器人主机上的绝对路径")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("父路径不是目录")
    child = root / name.strip()
    try:
        child.mkdir()
    except FileExistsError as exc:
        raise ValueError("同名文件或文件夹已存在，请使用其他名称") from exc
    except PermissionError as exc:
        raise ValueError("没有权限在此目录创建文件夹") from exc
    return {"path": str(child), "parent": str(root), "name": child.name}
