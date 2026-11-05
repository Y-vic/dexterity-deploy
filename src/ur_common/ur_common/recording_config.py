"""Shared validation for recording directory changes."""

from pathlib import Path
import json
import os
import tempfile


from ur_common.paths import workspace_root

DEFAULT_HIL_RECORDING_ROOT = str(workspace_root() / "data/episodes")


def normalize_recording_root(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("采集目录不能为空，必须使用绝对路径")
    root = Path(value.strip()).expanduser()
    if not root.is_absolute():
        raise ValueError("采集目录必须使用绝对路径")
    root = root.resolve()
    if root == Path(root.anchor):
        raise ValueError("采集目录不能是文件系统根目录")
    return root


class RecordingPreferences:
    """Recorder-owned settings, saved only after accepting a directory change."""

    def __init__(self, path=None):
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        self.path = Path(path) if path is not None else base / 'ur_sharpa' / 'recording.json'

    def read(self, default):
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return str(default)
        if not isinstance(value, dict):
            raise ValueError('数据保存目录记忆文件格式无效')
        return str(normalize_recording_root(value.get('recording_root')))

    def save(self, root):
        value = str(normalize_recording_root(str(root)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.recording-', delete=False) as stream:
                name = stream.name
                json.dump({'recording_root': value}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
