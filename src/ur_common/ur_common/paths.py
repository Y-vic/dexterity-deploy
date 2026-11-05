"""Locate the single Deploy workspace from source or a colcon installation."""
import os
from pathlib import Path


def workspace_root() -> Path:
    configured = os.environ.get('DEPLOY_ROOT') or os.environ.get('UR_SHARPA_ROOT')
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if ((candidate / 'src/nodes/ur_launch').is_dir()
                and (candidate / 'config/protection.yaml').is_file()):
            return candidate.resolve()
    raise RuntimeError('Cannot locate Deploy; source scripts/setup_env.sh first')
