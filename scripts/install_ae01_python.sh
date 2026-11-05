#!/usr/bin/env bash
# Install the vendor's sample dependencies and expose the installed SDK to Python 3.10.
set -euo pipefail
AE01_INSTALL_ROOT="${AE01_SDK_ROOT:-/opt/avatar-sdk}"
if [ ! -f "$AE01_INSTALL_ROOT/lib/python/avatar_sdk/__init__.py" ]; then
  echo "Avatar SDK missing at $AE01_INSTALL_ROOT; run the vendor AE01 install.sh first." >&2
  exit 1
fi
/usr/bin/python3 -m pip install --user -r "$AE01_INSTALL_ROOT/share/sample/python/requirements.txt"
# Vive still uses ZeroMQ after the Manus dependency directory is removed.
/usr/bin/python3 -m pip install --user pyzmq
/usr/bin/python3 - "$AE01_INSTALL_ROOT" <<'PY'
from pathlib import Path
import site
import sys

root = Path(sys.argv[1]).expanduser().resolve()
directory = Path(site.getusersitepackages())
directory.mkdir(parents=True, exist_ok=True)
path = directory / 'ur_sharpa_avatar_sdk.pth'
value = str(root / 'lib/python') + '\n'
if path.exists() and path.read_text() != value:
    raise SystemExit(f'Refusing to overwrite a different SDK path in {path}')
path.write_text(value)
print(f'Python SDK path: {path} -> {value.strip()}')
PY
/usr/bin/python3 -c 'import avatar_sdk, zenoh, zmq; print("Avatar SDK", avatar_sdk.__version__); print("Zenoh / Vive ZeroMQ import OK")'
