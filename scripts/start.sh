#!/usr/bin/env bash
set -eo pipefail
DEPLOY_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$DEPLOY_SCRIPT_ROOT/scripts/setup_env.sh"
cd "$DEPLOY_ROOT/runtime"
if [[ ! -f "$DEPLOY_ROOT/runtime/install/setup.bash" ]]; then
  echo "Run bash scripts/build.sh --executor sequential first" >&2
  exit 1
fi
exec ros2 launch ur_launch workbench.launch.py "$@"
