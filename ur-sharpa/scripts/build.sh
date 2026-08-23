#!/usr/bin/env bash
set -euo pipefail

UR_SHARPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$UR_SHARPA_ROOT/scripts/setup_env.sh"
set -u

/usr/bin/colcon --log-base "$UR_SHARPA_ROOT/log" build \
  --base-paths "$UR_SHARPA_ROOT/src" \
  --build-base "$UR_SHARPA_ROOT/build" \
  --install-base "$UR_SHARPA_ROOT/install" \
  --symlink-install "$@"

echo "Build complete. Run: source '$UR_SHARPA_ROOT/scripts/setup_env.sh'"
