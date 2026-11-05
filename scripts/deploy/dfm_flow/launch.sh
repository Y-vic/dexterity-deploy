#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: launch.sh <task_name> <run_id>" >&2
  exit 2
fi
bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$bundle_root/interface${PYTHONPATH:+:$PYTHONPATH}"
export SHARPA_DEPLOY_CALIBRATION="${SHARPA_DEPLOY_CALIBRATION:-$bundle_root/config/dfm_calibration.json}"
exec "${SHARPA_DEPLOY_PYTHON:-python3}" -m sharpa_interface.dfm.server "$1" "$2"
