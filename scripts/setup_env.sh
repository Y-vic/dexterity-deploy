#!/usr/bin/env bash
# Source this before building or launching UR SharpA.
# This file is sourced into the caller's shell.  Do not enable `errexit` here:
# doing so would make a later, unrelated `ros2` failure terminate that shell.
# Preserve nounset while temporarily disabling it for ROS/colcon setup files,
# which intentionally read optional variables.
UR_SHARPA_SETUP_RESTORE_NOUNSET=0
case $- in
  *u*)
    UR_SHARPA_SETUP_RESTORE_NOUNSET=1
    set +u
    ;;
esac

# This workstation commonly has ROS 1 Noetic in the parent shell. Do not let
# its paths leak into a ROS 2 Humble process.
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ROOT ROS_PACKAGE_PATH \
  ROS_MASTER_URI ROS_HOSTNAME ROS_IP ROS_ETC_DIR CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH \
  CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PATH="/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
hash -r
if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "[setup_env] ROS 2 Humble is missing: /opt/ros/humble/setup.bash" >&2
  if [ "$UR_SHARPA_SETUP_RESTORE_NOUNSET" -eq 1 ]; then
    set -u
  fi
  unset UR_SHARPA_SETUP_RESTORE_NOUNSET
  return 1 2>/dev/null || exit 1
fi
if source /opt/ros/humble/setup.bash; then
  :
else
  UR_SHARPA_SETUP_STATUS=$?
  echo "[setup_env] failed to source /opt/ros/humble/setup.bash (status=$UR_SHARPA_SETUP_STATUS)" >&2
  if [ "$UR_SHARPA_SETUP_RESTORE_NOUNSET" -eq 1 ]; then
    set -u
  fi
  unset UR_SHARPA_SETUP_RESTORE_NOUNSET
  return "$UR_SHARPA_SETUP_STATUS" 2>/dev/null || exit "$UR_SHARPA_SETUP_STATUS"
fi

UR_SHARPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UR_SHARPA_REPO_ROOT="$UR_SHARPA_ROOT"

export UR_SHARPA_ROOT
export UR_SHARPA_REPO_ROOT
export DEPLOY_ROOT="$UR_SHARPA_ROOT"
export TMPDIR="$DEPLOY_ROOT/runtime/tmp"
export ROS_HOME="$DEPLOY_ROOT/runtime/ros"
export ROS_LOG_DIR="$DEPLOY_ROOT/runtime/log/ros"
export XDG_CACHE_HOME="$DEPLOY_ROOT/runtime/cache"
export XDG_CONFIG_HOME="$DEPLOY_ROOT/runtime/config"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$ROS_HOME" "$ROS_LOG_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
export UR_SHARPA_EXTERNAL_ROOT="$UR_SHARPA_ROOT/external"
export UR_SHARPA_PYTHON_PATHS="$UR_SHARPA_EXTERNAL_ROOT/realsense-python:$UR_SHARPA_EXTERNAL_ROOT/sharpa-wave-sdk-5.0.7/python/python310"
export SHARPA_SDK_ROOT="$UR_SHARPA_EXTERNAL_ROOT/sharpa-wave-sdk-5.0.7"
export MANUS_SDK_ROOT="$UR_SHARPA_EXTERNAL_ROOT/sharpa-manus-sdk"
export MANUS_RETARGET_ROOT="$UR_SHARPA_EXTERNAL_ROOT/manus-retarget"
export MANUS_RETARGET_PYTHON="/usr/bin/python3"

# The external interface package is intentionally kept outside the ROS tree.
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$UR_SHARPA_REPO_ROOT/interface"

if [ -f "$UR_SHARPA_ROOT/runtime/install/setup.bash" ]; then
  if source "$UR_SHARPA_ROOT/runtime/install/setup.bash"; then
    :
  else
    UR_SHARPA_SETUP_STATUS=$?
    echo "[setup_env] failed to source $UR_SHARPA_ROOT/runtime/install/setup.bash (status=$UR_SHARPA_SETUP_STATUS)" >&2
    if [ "$UR_SHARPA_SETUP_RESTORE_NOUNSET" -eq 1 ]; then
      set -u
    fi
    unset UR_SHARPA_SETUP_RESTORE_NOUNSET
    return "$UR_SHARPA_SETUP_STATUS" 2>/dev/null || exit "$UR_SHARPA_SETUP_STATUS"
  fi
else
  echo "[setup_env] workspace is not built: $UR_SHARPA_ROOT/runtime/install/setup.bash" >&2
  echo "[setup_env] run: $UR_SHARPA_ROOT/scripts/build.sh" >&2
fi

if [ "$UR_SHARPA_SETUP_RESTORE_NOUNSET" -eq 1 ]; then
  set -u
fi
unset UR_SHARPA_SETUP_RESTORE_NOUNSET
