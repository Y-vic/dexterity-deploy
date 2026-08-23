#!/usr/bin/env bash
# Source this before building or launching UR SharpA.
set -eo pipefail

# This workstation commonly has ROS 1 Noetic in the parent shell. Do not let
# its paths leak into a ROS 2 Humble process.
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ROOT ROS_PACKAGE_PATH \
  ROS_MASTER_URI ROS_HOSTNAME ROS_IP ROS_ETC_DIR CMAKE_PREFIX_PATH PYTHONPATH \
  CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PATH="/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
hash -r
# ROS and generated colcon setup files intentionally read optional variables.
set +u
if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "ROS 2 Humble is missing: /opt/ros/humble/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi
source /opt/ros/humble/setup.bash
# Keep nounset disabled after sourcing ROS.  Colcon's generated setup scripts
# intentionally read optional variables such as COLCON_TRACE while unset.

UR_SHARPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UR_SHARPA_REPO_ROOT="$(cd "$UR_SHARPA_ROOT/.." && pwd)"

export UR_SHARPA_ROOT
export UR_SHARPA_REPO_ROOT
export UR_SHARPA_EXTERNAL_ROOT="$UR_SHARPA_ROOT/external"
export UR_SHARPA_PYTHON_PATHS="$UR_SHARPA_EXTERNAL_ROOT/realsense-python:$UR_SHARPA_EXTERNAL_ROOT/sharpa-wave-sdk-5.0.7/python/python310"
export SHARPA_SDK_ROOT="$UR_SHARPA_EXTERNAL_ROOT/sharpa-wave-sdk-5.0.7"
export MANUS_SDK_ROOT="$UR_SHARPA_EXTERNAL_ROOT/sharpa-manus-sdk"
export MANUS_RETARGET_ROOT="$UR_SHARPA_EXTERNAL_ROOT/manus-retarget"
export MANUS_RETARGET_PYTHON="/usr/bin/python3"

# The external interface package is intentionally kept outside the ROS tree.
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$UR_SHARPA_REPO_ROOT/interface"

if [ -f "$UR_SHARPA_ROOT/install/setup.bash" ]; then
  source "$UR_SHARPA_ROOT/install/setup.bash"
fi
