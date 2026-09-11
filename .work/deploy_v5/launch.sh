#!/bin/bash
# Start the SharpA v3 policy server for one registered run.
#
# The launcher is generic: WHICH checkpoint, config and normalizer get served is
# decided entirely by the run's entry in <task_name>/task.json, so a new run --
# another step, another experiment, a retrained normalizer -- needs a
# register_run.py call and no change here. Only a change of REPRESENTATION (a
# different action layout or state contract) would reach into this directory,
# because that changes what the numbers mean rather than which ones are loaded.
#
# Invoked BY THE WORKSTATION, never by hand in normal use. The dashboard's
# model_manager runs, over ssh and inside tmux:
#
#     bash scripts/deploy/sharpa_policy_server/launch.sh <task_name> <run_id>
#
# with these already exported (model_manager.py:_launch_remote):
#
#     SHARPA_DEPLOY_PORT              the port to listen on; the workstation
#                                     tunnels to it and polls /healthz
#     SHARPA_DEPLOY_ASSET_ROOT        checkpoint paths resolve against this
#     SHARPA_DEPLOY_TASK_CONFIG_ROOT  where <task_name>/task.json lives
#
# The workstation decides the port, so nothing here may hardcode one: a second
# run started after a failed cleanup would bind the old port and serve the wrong
# checkpoint under a healthy-looking /healthz.
#
# Run it by hand only to debug, and then export those three yourself.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <task_name> <run_id>" >&2
    exit 2
fi

TASK_NAME=$1
RUN_ID=$2

REPO_ROOT=/share/project/zjk/orca
PORT=${SHARPA_DEPLOY_PORT:?SHARPA_DEPLOY_PORT must be set by the caller}
ASSET_ROOT=${SHARPA_DEPLOY_ASSET_ROOT:-$REPO_ROOT}
TASK_CONFIG_ROOT=${SHARPA_DEPLOY_TASK_CONFIG_ROOT:-$REPO_ROOT/scripts/deploy/task_config}

# The training environment, byte for byte from run_mot_action_v5_head100h_16gpu.sh.
# Serving must import the same flagscale that fitted the checkpoint: a different
# hand_action_repr would silently mean a different 56D layout.
export PATH=/share/project/zhangqq/envs_conda/flagscale-robo/bin:$PATH
export CONDA_PREFIX=/share/project/zhangqq/envs_conda/flagscale-robo
export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}

# Same LD_PRELOAD/LD_LIBRARY_PATH as training. Dropping these produces an
# import-time C++ ABI error, not a wrong number, so it fails loudly -- but it
# fails inside tmux where only the workstation's log tail would show it.
export LD_PRELOAD=/share/project/lyx/miniconda3/envs/starVLA/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/share/project/lyx/miniconda3/envs/starVLA/lib:/usr/local/cuda-12.4/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}

# One GPU is enough for a 4B at batch 1; training's 16-GPU topology is
# irrelevant here. Pinned so a shared node cannot hand the process a busy card.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PyAV is chatty about long-GOP files; only errors are useful in a served log.
export LIBAV_LOG_LEVEL=0
export AV_LOG_LEVEL=quiet
export OTEL_SDK_DISABLED=true
# Nothing here should reach wandb. Serving is not a run.
export WANDB_MODE=disabled

cd "$REPO_ROOT"

echo "=== sharpa_policy_server launch ==="
echo "  task        : $TASK_NAME"
echo "  run_id      : $RUN_ID"
echo "  port        : $PORT"
echo "  asset_root  : $ASSET_ROOT"
echo "  task_config : $TASK_CONFIG_ROOT"
echo "  python      : $(command -v python)"
echo "  gpu         : ${CUDA_VISIBLE_DEVICES}"
echo

# exec, so tmux's pane tracks the server process itself. Without it the shell
# stays as pid 1 of the pane and the workstation's C-c cleanup would leave an
# orphaned python holding the GPU and the port.
exec python -u scripts/deploy/sharpa_policy_server/server.py \
    --task-name "$TASK_NAME" \
    --run-id "$RUN_ID" \
    --port "$PORT" \
    --asset-root "$ASSET_ROOT" \
    --task-config-root "$TASK_CONFIG_ROOT"
