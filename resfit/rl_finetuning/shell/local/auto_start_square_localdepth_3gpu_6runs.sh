#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
AUTO_LAUNCHER="${ROOT_DIR}/resfit/rl_finetuning/shell/local/auto_start_square_localdepth_paired.sh"

cd "${ROOT_DIR}"

export NUM_GPUS="${NUM_GPUS:-3}"
export RUN_NODEPTH=1
export PARALLEL_NODEPTH=1
export ALLOW_GPU_OVERSUBSCRIBE=1
export WANDB_GROUP="${WANDB_GROUP:-square_macro4710_localdepth_3gpu_6runs_$(date +%Y%m%d_%H%M%S)}"

bash "${AUTO_LAUNCHER}"
