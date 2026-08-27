#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
MODE="${MODE:-reactive}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_reactive_vs_ahr_${RUN_ID}}"

case "${MODE}" in
    reactive)
        BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_fixed10_perstep_rgb_cacheddepth_stable.sh"
        WANDB_NAME="${WANDB_NAME:-square_fixed10_perstep_rgb_cacheddepth_seed${SEED}}"
        ;;
    ahr)
        BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"
        WANDB_NAME="${WANDB_NAME:-square_ahr_4710_seed${SEED}}"
        ;;
    *)
        echo "Unknown MODE=${MODE}; expected reactive or ahr" >&2
        exit 2
        ;;
esac

export SEED WANDB_GROUP WANDB_NAME
exec bash "${BASE_SCRIPT}" seed="${SEED}" wandb.group="${WANDB_GROUP}" wandb.name="${WANDB_NAME}" "$@"
