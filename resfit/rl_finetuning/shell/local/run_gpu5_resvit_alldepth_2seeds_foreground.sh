#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"

GPU="${GPU:-5}"
SEEDS="${SEEDS:-1 3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${GPU}_resvit_alldepth_2seeds_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "GPU=${GPU}" | tee -a "${LAUNCHER_LOG}"
echo "SEEDS=${SEEDS}" | tee -a "${LAUNCHER_LOG}"
echo "W&B group=${WANDB_GROUP}" | tee -a "${LAUNCHER_LOG}"
echo "Log dir=${LOG_DIR}" | tee -a "${LAUNCHER_LOG}"

PIDS=()

for SEED in ${SEEDS}; do
    WANDB_NAME="square_macro4710_resvit_alldepth_fullpatch_seed${SEED}"
    RUN_LOG="${LOG_DIR}/resvit_alldepth_seed${SEED}_gpu${GPU}.log"
    PID_FILE="${LOG_DIR}/resvit_alldepth_seed${SEED}_gpu${GPU}.pid"

    echo "[$(date '+%F %T')] launching seed ${SEED}" | tee -a "${LAUNCHER_LOG}"

    CUDA_VISIBLE_DEVICES="${GPU}" bash "${BASE_SCRIPT}" \
        seed="${SEED}" \
        agent.use_residual_image_encoder=true \
        agent.use_base_act_encoder_state=false \
        agent.depth_anything_v2_patch_state.enabled=true \
        agent.depth_anything_v2_patch_state.selection_mode=all \
        agent.depth_anything_v2_patch_state.max_patches_per_camera=9999 \
        wandb.name="${WANDB_NAME}" \
        wandb.group="${WANDB_GROUP}" \
        wandb.notes="gpu${GPU}_resvit_full_image_depth_patch_tokens_vs_local_patch" \
        "$@" \
        >"${RUN_LOG}" 2>&1 &

    PID="$!"
    echo "${PID}" > "${PID_FILE}"
    PIDS+=("${PID}")
    echo "[$(date '+%F %T')] seed ${SEED} PID ${PID}, log ${RUN_LOG}" | tee -a "${LAUNCHER_LOG}"
done

echo "[$(date '+%F %T')] launched ${#PIDS[@]} runs, waiting" | tee -a "${LAUNCHER_LOG}"
echo "W&B group: https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}" | tee -a "${LAUNCHER_LOG}"

STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        STATUS=1
        echo "[$(date '+%F %T')] PID ${PID} failed" | tee -a "${LAUNCHER_LOG}"
    else
        echo "[$(date '+%F %T')] PID ${PID} finished" | tee -a "${LAUNCHER_LOG}"
    fi
done

exit "${STATUS}"
