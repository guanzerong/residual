#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${GPU}_actenc_localdepth_only_seed${SEED}_${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-square_macro4710_actenc_localdepth_seed${SEED}_parallel}"
WANDB_NOTES="${WANDB_NOTES:-gpu${GPU}_actenc_localdepth_only_parallel}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/actenc_localdepth_seed${SEED}_gpu${GPU}.log}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
PID_FILE="${LOG_DIR}/actenc_localdepth_seed${SEED}_gpu${GPU}.pid"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "GPU=${GPU}"
echo "SEED=${SEED}"
echo "W&B name=${WANDB_NAME}"
echo "W&B group=${WANDB_GROUP}"
echo "Log dir=${LOG_DIR}"

echo "[$(date '+%F %T')] starting actenc_localdepth on GPU ${GPU}" | tee -a "${LAUNCHER_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup bash "${BASE_SCRIPT}" \
    seed="${SEED}" \
    agent.use_residual_image_encoder=false \
    agent.use_base_act_encoder_state=true \
    agent.depth_anything_v2_patch_state.enabled=true \
    agent.depth_anything_v2_patch_state.max_patches_per_camera=20 \
    agent.depth_anything_v2_patch_state.token_scale_init=0.25 \
    agent.depth_anything_v2_patch_state.token_dropout=0.15 \
    wandb.name="${WANDB_NAME}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.notes="${WANDB_NOTES}" \
    "$@" \
    >"${RUN_LOG}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started actenc_localdepth PID: ${PID}"
echo "Run log: ${RUN_LOG}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "PID file: ${PID_FILE}"
echo "W&B group:"
echo "https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
echo
echo "Monitor:"
echo "  tail -f ${RUN_LOG}"
echo
echo "Stop this run if needed:"
echo "  kill ${PID}"
