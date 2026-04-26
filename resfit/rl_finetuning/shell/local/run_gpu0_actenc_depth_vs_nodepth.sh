#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${GPU}_actenc_depth_vs_nodepth_seed${SEED}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "GPU=${GPU}"
echo "SEED=${SEED}"
echo "W&B group=${WANDB_GROUP}"
echo "Log dir=${LOG_DIR}"

echo "[$(date '+%F %T')] starting actenc_nodepth" | tee -a "${LOG_DIR}/launcher.log"
CUDA_VISIBLE_DEVICES="${GPU}" bash "${BASE_SCRIPT}" \
    seed="${SEED}" \
    agent.use_residual_image_encoder=false \
    agent.use_base_act_encoder_state=true \
    agent.depth_anything_v2_patch_state.enabled=false \
    wandb.name="square_macro4710_actenc_nodepth_seed${SEED}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.notes="gpu${GPU}_same_seed_actenc_depth_vs_nodepth" \
    "$@" \
    >"${LOG_DIR}/actenc_nodepth_seed${SEED}_gpu${GPU}.log" 2>&1
echo "[$(date '+%F %T')] finished actenc_nodepth" | tee -a "${LOG_DIR}/launcher.log"

echo "[$(date '+%F %T')] starting actenc_localdepth" | tee -a "${LOG_DIR}/launcher.log"
CUDA_VISIBLE_DEVICES="${GPU}" bash "${BASE_SCRIPT}" \
    seed="${SEED}" \
    agent.use_residual_image_encoder=false \
    agent.use_base_act_encoder_state=true \
    agent.depth_anything_v2_patch_state.enabled=true \
    agent.depth_anything_v2_patch_state.max_patches_per_camera=20 \
    agent.depth_anything_v2_patch_state.token_scale_init=0.25 \
    agent.depth_anything_v2_patch_state.token_dropout=0.15 \
    wandb.name="square_macro4710_actenc_localdepth_seed${SEED}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.notes="gpu${GPU}_same_seed_actenc_depth_vs_nodepth" \
    "$@" \
    >"${LOG_DIR}/actenc_localdepth_seed${SEED}_gpu${GPU}.log" 2>&1
echo "[$(date '+%F %T')] finished actenc_localdepth" | tee -a "${LOG_DIR}/launcher.log"

echo "All GPU${GPU} ACT-encoder depth comparison runs finished."
echo "W&B group:"
echo "https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
echo "Logs: ${LOG_DIR}"
