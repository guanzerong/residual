#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
RUN_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/run_gpu0_actenc_localdepth_only.sh"

GPU="${GPU:-3}"
SEEDS="${SEEDS:-2 3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${GPU}_actenc_localdepth_stable_2seeds_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "GPU=${GPU}"
echo "SEEDS=${SEEDS}"
echo "W&B group=${WANDB_GROUP}"
echo "Log dir=${LOG_DIR}"

for SEED in ${SEEDS}; do
    WANDB_NAME="square_macro4710_actenc_localdepth_stable_p10_s0p1_drop0p2_utd4_seed${SEED}"
    RUN_LOG="${LOG_DIR}/actenc_localdepth_stable_seed${SEED}_gpu${GPU}.log"

    echo "[$(date '+%F %T')] launching seed ${SEED}" | tee -a "${LAUNCHER_LOG}"

    GPU="${GPU}" \
    SEED="${SEED}" \
    WANDB_GROUP="${WANDB_GROUP}" \
    WANDB_NAME="${WANDB_NAME}" \
    WANDB_NOTES="gpu${GPU}_actenc_localdepth_stable_2seeds" \
    LOG_DIR="${LOG_DIR}" \
    RUN_LOG="${RUN_LOG}" \
    bash "${RUN_SCRIPT}" \
        agent.depth_anything_v2_patch_state.max_patches_per_camera=10 \
        agent.depth_anything_v2_patch_state.token_scale_init=0.1 \
        agent.depth_anything_v2_patch_state.token_dropout=0.2 \
        algo.num_updates_per_iteration=4 \
        algo.learning_starts=20000 \
        algo.critic_warmup_steps=20000
done

echo "[$(date '+%F %T')] launched all seeds" | tee -a "${LAUNCHER_LOG}"
echo "W&B group:"
echo "https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
echo
echo "Monitor:"
for SEED in ${SEEDS}; do
    echo "  tail -f ${LOG_DIR}/actenc_localdepth_stable_seed${SEED}_gpu${GPU}.log"
done
