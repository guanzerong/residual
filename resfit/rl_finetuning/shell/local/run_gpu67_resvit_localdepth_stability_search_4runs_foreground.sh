#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"

GPU_SAFE="${GPU_SAFE:-6}"
GPU_BALANCED="${GPU_BALANCED:-7}"
SEEDS="${SEEDS:-1 2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu67_resvit_localdepth_stability_search_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
EXTRA_ARGS=("$@")

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

{
    echo "GPU_SAFE=${GPU_SAFE}"
    echo "GPU_BALANCED=${GPU_BALANCED}"
    echo "SEEDS=${SEEDS}"
    echo "W&B group=${WANDB_GROUP}"
    echo "Log dir=${LOG_DIR}"
    echo "W&B group link: https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
} | tee -a "${LAUNCHER_LOG}"

nvidia-smi > "${LOG_DIR}/nvidia_smi_before.txt" 2>&1 || true

PIDS=()

launch_safe() {
    local seed="$1"
    local gpu="${GPU_SAFE}"
    local tag="safe_p8_s0p05_w100k_zeroinit_drop0p10_utd3"
    local wandb_name="square_macro4710_resvit_localdepth_${tag}_seed${seed}"
    local run_log="${LOG_DIR}/${tag}_seed${seed}_gpu${gpu}.log"
    local pid_file="${LOG_DIR}/${tag}_seed${seed}_gpu${gpu}.pid"

    echo "[$(date '+%F %T')] launching ${tag} seed=${seed} gpu=${gpu}" | tee -a "${LAUNCHER_LOG}"

    CUDA_VISIBLE_DEVICES="${gpu}" bash "${BASE_SCRIPT}" \
        seed="${seed}" \
        agent.use_residual_image_encoder=true \
        agent.use_base_act_encoder_state=false \
        agent.depth_anything_v2_patch_state.enabled=true \
        agent.depth_anything_v2_patch_state.selection_mode=trajectory \
        agent.depth_anything_v2_patch_state.max_patches_per_camera=8 \
        agent.depth_anything_v2_patch_state.token_scale_init=0.05 \
        agent.depth_anything_v2_patch_state.token_scale_warmup_steps=100000 \
        agent.depth_anything_v2_patch_state.zero_init_projector=true \
        agent.depth_anything_v2_patch_state.token_dropout=0.10 \
        algo.num_updates_per_iteration=3 \
        algo.learning_starts=30000 \
        algo.critic_warmup_steps=30000 \
        algo.actor_lr_warmup_steps=30000 \
        agent.critic_lr=5e-5 \
        agent.critic_target_tau=0.003 \
        wandb.name="${wandb_name}" \
        wandb.group="${WANDB_GROUP}" \
        wandb.notes="safe_local_depth_zero_init_small_scale_long_warmup_low_utd" \
        "${EXTRA_ARGS[@]}" \
        > "${run_log}" 2>&1 &

    local pid="$!"
    echo "${pid}" > "${pid_file}"
    PIDS+=("${pid}")
    echo "[$(date '+%F %T')] PID ${pid}, log ${run_log}" | tee -a "${LAUNCHER_LOG}"
}

launch_balanced() {
    local seed="$1"
    local gpu="${GPU_BALANCED}"
    local tag="balanced_p12_s0p08_w70k_zeroinit_drop0p15_utd4"
    local wandb_name="square_macro4710_resvit_localdepth_${tag}_seed${seed}"
    local run_log="${LOG_DIR}/${tag}_seed${seed}_gpu${gpu}.log"
    local pid_file="${LOG_DIR}/${tag}_seed${seed}_gpu${gpu}.pid"

    echo "[$(date '+%F %T')] launching ${tag} seed=${seed} gpu=${gpu}" | tee -a "${LAUNCHER_LOG}"

    CUDA_VISIBLE_DEVICES="${gpu}" bash "${BASE_SCRIPT}" \
        seed="${seed}" \
        agent.use_residual_image_encoder=true \
        agent.use_base_act_encoder_state=false \
        agent.depth_anything_v2_patch_state.enabled=true \
        agent.depth_anything_v2_patch_state.selection_mode=trajectory \
        agent.depth_anything_v2_patch_state.max_patches_per_camera=12 \
        agent.depth_anything_v2_patch_state.token_scale_init=0.08 \
        agent.depth_anything_v2_patch_state.token_scale_warmup_steps=70000 \
        agent.depth_anything_v2_patch_state.zero_init_projector=true \
        agent.depth_anything_v2_patch_state.token_dropout=0.15 \
        algo.num_updates_per_iteration=4 \
        algo.learning_starts=30000 \
        algo.critic_warmup_steps=30000 \
        algo.actor_lr_warmup_steps=25000 \
        agent.critic_lr=7e-5 \
        agent.critic_target_tau=0.004 \
        wandb.name="${wandb_name}" \
        wandb.group="${WANDB_GROUP}" \
        wandb.notes="balanced_local_depth_zero_init_moderate_scale_warmup_moderate_utd" \
        "${EXTRA_ARGS[@]}" \
        > "${run_log}" 2>&1 &

    local pid="$!"
    echo "${pid}" > "${pid_file}"
    PIDS+=("${pid}")
    echo "[$(date '+%F %T')] PID ${pid}, log ${run_log}" | tee -a "${LAUNCHER_LOG}"
}

for seed in ${SEEDS}; do
    launch_safe "${seed}"
done

for seed in ${SEEDS}; do
    launch_balanced "${seed}"
done

echo "[$(date '+%F %T')] launched ${#PIDS[@]} runs, waiting" | tee -a "${LAUNCHER_LOG}"

STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        STATUS=1
        echo "[$(date '+%F %T')] PID ${pid} failed" | tee -a "${LAUNCHER_LOG}"
    else
        echo "[$(date '+%F %T')] PID ${pid} finished" | tee -a "${LAUNCHER_LOG}"
    fi
done

exit "${STATUS}"
