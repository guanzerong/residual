#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_resvit_depth_stable.sh"

GPU_SEED1="${GPU_SEED1:-6}"
GPU_SEED2="${GPU_SEED2:-7}"
SEED1="${SEED1:-1}"
SEED2="${SEED2:-2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu67_actenc_localdepth_resl1_paired_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
EXTRA_ARGS=("$@")

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

{
    echo "GPU_SEED1=${GPU_SEED1}"
    echo "GPU_SEED2=${GPU_SEED2}"
    echo "SEED1=${SEED1}"
    echo "SEED2=${SEED2}"
    echo "W&B group=${WANDB_GROUP}"
    echo "Log dir=${LOG_DIR}"
    echo "W&B group link: https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
} | tee -a "${LAUNCHER_LOG}"

nvidia-smi > "${LOG_DIR}/nvidia_smi_before.txt" 2>&1 || true

PIDS=()

launch_run() {
    local gpu="$1"
    local seed="$2"
    local variant="$3"
    local penalty_coef="$4"
    local penalty_target="$5"

    local tag="actenc_localdepth_${variant}_p8_s0p05_drop0p10_utd3_lr5e5"
    local wandb_name="square_macro4710_${tag}_seed${seed}"
    local run_log="${LOG_DIR}/${tag}_seed${seed}_gpu${gpu}.log"
    local pid_file="${LOG_DIR}/${tag}_seed${seed}_gpu${gpu}.pid"

    echo "[$(date '+%F %T')] launching ${variant} seed=${seed} gpu=${gpu}" | tee -a "${LAUNCHER_LOG}"

    CUDA_VISIBLE_DEVICES="${gpu}" bash "${BASE_SCRIPT}" \
        seed="${seed}" \
        agent.use_residual_image_encoder=false \
        agent.use_base_act_encoder_state=true \
        agent.depth_anything_v2_patch_state.enabled=true \
        agent.depth_anything_v2_patch_state.selection_mode=trajectory \
        agent.depth_anything_v2_patch_state.max_patches_per_camera=8 \
        agent.depth_anything_v2_patch_state.token_scale_init=0.05 \
        agent.depth_anything_v2_patch_state.token_dropout=0.10 \
        algo.num_updates_per_iteration=3 \
        algo.learning_starts=30000 \
        algo.critic_warmup_steps=30000 \
        algo.actor_lr_warmup_steps=20000 \
        agent.critic_lr=5e-5 \
        agent.critic_target_tau=0.003 \
        agent.residual_l1_penalty_coef="${penalty_coef}" \
        agent.residual_l1_penalty_target="${penalty_target}" \
        wandb.name="${wandb_name}" \
        wandb.group="${WANDB_GROUP}" \
        wandb.notes="actenc_localdepth_residual_l1_paired_${variant}" \
        "${EXTRA_ARGS[@]}" \
        > "${run_log}" 2>&1 &

    local pid="$!"
    echo "${pid}" > "${pid_file}"
    PIDS+=("${pid}")
    echo "[$(date '+%F %T')] PID ${pid}, log ${run_log}" | tee -a "${LAUNCHER_LOG}"
}

launch_pair_for_seed() {
    local gpu="$1"
    local seed="$2"

    launch_run "${gpu}" "${seed}" "baseline_resl1off" "0.0" "0.065"
    launch_run "${gpu}" "${seed}" "trust_resl1c1p0_t0p065" "1.0" "0.065"
}

launch_pair_for_seed "${GPU_SEED1}" "${SEED1}"
launch_pair_for_seed "${GPU_SEED2}" "${SEED2}"

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

nvidia-smi > "${LOG_DIR}/nvidia_smi_after.txt" 2>&1 || true
exit "${STATUS}"
