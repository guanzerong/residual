#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
RUNNER="${ROOT_DIR}/resfit/rl_finetuning/shell/local/run_square_reactive_vs_ahr.sh"
RESULT_ROOT="${RESULT_ROOT:-/data_all/gzr1/experiment_results/square_reactive_vs_ahr}"
CACHE_ROOT="${CACHE_ROOT:-/data_all/gzr1/experiment_cache/square_reactive_vs_ahr}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_reactive_vs_ahr_${RUN_ID}}"

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/wandb" "${CACHE_ROOT}"
export EVAL_METRICS_DIR="${RESULT_ROOT}"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export MPLCONFIGDIR="${RESULT_ROOT}/matplotlib"
export WANDB_GROUP RUN_ID
mkdir -p "${MPLCONFIGDIR}"

prewarm_cache() {
    local mode="$1"
    local gpu="$2"
    local log_file="${RESULT_ROOT}/logs/prewarm_${mode}.log"
    echo "[$(date --iso-8601=seconds)] prewarming ${mode} cache on GPU ${gpu}" | tee -a "${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" MODE="${mode}" SEED=0 CACHE_DIR="${CACHE_ROOT}" \
        bash "${RUNNER}" \
        algo.total_timesteps=0 \
        algo.learning_starts=4 \
        algo.critic_warmup_steps=0 \
        algo.actor_lr_warmup_steps=0 \
        eval_first=false \
        eval_final_num_episodes=0 \
        wandb.mode=disabled \
        save_video=false >>"${log_file}" 2>&1
    echo "[$(date --iso-8601=seconds)] ${mode} cache ready" | tee -a "${log_file}"
}

run_worker() {
    local mode="$1"
    local gpu="$2"
    shift 2
    local seeds=("$@")
    local log_file="${RESULT_ROOT}/logs/${mode}_gpu${gpu}.log"

    for seed in "${seeds[@]}"; do
        echo "[$(date --iso-8601=seconds)] starting ${mode} seed ${seed} on GPU ${gpu}" | tee -a "${log_file}"
        CUDA_VISIBLE_DEVICES="${gpu}" MODE="${mode}" SEED="${seed}" CACHE_DIR="${CACHE_ROOT}" \
            WANDB_NAME="square_${mode}_seed${seed}" \
            bash "${RUNNER}" wandb.mode=offline save_video=false >>"${log_file}" 2>&1
        echo "[$(date --iso-8601=seconds)] finished ${mode} seed ${seed}" | tee -a "${log_file}"
    done
}

# Build each method's shared offline replay cache once before concurrent seeds
# read it. Online warm-up caches remain seed-specific via their metadata hash.
prewarm_cache reactive 4 &
prewarm_reactive_pid=$!
prewarm_cache ahr 5 &
prewarm_ahr_pid=$!
wait "${prewarm_reactive_pid}"
wait "${prewarm_ahr_pid}"

run_worker reactive 4 0 2 4 &
reactive_even_pid=$!
run_worker ahr 5 0 2 4 &
ahr_even_pid=$!
run_worker reactive 6 1 3 &
reactive_odd_pid=$!
run_worker ahr 7 1 3 &
ahr_odd_pid=$!

status=0
wait "${reactive_even_pid}" || status=1
wait "${ahr_even_pid}" || status=1
wait "${reactive_odd_pid}" || status=1
wait "${ahr_odd_pid}" || status=1

if [[ "${status}" -eq 0 ]]; then
    touch "${RESULT_ROOT}/COMPLETE"
    python "${ROOT_DIR}/resfit/rl_finetuning/scripts/summarize_square_reactive_vs_ahr.py" "${RESULT_ROOT}" \
        | tee "${RESULT_ROOT}/summary.txt"
fi

exit "${status}"
