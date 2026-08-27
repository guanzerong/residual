#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
JOB_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/run_square_q2_groot_n15_job.sh"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-/data_all/gzr1/experiment_results/q2_groot_n15_square/residual_modes_1m_${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-square_residual_modes_1m_${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-offline}"
OFFLINE_CACHE_PATH="${OFFLINE_CACHE_PATH:-/data_all/gzr1/experiment_results/q2_groot_n15_square/formal_n15_20260810_1430/cache/ahr/offline_buffer_cache/5e58e4f0}"
GPUS_STRING="${GPUS:-1 2 3}"
EVAL_METRICS_DIR="${EVAL_METRICS_DIR:-${RESULT_ROOT}/metrics}"

read -r -a GPUS_ARRAY <<< "${GPUS_STRING}"
if [[ "${#GPUS_ARRAY[@]}" -lt 3 ]]; then
    echo "Need at least three GPU ids in GPUS (got: ${GPUS_STRING})" >&2
    exit 2
fi
if [[ ! -d "${OFFLINE_CACHE_PATH}" ]]; then
    echo "Offline cache does not exist: ${OFFLINE_CACHE_PATH}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}/logs"
cat > "${RESULT_ROOT}/allocation.txt" <<EOF
HC: GPU ${GPUS_ARRAY[0]} seed0, GPU ${GPUS_ARRAY[1]} seed1, GPU ${GPUS_ARRAY[2]} seed2
SP: GPU ${GPUS_ARRAY[0]} seed0, GPU ${GPUS_ARRAY[1]} seed1, GPU ${GPUS_ARRAY[2]} seed2
offline_cache: ${OFFLINE_CACHE_PATH}
timesteps: 1000000 primitive steps per run
EOF

run_one() {
    local method="$1"
    local gpu="$2"
    local seed="$3"
    local log_file="${RESULT_ROOT}/logs/${method}_gpu${gpu}_seed${seed}.log"
    echo "[$(date --iso-8601=seconds)] starting ${method} seed${seed} on GPU ${gpu}" | tee -a "${log_file}"
    GPU="${gpu}" METHOD="${method}" SEED="${seed}" \
        RUN_ID="${RUN_ID}" RESULT_ROOT="${RESULT_ROOT}" \
        WANDB_GROUP="${WANDB_GROUP}" WANDB_MODE="${WANDB_MODE}" \
        OFFLINE_CACHE_PATH="${OFFLINE_CACHE_PATH}" EVAL_METRICS_DIR="${EVAL_METRICS_DIR}" \
        bash "${JOB_SCRIPT}" \
        algo.total_timesteps=1000000 \
        offline_data.num_episodes=200 \
        eval_first=true \
        eval_interval_every_steps=20000 \
        eval_num_episodes=50 \
        eval_final_num_episodes=100 \
        save_video=false \
        >>"${log_file}" 2>&1
    echo "[$(date --iso-8601=seconds)] finished ${method} seed${seed}" | tee -a "${log_file}"
}

run_wave() {
    local method="$1"
    local pids=()
    local idx
    for idx in 0 1 2; do
        run_one "${method}" "${GPUS_ARRAY[$idx]}" "${idx}" &
        pids+=("$!")
    done

    local status=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

echo "[$(date --iso-8601=seconds)] launching HC wave"
run_wave hc
echo "[$(date --iso-8601=seconds)] HC wave complete; launching SP wave"
run_wave sp
echo "[$(date --iso-8601=seconds)] all residual-mode runs complete"
touch "${RESULT_ROOT}/COMPLETE"

RESIDUAL_PYTHON="${RESIDUAL_PYTHON:-/home/gzr1/miniconda3/envs/residual/bin/python}"
if [[ -f "${ROOT_DIR}/resfit/rl_finetuning/scripts/summarize_horizon_set_ablation.py" ]]; then
    "${RESIDUAL_PYTHON}" "${ROOT_DIR}/resfit/rl_finetuning/scripts/summarize_horizon_set_ablation.py" \
        "${RESULT_ROOT}" --json > "${RESULT_ROOT}/summary.json" || true
fi
