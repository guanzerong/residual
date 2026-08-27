#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
GROOT_ROOT="${GROOT_ROOT:-/data_all/gzr1/code/Isaac-GR00T-n1.5}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-${GROOT_ROOT}/outputs/gr00t_n15_robomimic_ph_lcsth_bs128_noacc_20260430_1008/checkpoint-20000}"
RESIDUAL_PYTHON="${RESIDUAL_PYTHON:-/home/gzr1/miniconda3/envs/residual/bin/python}"
SERVER_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/scripts/serve_groot_policy_zmq.py"
TRAIN_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_q2_groot_n15_train.sh"
GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required}"
SEED="${SEED:?SEED is required}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-$((8870 + GPU))}"
RESULT_ROOT="${RESULT_ROOT:-/data_all/gzr1/experiment_results/q2_groot_n15_square/${RUN_ID}}"
RUN_DIR="${RESULT_ROOT}/${METHOD}/seed${SEED}"
WANDB_GROUP="${WANDB_GROUP:-square_q2_groot_n15_${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-offline}"

case "${METHOD}" in
    ahr|sp|hc)
        CANDIDATE_HORIZONS='[4,8,12,14]'
        ;;
    ahr_full)
        CANDIDATE_HORIZONS='[1,2,3,4,5,6,7,8,9,10,11,12,13,14]'
        ;;
    *)
        CANDIDATE_HORIZONS='[]'
        ;;
esac

if [[ ! -f "${GROOT_MODEL_PATH}/config.json" ]]; then
    echo "Missing GR00T checkpoint: ${GROOT_MODEL_PATH}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}/metrics" "${RUN_DIR}/wandb" "${RESULT_ROOT}/cache/${METHOD}"
SERVER_LOG="${RUN_DIR}/server.log"
TRAIN_LOG="${RUN_DIR}/trainer.log"
GPU_LOG="${RUN_DIR}/gpu.csv"
START_UNIX="$(date +%s)"
STATUS="failed"
SERVER_PID=""
MONITOR_PID=""

cleanup() {
    local exit_code=$?
    [[ "${exit_code}" -eq 0 ]] && STATUS="complete"
    [[ -n "${MONITOR_PID}" ]] && kill "${MONITOR_PID}" 2>/dev/null || true
    [[ -n "${SERVER_PID}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    local end_unix
    end_unix="$(date +%s)"
    printf '{"status":"%s","exit_code":%d,"start_unix":%s,"end_unix":%s,"wall_clock_seconds":%s}\n' \
        "${STATUS}" "${exit_code}" "${START_UNIX}" "${end_unix}" "$((end_unix - START_UNIX))" \
        > "${RUN_DIR}/status.json"
}
trap cleanup EXIT INT TERM

GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
CONFIG_SHA="$(sha256sum "${GROOT_MODEL_PATH}/config.json" | awk '{print $1}')"
cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "task": "Square",
  "method": "${METHOD}",
  "seed": ${SEED},
  "gpu": ${GPU},
  "groot_wandb_run": "e6m6ioxw",
  "groot_model_path": "${GROOT_MODEL_PATH}",
  "groot_config_sha256": "${CONFIG_SHA}",
  "groot_predicted_horizon": 16,
  "max_execution_horizon": 14,
  "candidate_horizons": ${CANDIDATE_HORIZONS},
  "residual_git_commit": "${GIT_COMMIT}",
  "wandb_group": "${WANDB_GROUP}",
  "start_unix": ${START_UNIX}
}
EOF

nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -i "${GPU}" -l 5 > "${GPU_LOG}" 2>&1 &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    "${GROOT_PYTHON}" "${SERVER_SCRIPT}" \
    --groot-root "${GROOT_ROOT}" \
    --model-path "${GROOT_MODEL_PATH}" \
    --task square \
    --port "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

PYTHONPATH="${ROOT_DIR}" "${RESIDUAL_PYTHON}" \
    "${ROOT_DIR}/resfit/rl_finetuning/scripts/probe_groot_server.py" \
    --port "${PORT}" \
    --output "${RUN_DIR}/server_metadata.json" \
    > "${RUN_DIR}/server_probe.log" 2>&1

export METHOD SEED GROOT_ROOT GROOT_MODEL_PATH WANDB_GROUP WANDB_MODE
export GROOT_REMOTE_HOST=127.0.0.1 GROOT_REMOTE_PORT="${PORT}"
export WANDB_NAME="square_${METHOD}_seed${SEED}"
export EVAL_METRICS_DIR="${RUN_DIR}/metrics"
export WANDB_DIR="${RUN_DIR}/wandb"
export CACHE_DIR="${RESULT_ROOT}/cache/${METHOD}"
export MPLCONFIGDIR="${RUN_DIR}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -v -o "${RUN_DIR}/trainer.time.txt" \
    bash "${TRAIN_SCRIPT}" "$@" \
    > "${TRAIN_LOG}" 2>&1

STATUS="complete"
