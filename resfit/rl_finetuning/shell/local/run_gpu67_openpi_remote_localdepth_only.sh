#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
SERVER_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/scripts/serve_openpi_policy_ws.py"
TRAIN_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_openpi_remote_localdepth_stable.sh"

SERVER_GPU="${SERVER_GPU:-6}"
TRAIN_GPU="${TRAIN_GPU:-7}"
SEED="${SEED:-1}"
OPENPI_PORT="${OPENPI_PORT:-8766}"
OPENPI_ROOT="${OPENPI_ROOT:-/data_all/gzr1/openpi}"
OPENPI_TRAIN_CONFIG="${OPENPI_TRAIN_CONFIG:-pi05_robomimic_lcsth}"
OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-/data_all/gzr1/openpi/checkpoints/pi05_robomimic_lcsth/pi05_robomimic_lcsth_lora_20260427_1625/10000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${SERVER_GPU}${TRAIN_GPU}_openpi_remote_localdepth_seed${SEED}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
SERVER_LOG="${LOG_DIR}/openpi_server_gpu${SERVER_GPU}.log"
TRAIN_LOG="${LOG_DIR}/openpi_remote_localdepth_seed${SEED}_gpu${TRAIN_GPU}.log"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
SERVER_PID_FILE="${LOG_DIR}/openpi_server_gpu${SERVER_GPU}.pid"
TRAIN_PID_FILE="${LOG_DIR}/openpi_remote_localdepth_seed${SEED}_gpu${TRAIN_GPU}.pid"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

echo "SERVER_GPU=${SERVER_GPU}" | tee -a "${LAUNCHER_LOG}"
echo "TRAIN_GPU=${TRAIN_GPU}" | tee -a "${LAUNCHER_LOG}"
echo "SEED=${SEED}" | tee -a "${LAUNCHER_LOG}"
echo "OPENPI_PORT=${OPENPI_PORT}" | tee -a "${LAUNCHER_LOG}"
echo "W&B group=${WANDB_GROUP}" | tee -a "${LAUNCHER_LOG}"
echo "Log dir=${LOG_DIR}" | tee -a "${LAUNCHER_LOG}"

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
TF_FORCE_GPU_ALLOW_GROWTH=true \
nohup /data_all/gzr1/openpi/.venv/bin/python "${SERVER_SCRIPT}" \
    --openpi-root "${OPENPI_ROOT}" \
    --config "${OPENPI_TRAIN_CONFIG}" \
    --checkpoint-dir "${OPENPI_CHECKPOINT_DIR}" \
    --task square \
    --port "${OPENPI_PORT}" \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID="$!"
echo "${SERVER_PID}" > "${SERVER_PID_FILE}"
echo "[$(date '+%F %T')] started OpenPI server PID ${SERVER_PID}" | tee -a "${LAUNCHER_LOG}"

for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${OPENPI_PORT}/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if ! curl -fsS "http://127.0.0.1:${OPENPI_PORT}/healthz" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] OpenPI server failed health check" | tee -a "${LAUNCHER_LOG}"
    exit 1
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
OPENPI_REMOTE_HOST=127.0.0.1 \
OPENPI_REMOTE_PORT="${OPENPI_PORT}" \
nohup bash "${TRAIN_SCRIPT}" \
    seed="${SEED}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="square_macro4710_openpi_remote_localdepth_seed${SEED}" \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID="$!"
echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
echo "[$(date '+%F %T')] started trainer PID ${TRAIN_PID}" | tee -a "${LAUNCHER_LOG}"

echo "Server log: ${SERVER_LOG}"
echo "Train log: ${TRAIN_LOG}"
echo "W&B group:"
echo "https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
