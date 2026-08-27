#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
SERVER_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/scripts/serve_groot_policy_zmq.py"
TRAIN_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_groot_remote_localdepth_stable.sh"

SERVER_GPU="${SERVER_GPU:-6}"
TRAIN_GPU="${TRAIN_GPU:-7}"
SEED="${SEED:-1}"
GROOT_PORT="${GROOT_PORT:-8775}"
GROOT_ROOT="${GROOT_ROOT:-/data_all/gzr1/code/Isaac-GR00T-n1.5}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-nvidia/GR00T-N1.5-3B}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${SERVER_GPU}${TRAIN_GPU}_groot_remote_localdepth_seed${SEED}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
SERVER_LOG="${LOG_DIR}/groot_server_gpu${SERVER_GPU}.log"
TRAIN_LOG="${LOG_DIR}/groot_remote_localdepth_seed${SEED}_gpu${TRAIN_GPU}.log"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
SERVER_PID_FILE="${LOG_DIR}/groot_server_gpu${SERVER_GPU}.pid"
TRAIN_PID_FILE="${LOG_DIR}/groot_remote_localdepth_seed${SEED}_gpu${TRAIN_GPU}.pid"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

echo "SERVER_GPU=${SERVER_GPU}" | tee -a "${LAUNCHER_LOG}"
echo "TRAIN_GPU=${TRAIN_GPU}" | tee -a "${LAUNCHER_LOG}"
echo "SEED=${SEED}" | tee -a "${LAUNCHER_LOG}"
echo "GROOT_PORT=${GROOT_PORT}" | tee -a "${LAUNCHER_LOG}"
echo "W&B group=${WANDB_GROUP}" | tee -a "${LAUNCHER_LOG}"
echo "Log dir=${LOG_DIR}" | tee -a "${LAUNCHER_LOG}"

PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
nohup "${GROOT_PYTHON}" "${SERVER_SCRIPT}" \
    --groot-root "${GROOT_ROOT}" \
    --model-path "${GROOT_MODEL_PATH}" \
    --task square \
    --port "${GROOT_PORT}" \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID="$!"
echo "${SERVER_PID}" > "${SERVER_PID_FILE}"
echo "[$(date '+%F %T')] started GR00T server PID ${SERVER_PID}" | tee -a "${LAUNCHER_LOG}"

sleep 15

PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
GROOT_REMOTE_HOST=127.0.0.1 \
GROOT_REMOTE_PORT="${GROOT_PORT}" \
nohup bash "${TRAIN_SCRIPT}" \
    seed="${SEED}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="square_macro4710_groot_remote_localdepth_seed${SEED}" \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID="$!"
echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
echo "[$(date '+%F %T')] started trainer PID ${TRAIN_PID}" | tee -a "${LAUNCHER_LOG}"

echo "Server log: ${SERVER_LOG}"
echo "Train log: ${TRAIN_LOG}"
echo "W&B group:"
echo "https://wandb.ai/2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3/groups/${WANDB_GROUP}"
