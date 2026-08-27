#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
BASE_SCRIPT="${ROOT_DIR}/resfit/rl_finetuning/shell/local/square_macro4710_openpi_localdepth_stable.sh"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_gpu${GPU}_openpi_localdepth_only_seed${SEED}_${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-square_macro4710_openpi_localdepth_seed${SEED}}"
WANDB_NOTES="${WANDB_NOTES:-gpu${GPU}_openpi_localdepth_only}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${WANDB_GROUP}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/openpi_localdepth_seed${SEED}_gpu${GPU}.log}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
PID_FILE="${PID_FILE:-${LOG_DIR}/openpi_localdepth_seed${SEED}_gpu${GPU}.pid}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "GPU=${GPU}" | tee -a "${LAUNCHER_LOG}"
echo "SEED=${SEED}" | tee -a "${LAUNCHER_LOG}"
echo "W&B name=${WANDB_NAME}" | tee -a "${LAUNCHER_LOG}"
echo "W&B group=${WANDB_GROUP}" | tee -a "${LAUNCHER_LOG}"
echo "Log dir=${LOG_DIR}" | tee -a "${LAUNCHER_LOG}"

echo "[$(date '+%F %T')] starting openpi_localdepth on GPU ${GPU}" | tee -a "${LAUNCHER_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup bash "${BASE_SCRIPT}" \
    seed="${SEED}" \
    wandb.name="${WANDB_NAME}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.notes="${WANDB_NOTES}" \
    "$@" \
    >"${RUN_LOG}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started openpi_localdepth PID: ${PID}"
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
