#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change"
GROOT_ROOT="/data_all/gzr1/code/Isaac-GR00T-n1.5"
MODEL_PATH="/data_all/gzr1/code/Isaac-GR00T-n1.5/outputs/gr00t_n15_robomimic_ph_lcsth_bs128_noacc_20260430_1008/checkpoint-20000"
PORT="${GROOT_REMOTE_PORT:-8780}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${GROOT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

"/data_all/gzr1/code/Isaac-GR00T-n1.5/.venv/bin/python" -u \
  resfit/rl_finetuning/scripts/serve_groot_policy_zmq.py \
  --groot-root "${GROOT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --task square \
  --token-target-count 32 \
  --port "${PORT}" \
  --base-image-key observation.images.agentview \
  --wrist-image-key observation.images.robot0_eye_in_hand
