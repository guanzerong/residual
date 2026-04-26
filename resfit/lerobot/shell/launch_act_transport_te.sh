#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gzr1/miniconda3/envs/residual/bin/python}"
CACHE_DIR="${CACHE_DIR:-/data_all/gzr1/cache}"
export CACHE_DIR

SEED="${SEED:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-200000}"
ROLLOUT_FREQ="${ROLLOUT_FREQ:-20000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-8}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-50}"
WANDB_GROUP="${WANDB_GROUP:-transport_ph_act_te_${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-transport_ph_act_te_seed${SEED}_${RUN_ID}}"

"${PYTHON_BIN}" -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-ph-transport-image \
    --policy act \
    --policy_kwargs "chunk_size=8,n_action_steps=1,temporal_ensemble_coeff=0.01" \
    --batch_size 256 \
    --num_workers 8 \
    --wandb_project robomimic-transport-ph-bc \
    --wandb_enable \
    --wandb_name "${WANDB_NAME}" \
    --wandb_group "${WANDB_GROUP}" \
    --eval_env Transport \
    --rollout_freq "${ROLLOUT_FREQ}" \
    --steps "${STEPS}" \
    --policy_cameras agentview robot0_eye_in_hand robot1_eye_in_hand \
    --eval_camera_size 84 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    --eval_num_episodes "${EVAL_NUM_EPISODES}" \
    --log_freq 100 \
    --save_freq "${SAVE_FREQ}" \
    --seed "${SEED}" \
    "$@"
