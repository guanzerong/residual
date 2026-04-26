#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gzr1/miniconda3/envs/residual/bin/python}"

# Launch ACT BC training for the two-arm Robomimic Transport PH task.
# The PH dataset contains 84x84 cameras:
#   observation.images.agentview
#   observation.images.robot0_eye_in_hand
#   observation.images.robot1_eye_in_hand

"${PYTHON_BIN}" -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-ph-transport-image \
    --policy act \
    --policy_kwargs "chunk_size=8,n_action_steps=8" \
    --batch_size 256 \
    --num_workers 8 \
    --wandb_project robomimic-transport-ph-bc \
    --wandb_enable \
    --eval_env Transport \
    --rollout_freq 5000 \
    --steps 100000 \
    --policy_cameras agentview robot0_eye_in_hand robot1_eye_in_hand \
    --eval_camera_size 84 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 10000 \
    "$@"
