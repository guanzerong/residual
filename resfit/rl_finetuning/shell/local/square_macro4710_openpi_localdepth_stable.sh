#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
DA2_ROOT="${ROOT_DIR}/third_party/Depth-Anything-V2"
OPENPI_ROOT="${OPENPI_ROOT:-/data_all/gzr1/openpi}"
PYTHON_BIN="${PYTHON_BIN:-/data_all/gzr1/openpi/.venv/bin/python}"
OPENPI_TRAIN_CONFIG="${OPENPI_TRAIN_CONFIG:-pi05_robomimic_lcsth}"
OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-/data_all/gzr1/openpi/checkpoints/pi05_robomimic_lcsth/pi05_robomimic_lcsth_lora_20260427_1625/10000}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    base_policy.provider=openpi \
    base_policy.openpi_root="${OPENPI_ROOT}" \
    base_policy.openpi_train_config="${OPENPI_TRAIN_CONFIG}" \
    base_policy.openpi_checkpoint_dir="${OPENPI_CHECKPOINT_DIR}" \
    base_policy.openpi_token_pool_size=4 \
    base_policy.openpi_base_image_key=observation.images.agentview \
    base_policy.openpi_left_wrist_image_key=observation.images.robot0_eye_in_hand \
    base_policy.openpi_right_wrist_image_key=null \
    offline_data.name=ankile/robomimic-ph-square-image \
    offline_data.num_episodes=200 \
    offline_data.use_base_policy_for_base_actions=true \
    algo.macro_action_horizon=10 \
    algo.adaptive_macro_enabled=true \
    algo.adaptive_macro_horizons=[4,7,10] \
    algo.adaptive_macro_offline_stride=2 \
    algo.adaptive_macro_horizon_entropy_reg=0.0 \
    algo.total_timesteps=1000000 \
    algo.prefetch_batches=4 \
    algo.gamma=0.996 \
    algo.learning_starts=30000 \
    algo.critic_warmup_steps=30000 \
    algo.actor_lr_warmup_steps=20000 \
    algo.num_updates_per_iteration=3 \
    algo.actor_updates_per_iteration=1 \
    algo.stddev_max=0.02 \
    algo.stddev_min=0.02 \
    algo.buffer_size=300000 \
    agent.actor_lr=1e-6 \
    agent.critic_lr=5e-5 \
    agent.critic_target_tau=0.003 \
    agent.use_residual_image_encoder=false \
    agent.use_base_act_encoder_state=false \
    agent.use_base_policy_encoder_state=true \
    agent.depth_anything_v2_patch_state.enabled=true \
    agent.depth_anything_v2_patch_state.encoder=vits \
    agent.depth_anything_v2_patch_state.freeze_encoder=true \
    agent.depth_anything_v2_patch_state.source_root="${DA2_ROOT}" \
    agent.depth_anything_v2_patch_state.selection_mode=trajectory \
    agent.depth_anything_v2_patch_state.max_patches_per_camera=8 \
    agent.depth_anything_v2_patch_state.token_scale_init=0.05 \
    agent.depth_anything_v2_patch_state.token_dropout=0.10 \
    eval_interval_every_steps=20000 \
    wandb.project=robomimic-square-ph-residual-td3 \
    wandb.name=square_macro4710_openpi_localdepth_stable \
    wandb.group=macro4710_openpi_localdepth_stable \
    wandb.mode=online \
    "$@"
