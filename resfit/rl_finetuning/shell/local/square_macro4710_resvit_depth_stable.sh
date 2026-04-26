#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
DA2_ROOT="${ROOT_DIR}/third_party/Depth-Anything-V2"
PYTHON_BIN="${PYTHON_BIN:-/home/gzr1/miniconda3/envs/residual/bin/python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    base_policy.wandb_id=square-ph-bc/i9tt1t4a \
    base_policy.wt_type=latest \
    base_policy.wt_version=v41 \
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
    algo.learning_starts=12000 \
    algo.critic_warmup_steps=12000 \
    algo.actor_lr_warmup_steps=10000 \
    algo.num_updates_per_iteration=7 \
    algo.actor_updates_per_iteration=1 \
    algo.stddev_max=0.02 \
    algo.stddev_min=0.02 \
    algo.buffer_size=300000 \
    agent.actor_lr=1e-6 \
    agent.critic_lr=1e-4 \
    agent.critic_target_tau=0.005 \
    agent.use_residual_image_encoder=true \
    agent.use_base_act_encoder_state=false \
    agent.depth_anything_v2_patch_state.enabled=true \
    agent.depth_anything_v2_patch_state.encoder=vits \
    agent.depth_anything_v2_patch_state.freeze_encoder=true \
    agent.depth_anything_v2_patch_state.source_root="${DA2_ROOT}" \
    agent.depth_anything_v2_patch_state.max_patches_per_camera=20 \
    agent.depth_anything_v2_patch_state.token_scale_init=0.25 \
    agent.depth_anything_v2_patch_state.token_dropout=0.15 \
    eval_interval_every_steps=20000 \
    wandb.project=robomimic-square-ph-residual-td3 \
    wandb.name=square_macro4710_adaptive_resvit_localdepth_stable \
    wandb.group=macro4710_adaptive_localdepth_stable \
    wandb.mode=online \
    "$@"
