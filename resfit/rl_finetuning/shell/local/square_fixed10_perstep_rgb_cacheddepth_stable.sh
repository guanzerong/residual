#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
DA2_ROOT="${ROOT_DIR}/third_party/Depth-Anything-V2"
PYTHON_BIN="${PYTHON_BIN:-/home/gzr1/miniconda3/envs/residual/bin/python}"
SEED="${SEED:-0}"
WANDB_GROUP="${WANDB_GROUP:-square_fixed10_perstep_rgb_cacheddepth}"
WANDB_NAME="${WANDB_NAME:-square_fixed10_perstep_rgb_cacheddepth_seed${SEED}}"
EVAL_METRICS_DIR="${EVAL_METRICS_DIR:-/data_all/gzr1/experiment_results/square_reactive_vs_ahr}"
export EVAL_METRICS_DIR

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    seed="${SEED}" \
    base_policy.local_path=/data_all/gzr1/.wandb/artifacts/run_i9tt1t4a_latest:v41/policy \
    base_policy.wandb_id=square-ph-bc/i9tt1t4a \
    base_policy.wt_type=latest \
    base_policy.wt_version=v41 \
    offline_data.name=ankile/robomimic-ph-square-image \
    offline_data.num_episodes=200 \
    offline_data.cache_loader_batch_size=1 \
    offline_data.use_base_policy_for_base_actions=true \
    algo.macro_action_horizon=1 \
    algo.fixed_proposal_horizon=10 \
    algo.adaptive_macro_enabled=false \
    algo.n_step=1 \
    algo.update_every_n_steps=7 \
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
    eval_num_envs=1 \
    eval_num_episodes=50 \
    eval_final_num_episodes=100 \
    wandb.project=robomimic-square-ph-residual-td3 \
    wandb.name="${WANDB_NAME}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.mode=online \
    "$@"
