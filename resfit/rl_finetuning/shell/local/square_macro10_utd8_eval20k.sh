#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    base_policy.wandb_id=square-ph-bc/i9tt1t4a \
    base_policy.wt_type=latest \
    base_policy.wt_version=latest \
    offline_data.name=ankile/robomimic-ph-square-image \
    offline_data.num_episodes=200 \
    offline_data.use_base_policy_for_base_actions=true \
    algo.macro_action_horizon=10 \
    algo.total_timesteps=1000000 \
    algo.prefetch_batches=4 \
    algo.gamma=0.995 \
    algo.learning_starts=20000 \
    algo.critic_warmup_steps=20000 \
    algo.num_updates_per_iteration=8 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    algo.buffer_size=300000 \
    eval_interval_every_steps=20000 \
    wandb.project=robomimic-square-ph-residual-td3 \
    wandb.name=square_macro10_utd8_warm20k_eval20k \
    wandb.group=macro10 \
    wandb.mode=online
