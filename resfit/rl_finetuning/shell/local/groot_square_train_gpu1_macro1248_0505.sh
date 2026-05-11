#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change"
PYTHON_BIN="/home/gzr1/miniconda3/envs/residual1/bin/python"
MODEL_PATH="/data_all/gzr1/code/Isaac-GR00T-n1.5/outputs/gr00t_n15_robomimic_ph_lcsth_bs128_noacc_20260430_1008/checkpoint-20000"
GROOT_DATASET="/data_all/gzr1/openpi/datasets/robomimic_ph__lift_can_square_toolhang_lerobot"
PORT="${GROOT_REMOTE_PORT:-8787}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/deps/robosuite:${ROOT_DIR}/deps/lerobot:${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=300
export PYTHONUNBUFFERED=1
export HF_HOME=/data_all/gzr1/.cache/huggingface
export HF_DATASETS_CACHE=/data_all/gzr1/.cache/huggingface/datasets
export MPLCONFIGDIR=/tmp/matplotlib-gzr1
export RESFIT_KEEP_RUN_DIR=1
mkdir -p "${MPLCONFIGDIR}" "${HF_DATASETS_CACHE}"

"${PYTHON_BIN}" -u -m resfit.rl_finetuning.scripts.train_residual_td3 \
  --config-name=residual_td3_square_config \
  base_policy.provider=groot_remote \
  base_policy.groot_root=/data_all/gzr1/code/Isaac-GR00T-n1.5 \
  base_policy.groot_model_path="${MODEL_PATH}" \
  base_policy.groot_remote_host=127.0.0.1 \
  base_policy.groot_remote_port="${PORT}" \
  base_policy.groot_token_target_count=32 \
  base_policy.groot_base_image_key=observation.images.agentview \
  base_policy.groot_wrist_image_key=observation.images.robot0_eye_in_hand \
  camera_size=224 \
  offline_data.name="${GROOT_DATASET}" \
  offline_data.format=groot_lerobot \
  offline_data.task_index=2 \
  offline_data.num_episodes=200 \
  offline_data.use_base_policy_for_base_actions=true \
  offline_data.base_policy_batch_size=32 \
  algo.offline_fraction=0.5 \
  algo.macro_action_horizon=8 \
  algo.adaptive_macro_enabled=true \
  'algo.adaptive_macro_horizons=[1,2,4,8]' \
  algo.adaptive_macro_offline_stride=2 \
  algo.adaptive_macro_horizon_entropy_reg=0.005 \
  algo.adaptive_macro_horizon_value_ce_coef=0.1 \
  algo.adaptive_macro_horizon_value_temperature=0.1 \
  algo.adaptive_macro_horizon_conditioned_actions=true \
  algo.adaptive_macro_horizon_exploration_initial=0.3 \
  algo.adaptive_macro_horizon_exploration_final=0.05 \
  algo.adaptive_macro_horizon_exploration_steps=300000 \
  algo.total_timesteps=1000000 \
  algo.prefetch_batches=4 \
  algo.gamma=0.996 \
  algo.learning_starts=50000 \
  algo.critic_warmup_steps=50000 \
  algo.actor_lr_warmup_steps=0 \
  algo.num_updates_per_iteration=6 \
  algo.actor_updates_per_iteration=1 \
  algo.random_action_noise_scale=0.05 \
  algo.stddev_max=0.02 \
  algo.stddev_min=0.02 \
  "algo.stddev_schedule='linear(0.02,0.02,300000)'" \
  algo.buffer_size=300000 \
  agent.use_groot_features=true \
  agent.actor.spatial_emb=1024 \
  agent.critic.fuse_patch=0 \
  agent.actor_lr=1e-6 \
  agent.critic_lr=1e-4 \
  agent.critic_target_tau=0.005 \
  eval_first=false \
  eval_interval_every_steps=20000 \
  save_video=false \
  wandb.project=robomimic-square-ph-groot-residual-td3 \
  wandb.name=square_macro1248_grootn15_ckpt20000_off50_grootfeat_img224_0505_gpu1 \
  wandb.group=grootn15_macro1248_off50_grootfeat \
  wandb.mode=online
