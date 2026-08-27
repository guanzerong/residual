#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
DA2_ROOT="${ROOT_DIR}/third_party/Depth-Anything-V2"
PYTHON_BIN="${PYTHON_BIN:-/home/gzr1/miniconda3/envs/residual/bin/python}"
GROOT_ROOT="${GROOT_ROOT:-/data_all/gzr1/code/Isaac-GR00T-n1.5}"
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-${GROOT_ROOT}/outputs/gr00t_n15_robomimic_ph_lcsth_bs128_noacc_20260430_1008/checkpoint-20000}"
GROOT_REMOTE_HOST="${GROOT_REMOTE_HOST:-127.0.0.1}"
GROOT_REMOTE_PORT="${GROOT_REMOTE_PORT:-8775}"
METHOD="${METHOD:-ahr}"
SEED="${SEED:-0}"
WANDB_GROUP="${WANDB_GROUP:-square_q2_groot_n15}"
WANDB_NAME="${WANDB_NAME:-square_${METHOD}_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
RESIDUAL_MODE="shared_prefix"
USE_BASE_POLICY_ENCODER_STATE="true"
USE_DEPTH_PATCH_STATE="true"
EVAL_METRICS_DIR="${EVAL_METRICS_DIR:-/data_all/gzr1/experiment_results/q2_groot_n15_square/metrics}"
export EVAL_METRICS_DIR

case "${METHOD}" in
    fixed4)
        METHOD_ARGS=(
            algo.macro_action_horizon=4
            algo.fixed_proposal_horizon=0
            algo.adaptive_macro_enabled=false
        )
        CACHE_BATCH_SIZE=8
        ;;
    fixed14)
        METHOD_ARGS=(
            algo.macro_action_horizon=14
            algo.fixed_proposal_horizon=0
            algo.adaptive_macro_enabled=false
        )
        CACHE_BATCH_SIZE=8
        ;;
    reactive)
        METHOD_ARGS=(
            algo.macro_action_horizon=1
            algo.fixed_proposal_horizon=14
            algo.adaptive_macro_enabled=false
            algo.n_step=1
        )
        CACHE_BATCH_SIZE=1
        ;;
    ahr)
        METHOD_ARGS=(
            algo.macro_action_horizon=14
            algo.fixed_proposal_horizon=0
            algo.adaptive_macro_enabled=true
            algo.adaptive_macro_horizons=[4,8,12,14]
            algo.adaptive_macro_offline_stride=2
            algo.adaptive_macro_horizon_entropy_reg=0.0
        )
        CACHE_BATCH_SIZE=8
        ;;
    sp)
        # Shared-prefix baseline: one residual chunk is generated and the
        # selected horizon only masks its prefix.
        METHOD_ARGS=(
            algo.macro_action_horizon=14
            algo.fixed_proposal_horizon=0
            algo.adaptive_macro_enabled=true
            algo.adaptive_macro_horizons=[4,8,12,14]
            algo.adaptive_macro_offline_stride=2
            algo.adaptive_macro_horizon_entropy_reg=0.0
        )
        RESIDUAL_MODE="shared_prefix"
        USE_BASE_POLICY_ENCODER_STATE="false"
        USE_DEPTH_PATCH_STATE="false"
        CACHE_BATCH_SIZE=8
        ;;
    hc)
        # Horizon-conditioned variant: the actor emits K residual candidates
        # with one shared decoder plus a duration embedding.
        METHOD_ARGS=(
            algo.macro_action_horizon=14
            algo.fixed_proposal_horizon=0
            algo.adaptive_macro_enabled=true
            algo.adaptive_macro_horizons=[4,8,12,14]
            algo.adaptive_macro_offline_stride=2
            algo.adaptive_macro_horizon_entropy_reg=0.0
        )
        RESIDUAL_MODE="horizon_conditioned"
        USE_BASE_POLICY_ENCODER_STATE="false"
        USE_DEPTH_PATCH_STATE="false"
        CACHE_BATCH_SIZE=8
        ;;
    ahr_full)
        METHOD_ARGS=(
            algo.macro_action_horizon=14
            algo.fixed_proposal_horizon=0
            algo.adaptive_macro_enabled=true
            algo.adaptive_macro_horizons=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            algo.adaptive_macro_offline_stride=2
            algo.adaptive_macro_horizon_entropy_reg=0.0
        )
        CACHE_BATCH_SIZE=8
        ;;
    *)
        echo "Unknown METHOD=${METHOD}; expected fixed4, fixed14, reactive, ahr, sp, hc, or ahr_full" >&2
        exit 2
        ;;
esac

cd "${ROOT_DIR}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_square_config \
    seed="${SEED}" \
    base_policy.provider=groot_remote \
    base_policy.groot_root="${GROOT_ROOT}" \
    base_policy.groot_model_path="${GROOT_MODEL_PATH}" \
    base_policy.groot_remote_host="${GROOT_REMOTE_HOST}" \
    base_policy.groot_remote_port="${GROOT_REMOTE_PORT}" \
    base_policy.groot_token_target_count=32 \
    base_policy.groot_base_image_key=observation.images.agentview \
    base_policy.groot_wrist_image_key=observation.images.robot0_eye_in_hand \
    offline_data.name=ankile/robomimic-ph-square-image \
    offline_data.num_episodes=200 \
    offline_data.cache_loader_batch_size="${CACHE_BATCH_SIZE}" \
    offline_data.cache_loader_num_workers=0 \
    offline_data.use_base_policy_for_base_actions=true \
    "${METHOD_ARGS[@]}" \
    algo.total_timesteps=1000000 \
    algo.prefetch_batches=4 \
    algo.gamma=0.996 \
    algo.learning_starts=12000 \
    algo.critic_warmup_steps=12000 \
    algo.actor_lr_warmup_steps=10000 \
    algo.num_updates_per_iteration=7 \
    algo.actor_updates_per_iteration=1 \
    algo.updates_per_primitive_step=1.0 \
    algo.actor_update_every_n_updates=7 \
    algo.adaptive_residual_mode="${RESIDUAL_MODE}" \
    algo.stddev_max=0.02 \
    algo.stddev_min=0.02 \
    'algo.stddev_schedule=str("linear(0.02,0.02,300000)")' \
    algo.buffer_size=300000 \
    agent.actor_lr=1e-6 \
    agent.critic_lr=1e-4 \
    agent.critic_target_tau=0.005 \
    agent.use_residual_image_encoder=true \
    agent.use_base_act_encoder_state=false \
    agent.use_base_policy_encoder_state="${USE_BASE_POLICY_ENCODER_STATE}" \
    agent.depth_anything_v2_conditioning.enabled=false \
    agent.depth_anything_v2_patch_state.enabled="${USE_DEPTH_PATCH_STATE}" \
    agent.depth_anything_v2_patch_state.encoder=vits \
    agent.depth_anything_v2_patch_state.freeze_encoder=true \
    agent.depth_anything_v2_patch_state.source_root="${DA2_ROOT}" \
    agent.depth_anything_v2_patch_state.selection_mode=trajectory \
    agent.depth_anything_v2_patch_state.max_patches_per_camera=8 \
    agent.depth_anything_v2_patch_state.token_scale_init=0.05 \
    agent.depth_anything_v2_patch_state.token_dropout=0.10 \
    eval_interval_every_steps=20000 \
    eval_num_envs=1 \
    eval_num_episodes=50 \
    eval_final_num_episodes=100 \
    wandb.project=robomimic-square-ph-residual-td3 \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="${WANDB_NAME}" \
    wandb.mode="${WANDB_MODE}" \
    "$@"
