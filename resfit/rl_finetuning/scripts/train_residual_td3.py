# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os

# Cap all BLAS/OpenMP threadpools (critical to set before importing numpy/torch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Stop threads from spin-waiting
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import hashlib
import json
import logging
import pprint
import random
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import tensordict
import torch
import torchrl
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.utils.groot_adapter import GROOTRemoteBasePolicyAdapter
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.openpi_adapter import OpenPIBasePolicyAdapter, OpenPIRemoteBasePolicyAdapter
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper


# -----------------------------------------------------------------------------
# Timing utility --------------------------------------------------------------
# -----------------------------------------------------------------------------
class TrainingTimer:
    """Simple timing utility for measuring training stage proportions."""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()

    @contextmanager
    def time(self, stage_name: str):
        """Context manager to time a specific training stage."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[stage_name].append(elapsed)

    def get_timing_stats(self) -> dict[str, float]:
        """Get timing statistics as percentages of total time."""
        if not self.times:
            return {}

        # Calculate total time across all stages
        total_time = sum(sum(times) for times in self.times.values())
        if total_time == 0:
            return {}

        # Calculate percentages and averages
        stats = {}
        for stage_name, times_list in self.times.items():
            stage_total = sum(times_list)
            stage_avg = stage_total / len(times_list) if times_list else 0
            stage_percentage = (stage_total / total_time) * 100

            stats[f"timing/{stage_name}_percentage"] = stage_percentage
            stats[f"timing/{stage_name}_avg_ms"] = stage_avg * 1000  # Convert to ms
            stats[f"timing/{stage_name}_total_s"] = stage_total

        return stats

    def reset(self):
        """Reset all timing data."""
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()


# -----------------------------------------------------------------------------
# Logging configuration -------------------------------------------------------
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Hugging Face buffer cache helpers (global) ---------------------------------
# -----------------------------------------------------------------------------
OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
ONLINE_HF_REPO = os.environ.get("HF_ONLINE_BUFFER_REPO", None)

if OFFLINE_HF_REPO is not None:
    logger.info(f"Using offline buffer from {OFFLINE_HF_REPO}")
if ONLINE_HF_REPO is not None:
    logger.info(f"Using online buffer from {ONLINE_HF_REPO}")

# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()

# Dedicated sub-folders for the different cache types -----------------------
OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_buffer_cache"
ONLINE_CACHE_DIR = _CACHE_ROOT / "online_buffer_cache"


# -----------------------------------------------------------------------------
# Repository-local imports ------------------------------------------------------
# -----------------------------------------------------------------------------
os.environ["MUJOCO_GL"] = "egl"

if "MUJOCO_EGL_DEVICE_ID" in os.environ:
    del os.environ["MUJOCO_EGL_DEVICE_ID"]


def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
    add_batch_dim: bool = True,
    depth_cls_cache_fn=None,
    depth_patch_cache_fn=None,
    base_act_encoder_cache_fn=None,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)

    def _keep_observation_key(key: str) -> bool:
        return (
            key in obs_keys_set
            or key.startswith("observation.depth_cls.")
            or key.startswith("observation.depth_patch_tokens.")
            or key == "observation.base_act_encoder_tokens"
        )
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if _keep_observation_key(k)}
        next_obs_i = {k: v for k, v in next_obs_i.items() if _keep_observation_key(k)}
        if depth_cls_cache_fn is not None:
            curr_obs_i.update(depth_cls_cache_fn(curr_obs_i))
            next_obs_i.update(depth_cls_cache_fn(next_obs_i))
        if depth_patch_cache_fn is not None:
            curr_obs_i.update(depth_patch_cache_fn(curr_obs_i))
            next_obs_i.update(depth_patch_cache_fn(next_obs_i))
        if base_act_encoder_cache_fn is not None:
            curr_obs_i.update(base_act_encoder_cache_fn(curr_obs_i))
            next_obs_i.update(base_act_encoder_cache_fn(next_obs_i))
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
            },
            batch_size=[],
        )
        if "gamma" in info:
            td.set("gamma", info["gamma"][i].detach().cpu())
        if "nonterminal" in info:
            td.set("nonterminal", info["nonterminal"][i].detach().cpu())
        if "chosen_horizon" in info:
            td.set("chosen_horizon", info["chosen_horizon"][i].detach().cpu())
        if "executed_horizon" in info:
            td.set("executed_horizon", info["executed_horizon"][i].detach().cpu())

        online_rb.add(td.unsqueeze(0) if add_batch_dim else td)


def _align_replay_batch_keys(lhs: TensorDict, rhs: TensorDict) -> None:
    """Make two sampled replay batches concatenable at every TensorDict level.

    Caches may contain frozen feature fields produced by an older encoder
    configuration (for example DA2 patches), while a new run intentionally
    disables those fields.  TensorDict concatenation is strict about keys, so
    remove fields that are absent from either side before mixing the batches.
    The operation is in-place and preserves all common transition fields.
    """

    common_keys = set(lhs.keys()).intersection(rhs.keys())
    for key in list(lhs.keys()):
        if key not in common_keys:
            lhs.pop(key)
    for key in list(rhs.keys()):
        if key not in common_keys:
            rhs.pop(key)

    for key in common_keys:
        left_value = lhs.get(key)
        right_value = rhs.get(key)
        if isinstance(left_value, TensorDict) and isinstance(right_value, TensorDict):
            _align_replay_batch_keys(left_value, right_value)


def _flatten_or_pad_action_chunk(
    action_chunk: torch.Tensor,
    *,
    chunk_horizon: int,
    primitive_action_dim: int,
) -> torch.Tensor:
    """Pad a variable-length primitive action chunk and flatten it into a single macro action vector."""
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action_chunk to have shape [T, A], got {tuple(action_chunk.shape)}")

    padded = torch.zeros(
        (chunk_horizon, primitive_action_dim),
        dtype=action_chunk.dtype,
        device=action_chunk.device,
    )
    valid_steps = min(action_chunk.shape[0], chunk_horizon)
    if valid_steps > 0:
        padded[:valid_steps] = action_chunk[:valid_steps]
    return padded.reshape(-1)


def _discounted_sum(rewards: list[torch.Tensor], gamma: float) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=rewards[0].device if rewards else "cpu")
    for step_idx, reward in enumerate(rewards):
        total = total + (gamma**step_idx) * reward.float()
    return total


def _primitive_step_count_from_info(info: dict, default_num_envs: int) -> int:
    primitive_steps = info.get("primitive_steps")
    if primitive_steps is None:
        return default_num_envs
    if isinstance(primitive_steps, torch.Tensor):
        return int(primitive_steps.sum().item())
    primitive_steps_arr = np.asarray(primitive_steps)
    return int(primitive_steps_arr.sum())


def _load_base_policies(
    cfg,
    *,
    task_name: str,
    state_dim: int,
    device: torch.device,
):
    provider = str(getattr(cfg.base_policy, "provider", "wandb_act")).lower()
    if provider == "wandb_act":
        from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy

        if cfg.base_policy.local_path:
            policy_dir = Path(cfg.base_policy.local_path).expanduser().resolve()
            if not (policy_dir / "config.json").exists():
                raise FileNotFoundError(f"Local base policy is missing config.json: {policy_dir}")
        else:
            policy_dir, _ = download_policy_from_wandb(
                cfg.base_policy.wandb_id,
                step=cfg.base_policy.wt_type,
                artifact_version=cfg.base_policy.wt_version,
            )

        base_policy = load_policy(policy_dir)
        base_policy.to(device)
        base_policy.eval()
        eval_base_policy = load_policy(policy_dir)
        eval_base_policy.to(device)
        eval_base_policy.eval()
        identity = {
            "provider": provider,
            "wandb_id": cfg.base_policy.wandb_id,
            "wt_type": cfg.base_policy.wt_type,
            "wt_version": cfg.base_policy.wt_version,
            "local_path": str(policy_dir) if cfg.base_policy.local_path else None,
        }
        return base_policy, eval_base_policy, identity

    if provider == "openpi":
        if cfg.base_policy.openpi_train_config is None:
            raise ValueError("base_policy.openpi_train_config is required when provider='openpi'.")
        if cfg.base_policy.openpi_checkpoint_dir is None:
            raise ValueError("base_policy.openpi_checkpoint_dir is required when provider='openpi'.")

        adapter_kwargs = dict(
            openpi_root=cfg.base_policy.openpi_root,
            train_config_name=cfg.base_policy.openpi_train_config,
            checkpoint_dir=cfg.base_policy.openpi_checkpoint_dir,
            state_dim=state_dim,
            task_name=task_name,
            token_pool_size=int(cfg.base_policy.openpi_token_pool_size),
            base_image_key=cfg.base_policy.openpi_base_image_key,
            left_wrist_image_key=cfg.base_policy.openpi_left_wrist_image_key,
            right_wrist_image_key=cfg.base_policy.openpi_right_wrist_image_key,
            default_prompt=cfg.base_policy.openpi_default_prompt,
        )
        base_policy = OpenPIBasePolicyAdapter(**adapter_kwargs)
        base_policy.eval()
        eval_base_policy = base_policy.clone_for_eval()
        eval_base_policy.eval()
        identity = {
            "provider": provider,
            "train_config": cfg.base_policy.openpi_train_config,
            "checkpoint_dir": str(cfg.base_policy.openpi_checkpoint_dir),
            "token_pool_size": int(cfg.base_policy.openpi_token_pool_size),
        }
        return base_policy, eval_base_policy, identity

    if provider == "openpi_remote":
        adapter_kwargs = dict(
            openpi_root=cfg.base_policy.openpi_root,
            host=cfg.base_policy.openpi_remote_host,
            port=int(cfg.base_policy.openpi_remote_port),
            state_dim=state_dim,
            token_pool_size=int(cfg.base_policy.openpi_token_pool_size),
            base_image_key=cfg.base_policy.openpi_base_image_key,
            left_wrist_image_key=cfg.base_policy.openpi_left_wrist_image_key,
            right_wrist_image_key=cfg.base_policy.openpi_right_wrist_image_key,
        )
        base_policy = OpenPIRemoteBasePolicyAdapter(**adapter_kwargs)
        base_policy.eval()
        eval_base_policy = base_policy.clone_for_eval()
        eval_base_policy.eval()
        identity = {
            "provider": provider,
            "host": cfg.base_policy.openpi_remote_host,
            "port": int(cfg.base_policy.openpi_remote_port),
            "token_pool_size": int(cfg.base_policy.openpi_token_pool_size),
        }
        return base_policy, eval_base_policy, identity

    if provider == "groot_remote":
        adapter_kwargs = dict(
            groot_root=cfg.base_policy.groot_root,
            host=cfg.base_policy.groot_remote_host,
            port=int(cfg.base_policy.groot_remote_port),
            state_dim=state_dim,
            base_image_key=cfg.base_policy.groot_base_image_key,
            wrist_image_key=cfg.base_policy.groot_wrist_image_key,
        )
        base_policy = GROOTRemoteBasePolicyAdapter(**adapter_kwargs)
        base_policy.eval()
        eval_base_policy = base_policy.clone_for_eval()
        eval_base_policy.eval()
        identity = {
            "provider": provider,
            "model_path": cfg.base_policy.groot_model_path,
            "host": cfg.base_policy.groot_remote_host,
            "port": int(cfg.base_policy.groot_remote_port),
            "token_target_count": int(cfg.base_policy.groot_token_target_count),
        }
        return base_policy, eval_base_policy, identity

    raise ValueError(f"Unsupported base_policy.provider={provider!r}")


def _append_base_encoder_cache_meta(
    cache_meta: dict,
    *,
    agent,
    base_policy,
    base_policy_identity: dict[str, object],
    state_standardizer: StateStandardizer,
) -> None:
    if not agent.uses_base_act_encoder_state:
        return

    cache_meta.update(
        {
            "base_act_encoder_cache_version": 1,
            "base_act_encoder_cache_dtype": "float16",
            "base_act_encoder_tokens": agent.base_act_encoder_num_tokens,
            "base_act_encoder_token_dim": int(agent.base_act_encoder_projector[0].in_features),
            "base_act_encoder_projected_dim": int(agent.base_act_encoder_projector[0].out_features),
            "base_act_encoder_image_keys": list(base_policy.config.image_features.keys()),
            "base_act_encoder_state_mean": state_standardizer.mean.detach().cpu().tolist(),
            "base_act_encoder_state_std": state_standardizer.std.detach().cpu().tolist(),
        }
    )
    provider = str(base_policy_identity.get("provider", "wandb_act"))
    if provider == "wandb_act":
        cache_meta.update(
            {
                "base_act_encoder_policy_wandb_id": base_policy_identity.get("wandb_id"),
                "base_act_encoder_policy_wt_type": base_policy_identity.get("wt_type"),
                "base_act_encoder_policy_wt_version": base_policy_identity.get("wt_version"),
            }
        )
    else:
        cache_meta.update(
            {
                "base_act_encoder_policy_provider": provider,
                "base_act_encoder_model_path": base_policy_identity.get("model_path"),
                "base_act_encoder_openpi_train_config": base_policy_identity.get("train_config"),
                "base_act_encoder_openpi_checkpoint_dir": base_policy_identity.get("checkpoint_dir"),
                "base_act_encoder_openpi_token_pool_size": base_policy_identity.get("token_pool_size"),
                "base_act_encoder_groot_token_target_count": base_policy_identity.get("token_target_count"),
                "base_act_encoder_openpi_remote_host": base_policy_identity.get("host"),
                "base_act_encoder_openpi_remote_port": base_policy_identity.get("port"),
            }
        )


# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------
def main(cfg: ResidualTD3DexmgConfig):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    dataset = LeRobotDataset(cfg.offline_data.name)

    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )

    # Create state standardizer from dataset statistics
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    state_dim = int(state_standardizer.mean.numel())

    # ---------------------------------------------------------------------
    # Load the behaviour-cloning policy that will serve as the "base" policy
    # for residual learning.
    # ---------------------------------------------------------------------
    assert "base_policy" in cfg, "Base policy configuration is required"
    base_policy, eval_base_policy, base_policy_identity = _load_base_policies(
        cfg,
        task_name=cfg.task,
        state_dim=state_dim,
        device=device,
    )

    # Extract the configuration from base policy
    base_cfg = base_policy.config

    if isinstance(base_cfg, ACTConfig) or str(getattr(cfg.base_policy, "provider", "wandb_act")).lower() in {
        "openpi",
        "openpi_remote",
        "groot_remote",
    }:
        cfg.actor_name = "residual_act"
    else:
        raise ValueError(f"Unknown base policy type: {type(base_cfg)}")

    macro_action_horizon = int(cfg.algo.macro_action_horizon)
    use_macro_actions = macro_action_horizon > 1
    fixed_proposal_horizon = int(cfg.algo.fixed_proposal_horizon)
    use_fixed_proposal_reactive = fixed_proposal_horizon > 1
    adaptive_horizons = tuple(sorted(int(h) for h in cfg.algo.adaptive_macro_horizons)) if cfg.algo.adaptive_macro_enabled else ()
    use_adaptive_macro_actions = use_macro_actions and bool(adaptive_horizons)
    adaptive_residual_mode = str(getattr(cfg.algo, "adaptive_residual_mode", "shared_prefix"))
    if adaptive_residual_mode not in {"shared_prefix", "horizon_conditioned"}:
        raise ValueError(
            "algo.adaptive_residual_mode must be 'shared_prefix' or 'horizon_conditioned', "
            f"got {adaptive_residual_mode!r}"
        )
    if adaptive_residual_mode == "horizon_conditioned" and not use_adaptive_macro_actions:
        raise ValueError("horizon_conditioned residuals require adaptive macro actions.")
    if use_fixed_proposal_reactive and use_macro_actions:
        raise ValueError("fixed_proposal_horizon requires algo.macro_action_horizon=1.")
    if use_fixed_proposal_reactive and cfg.algo.adaptive_macro_enabled:
        raise ValueError("fixed_proposal_horizon cannot be combined with adaptive macro actions.")
    if cfg.algo.adaptive_macro_enabled and not use_macro_actions:
        raise ValueError("adaptive_macro_enabled requires macro_action_horizon > 1.")
    if use_adaptive_macro_actions and adaptive_horizons[-1] != macro_action_horizon:
        raise ValueError(
            "adaptive_macro_horizons must end at macro_action_horizon. "
            f"Got adaptive_macro_horizons={adaptive_horizons}, macro_action_horizon={macro_action_horizon}."
        )

    replay_n_step = 1 if use_macro_actions else cfg.algo.n_step
    replay_gamma = cfg.algo.gamma**macro_action_horizon if (use_macro_actions and not use_adaptive_macro_actions) else cfg.algo.gamma
    if use_macro_actions:
        print(
            "Macro residual mode enabled: "
            f"horizon={macro_action_horizon}, replay_n_step={replay_n_step}, replay_gamma={replay_gamma:.6f}"
        )
    if use_adaptive_macro_actions:
        print(
            "Adaptive macro horizon choices enabled: "
            f"{adaptive_horizons} (residual_mode={adaptive_residual_mode})"
        )
    if use_fixed_proposal_reactive:
        print(
            "Per-step reactive fixed-proposal mode enabled: "
            f"proposal_horizon={fixed_proposal_horizon}, residual_horizon=1"
        )

    def get_envs(
        env_name: str,
        num_envs: int,
        base_policy: ACTPolicy,
        device: str,
        video_key: str,
        debug: bool,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
    ):
        assert action_scaler is not None, "action_scaler must be provided for consistent normalization"
        assert state_standardizer is not None, "state_standardizer must be provided for consistent normalization"

        # Create the vectorized environment
        vec_env = create_vectorized_env(
            env_name=env_name,
            num_envs=num_envs,
            device=device,
            video_key=video_key,
            debug=debug,
        )

        # Wrap it with the base policy wrapper
        return BasePolicyVecEnvWrapper(
            vec_env=vec_env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            macro_action_horizon=macro_action_horizon,
            macro_discount_gamma=cfg.algo.gamma,
            adaptive_horizons=adaptive_horizons if use_adaptive_macro_actions else (),
            fixed_proposal_horizon=fixed_proposal_horizon if use_fixed_proposal_reactive else 0,
        )

    # ---------------------------------------------------------------------
    # Seeding (must be done before environment creation) ------------------
    # ---------------------------------------------------------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # CUDA seeding for multi-GPU reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    print(f"Set random seed to {cfg.seed}")

    # ---------------------------------------------------------------------
    # Environment setup ----------------------------------------------------
    # ---------------------------------------------------------------------
    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    env = get_envs(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        base_policy=base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )
    cfg.eval_num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes)
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    cfg.eval_num_envs = min(num_cpus_available, cfg.eval_num_envs)
    if use_macro_actions or use_fixed_proposal_reactive:
        cfg.eval_num_envs = 1

    eval_env = get_envs(
        env_name=cfg.task,
        num_envs=cfg.eval_num_envs,
        base_policy=eval_base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )

    # Seed environments explicitly for reproducibility
    if hasattr(env, "seed"):
        env.seed(cfg.seed)
    if hasattr(eval_env, "seed"):
        eval_env.seed(cfg.seed + 1)  # Use different seed for eval env to avoid correlation

    # ---------------------------------------------------------------------
    # Observation / action dimensions -------------------------------------
    # ---------------------------------------------------------------------
    # Determine which image keys (camera observations) will be used. The
    # configuration can specify either a single camera name (str) or a list of
    # names.
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    base_action_dim = env.base_action_dim if hasattr(env, "base_action_dim") else action_dim

    lowdim_keys = ["observation.state", "observation.base_action"]

    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
        base_action_dim=base_action_dim,
        primitive_action_dim=env.primitive_action_dim if hasattr(env, "primitive_action_dim") else None,
        adaptive_horizons=adaptive_horizons if use_adaptive_macro_actions else None,
        adaptive_residual_mode=adaptive_residual_mode,
        adaptive_horizon_entropy_reg=cfg.algo.adaptive_macro_horizon_entropy_reg,
        task_name=cfg.task,
        action_scaler_min=action_scaler.limits.min.detach().cpu(),
        action_scaler_max=action_scaler.limits.max.detach().cpu(),
        state_mean=state_standardizer.mean.detach().cpu(),
        state_std=state_standardizer.std.detach().cpu(),
        base_policy=base_policy,
    )
    depth_cls_cache_fn = None
    offline_depth_cls_cache_fn = None
    depth_patch_cache_fn = None
    offline_depth_patch_cache_fn = None
    base_act_encoder_cache_fn = None
    offline_base_act_encoder_cache_fn = None
    if agent.uses_depth_anything_v2_conditioning:
        print("Depth CLS replay caching enabled.")
        depth_cls_cache_keys = tuple(agent.get_depth_cache_keys())

        def depth_cls_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if all(key in obs_dict for key in depth_cls_cache_keys):
                return {}
            return agent.compute_depth_cls_cache(obs_dict, cpu=False)

        def offline_depth_cls_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if all(key in obs_dict for key in depth_cls_cache_keys):
                return {}
            return agent.compute_depth_cls_cache(obs_dict, cpu=True)

    if agent.uses_depth_patch_state:
        print("Depth patch replay caching enabled.")
        depth_patch_cache_keys = tuple(agent.get_depth_patch_cache_keys())

        def depth_patch_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if all(key in obs_dict for key in depth_patch_cache_keys):
                return {}
            return agent.compute_depth_patch_cache(obs_dict, cpu=False)

        def offline_depth_patch_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if all(key in obs_dict for key in depth_patch_cache_keys):
                return {}
            return agent.compute_depth_patch_cache(obs_dict, cpu=True)

    if agent.uses_base_act_encoder_state:
        print("Base policy encoder replay caching enabled.")

        def base_act_encoder_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if agent.base_act_encoder_cache_key in obs_dict:
                return {}
            return agent.compute_base_act_encoder_cache(obs_dict, cpu=False)

        def offline_base_act_encoder_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if agent.base_act_encoder_cache_key in obs_dict:
                return {}
            return agent.compute_base_act_encoder_cache(obs_dict, cpu=True)

    joint_proposal_fn = None
    if agent.uses_base_act_encoder_state:
        joint_infer = getattr(base_policy, "infer_action_chunk_and_observation_features", None)
        if callable(joint_infer):

            def joint_proposal_fn(raw_obs: dict[str, torch.Tensor], n_action_steps: int):
                actions, tokens = joint_infer(raw_obs, n_action_steps=n_action_steps)
                context = {
                    agent.base_act_encoder_cache_key: tokens.detach().to(dtype=torch.float16)
                }
                return actions, context

    if use_fixed_proposal_reactive or macro_action_horizon > 1:
        env.set_proposal_context_fns(
            depth_context_fn=depth_patch_cache_fn or depth_cls_cache_fn,
            base_context_fn=base_act_encoder_cache_fn,
            joint_proposal_fn=joint_proposal_fn,
        )
        eval_env.set_proposal_context_fns(
            depth_context_fn=depth_patch_cache_fn or depth_cls_cache_fn,
            base_context_fn=base_act_encoder_cache_fn,
            joint_proposal_fn=joint_proposal_fn,
        )

    horizon = env.vec_env.metadata["horizon"]

    # Set up actor learning rate warmup
    actor_updates = 0
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup enabled: 0.0 -> {cfg.agent.actor_lr:.2e} "
            f"over {cfg.algo.actor_lr_warmup_steps} actor updates"
        )

    # ---------------------------------------------------------------------
    # Replay buffers -------------------------------------------------------
    # ---------------------------------------------------------------------
    # -----------------------------------------------------------------
    # Use TensorDictPrioritizedReplayBuffer for unified PER support
    # For uniform sampling, we'll use alpha=0 and beta=0, and never update priorities
    # -----------------------------------------------------------------
    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("Online-only training mode: offline_fraction=0.0")

    # Use TensorDictPrioritizedReplayBuffer with optimized prefetching
    online_rb_kwargs = dict(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=online_batch_size,
    )
    if not use_adaptive_macro_actions:
        online_rb_kwargs["transform"] = MultiStepTransform(n_steps=replay_n_step, gamma=replay_gamma)
    online_rb = TensorDictPrioritizedReplayBuffer(**online_rb_kwargs)

    # ------------------------------------------------------------------
    # Caching layer for online replay buffer ----------------------------
    # ------------------------------------------------------------------
    online_cache_meta = {
        "task": cfg.task,
        "seed": cfg.seed,
        "image_keys": image_keys,
        "n_step": replay_n_step,
        "gamma": replay_gamma,
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        "macro_action_horizon": macro_action_horizon,
        "fixed_proposal_horizon": fixed_proposal_horizon,
        "fixed_proposal_reactive": use_fixed_proposal_reactive,
        "adaptive_macro_horizons": adaptive_horizons,
        "adaptive_residual_mode": adaptive_residual_mode,
        "uses_sampled_gamma": use_adaptive_macro_actions,
        "adaptive_macro_buffer_format_version": 2 if use_adaptive_macro_actions else 1,
        # Include random action noise scale to prevent mixing data from different noise levels
        "random_action_noise_scale": cfg.algo.random_action_noise_scale,
        # Normalization parameters for consistency
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "normalized_actions": True,
        "agent_vit_depth": cfg.agent.vit.depth,
        "agent_use_residual_image_encoder": cfg.agent.use_residual_image_encoder,
        "agent_use_base_act_encoder_state": cfg.agent.use_base_act_encoder_state,
        "cached_base_act_encoder_tokens": agent.uses_base_act_encoder_state,
        "depth_anything_v2_conditioning_enabled": agent.uses_depth_anything_v2_conditioning,
        "cached_depth_cls": agent.uses_depth_anything_v2_conditioning,
        "depth_anything_v2_patch_state_enabled": agent.uses_depth_patch_state,
        "cached_depth_patch_tokens": agent.uses_depth_patch_state,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if base_policy_identity.get("provider") != "wandb_act":
        online_cache_meta["agent_use_base_policy_encoder_state"] = getattr(cfg.agent, "use_base_policy_encoder_state", False)
        online_cache_meta["base_policy_provider"] = base_policy_identity.get("provider")
        online_cache_meta["base_policy_model_path"] = base_policy_identity.get("model_path")
        online_cache_meta["base_policy_token_target_count"] = base_policy_identity.get("token_target_count")
        online_cache_meta["base_policy_openpi_remote_host"] = base_policy_identity.get("host")
        online_cache_meta["base_policy_openpi_remote_port"] = base_policy_identity.get("port")
    if agent.uses_depth_anything_v2_conditioning:
        online_cache_meta["depth_anything_v2_encoder"] = cfg.agent.depth_anything_v2_conditioning.encoder
        online_cache_meta["depth_anything_v2_num_conditioned_layers"] = (
            cfg.agent.depth_anything_v2_conditioning.num_conditioned_layers
        )
    if agent.uses_depth_patch_state:
        depth_patch_cfg = cfg.agent.depth_anything_v2_patch_state
        online_cache_meta.update(
            {
                "depth_patch_cache_version": 1,
                "depth_patch_cache_dtype": "float16",
                "depth_patch_encoder": depth_patch_cfg.encoder,
                "depth_patch_weights": depth_patch_cfg.weights,
                "depth_patch_num_intermediate_layers": depth_patch_cfg.num_intermediate_layers,
                "depth_patch_feature_layer": depth_patch_cfg.feature_layer,
                "depth_patch_resize_to": depth_patch_cfg.resize_to,
                "depth_patch_mean": list(depth_patch_cfg.mean),
                "depth_patch_std": list(depth_patch_cfg.std),
                "depth_patch_camera_keys": list(agent.depth_patch_camera_keys),
                "depth_patch_selection_mode": agent.depth_patch_selection_mode,
                "depth_patch_max_tokens_per_camera": agent.depth_patch_max_tokens_per_camera,
                "depth_patch_action_scaler_min": action_scaler.limits.min.detach().cpu().tolist(),
                "depth_patch_action_scaler_max": action_scaler.limits.max.detach().cpu().tolist(),
                "depth_patch_state_mean": state_standardizer.mean.detach().cpu().tolist(),
                "depth_patch_state_std": state_standardizer.std.detach().cpu().tolist(),
            }
        )
    _append_base_encoder_cache_meta(
        online_cache_meta,
        agent=agent,
        base_policy=base_policy,
        base_policy_identity=base_policy_identity,
        state_standardizer=state_standardizer,
    )
    if cfg.algo.sampling_strategy == "prioritized_replay":
        online_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        online_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(online_cache_meta)

    # Create the W&B run before any long cache population / warm-up work so
    # launched jobs are visible immediately and setup failures are attributable.
    _hp_parts: list[str] = [
        cfg.task,
        f"n{replay_n_step}",
        f"utd{cfg.algo.num_updates_per_iteration}",
        f"buf{cfg.algo.buffer_size}",
    ]

    if use_macro_actions:
        _hp_parts.append(f"macro{macro_action_horizon}")
    if use_fixed_proposal_reactive:
        _hp_parts.append(f"fixedproposal{fixed_proposal_horizon}-reactive1")
    if use_adaptive_macro_actions:
        _hp_parts.append("adaptive" + "-".join(str(h) for h in adaptive_horizons))
        _hp_parts.append(f"residual{adaptive_residual_mode}")

    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None and cfg.algo.offline_fraction > 0.0:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")
    elif cfg.algo.offline_fraction == 0.0:
        _hp_parts.append("online_only")

    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")

    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)

    wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )
    wandb.summary["environment/horizon"] = env.vec_env.metadata["horizon"]

    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha1(_online_meta_str.encode()).hexdigest()[:8]  # noqa: S324
    # Base local path for the online buffer ------------------------------
    online_cache_dir = ONLINE_CACHE_DIR / online_cache_hash

    # Attempt to download/extract from HF every run (no-op if already cached)
    dl_dir = None
    if ONLINE_HF_REPO is not None:
        print(f"Attempting to download online buffer {online_cache_hash} from {ONLINE_HF_REPO}...")
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, ONLINE_CACHE_DIR)
    if dl_dir is not None:
        online_cache_dir = dl_dir

    loaded_online_from_cache = False
    if online_cache_dir.exists():
        print(f"{online_cache_dir} found on disk. Attempting to load...")
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, online_cache_dir)
        loaded_online_from_cache = True
        print(f"Loaded online buffer from cache at {online_cache_dir} (size={len(online_rb)})")

    # Offline data is required for normalization, but can be unused for training if offline_fraction=0
    assert cfg.offline_data is not None and cfg.offline_data.num_episodes is not None

    # Dataset and normalization already loaded above - use existing dataset

    # Use actual dataset metadata for precise buffer sizing
    if cfg.offline_data.num_episodes is not None:
        # Only use subset of episodes if specified
        selected_episode_lengths = [
            dataset.meta.episodes[ep_idx]["length"]
            for ep_idx in range(min(cfg.offline_data.num_episodes, dataset.meta.total_episodes))
        ]
        total_frames = sum(selected_episode_lengths)
        num_episodes = cfg.offline_data.num_episodes
    else:
        # Use entire dataset
        total_frames = dataset.meta.total_frames
        num_episodes = dataset.meta.total_episodes
        selected_episode_lengths = [dataset.meta.episodes[ep_idx]["length"] for ep_idx in range(dataset.meta.total_episodes)]

    # Calculate transitions: single-step uses frame pairs, macro mode aggregates them in stride-H windows.
    if use_adaptive_macro_actions:
        min_horizon = min(adaptive_horizons)
        offline_stride = max(1, int(cfg.algo.adaptive_macro_offline_stride))
        estimated_transitions = 0
        for episode_length in selected_episode_lengths:
            primitive_transitions = max(0, episode_length - 1)
            if primitive_transitions < min_horizon:
                continue
            estimated_starts = max(0, ((primitive_transitions - min_horizon) // offline_stride) + 1)
            estimated_transitions += estimated_starts * len(adaptive_horizons)
    elif use_macro_actions:
        estimated_transitions = sum(
            max(0, (episode_length - 1 + macro_action_horizon - 1) // macro_action_horizon)
            for episode_length in selected_episode_lengths
        )
    else:
        estimated_transitions = max(0, total_frames - num_episodes)

    print("Dataset buffer sizing:")
    print(f"  Total frames to process: {total_frames}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Estimated transitions: {estimated_transitions}")

    # Calculate buffer size for simplified approach (1 transition per frame pair)
    max_offline_transitions = (
        estimated_transitions if cfg.algo.offline_fraction > 0.0 else 1
    )  # Minimum size for online-only mode
    if cfg.algo.offline_fraction > 0.0:
        print(f"Offline buffer sized for GT-as-base approach: {max_offline_transitions} transitions")
    else:
        print("Online-only mode: creating minimal offline buffer (unused)")

    offline_rb_kwargs = dict(
        storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=max(offline_batch_size, 1),
    )
    if not use_adaptive_macro_actions:
        offline_rb_kwargs["transform"] = MultiStepTransform(n_steps=replay_n_step, gamma=replay_gamma)
    offline_rb = TensorDictPrioritizedReplayBuffer(**offline_rb_kwargs)

    # Normalization functions already defined above - use them

    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------
    def _populate_offline_buffer(
        dataset: LeRobotDataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        loader_batch_size: int = 1,
        loader_num_workers: int = 0,
        use_base_policy_for_base_actions: bool = False,
        base_policy: ACTPolicy | None = None,
        macro_action_horizon: int = 1,
        fixed_proposal_horizon: int = 0,
        macro_discount_gamma: float = 0.99,
        adaptive_horizons: tuple[int, ...] = (),
        adaptive_macro_offline_stride: int = 1,
        depth_cls_cache_fn=None,
        depth_patch_cache_fn=None,
        base_act_encoder_cache_fn=None,
    ) -> int:
        """
        Iterates through *dataset* sequentially, converts consecutive frames
        into residual RL transitions and pushes them into *rb*.

        Two modes:
        1. GT-as-base (use_base_policy_for_base_actions=False):
           Uses GT actions as both the base action (in observations) and the target action
           (in transitions). Teaches residual policy to output zero: residual = GT - GT = 0

        2. Base-policy-as-base (use_base_policy_for_base_actions=True):
           Uses base policy to generate base actions and GT actions as targets.
           More consistent with online training: residual = GT - base_policy_action

        Returns the number of transitions added.
        """
        if use_base_policy_for_base_actions and base_policy is None:
            raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")
        if macro_action_horizon > 1 and not use_base_policy_for_base_actions:
            raise ValueError(
                "Macro offline buffer construction currently requires "
                "offline_data.use_base_policy_for_base_actions=True."
            )
        use_fixed_proposal = fixed_proposal_horizon > 1
        if use_fixed_proposal and macro_action_horizon != 1:
            raise ValueError("Fixed-proposal offline construction requires macro_action_horizon=1.")
        if use_fixed_proposal and not use_base_policy_for_base_actions:
            raise ValueError(
                "Fixed-proposal offline construction requires offline_data.use_base_policy_for_base_actions=True."
            )
        if use_fixed_proposal and int(loader_batch_size) != 1:
            raise ValueError("Fixed-proposal offline construction currently requires loader_batch_size=1.")

        # Populate buffer from pre-loaded dataset
        print("Populating offline buffer from dataset...")
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(loader_batch_size)),
            shuffle=False,
            num_workers=max(0, int(loader_num_workers)),
            pin_memory=torch.cuda.is_available(),
        )

        episode_cache: dict[int, dict] = {}
        episode_macro_cache: dict[int, list[TensorDict]] = {}
        fixed_proposal_cache: dict[int, dict] = {}
        transitions = 0
        use_adaptive_macro = macro_action_horizon > 1 and len(adaptive_horizons) > 0
        offline_stride = max(1, int(adaptive_macro_offline_stride))
        min_adaptive_horizon = min(adaptive_horizons) if use_adaptive_macro else 1
        joint_base_policy_infer = None
        if (
            use_base_policy_for_base_actions
            and base_policy is not None
            and base_act_encoder_cache_fn is not None
        ):
            joint_base_policy_infer = getattr(base_policy, "infer_action_chunk_and_observation_features", None)
            if not callable(joint_base_policy_infer):
                joint_base_policy_infer = None

        def _flush_macro_episode(ep_idx: int) -> None:
            nonlocal transitions
            primitive_transitions = episode_macro_cache.pop(ep_idx, [])
            if use_adaptive_macro:
                if len(primitive_transitions) < min_adaptive_horizon:
                    return
                for start_idx in range(0, len(primitive_transitions) - min_adaptive_horizon + 1, offline_stride):
                    for horizon_idx, horizon in enumerate(adaptive_horizons):
                        window = primitive_transitions[start_idx : start_idx + horizon]
                        if len(window) < horizon:
                            continue

                        action_chunk = torch.stack([td["action"] for td in window], dim=0)
                        macro_action = _flatten_or_pad_action_chunk(
                            action_chunk,
                            chunk_horizon=macro_action_horizon,
                            primitive_action_dim=action_scaler.limits.min.numel(),
                        )
                        horizon_onehot = torch.zeros(len(adaptive_horizons), dtype=macro_action.dtype)
                        horizon_onehot[horizon_idx] = 1.0
                        macro_action = torch.cat([macro_action, horizon_onehot], dim=0)
                        macro_reward = _discounted_sum([td["next"]["reward"] for td in window], macro_discount_gamma)
                        done_tensor = window[-1]["next"]["done"].clone()

                        macro_transition = TensorDict(
                            {
                                "obs": window[0]["obs"].clone(),
                                "action": macro_action,
                                "next": TensorDict(
                                    {
                                        "obs": window[-1]["next"]["obs"].clone(),
                                        "done": done_tensor,
                                        "reward": macro_reward,
                                    },
                                    batch_size=[],
                                ),
                                "gamma": torch.tensor(macro_discount_gamma**horizon, dtype=torch.float32),
                                "nonterminal": (~done_tensor.bool()).clone(),
                                "chosen_horizon": torch.tensor(horizon, dtype=torch.long),
                                "executed_horizon": torch.tensor(horizon, dtype=torch.long),
                                "_priority": torch.tensor(10.0, dtype=torch.float32),
                            },
                            batch_size=[],
                        )

                        rb.add(macro_transition)
                        transitions += 1
                return

            for start_idx in range(0, len(primitive_transitions), macro_action_horizon):
                window = primitive_transitions[start_idx : start_idx + macro_action_horizon]
                if not window:
                    continue

                action_chunk = torch.stack([td["action"] for td in window], dim=0)
                macro_action = _flatten_or_pad_action_chunk(
                    action_chunk,
                    chunk_horizon=macro_action_horizon,
                    primitive_action_dim=action_scaler.limits.min.numel(),
                )
                macro_reward = _discounted_sum([td["next"]["reward"] for td in window], macro_discount_gamma)

                macro_transition = TensorDict(
                    {
                        "obs": window[0]["obs"].clone(),
                        "action": macro_action,
                        "next": TensorDict(
                            {
                                "obs": window[-1]["next"]["obs"].clone(),
                                "done": window[-1]["next"]["done"].clone(),
                                "reward": macro_reward,
                            },
                            batch_size=[],
                        ),
                        "_priority": torch.tensor(10.0, dtype=torch.float32),
                    },
                    batch_size=[],
                ).unsqueeze(0)

                rb.add(macro_transition)
                transitions += 1

        for sample in tqdm(loader, desc="Processing offline dataset"):
            batch_episode_index = sample["episode_index"]
            original_batch_size = int(batch_episode_index.shape[0])
            valid_batch_size = original_batch_size
            if num_episodes is not None:
                valid_batch_size = 0
                for ep_idx in batch_episode_index.tolist():
                    if int(ep_idx) >= num_episodes:
                        break
                    valid_batch_size += 1
                if valid_batch_size == 0:
                    break

            sample = {
                key: value[:valid_batch_size] if isinstance(value, torch.Tensor) and value.dim() > 0 else value
                for key, value in sample.items()
            }

            _gt_action_batch: torch.Tensor = sample["action"].float()
            gt_action_scaled_batch = action_scaler.scale(_gt_action_batch)
            done_flags = sample["next.done"].bool().tolist()

            cached_base_encoder_tokens_batch = None
            fixed_context_batch: dict[str, torch.Tensor] | None = None
            if use_base_policy_for_base_actions:
                raw_obs = {}
                for k in sample:
                    if "observation" in k:
                        raw_obs[k] = sample[k].to(device)

                with torch.no_grad():
                    if use_fixed_proposal:
                        if valid_batch_size != 1:
                            raise RuntimeError("Fixed-proposal batches must contain exactly one sequential frame.")
                        ep_idx = int(sample["episode_index"][0].item())
                        proposal_state = fixed_proposal_cache.get(ep_idx)
                        if proposal_state is None or int(proposal_state["index"]) >= fixed_proposal_horizon:
                            if proposal_state is None:
                                base_policy.reset()

                            if joint_base_policy_infer is not None:
                                base_action_chunk, base_encoder_tokens = joint_base_policy_infer(
                                    raw_obs,
                                    n_action_steps=fixed_proposal_horizon,
                                )
                                cached_base_encoder_tokens_batch = (
                                    base_encoder_tokens.detach().to(dtype=torch.float16).cpu()
                                )
                            else:
                                base_action_chunk = base_policy.select_action_chunk(
                                    raw_obs,
                                    n_action_steps=fixed_proposal_horizon,
                                )

                            scaled_plan = action_scaler.scale(base_action_chunk).cpu()
                            proposal_obs = {
                                "observation.state": state_standardizer.standardize(
                                    sample["observation.state"].float()
                                ),
                                "observation.base_action": scaled_plan.reshape(1, -1),
                            }
                            for image_key in image_keys:
                                proposal_obs[image_key] = sample[image_key]

                            proposal_context: dict[str, torch.Tensor] = {}
                            if depth_cls_cache_fn is not None:
                                proposal_context.update(depth_cls_cache_fn(proposal_obs))
                            if depth_patch_cache_fn is not None:
                                proposal_context.update(depth_patch_cache_fn(proposal_obs))
                            if base_act_encoder_cache_fn is not None:
                                if cached_base_encoder_tokens_batch is not None:
                                    proposal_context[agent.base_act_encoder_cache_key] = cached_base_encoder_tokens_batch
                                else:
                                    proposal_context.update(base_act_encoder_cache_fn(proposal_obs))

                            proposal_state = {
                                "plan": scaled_plan,
                                "index": 0,
                                "context": proposal_context,
                            }
                            fixed_proposal_cache[ep_idx] = proposal_state

                        proposal_index = int(proposal_state["index"])
                        base_action_scaled_batch = proposal_state["plan"][:, proposal_index]
                        fixed_context_batch = proposal_state["context"]
                        proposal_state["index"] = proposal_index + 1
                    else:
                        requested_steps = macro_action_horizon if macro_action_horizon > 1 else 1
                        if joint_base_policy_infer is not None:
                            base_action_chunk, base_encoder_tokens = joint_base_policy_infer(
                                raw_obs,
                                n_action_steps=requested_steps,
                            )
                            if requested_steps > 1:
                                base_action_scaled_batch = action_scaler.scale(base_action_chunk).cpu().reshape(
                                    valid_batch_size,
                                    -1,
                                )
                            else:
                                base_action_scaled_batch = action_scaler.scale(base_action_chunk[:, 0, :]).cpu()
                            cached_base_encoder_tokens_batch = base_encoder_tokens.detach().to(dtype=torch.float16).cpu()
                        elif macro_action_horizon > 1:
                            base_action = base_policy.select_action_chunk(
                                raw_obs,
                                n_action_steps=macro_action_horizon,
                            )
                            base_action_scaled_batch = action_scaler.scale(base_action).cpu().reshape(valid_batch_size, -1)
                        else:
                            base_action = base_policy.select_action(raw_obs)
                            base_action_scaled_batch = action_scaler.scale(base_action).cpu()
            else:
                base_action_scaled_batch = gt_action_scaled_batch

            curr_obs_batch = {
                "observation.state": state_standardizer.standardize(sample["observation.state"].float()),
                "observation.base_action": base_action_scaled_batch,
            }
            for k in image_keys:
                curr_obs_batch[k] = sample[k]
            if fixed_context_batch is not None:
                curr_obs_batch.update(fixed_context_batch)
            else:
                if depth_cls_cache_fn is not None:
                    curr_obs_batch.update(depth_cls_cache_fn(curr_obs_batch))
                if depth_patch_cache_fn is not None:
                    curr_obs_batch.update(depth_patch_cache_fn(curr_obs_batch))
                if base_act_encoder_cache_fn is not None:
                    if cached_base_encoder_tokens_batch is not None:
                        curr_obs_batch[agent.base_act_encoder_cache_key] = cached_base_encoder_tokens_batch
                    else:
                        curr_obs_batch.update(base_act_encoder_cache_fn(curr_obs_batch))

            for batch_idx in range(valid_batch_size):
                ep_idx = int(sample["episode_index"][batch_idx].item())
                done_flag = bool(done_flags[batch_idx])
                gt_action_scaled = gt_action_scaled_batch[batch_idx]

                curr_obs = {}
                for key, value in curr_obs_batch.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == valid_batch_size:
                        curr_obs[key] = value[batch_idx]
                    else:
                        curr_obs[key] = value

                to_uint8(curr_obs, image_keys)

                if ep_idx in episode_cache:
                    prev_obs = episode_cache[ep_idx]["obs"]
                    prev_action_scaled = episode_cache[ep_idx]["action"]
                    transition = TensorDict(
                        {
                            "obs": TensorDict(prev_obs, batch_size=[]),
                            "action": prev_action_scaled,
                            "next": TensorDict(
                                {
                                    "obs": TensorDict(curr_obs, batch_size=[]),
                                    "done": torch.tensor(done_flag, dtype=torch.bool),
                                    "reward": torch.tensor(float(done_flag), dtype=torch.float32),
                                },
                                batch_size=[],
                            ),
                            "_priority": torch.tensor(10.0, dtype=torch.float32),
                        },
                        batch_size=[],
                    ).unsqueeze(0)

                    if macro_action_horizon > 1:
                        episode_macro_cache.setdefault(ep_idx, []).append(transition.squeeze(0).clone())
                        if done_flag:
                            _flush_macro_episode(ep_idx)
                    else:
                        rb.add(transition)
                        transitions += 1

                episode_cache[ep_idx] = {
                    "obs": curr_obs,
                    "action": gt_action_scaled,
                }

                if done_flag:
                    episode_cache.pop(ep_idx, None)
                    fixed_proposal_cache.pop(ep_idx, None)

            if num_episodes is not None and valid_batch_size < original_batch_size:
                break

        if macro_action_horizon > 1:
            for ep_idx in sorted(episode_macro_cache):
                _flush_macro_episode(ep_idx)

        # Log final statistics
        print(f"Added {transitions} transitions")

        return transitions

    # ------------------------------------------------------------------
    # Caching layer for offline replay buffer ---------------------------
    # ------------------------------------------------------------------
    # Build a metadata dictionary that uniquely identifies the buffer
    offline_cache_meta = {
        "task": cfg.task,
        "dataset_name": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "image_keys": image_keys,
        "n_step": replay_n_step,
        "gamma": replay_gamma,
        "macro_action_horizon": macro_action_horizon,
        "fixed_proposal_horizon": fixed_proposal_horizon,
        "fixed_proposal_reactive": use_fixed_proposal_reactive,
        "adaptive_macro_horizons": adaptive_horizons,
        "uses_sampled_gamma": use_adaptive_macro_actions,
        "adaptive_macro_buffer_format_version": 2 if use_adaptive_macro_actions else 1,
        "base_policy_provider": base_policy_identity.get("provider"),
        "sampling_strategy": cfg.algo.sampling_strategy,
        "normalized_actions": True,
        "batch_size": offline_batch_size,
        "adaptive_macro_offline_stride": cfg.algo.adaptive_macro_offline_stride,
        "agent_vit_depth": cfg.agent.vit.depth,
        "agent_use_residual_image_encoder": cfg.agent.use_residual_image_encoder,
        "agent_use_base_act_encoder_state": cfg.agent.use_base_act_encoder_state,
        "cached_base_act_encoder_tokens": agent.uses_base_act_encoder_state,
        "depth_anything_v2_conditioning_enabled": agent.uses_depth_anything_v2_conditioning,
        "cached_depth_cls": agent.uses_depth_anything_v2_conditioning,
        "depth_anything_v2_patch_state_enabled": agent.uses_depth_patch_state,
        "cached_depth_patch_tokens": agent.uses_depth_patch_state,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if base_policy_identity.get("provider") != "wandb_act":
        offline_cache_meta["agent_use_base_policy_encoder_state"] = getattr(
            cfg.agent,
            "use_base_policy_encoder_state",
            False,
        )
    if agent.uses_depth_anything_v2_conditioning:
        offline_cache_meta["depth_anything_v2_encoder"] = cfg.agent.depth_anything_v2_conditioning.encoder
        offline_cache_meta["depth_anything_v2_num_conditioned_layers"] = (
            cfg.agent.depth_anything_v2_conditioning.num_conditioned_layers
        )
    if agent.uses_depth_patch_state:
        depth_patch_cfg = cfg.agent.depth_anything_v2_patch_state
        offline_cache_meta.update(
            {
                "depth_patch_cache_version": 1,
                "depth_patch_cache_dtype": "float16",
                "depth_patch_encoder": depth_patch_cfg.encoder,
                "depth_patch_weights": depth_patch_cfg.weights,
                "depth_patch_num_intermediate_layers": depth_patch_cfg.num_intermediate_layers,
                "depth_patch_feature_layer": depth_patch_cfg.feature_layer,
                "depth_patch_resize_to": depth_patch_cfg.resize_to,
                "depth_patch_mean": list(depth_patch_cfg.mean),
                "depth_patch_std": list(depth_patch_cfg.std),
                "depth_patch_camera_keys": list(agent.depth_patch_camera_keys),
                "depth_patch_selection_mode": agent.depth_patch_selection_mode,
                "depth_patch_max_tokens_per_camera": agent.depth_patch_max_tokens_per_camera,
                "depth_patch_action_scaler_min": action_scaler.limits.min.detach().cpu().tolist(),
                "depth_patch_action_scaler_max": action_scaler.limits.max.detach().cpu().tolist(),
                "depth_patch_state_mean": state_standardizer.mean.detach().cpu().tolist(),
                "depth_patch_state_std": state_standardizer.std.detach().cpu().tolist(),
            }
        )
    if base_policy_identity.get("provider") == "wandb_act":
        offline_cache_meta["base_policy_wandb_id"] = base_policy_identity.get("wandb_id")
    else:
        offline_cache_meta["base_policy_model_path"] = base_policy_identity.get("model_path")
        offline_cache_meta["base_policy_openpi_train_config"] = base_policy_identity.get("train_config")
        offline_cache_meta["base_policy_openpi_checkpoint_dir"] = base_policy_identity.get("checkpoint_dir")
        offline_cache_meta["base_policy_openpi_token_pool_size"] = base_policy_identity.get("token_pool_size")
        offline_cache_meta["base_policy_groot_token_target_count"] = base_policy_identity.get("token_target_count")
        offline_cache_meta["base_policy_openpi_remote_host"] = base_policy_identity.get("host")
        offline_cache_meta["base_policy_openpi_remote_port"] = base_policy_identity.get("port")
    _append_base_encoder_cache_meta(
        offline_cache_meta,
        agent=agent,
        base_policy=base_policy,
        base_policy_identity=base_policy_identity,
        state_standardizer=state_standardizer,
    )
    if cfg.algo.sampling_strategy == "prioritized_replay":
        offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(offline_cache_meta)

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(offline_cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer.  ``OFFLINE_CACHE_PATH`` is useful when
    # two residual architectures intentionally share the same base-trajectory
    # dataset (for example SP vs HC); it avoids regenerating expensive remote
    # base-policy actions just because the actor parameterization changed.
    cache_dir = OFFLINE_CACHE_DIR / cache_hash
    offline_cache_override = os.environ.get("OFFLINE_CACHE_PATH")
    if offline_cache_override:
        cache_dir = Path(offline_cache_override).expanduser().resolve()
        print(f"Using explicit offline cache override: {cache_dir}")

    # Try to download/extract from the Hub (will no-op if file not there)
    downloaded_dir = None
    if OFFLINE_HF_REPO is not None:
        print(f"Attempting to download offline buffer {cache_hash} from {OFFLINE_HF_REPO}...")
        downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
    if downloaded_dir is not None:
        cache_dir = downloaded_dir  # use extracted location

    loaded_from_cache = False
    added = 0

    if cfg.algo.offline_fraction > 0.0:
        # Only populate offline buffer if we're using offline data
        if cache_dir.exists():
            print(f"{cache_dir} found on disk. Attempting to load...")
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(f"Loaded offline buffer from cache at {cache_dir} (size={len(offline_rb)})")

        if not loaded_from_cache:
            added = _populate_offline_buffer(
                dataset=dataset,
                rb=offline_rb,
                image_keys=image_keys,
                num_episodes=cfg.offline_data.num_episodes,
                loader_batch_size=cfg.offline_data.cache_loader_batch_size,
                loader_num_workers=cfg.offline_data.cache_loader_num_workers,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
                macro_action_horizon=macro_action_horizon,
                fixed_proposal_horizon=fixed_proposal_horizon if use_fixed_proposal_reactive else 0,
                macro_discount_gamma=cfg.algo.gamma,
                adaptive_horizons=adaptive_horizons if use_adaptive_macro_actions else (),
                adaptive_macro_offline_stride=cfg.algo.adaptive_macro_offline_stride,
                depth_cls_cache_fn=offline_depth_cls_cache_fn,
                depth_patch_cache_fn=offline_depth_patch_cache_fn,
                base_act_encoder_cache_fn=offline_base_act_encoder_cache_fn,
            )

            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")

            # Save buffer to disk for future runs + upload to Hub ----------------
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)

            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)

            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
        else:
            added = len(offline_rb)
    else:
        print("Skipping offline buffer population for online-only training")

    # ------------------------------------------------------------------
    # Warm-up phase (random policy) --------------------------------------
    # ------------------------------------------------------------------

    warmup_primitive_steps = 0
    if warmup_primitive_steps < cfg.algo.learning_starts and not loaded_online_from_cache:
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - warmup_primitive_steps} random steps…")
        warmup_start_time = time.perf_counter()
        warmup_decisions = 0
        warmup_horizon_counts = {horizon: 0 for horizon in adaptive_horizons}
        obs, _ = env.reset()
        # --------------------------------------------------------------
        # Logging helper: print progress every 1 000 collected transitions
        # --------------------------------------------------------------
        next_log_threshold = 1000  # first threshold for progress message

        reward_sum = 0.0
        episode_count = 0.0
        success_count = 0.0

        while warmup_primitive_steps < cfg.algo.learning_starts:
            if use_adaptive_macro_actions:
                horizon_idx = torch.randint(len(adaptive_horizons), (cfg.num_envs,), device=device)
                horizon_onehot = torch.nn.functional.one_hot(horizon_idx, num_classes=len(adaptive_horizons)).float()
                for sampled_idx in horizon_idx.detach().cpu().tolist():
                    warmup_horizon_counts[adaptive_horizons[int(sampled_idx)]] += 1
            warmup_decisions += cfg.num_envs
            if cfg.algo.use_base_policy_for_warmup:
                # Use base policy action + noise (residual exploration)
                # Since the environment wrapper always adds base_action to residual_action,
                # we just need to provide the noise as the residual action
                rand_residual = (
                    torch.rand((cfg.num_envs, base_action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
                rand_actions = (
                    torch.cat([rand_residual, horizon_onehot], dim=-1) if use_adaptive_macro_actions else rand_residual
                )
            else:
                # Pure uniform random actions - need to cancel out the base policy action
                # Since env does: combined = base_action + residual_action
                # To get pure random: residual_action = random - base_action
                base_action = obs["observation.base_action"]  # Already normalized to [-1, 1]
                pure_random = (
                    torch.rand((cfg.num_envs, base_action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
                rand_residual = pure_random - base_action
                rand_actions = (
                    torch.cat([rand_residual, horizon_onehot], dim=-1) if use_adaptive_macro_actions else rand_residual
                )

            next_obs, reward, terminated, truncated, info = env.step(rand_actions)
            done = terminated | truncated
            step_increment = _primitive_step_count_from_info(info, cfg.num_envs)

            reward_sum += reward.sum().item()
            episode_count += done.float().sum().item()
            if "macro_success" in info:
                success_count += info["macro_success"].float().sum().item()
            else:
                success_count += reward.sum().item()

            # Use the executed combined action returned by the environment
            combined_action = info["scaled_action"]
            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
                add_batch_dim=not use_adaptive_macro_actions,
                depth_cls_cache_fn=depth_cls_cache_fn,
                depth_patch_cache_fn=depth_patch_cache_fn,
                base_act_encoder_cache_fn=base_act_encoder_cache_fn,
            )
            warmup_primitive_steps += step_increment

            # ----------------------------------------------------------
            # Progress logging (every ~1 000 transitions) --------------
            # ----------------------------------------------------------
            if warmup_primitive_steps >= next_log_threshold:
                success_rate = success_count / episode_count if episode_count > 0 else 0.0
                print(
                    f"[Warm-up] {warmup_primitive_steps} / {cfg.algo.learning_starts} "
                    f"primitive steps collected, reward_sum={reward_sum:.2f}, "
                    f"success_rate={success_rate:.3f} ({success_count}/{episode_count})"
                )
                next_log_threshold += 1000

            obs = next_obs  # roll state

        warmup_wall_clock_seconds = time.perf_counter() - warmup_start_time
        print(
            "Warm-up summary: "
            f"primitive_steps={warmup_primitive_steps}, decisions={warmup_decisions}, "
            f"wall_clock_seconds={warmup_wall_clock_seconds:.3f}, "
            f"candidate_counts={warmup_horizon_counts}"
        )
        wandb.summary["warmup/primitive_steps"] = warmup_primitive_steps
        wandb.summary["warmup/decision_count"] = warmup_decisions
        wandb.summary["warmup/wall_clock_seconds"] = warmup_wall_clock_seconds
        for horizon, count in warmup_horizon_counts.items():
            wandb.summary[f"warmup/candidate_count_{horizon}"] = count

        # Persist freshly-collected buffer (local + HF) --------------------
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        optimized_replay_buffer_dumps(online_rb, online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        loaded_online_from_cache = True  # treat as cached going forward

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"

    # Create subdirectories for models and outputs
    model_save_dir = run_cache_dir / "models"
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()

    global_step = 0
    update_budget = 0.0
    gradient_updates = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0
    metrics: dict = {}

    train_start_time = time.time()

    # Initialize timing utility
    training_timer = TrainingTimer()

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""
        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            with training_timer.time("batch_sampling"):
                # Sample batches from replay buffers
                online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(device, non_blocking=True)

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training
                    offline_batch = offline_rb.sample(offline_batch_size)
                    offline_batch = offline_batch.to(device, non_blocking=True)
                    _align_replay_batch_keys(online_batch, offline_batch)
                    batch = torch.cat([online_batch, offline_batch], dim=0)
                else:
                    # Online-only training
                    batch = online_batch

            # Only update critic during warmup (update_actor=False)
            with training_timer.time("gradient_update"):
                metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)

            # Update priorities for prioritized experience replay
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                # Update priorities in the batch for priority updates
                td_errors = metrics["_td_errors"]
                batch["_priority"] = td_errors

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training - update both buffers
                    online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                    # Update online buffer priorities
                    if online_batch_size_actual > 0:
                        online_batch_subset = batch[:online_batch_size_actual]
                        online_rb.update_tensordict_priority(online_batch_subset)

                    # Update offline buffer priorities
                    if online_batch_size_actual < len(batch):
                        offline_batch_subset = batch[online_batch_size_actual:]
                        offline_rb.update_tensordict_priority(offline_batch_subset)
                else:
                    # Online-only training - update only online buffer
                    online_rb.update_tensordict_priority(batch)

            # Progress logging
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )

    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    if cfg.algo.critic_warmup_steps > 0:
        print(f"Critic warmup: running {cfg.algo.critic_warmup_steps} critic-only updates...")
        _run_critic_warmup(
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            cfg=cfg,
            device=device,
            training_timer=training_timer,
            online_batch_size=online_batch_size,
            offline_batch_size=offline_batch_size,
        )
        print("Critic warmup completed.")

    def _run_evaluation(step_value: int, num_episodes: int | None = None) -> None:
        nonlocal best_eval_success_rate
        evaluation_episodes = cfg.eval_num_episodes if num_episodes is None else int(num_episodes)
        with training_timer.time("evaluation"):
            eval_metrics = run_dexmg_evaluation(
                env=eval_env,
                agent=agent,
                num_episodes=evaluation_episodes,
                device=device,
                global_step=step_value,
                save_video=cfg.save_video,
                save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                run_name=run_name,
                output_dir=outputs_dir,
            )

        current_success_rate = eval_metrics["eval/success_rate"]
        if current_success_rate > best_eval_success_rate:
            print(f"🎉 New best success rate: {current_success_rate:.4f} (prev: {best_eval_success_rate:.4f})")
            best_eval_success_rate = current_success_rate

    if cfg.eval_first:
        _run_evaluation(0)

    next_eval_step = cfg.eval_interval_every_steps
    next_log_step = cfg.log_freq

    while global_step < cfg.algo.total_timesteps:
        iter_start = time.time()
        # ------------------------------------------------------------------
        # (1) Collect action + Environment step ---------------------------
        # ------------------------------------------------------------------
        with training_timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

            if cfg.algo.progressive_clipping_steps > 0:
                clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                action = action * clip_factor

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            step_increment = _primitive_step_count_from_info(info, cfg.num_envs)
        if done.any():
            episode_count += done.float().sum().item()
            # Extract episode information from final_info
            final_info = info["final_info"]
            episode_steps = final_info["episode_steps"]
            episode_indices = final_info["_episode_steps"]

            if "undiscounted_reward" in info:
                episode_rewards = info["undiscounted_reward"].cpu().numpy()[episode_indices]
                episode_return = float(np.mean(episode_rewards))
            else:
                # Calculate discounted episode return
                discount_factor = cfg.algo.gamma ** episode_steps[episode_indices]
                episode_rewards = reward.cpu().numpy()[episode_indices]
                episode_return = float(np.mean(discount_factor * episode_rewards))

            wandb.log(
                {
                    "training/episode_return": episode_return,
                    "training/episode_steps": episode_steps,
                    "training/episode_count": episode_count,
                },
                step=global_step,
            )

        # Add to online replay buffer --------------------------------------
        # Use the executed combined action returned by the environment
        combined_action = info["scaled_action"]
        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=combined_action,
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
            add_batch_dim=not use_adaptive_macro_actions,
            depth_cls_cache_fn=depth_cls_cache_fn,
            depth_patch_cache_fn=depth_patch_cache_fn,
            base_act_encoder_cache_fn=base_act_encoder_cache_fn,
        )

        obs = next_obs  # roll

        global_step += step_increment

        # ------------------------------------------------------------------
        # (3) Periodic evaluation ------------------------------------------
        # ------------------------------------------------------------------
        if global_step >= next_eval_step:
            _run_evaluation(global_step)
            while next_eval_step <= global_step:
                next_eval_step += cfg.eval_interval_every_steps

        # ------------------------------------------------------------------
        # (4) Updates -------------------------------------------------------
        # ------------------------------------------------------------------
        if cfg.algo.updates_per_primitive_step > 0.0:
            update_budget += step_increment * cfg.algo.updates_per_primitive_step
            updates_this_iteration = int(update_budget)
            update_budget -= updates_this_iteration
            should_update = updates_this_iteration > 0
        else:
            should_update = global_step % cfg.algo.update_every_n_steps == 0 or global_step == step_increment
            updates_this_iteration = cfg.algo.num_updates_per_iteration if should_update else 0

        if should_update:
            i = 0
            actor_update_cadence = max(
                1,
                cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration,
            )
            # Normal training loop - critic is already warmed up
            while i < updates_this_iteration:
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
                with training_timer.time("batch_sampling"):
                    # Sample batches from replay buffers
                    online_batch = online_rb.sample(online_batch_size)
                    online_batch = online_batch.to(device, non_blocking=True)

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training
                        offline_batch = offline_rb.sample(offline_batch_size)
                        offline_batch = offline_batch.to(device, non_blocking=True)
                        _align_replay_batch_keys(online_batch, offline_batch)
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        # Online-only training
                        batch = online_batch

                # Update actor on the last iteration of each update cycle
                if cfg.algo.actor_update_every_n_updates > 0:
                    update_actor = (gradient_updates + 1) % cfg.algo.actor_update_every_n_updates == 0
                else:
                    update_actor = (i + 1) % actor_update_cadence == 0

                # Apply actor learning rate warmup
                if update_actor:
                    if cfg.algo.actor_lr_warmup_steps > 0:
                        # Calculate current LR with linear warmup from 0 to target
                        warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                        current_lr = cfg.agent.actor_lr * warmup_progress
                        for param_group in agent.actor_opt.param_groups:
                            param_group["lr"] = current_lr

                    actor_updates += 1

                with training_timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)
                gradient_updates += 1

                # Update priorities for prioritized experience replay
                if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    # Update priorities in the batch for priority updates
                    td_errors = metrics["_td_errors"]
                    batch["_priority"] = td_errors

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training - update both buffers
                        online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                        # Update online buffer priorities
                        if online_batch_size_actual > 0:
                            online_batch_subset = batch[:online_batch_size_actual]
                            online_rb.update_tensordict_priority(online_batch_subset)

                        # Update offline buffer priorities
                        if online_batch_size_actual < len(batch):
                            offline_batch_subset = batch[online_batch_size_actual:]
                            offline_rb.update_tensordict_priority(offline_batch_subset)
                    else:
                        # Online-only training - update only online buffer
                        online_rb.update_tensordict_priority(batch)

                terminal_rewards = batch["next"]["reward"][~batch["nonterminal"]]
                # A replay batch may contain no terminal transition.  Keep
                # this diagnostic finite without changing the TD target.
                metrics["data/batch_terminal_R"] = (
                    terminal_rewards.mean()
                    if terminal_rewards.numel()
                    else torch.zeros((), device=batch["next"]["reward"].device)
                )
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

                i += 1

        training_cum_time += time.time() - iter_start

        # ------------------------------------------------------------------
        # (6) Logging -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step >= next_log_step:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

            # Prepare base logging dict
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                "timing/training_total_time": time.time() - train_start_time,
                "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                "training/gradient_updates": gradient_updates,
                "training/actor_updates": actor_updates,
            }

            # Add timing statistics
            timing_stats = training_timer.get_timing_stats()
            log_dict.update(timing_stats)

            # Add metrics, filtering out internal data
            filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            log_dict.update(filtered_metrics)

            # Compute residual action statistics only when logging
            if "_actions" in metrics:
                actions = metrics["_actions"]
                # Compute L1/L2 magnitudes (only during logging to save computation)
                residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))
            else:
                residual_l1_magnitude = None
                residual_l2_magnitude = None

            # Add Q values histogram when available
            if "_target_q" in metrics:
                target_q = metrics["_target_q"]
                log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

            if cfg.algo.progressive_clipping_steps > 0:
                log_dict["training/progressive_clipping_factor"] = clip_factor

            wandb.log(log_dict, step=global_step)

            # Enhanced print statement with residual action magnitudes, gradient norms, and actor LR
            current_actor_lr = agent.actor_opt.param_groups[0]["lr"]

            if "train/actor_loss_base" in metrics:
                actor_loss_str = f"actor_loss_base={metrics['train/actor_loss_base']:.4f}"
                print_msg = (
                    f"[{global_step}] {actor_loss_str} "
                    f"critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e}"
                )
            else:
                # During critic warmup, actor might not be updated
                print_msg = (
                    f"[{global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e} (actor not updated)"
                )
            if residual_l1_magnitude is not None and residual_l2_magnitude is not None:
                print_msg += f" residual_l1={residual_l1_magnitude:.4f} residual_l2={residual_l2_magnitude:.4f}"

            # Add gradient norms to print statement
            if "train/actor_grad_norm" in metrics:
                print_msg += f" actor_grad_norm={metrics['train/actor_grad_norm']:.4f}"

            # Add L2 penalty if active
            if "train/actor_l2_penalty" in metrics:
                print_msg += f" l2_penalty={metrics['train/actor_l2_penalty']:.4f}"

            # Add timing percentages to print statement
            if timing_stats:
                env_pct = timing_stats.get("timing/env_step_percentage", 0)
                grad_pct = timing_stats.get("timing/gradient_update_percentage", 0)
                batch_pct = timing_stats.get("timing/batch_sampling_percentage", 0)
                eval_pct = timing_stats.get("timing/evaluation_percentage", 0)
                print_msg += (
                    f" | Time%: env={env_pct:.1f} grad={grad_pct:.1f} batch={batch_pct:.1f} eval={eval_pct:.1f}"
                )

            print(print_msg)
            while next_log_step <= global_step:
                next_log_step += cfg.log_freq

    if cfg.eval_final_num_episodes > 0:
        print(f"Running final evaluation with {cfg.eval_final_num_episodes} episodes...")
        _run_evaluation(global_step, num_episodes=cfg.eval_final_num_episodes)

    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if run_cache_dir.exists():
        print(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        print("Run directory cleaned up successfully.")


# -----------------------------------------------------------------------------
# Hydra entry point -----------------------------------------------------------
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_name="residual_td3_dexmg_config")
def hydra_entry(cfg: ResidualTD3DexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
