# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import hashlib
import json
import logging
import os
import pprint
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import tensordict
import torch
import torchrl
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.rl_finetuning.config.residual_sac_flow import ResidualSACFlowDexmgConfig
from resfit.rl_finetuning.residual_sac_flow.agent import ResidualSACFlowAgent
from resfit.rl_finetuning.residual_sac_flow.evaluate import run_residual_sac_flow_evaluation
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ["MUJOCO_GL"] = "egl"
if "MUJOCO_EGL_DEVICE_ID" in os.environ:
    del os.environ["MUJOCO_EGL_DEVICE_ID"]

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT)).expanduser().resolve()
OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_buffer_cache"


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
    replay_buffer: TensorDictPrioritizedReplayBuffer,
) -> None:
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for env_idx in range(num_envs):
        if done[env_idx] and "final_obs" in info and info["final_obs"][env_idx] is not None:
            final_obs_dict = info["final_obs"][env_idx]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[env_idx] for k, v in next_obs.items()}

        curr_obs_i = {k: v[env_idx] for k, v in obs.items()}
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[env_idx],
                        "reward": reward[env_idx],
                    },
                    batch_size=[],
                ),
                "action": actions[env_idx],
                "_priority": torch.tensor(10.0, dtype=torch.float32),
            },
            batch_size=[],
        ).unsqueeze(0)
        replay_buffer.add(td)


def _seed_everything(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def _build_env(
    *,
    env_name: str,
    num_envs: int,
    base_policy: ACTPolicy,
    device: str,
    video_key: str,
    debug: bool,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
):
    vec_env = create_vectorized_env(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        video_key=video_key,
        debug=debug,
    )
    return BasePolicyVecEnvWrapper(
        vec_env=vec_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )


def _populate_offline_buffer(
    *,
    dataset: LeRobotDataset,
    replay_buffer: TensorDictPrioritizedReplayBuffer,
    image_keys: list[str],
    state_standardizer: StateStandardizer,
    action_scaler: ActionScaler,
    num_episodes: int | None,
    use_base_policy_for_base_actions: bool,
    base_policy: ACTPolicy | None,
    device: torch.device,
) -> int:
    if use_base_policy_for_base_actions and base_policy is None:
        raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

    print("Populating offline replay buffer...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    episode_cache: dict[int, dict] = {}
    transitions = 0
    current_episode: int | None = None

    for sample in tqdm(loader, desc="Processing offline dataset"):
        ep_idx = int(sample["episode_index"].item())
        if num_episodes is not None and ep_idx >= num_episodes:
            break

        if current_episode != ep_idx:
            if base_policy is not None:
                base_policy.reset()
            current_episode = ep_idx

        gt_action = sample["action"].float().squeeze(0)
        gt_action_scaled = action_scaler.scale(gt_action)
        done_flag = bool(sample["next.done"].item())

        if use_base_policy_for_base_actions:
            raw_obs = {k: sample[k].to(device) for k in sample if "observation" in k}
            with torch.no_grad():
                base_action = base_policy.select_action(raw_obs)
            base_action_scaled = action_scaler.scale(base_action.squeeze(0).cpu())
        else:
            base_action_scaled = gt_action_scaled

        curr_obs = {
            "observation.state": state_standardizer.standardize(sample["observation.state"].float().squeeze(0)),
            "observation.base_action": base_action_scaled,
        }
        for key in image_keys:
            curr_obs[key] = sample[key].squeeze(0)
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
            replay_buffer.add(transition)
            transitions += 1

        episode_cache[ep_idx] = {
            "obs": curr_obs,
            "action": gt_action_scaled,
        }

    print(f"Added {transitions} offline transitions")
    return transitions


def _sample_mixed_batch(
    *,
    online_rb: TensorDictPrioritizedReplayBuffer,
    offline_rb: TensorDictPrioritizedReplayBuffer | None,
    online_batch_size: int,
    offline_batch_size: int,
    device: torch.device,
) -> tuple[TensorDict, int]:
    parts: list[TensorDict] = []
    online_count = 0

    if online_batch_size > 0:
        online_batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
        parts.append(online_batch)
        online_count = len(online_batch)

    if offline_batch_size > 0:
        if offline_rb is None:
            raise ValueError("offline_rb is required when offline_batch_size > 0")
        offline_batch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
        parts.append(offline_batch)

    if not parts:
        raise ValueError("At least one of online_batch_size or offline_batch_size must be > 0")

    if len(parts) == 1:
        return parts[0], online_count
    return torch.cat(parts, dim=0), online_count


def _update_priorities(
    *,
    batch: TensorDict,
    online_rb: TensorDictPrioritizedReplayBuffer,
    offline_rb: TensorDictPrioritizedReplayBuffer | None,
    online_count: int,
    metrics: dict,
    offline_count: int,
) -> None:
    if "_td_errors" not in metrics:
        return

    batch["_priority"] = metrics["_td_errors"]
    if online_count > 0:
        online_rb.update_tensordict_priority(batch[:online_count])
    if offline_count > 0 and offline_rb is not None:
        offline_rb.update_tensordict_priority(batch[online_count : online_count + offline_count])


def _save_checkpoint(path: Path, agent: ResidualSACFlowAgent, cfg: ResidualSACFlowDexmgConfig, step: int) -> None:
    payload = {
        "step": step,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "agent": agent.state_dict(),
    }
    torch.save(payload, path)


def _metric_value(metrics: dict, key: str, default: float = float("nan")) -> float:
    value = metrics.get(key, default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)


def _metric_value_str(metrics: dict, key: str) -> str:
    if key not in metrics:
        return "na"
    return f"{_metric_value(metrics, key):.4f}"


def _format_train_status(
    *,
    global_step: int,
    metrics: dict,
    actor_lr: float,
) -> str:
    residual = metrics.get("_actions", None)
    residual_l1 = "na"
    residual_l2 = "na"
    if isinstance(residual, torch.Tensor):
        residual = residual.float()
        residual_l1 = f"{residual.abs().mean().item():.4f}"
        residual_l2 = f"{residual.pow(2).mean().sqrt().item():.4f}"

    return (
        f"[{global_step}] "
        f"actor_loss={_metric_value_str(metrics, 'train/actor_loss')} "
        f"critic_loss={_metric_value_str(metrics, 'train/critic_loss')} "
        f"actor_lr={actor_lr:.2e} "
        f"alpha={_metric_value_str(metrics, 'train/alpha')} "
        f"residual_l1={residual_l1} "
        f"residual_l2={residual_l2} "
        f"actor_grad_norm={_metric_value_str(metrics, 'train/actor_grad_norm')}"
    )


def main(cfg: ResidualSACFlowDexmgConfig) -> None:
    requested_device = cfg.agent.device
    if requested_device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{requested_device}' but CUDA is not available")
    else:
        device_str = requested_device
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    _seed_everything(cfg.seed, cfg.torch_deterministic)
    print(f"Set random seed to {cfg.seed}")

    assert "base_policy" in cfg, "Base policy configuration is required"
    policy_dir, _ = download_policy_from_wandb(
        cfg.base_policy.wandb_id,
        step=cfg.base_policy.wt_type,
        artifact_version=cfg.base_policy.wt_version,
    )
    base_policy: ACTPolicy = load_policy(policy_dir)
    eval_base_policy: ACTPolicy = load_policy(policy_dir)
    base_policy.to(device).eval()
    eval_base_policy.to(device).eval()

    base_cfg = base_policy.config
    if not isinstance(base_cfg, ACTConfig):
        raise ValueError(f"Only ACT base policies are supported, got {type(base_cfg)}")

    print("Loading dataset and normalization statistics...")
    dataset = LeRobotDataset(cfg.offline_data.name)
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=cfg.agent.action_range_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )

    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    env = _build_env(
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
    num_cpus_available = max((os.cpu_count() or 1) - 1, 1)
    cfg.eval_num_envs = min(cfg.eval_num_envs, num_cpus_available)
    eval_env = _build_env(
        env_name=cfg.task,
        num_envs=cfg.eval_num_envs,
        base_policy=eval_base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )

    if hasattr(env, "seed"):
        env.seed(cfg.seed)
    if hasattr(eval_env, "seed"):
        eval_env.seed(cfg.seed + 1)

    if isinstance(cfg.rl_camera, str):
        image_keys = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    lowdim_keys = ["observation.state", "observation.base_action"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]

    cfg.agent.device = device_str
    agent = ResidualSACFlowAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
    )

    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = cfg.algo.batch_size - online_batch_size

    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=max(1, online_batch_size),
    )

    offline_rb = None
    if offline_batch_size > 0:
        if cfg.offline_data.num_episodes is not None:
            total_frames = sum(
                dataset.meta.episodes[ep_idx]["length"]
                for ep_idx in range(min(cfg.offline_data.num_episodes, dataset.meta.total_episodes))
            )
            num_episodes = cfg.offline_data.num_episodes
        else:
            total_frames = dataset.meta.total_frames
            num_episodes = dataset.meta.total_episodes

        estimated_transitions = max(0, total_frames - num_episodes)
        max_offline_transitions = estimated_transitions if cfg.algo.offline_fraction > 0.0 else 1
        offline_rb = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=max(max_offline_transitions, 1), device="cpu"),
            alpha=alpha,
            beta=beta,
            eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
            pin_memory=True,
            prefetch=cfg.algo.prefetch_batches,
            batch_size=max(1, offline_batch_size),
        )
        offline_cache_meta = {
            "task": cfg.task,
            "dataset_name": cfg.offline_data.name,
            "num_episodes": cfg.offline_data.num_episodes,
            "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
            "min_action_range": cfg.offline_data.min_action_range,
            "min_state_std": cfg.offline_data.min_state_std,
            "image_keys": image_keys,
            "n_step": cfg.algo.n_step,
            "gamma": cfg.algo.gamma,
            "base_policy_wandb_id": cfg.base_policy.wandb_id,
            "sampling_strategy": cfg.algo.sampling_strategy,
            "normalized_actions": True,
            "batch_size": offline_batch_size,
            "torchrl_version": torchrl.__version__,
            "tensordict_version": tensordict.__version__,
        }
        if cfg.algo.sampling_strategy == "prioritized_replay":
            offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
            offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

        meta_str = json.dumps(offline_cache_meta, sort_keys=True)
        cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324
        cache_dir = OFFLINE_CACHE_DIR / cache_hash

        downloaded_dir = None
        if OFFLINE_HF_REPO is not None:
            print(f"Attempting to download offline buffer {cache_hash} from {OFFLINE_HF_REPO}...")
            downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
        if downloaded_dir is not None:
            cache_dir = downloaded_dir

        loaded_from_cache = False
        if cfg.algo.offline_fraction > 0.0 and cache_dir.exists():
            print(f"{cache_dir} found on disk. Attempting to load...")
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(f"Loaded offline buffer from cache at {cache_dir} (size={len(offline_rb)})")

        if cfg.algo.offline_fraction > 0.0 and not loaded_from_cache:
            added = _populate_offline_buffer(
                dataset=dataset,
                replay_buffer=offline_rb,
                image_keys=image_keys,
                state_standardizer=state_standardizer,
                action_scaler=action_scaler,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
                device=device,
            )
            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)
            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)
            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{cfg.task}__seed{cfg.seed}"
    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(wandb_config, dict)

    obs, _ = env.reset()
    if cfg.algo.learning_starts > 0:
        warmup_log_interval = max(1, min(1_000, cfg.algo.learning_starts // 10))
        next_warmup_log = warmup_log_interval
        warmup_start_time = time.time()
        print(f"Collecting {cfg.algo.learning_starts} online warmup transitions...")
        warmup_mode = "base_plus_noise" if cfg.algo.use_base_policy_for_warmup else "uniform_final_action"
        print(f"Warmup mode: {warmup_mode}, noise_scale={cfg.algo.random_action_noise_scale}")
        while len(online_rb) < cfg.algo.learning_starts:
            if cfg.algo.use_base_policy_for_warmup:
                residual_action = (torch.rand((cfg.num_envs, action_dim), device=device) * 2.0 - 1.0)
                residual_action = residual_action * cfg.algo.random_action_noise_scale
                final_action = torch.clamp(obs["observation.base_action"] + residual_action, -1.0, 1.0)
                residual_action = final_action - obs["observation.base_action"]
            else:
                final_action = (torch.rand((cfg.num_envs, action_dim), device=device) * 2.0 - 1.0)
                final_action = final_action * cfg.algo.random_action_noise_scale
                final_action = torch.clamp(final_action, -1.0, 1.0)
                residual_action = final_action - obs["observation.base_action"]

            next_obs, reward, terminated, truncated, info = env.step(residual_action)
            done = terminated | truncated
            executed_action = info["scaled_action"]
            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=executed_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                replay_buffer=online_rb,
            )
            obs = next_obs

            collected = len(online_rb)
            if collected >= next_warmup_log or collected == cfg.algo.learning_starts:
                elapsed = time.time() - warmup_start_time
                rate = collected / max(elapsed, 1e-6)
                progress = 100.0 * collected / max(cfg.algo.learning_starts, 1)
                print(
                    f"Warmup progress: {collected}/{cfg.algo.learning_starts} "
                    f"({progress:.1f}%), {rate:.1f} transitions/s"
                )
                next_warmup_log += warmup_log_interval

        print(f"Warmup done. Online buffer size = {len(online_rb)}")

    print("Launching residual SAC-flow with config:")
    pprint.pprint(wandb_config)

    wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )
    wandb.summary["environment/horizon"] = env.metadata["horizon"]

    output_root = _CACHE_ROOT
    run_dir = output_root / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}"
    model_dir = run_dir / "models"
    output_dir = run_dir / "outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_success_rate = -1.0
    global_step = 0
    update_step = 0
    obs, _ = env.reset()
    train_start_time = time.time()

    if cfg.eval_first:
        eval_metrics = run_residual_sac_flow_evaluation(
            env=eval_env,
            agent=agent,
            num_episodes=cfg.eval_num_episodes,
            device=device,
            global_step=global_step,
            save_video=cfg.save_video,
            run_name=run_name,
            output_dir=output_dir,
        )
        best_success_rate = eval_metrics["eval/success_rate"]
        _save_checkpoint(model_dir / "best.pt", agent, cfg, global_step)
        wandb.summary["best_eval/success_rate"] = best_success_rate
        print(
            f"Eval step {global_step}: success_rate={eval_metrics['eval/success_rate']:.4f} "
            f"mean_return={eval_metrics['eval/mean_return']:.4f} "
            f"mean_success_len={eval_metrics['eval/mean_successful_episode_length']:.1f}"
        )

    if cfg.algo.critic_warmup_steps > 0:
        print(f"Running {cfg.algo.critic_warmup_steps} critic warmup updates...")
        for warmup_idx in range(cfg.algo.critic_warmup_steps):
            batch, online_count = _sample_mixed_batch(
                online_rb=online_rb,
                offline_rb=offline_rb,
                online_batch_size=online_batch_size,
                offline_batch_size=offline_batch_size,
                device=device,
            )
            metrics = agent.update(
                batch,
                update_actor=False,
                deterministic_backup=cfg.algo.critic_warmup_deterministic_backup,
                use_entropy_backup=cfg.algo.critic_warmup_use_entropy_backup,
                clip_q_target_to_reward_range=cfg.algo.critic_warmup_clip_q_target,
            )
            agent.soft_update_targets()
            if cfg.algo.sampling_strategy == "prioritized_replay":
                _update_priorities(
                    batch=batch,
                    online_rb=online_rb,
                    offline_rb=offline_rb,
                    online_count=online_count,
                    offline_count=len(batch) - online_count,
                    metrics=metrics,
                )
            if warmup_idx % 100 == 0:
                print(
                    f"Critic warmup {warmup_idx}/{cfg.algo.critic_warmup_steps} "
                    f"critic_loss={metrics['train/critic_loss']:.4f}"
                )

    while global_step <= cfg.algo.total_timesteps:
        with torch.no_grad():
            residual_action = agent.act(obs, eval_mode=False, cpu=False)

        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        done = terminated | truncated
        executed_action = info["scaled_action"]

        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=executed_action,
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            replay_buffer=online_rb,
        )
        obs = next_obs
        global_step += cfg.num_envs

        if global_step % cfg.eval_interval_every_steps == 0 and (cfg.eval_first or global_step > 0):
            print(f"Evaluating {cfg.eval_num_episodes} episodes...")
            eval_metrics = run_residual_sac_flow_evaluation(
                env=eval_env,
                agent=agent,
                num_episodes=cfg.eval_num_episodes,
                device=device,
                global_step=global_step,
                save_video=cfg.save_video,
                run_name=run_name,
                output_dir=output_dir,
            )
            success_rate = eval_metrics["eval/success_rate"]
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                _save_checkpoint(model_dir / "best.pt", agent, cfg, global_step)
                wandb.summary["best_eval/success_rate"] = best_success_rate
            print(
                f"Eval step {global_step}: success_rate={eval_metrics['eval/success_rate']:.4f} "
                f"mean_return={eval_metrics['eval/mean_return']:.4f} "
                f"mean_success_len={eval_metrics['eval/mean_successful_episode_length']:.1f}"
            )

        if global_step % cfg.algo.update_every_n_steps == 0:
            for _ in range(cfg.algo.num_updates_per_iteration):
                batch, online_count = _sample_mixed_batch(
                    online_rb=online_rb,
                    offline_rb=offline_rb,
                    online_batch_size=online_batch_size,
                    offline_batch_size=offline_batch_size,
                    device=device,
                )
                update_actor = (
                    global_step >= cfg.algo.actor_learning_starts
                    and (update_step + 1) % cfg.algo.actor_update_frequency == 0
                )
                metrics = agent.update(batch, update_actor=update_actor)
                update_step += 1
                if update_step % cfg.algo.target_update_frequency == 0:
                    agent.soft_update_targets()
                if cfg.algo.sampling_strategy == "prioritized_replay":
                    _update_priorities(
                        batch=batch,
                        online_rb=online_rb,
                        offline_rb=offline_rb,
                        online_count=online_count,
                        offline_count=len(batch) - online_count,
                        metrics=metrics,
                    )

        if global_step % cfg.log_freq == 0:
            elapsed = time.time() - train_start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb) if offline_rb is not None else 0,
                "timing/training_total_time": elapsed,
                "timing/aggregate_steps_per_second": global_step / elapsed if elapsed > 0 else 0.0,
                "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
            }
            if "metrics" in locals():
                for key, value in metrics.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, (float, int)):
                        log_dict[key] = value
                if "_actions" in metrics:
                    log_dict["histograms/residual_actions"] = wandb.Histogram(metrics["_actions"].numpy().reshape(-1))
                if "_combined_actions" in metrics:
                    log_dict["histograms/final_actions"] = wandb.Histogram(
                        metrics["_combined_actions"].numpy().reshape(-1)
                    )
            wandb.log(log_dict, step=global_step)

            if "metrics" in locals():
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    + _format_train_status(
                        global_step=global_step,
                        metrics=metrics,
                        actor_lr=agent.actor_opt.param_groups[0]["lr"],
                    )
                )

    _save_checkpoint(model_dir / "latest.pt", agent, cfg, global_step)
    env.close()
    eval_env.close()
    wandb.finish()


@hydra.main(version_base=None, config_name="residual_sac_flow_coffee_config")
def hydra_entry(cfg: ResidualSACFlowDexmgConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
