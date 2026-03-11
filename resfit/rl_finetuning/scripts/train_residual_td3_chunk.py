# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import logging
import hashlib
import json
import pprint
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
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
from resfit.rl_finetuning.config.residual_td3_chunk import ResidualTD3ChunkDexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.hugging_face import optimized_replay_buffer_dumps, optimized_replay_buffer_loads
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.residual_chunk_env_wrapper import ResidualChunkVecEnvWrapper


class TrainingTimer:
    """Simple timing utility for measuring training stage proportions."""

    def __init__(self):
        self.times = defaultdict(list)

    @contextmanager
    def time(self, stage_name: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[stage_name].append(elapsed)

    def get_timing_stats(self) -> dict[str, float]:
        if not self.times:
            return {}
        total_time = sum(sum(times) for times in self.times.values())
        if total_time == 0:
            return {}

        stats = {}
        for stage_name, times_list in self.times.items():
            stage_total = sum(times_list)
            stage_avg = stage_total / len(times_list) if times_list else 0
            stats[f"timing/{stage_name}_percentage"] = (stage_total / total_time) * 100
            stats[f"timing/{stage_name}_avg_ms"] = stage_avg * 1000
            stats[f"timing/{stage_name}_total_s"] = stage_total
        return stats

    def reset(self):
        self.times = defaultdict(list)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
os.environ["MUJOCO_GL"] = "egl"
if "MUJOCO_EGL_DEVICE_ID" in os.environ:
    del os.environ["MUJOCO_EGL_DEVICE_ID"]

_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()
CHUNK_OFFLINE_CACHE_DIR = _CACHE_ROOT / "chunk_offline_buffer_cache"


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
) -> None:
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            next_obs_i = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device))
                for k, v in info["final_obs"][i].items()
            }
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}
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
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "_priority": torch.tensor(10.0, dtype=torch.float32),
            },
            batch_size=[],
        ).unsqueeze(0)
        online_rb.add(td)


def _get_envs(
    *,
    env_name: str,
    num_envs: int,
    base_policy: ACTPolicy,
    device: str,
    video_key: str,
    debug: bool,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    chunk_success_threshold: float,
):
    vec_env = create_vectorized_env(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        video_key=video_key,
        debug=debug,
    )
    return ResidualChunkVecEnvWrapper(
        vec_env=vec_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=macro_horizon,
        base_plan_horizon=base_plan_horizon,
        chunk_success_threshold=chunk_success_threshold,
    )


def _sample_batch(online_rb, batch_size: int, device: torch.device) -> TensorDict:
    batch = online_rb.sample(batch_size)
    return batch.to(device, non_blocking=True)


def _sample_mixed_batch(
    *,
    online_rb: TensorDictPrioritizedReplayBuffer,
    online_batch_size: int,
    offline_rb: TensorDictPrioritizedReplayBuffer | None,
    offline_batch_size: int,
    device: torch.device,
) -> TensorDict:
    batches: list[TensorDict] = []
    if online_batch_size > 0:
        batches.append(_sample_batch(online_rb, online_batch_size, device))
    if offline_rb is not None and offline_batch_size > 0:
        batches.append(_sample_batch(offline_rb, offline_batch_size, device))
    if not batches:
        raise ValueError("At least one of online_batch_size or offline_batch_size must be positive.")
    if len(batches) == 1:
        return batches[0]
    return torch.cat(batches, dim=0)


def _format_progress(current: int, total: int) -> str:
    if total <= 0:
        return str(current)
    return f"{current}/{total} ({100.0 * current / total:.1f}%)"


def _read_cached_storage_capacity(cache_dir: Path) -> int | None:
    metadata_path = cache_dir / "storage" / "meta.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path) as f:
        metadata = json.load(f)
    shape = metadata.get("shape")
    if isinstance(shape, list) and shape:
        return int(shape[0])
    return None


def _pad_action_chunk(actions: list[torch.Tensor], chunk_len: int, action_dim: int) -> torch.Tensor:
    if not actions:
        return torch.zeros(chunk_len, action_dim, dtype=torch.float32)
    stacked = torch.stack(actions, dim=0)
    if stacked.shape[0] >= chunk_len:
        return stacked[:chunk_len]
    pad = stacked[-1:].repeat(chunk_len - stacked.shape[0], 1)
    return torch.cat([stacked, pad], dim=0)


def _build_offline_obs(
    *,
    sample: dict,
    image_keys: list[str],
    state_standardizer: StateStandardizer,
    base_action_flat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    obs = {
        "observation.state": state_standardizer.standardize(sample["observation.state"].float()),
        "observation.base_action": base_action_flat.float(),
    }
    for key in image_keys:
        obs[key] = sample[key]
    to_uint8(obs, image_keys)
    return obs


def _build_offline_terminal_obs(
    *,
    sample: dict,
    image_keys: list[str],
    state_standardizer: StateStandardizer,
    macro_action_dim: int,
) -> dict[str, torch.Tensor]:
    obs = {
        "observation.state": state_standardizer.standardize(sample["observation.state"].float()),
        "observation.base_action": torch.zeros(macro_action_dim, dtype=torch.float32),
    }
    for key in image_keys:
        obs[key] = sample[key]
    to_uint8(obs, image_keys)
    return obs


def _plan_offline_base_actions(
    *,
    sample: dict,
    base_policy: ACTPolicy,
    action_scaler: ActionScaler,
    base_plan_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    raw_obs = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor) and key.startswith("observation.")
    }
    with torch.no_grad():
        if hasattr(base_policy, "select_action_chunk"):
            base_actions = base_policy.select_action_chunk(raw_obs, n_steps=base_plan_horizon)
            base_nactions = action_scaler.scale(base_actions)
        else:
            base_nactions = base_policy.select_action_chunk_normalized(raw_obs, n_steps=base_plan_horizon)
    return base_nactions.squeeze(0).cpu()


def _flush_offline_episode(
    *,
    episode_samples: list[dict],
    rb: TensorDictPrioritizedReplayBuffer,
    image_keys: list[str],
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    use_base_policy_for_base_actions: bool,
    base_policy: ACTPolicy | None,
    device: torch.device,
) -> int:
    if not episode_samples:
        return 0

    primitive_action_dim = int(episode_samples[0]["action"].numel())
    macro_action_dim = primitive_action_dim * macro_horizon
    gt_actions_scaled = [action_scaler.scale(sample["action"].float()) for sample in episode_samples]

    transitions = 0
    step_idx = 0
    base_plan: torch.Tensor | None = None

    if use_base_policy_for_base_actions:
        if base_policy is None:
            raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")
        base_policy.reset()

    while step_idx < len(episode_samples):
        current_sample = episode_samples[step_idx]
        max_steps = min(macro_horizon, len(episode_samples) - step_idx)
        executed_steps = max_steps
        terminal_idx: int | None = None
        for offset in range(max_steps):
            if bool(episode_samples[step_idx + offset]["next.done"]):
                executed_steps = offset + 1
                terminal_idx = step_idx + offset
                break

        if use_base_policy_for_base_actions:
            if base_plan is None or base_plan.shape[0] < macro_horizon:
                base_plan = _plan_offline_base_actions(
                    sample=current_sample,
                    base_policy=base_policy,
                    action_scaler=action_scaler,
                    base_plan_horizon=base_plan_horizon,
                    device=device,
                )
            base_chunk = base_plan[:macro_horizon].clone()
        else:
            base_chunk = _pad_action_chunk(
                gt_actions_scaled[step_idx : step_idx + max_steps], macro_horizon, primitive_action_dim
            )

        combined_chunk = base_chunk.clone()
        combined_chunk[:executed_steps] = torch.stack(
            gt_actions_scaled[step_idx : step_idx + executed_steps], dim=0
        )
        curr_obs = _build_offline_obs(
            sample=current_sample,
            image_keys=image_keys,
            state_standardizer=state_standardizer,
            base_action_flat=base_chunk.reshape(-1),
        )

        if terminal_idx is not None:
            done = torch.tensor(True, dtype=torch.bool)
            reward = torch.tensor(1.0, dtype=torch.float32)
            next_obs = _build_offline_terminal_obs(
                sample=episode_samples[terminal_idx],
                image_keys=image_keys,
                state_standardizer=state_standardizer,
                macro_action_dim=macro_action_dim,
            )
            base_plan = None
            if use_base_policy_for_base_actions and base_policy is not None:
                base_policy.reset()
        else:
            done = torch.tensor(False, dtype=torch.bool)
            reward = torch.tensor(0.0, dtype=torch.float32)
            next_step_idx = step_idx + executed_steps
            next_sample = episode_samples[next_step_idx]
            if use_base_policy_for_base_actions:
                assert base_plan is not None
                base_plan = base_plan[executed_steps:]
                if base_plan.shape[0] < macro_horizon:
                    base_plan = _plan_offline_base_actions(
                        sample=next_sample,
                        base_policy=base_policy,
                        action_scaler=action_scaler,
                        base_plan_horizon=base_plan_horizon,
                        device=device,
                    )
                next_base_chunk = base_plan[:macro_horizon].clone()
            else:
                remaining = min(macro_horizon, len(episode_samples) - next_step_idx)
                next_base_chunk = _pad_action_chunk(
                    gt_actions_scaled[next_step_idx : next_step_idx + remaining],
                    macro_horizon,
                    primitive_action_dim,
                )
            next_obs = _build_offline_obs(
                sample=next_sample,
                image_keys=image_keys,
                state_standardizer=state_standardizer,
                base_action_flat=next_base_chunk.reshape(-1),
            )

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs, batch_size=[]),
                        "done": done,
                        "reward": reward,
                    },
                    batch_size=[],
                ),
                "action": combined_chunk.reshape(-1),
                "_priority": torch.tensor(10.0, dtype=torch.float32),
            },
            batch_size=[],
        ).unsqueeze(0)
        rb.add(td)
        transitions += 1
        step_idx += executed_steps

    return transitions


def populate_chunk_offline_buffer(
    *,
    dataset: LeRobotDataset,
    rb: TensorDictPrioritizedReplayBuffer,
    image_keys: list[str],
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    num_episodes: int | None,
    use_base_policy_for_base_actions: bool,
    base_policy: ACTPolicy | None,
    device: torch.device,
) -> int:
    if use_base_policy_for_base_actions and base_policy is None:
        raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    current_episode: int | None = None
    episode_samples: list[dict] = []
    transitions = 0

    for sample in tqdm(loader, desc="Processing chunk offline dataset"):
        ep_idx = int(sample["episode_index"].item())
        if num_episodes is not None and ep_idx >= num_episodes:
            break

        squeezed = {
            key: (value.squeeze(0).cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in sample.items()
        }

        if current_episode is None:
            current_episode = ep_idx
        elif ep_idx != current_episode:
            transitions += _flush_offline_episode(
                episode_samples=episode_samples,
                rb=rb,
                image_keys=image_keys,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
                macro_horizon=macro_horizon,
                base_plan_horizon=base_plan_horizon,
                use_base_policy_for_base_actions=use_base_policy_for_base_actions,
                base_policy=base_policy,
                device=device,
            )
            episode_samples = []
            current_episode = ep_idx

        episode_samples.append(squeezed)

    transitions += _flush_offline_episode(
        episode_samples=episode_samples,
        rb=rb,
        image_keys=image_keys,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=macro_horizon,
        base_plan_horizon=base_plan_horizon,
        use_base_policy_for_base_actions=use_base_policy_for_base_actions,
        base_policy=base_policy,
        device=device,
    )
    return transitions


def main(cfg: ResidualTD3ChunkDexmgConfig):
    if cfg.num_envs != 1:
        raise ValueError("Chunk residual TD3 currently supports exactly one training environment.")
    if not 0.0 <= cfg.algo.offline_fraction <= 1.0:
        raise ValueError("algo.offline_fraction must be in [0, 1].")
    if cfg.algo.offline_fraction > 0.0 and cfg.offline_data is None:
        raise ValueError("offline_data must be provided when algo.offline_fraction > 0.")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    policy_dir, _ = download_policy_from_wandb(
        cfg.base_policy.wandb_id,
        step=cfg.base_policy.wt_type,
        artifact_version=cfg.base_policy.wt_version,
    )
    base_policy: ACTPolicy = load_policy(policy_dir)
    base_policy.to(device)
    base_policy.eval()
    eval_base_policy: ACTPolicy = load_policy(policy_dir)
    eval_base_policy.to(device)
    eval_base_policy.eval()

    if not isinstance(base_policy.config, ACTConfig):
        raise ValueError(f"Unknown base policy type: {type(base_policy.config)}")
    if cfg.algo.base_plan_horizon > base_policy.config.chunk_size:
        raise ValueError(
            f"base_plan_horizon={cfg.algo.base_plan_horizon} exceeds ACT chunk_size={base_policy.config.chunk_size}."
        )
    offline_base_policy: ACTPolicy | None = None
    if cfg.algo.offline_fraction > 0.0 and cfg.offline_data is not None and cfg.offline_data.use_base_policy_for_base_actions:
        offline_base_policy = load_policy(policy_dir)
        offline_base_policy.to(device)
        offline_base_policy.eval()

    dataset = LeRobotDataset(cfg.offline_data.name)
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )

    env = _get_envs(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        base_policy=base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=cfg.algo.macro_horizon,
        base_plan_horizon=cfg.algo.base_plan_horizon,
        chunk_success_threshold=cfg.algo.chunk_success_threshold,
    )
    cfg.eval_num_envs = 1
    eval_env = _get_envs(
        env_name=cfg.task,
        num_envs=cfg.eval_num_envs,
        base_policy=eval_base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=cfg.algo.macro_horizon,
        base_plan_horizon=cfg.algo.base_plan_horizon,
        chunk_success_threshold=cfg.algo.chunk_success_threshold,
    )

    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    lowdim_keys = ["observation.state", "observation.base_action"]
    gamma_macro = cfg.algo.gamma ** cfg.algo.macro_horizon
    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = cfg.algo.batch_size - online_batch_size
    if cfg.algo.offline_fraction > 0.0 and offline_batch_size == 0:
        raise ValueError("offline_fraction > 0 but computed offline_batch_size is 0. Increase batch_size or offline_fraction.")

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=gamma_macro),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=cfg.algo.batch_size,
    )
    offline_rb: TensorDictPrioritizedReplayBuffer | None = None
    if offline_batch_size > 0:
        offline_rb = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
            alpha=alpha,
            beta=beta,
            eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=gamma_macro),
            pin_memory=True,
            prefetch=cfg.algo.prefetch_batches,
            batch_size=max(offline_batch_size, 1),
        )

    if cfg.algo.offline_fraction > 0.0:
        offline_cache_meta = {
            "task": cfg.task,
            "dataset_name": cfg.offline_data.name,
            "num_episodes": cfg.offline_data.num_episodes,
            "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
            "min_action_range": cfg.offline_data.min_action_range,
            "min_state_std": cfg.offline_data.min_state_std,
            "image_keys": image_keys,
            "macro_horizon": cfg.algo.macro_horizon,
            "base_plan_horizon": cfg.algo.base_plan_horizon,
            "n_step": cfg.algo.n_step,
            "gamma_macro": gamma_macro,
            "base_policy_wandb_id": cfg.base_policy.wandb_id,
            "sampling_strategy": cfg.algo.sampling_strategy,
            "normalized_actions": True,
            "total_batch_size": cfg.algo.batch_size,
            "offline_batch_size": offline_batch_size,
            "buffer_size": cfg.algo.buffer_size,
            "torchrl_version": torchrl.__version__,
        }
        if cfg.algo.sampling_strategy == "prioritized_replay":
            offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
            offline_cache_meta["priority_beta"] = cfg.algo.priority_beta
        pprint.pprint(offline_cache_meta)

        offline_meta_str = json.dumps(offline_cache_meta, sort_keys=True)
        offline_cache_hash = hashlib.sha1(offline_meta_str.encode()).hexdigest()[:8]  # noqa: S324
        offline_cache_dir = CHUNK_OFFLINE_CACHE_DIR / offline_cache_hash
        loaded_offline_from_cache = False

        if offline_rb is None:
            raise RuntimeError("offline replay buffer is not initialized.")
        if offline_cache_dir.exists():
            cached_storage_capacity = _read_cached_storage_capacity(offline_cache_dir)
            if cached_storage_capacity is not None and cached_storage_capacity != cfg.algo.buffer_size:
                print(
                    "Skipping incompatible chunk offline cache: "
                    f"storage_capacity={cached_storage_capacity}, expected_buffer_size={cfg.algo.buffer_size}. "
                    "A new cache will be built."
                )
            else:
                print(f"{offline_cache_dir} found on disk. Attempting to load chunk offline buffer...")
                offline_rb.sampler._empty()
                optimized_replay_buffer_loads(offline_rb, offline_cache_dir)
                loaded_offline_from_cache = True
                print(f"Loaded chunk offline buffer from cache at {offline_cache_dir} (size={len(offline_rb)})")

        if not loaded_offline_from_cache:
            print("Populating chunk offline buffer from dataset...")
            added = populate_chunk_offline_buffer(
                dataset=dataset,
                rb=offline_rb,
                image_keys=image_keys,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
                macro_horizon=cfg.algo.macro_horizon,
                base_plan_horizon=cfg.algo.base_plan_horizon,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=offline_base_policy,
                device=device,
            )
            print(f"Added {added} chunk offline transitions (size={len(offline_rb)})")
            offline_cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, offline_cache_dir)
            with open(offline_cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{cfg.task}_chunk{cfg.algo.macro_horizon}_plan{cfg.algo.base_plan_horizon}__seed{cfg.seed}"
    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(wandb_config, dict)
    wandb_config["wandb"].pop("notes", None)
    wandb_config["algo"]["gamma_macro"] = gamma_macro

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
    wandb.summary["environment/horizon"] = env.vec_env.metadata["horizon"]

    outputs_dir = _CACHE_ROOT / f"chunk_outputs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cfg.task}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    training_timer = TrainingTimer()
    actor_updates = 0
    latest_metrics: dict[str, float] = {}

    obs, _ = env.reset()
    global_step = 0
    best_eval_success_rate = 0.0
    episode_count = 0
    training_cum_time = 0.0
    train_start_time = time.time()
    collected_warmup_steps = 0
    warmup_start_time = time.time()
    warmup_log_interval = max(1, min(cfg.log_freq * 10, max(cfg.algo.learning_starts // 10, 1)))
    next_warmup_log = min(warmup_log_interval, cfg.algo.learning_starts)

    if cfg.algo.learning_starts > 0:
        print(
            "Starting env warmup: "
            f"target_steps={cfg.algo.learning_starts}, "
            f"macro_horizon={cfg.algo.macro_horizon}, "
            f"log_interval={warmup_log_interval}"
        )

    while collected_warmup_steps < cfg.algo.learning_starts:
        if cfg.algo.use_base_policy_for_warmup:
            rand_actions = (
                torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
            ) * cfg.algo.random_action_noise_scale
        else:
            base_action = obs["observation.base_action"]
            pure_random = (
                torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
            ) * cfg.algo.random_action_noise_scale
            rand_actions = pure_random - base_action

        next_obs, reward, terminated, truncated, info = env.step(rand_actions)
        done = terminated | truncated
        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=info["scaled_action"],
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
        )
        collected_warmup_steps += int(info["executed_steps"][0].item())
        obs = next_obs
        if collected_warmup_steps >= next_warmup_log or collected_warmup_steps >= cfg.algo.learning_starts:
            print(
                "Env warmup: "
                f"{_format_progress(collected_warmup_steps, cfg.algo.learning_starts)}, "
                f"replay_size={len(online_rb)}"
            )
            next_warmup_log += warmup_log_interval

    if cfg.algo.learning_starts > 0:
        print(
            "Env warmup complete: "
            f"steps={collected_warmup_steps}, "
            f"replay_size={len(online_rb)}, "
            f"elapsed={time.time() - warmup_start_time:.1f}s"
        )

    def _run_critic_warmup():
        print(
            "Starting critic warmup: "
            f"updates={cfg.algo.critic_warmup_steps}, "
            f"batch_size={cfg.algo.batch_size}"
        )
        for i in range(cfg.algo.critic_warmup_steps):
            with training_timer.time("batch_sampling"):
                batch = _sample_mixed_batch(
                    online_rb=online_rb,
                    online_batch_size=online_batch_size,
                    offline_rb=offline_rb,
                    offline_batch_size=offline_batch_size,
                    device=device,
                )
            with training_timer.time("gradient_update"):
                metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                batch["_priority"] = metrics["_td_errors"]
                if online_batch_size > 0:
                    online_rb.update_tensordict_priority(batch[:online_batch_size])
                if offline_rb is not None and offline_batch_size > 0:
                    offline_rb.update_tensordict_priority(batch[online_batch_size:])
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )
        print(f"Critic warmup complete: updates={cfg.algo.critic_warmup_steps}")

    if cfg.algo.critic_warmup_steps > 0:
        _run_critic_warmup()

    def _run_evaluation(step_value: int) -> None:
        nonlocal best_eval_success_rate
        print(f"Starting evaluation: step={step_value}, episodes={cfg.eval_num_episodes}")
        eval_metrics = run_dexmg_evaluation(
            env=eval_env,
            agent=agent,
            num_episodes=cfg.eval_num_episodes,
            device=device,
            global_step=step_value,
            save_video=cfg.save_video,
            save_q_plots=cfg.save_video,
            run_name=run_name,
            output_dir=outputs_dir,
        )
        current_success_rate = eval_metrics["eval/success_rate"]
        print(
            "Evaluation complete: "
            f"step={step_value}, "
            f"success_rate={current_success_rate:.4f}, "
            f"mean_return={eval_metrics['eval/mean_return']:.4f}"
        )
        if current_success_rate > best_eval_success_rate:
            best_eval_success_rate = current_success_rate

    if cfg.eval_first:
        with training_timer.time("evaluation"):
            _run_evaluation(0)
    next_eval_step = cfg.eval_interval_every_steps
    next_log_step = cfg.log_freq

    while global_step <= cfg.algo.total_timesteps:
        iter_start = time.time()
        prev_global_step = global_step

        with training_timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
            if cfg.algo.progressive_clipping_steps > 0:
                clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                action = action * clip_factor
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

        executed_steps = int(info["executed_steps"][0].item())
        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=info["scaled_action"],
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
        )
        obs = next_obs
        global_step += executed_steps

        if done.any():
            episode_count += done.float().sum().item()
            log_dict = {
                "training/episode_success": reward[0].item(),
                "training/episode_count": episode_count,
                "training/executed_steps": executed_steps,
            }
            if "final_info" in info and isinstance(info["final_info"], dict) and "episode_steps" in info["final_info"]:
                episode_steps = info["final_info"]["episode_steps"]
                if isinstance(episode_steps, torch.Tensor):
                    log_dict["training/episode_steps"] = episode_steps.float().mean().item()
                elif isinstance(episode_steps, np.ndarray):
                    log_dict["training/episode_steps"] = float(np.mean(episode_steps))
                else:
                    log_dict["training/episode_steps"] = float(episode_steps)
            wandb.log(log_dict, step=global_step)

        if global_step >= next_eval_step:
            with training_timer.time("evaluation"):
                _run_evaluation(global_step)
            while next_eval_step <= global_step:
                next_eval_step += cfg.eval_interval_every_steps

        update_triggers = (global_step // cfg.algo.update_every_n_steps) - (
            prev_global_step // cfg.algo.update_every_n_steps
        )
        for _ in range(max(update_triggers, 0)):
            i = 0
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            while i < cfg.algo.num_updates_per_iteration:
                with training_timer.time("batch_sampling"):
                    batch = _sample_mixed_batch(
                        online_rb=online_rb,
                        online_batch_size=online_batch_size,
                        offline_rb=offline_rb,
                        offline_batch_size=offline_batch_size,
                        device=device,
                    )
                update_actor = (i + 1) % actor_update_cadence == 0
                if update_actor and cfg.algo.actor_lr_warmup_steps > 0:
                    warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                    current_lr = cfg.agent.actor_lr * warmup_progress
                    for param_group in agent.actor_opt.param_groups:
                        param_group["lr"] = current_lr
                if update_actor:
                    actor_updates += 1

                with training_timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)
                if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    batch["_priority"] = metrics["_td_errors"]
                    if online_batch_size > 0:
                        online_rb.update_tensordict_priority(batch[:online_batch_size])
                    if offline_rb is not None and offline_batch_size > 0:
                        offline_rb.update_tensordict_priority(batch[online_batch_size:])
                if (~batch["nonterminal"]).any():
                    metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean().item()
                else:
                    metrics["data/batch_terminal_R"] = 0.0
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean().item()
                latest_metrics = metrics
                i += 1

        training_cum_time += time.time() - iter_start
        if global_step >= next_log_step and latest_metrics:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "replay/online_size": len(online_rb),
                "replay/offline_size": len(offline_rb) if offline_rb is not None else 0,
                "algo/gamma_macro": gamma_macro,
            }
            for key, value in latest_metrics.items():
                if not key.startswith("_"):
                    log_dict[key] = value
            log_dict.update(training_timer.get_timing_stats())
            wandb.log(log_dict, step=global_step)
            actor_loss_total = log_dict.get("train/actor_loss_total")
            actor_loss_str = f"{actor_loss_total:.4f}" if actor_loss_total is not None else "not_updated"
            print(
                "Train log: "
                f"step={global_step}, "
                f"online_replay_size={len(online_rb)}, "
                f"offline_replay_size={len(offline_rb) if offline_rb is not None else 0}, "
                f"SPS={sps}, "
                f"critic_loss={log_dict.get('train/critic_loss', float('nan')):.4f}, "
                f"actor_loss_total={actor_loss_str}, "
                f"terminal_share={log_dict.get('data/terminal_share', float('nan')):.4f}"
            )
            while next_log_step <= global_step:
                next_log_step += cfg.log_freq
            training_timer.reset()

    print(f"Chunk residual TD3 training finished in {time.time() - train_start_time:.2f} seconds.")


@hydra.main(version_base=None, config_name="residual_td3_chunk_dexmg_config")
def hydra_entry(cfg: ResidualTD3ChunkDexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
