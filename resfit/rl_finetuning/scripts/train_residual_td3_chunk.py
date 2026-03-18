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
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchrl
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader, Subset
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.lerobot.utils.openpi_chunk_policy import OpenPIChunkPolicy
from resfit.lerobot.utils.task_prompts import infer_task_prompt
from resfit.rl_finetuning.config.residual_td3_chunk import ResidualTD3ChunkDexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.checkpoint import save_checkpoint
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.hugging_face import optimized_replay_buffer_dumps, optimized_replay_buffer_loads
from resfit.rl_finetuning.utils.libero_eval_env import (
    LiberoEvalVecEnvWrapper,
    ensure_libero_available,
    get_libero_max_steps,
    get_libero_task_suite,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.residual_chunk_env_wrapper import ResidualChunkVecEnvWrapper

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
OPENPI_IMAGE_KEY_MAP = {
    "image": "observation.images.agentview",
    "wrist_image": "observation.images.robot0_eye_in_hand",
}


def _get_stats_feature_keys(dataset_schema: str) -> tuple[str, str]:
    if dataset_schema == "resfit":
        return "action", "observation.state"
    if dataset_schema == "openpi":
        return "actions", "state"
    raise ValueError(f"Unsupported offline dataset_schema={dataset_schema!r}.")


def _get_selected_episode_bounds(*, episode_start: int, num_episodes: int | None) -> tuple[int, int | None]:
    episode_end = None if num_episodes is None else episode_start + num_episodes
    return episode_start, episode_end


def _episode_is_selected(*, episode_index: int, episode_start: int, episode_end: int | None) -> bool:
    if episode_index < episode_start:
        return False
    if episode_end is not None and episode_index >= episode_end:
        return False
    return True


def _resolve_local_dataset_root(dataset_name: str) -> Path | None:
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()
    return None


def _load_episode_stats_records(dataset_root: Path) -> list[dict[str, Any]]:
    episodes_stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        raise FileNotFoundError(f"Episode stats file not found: {episodes_stats_path}")
    records: list[dict[str, Any]] = []
    with open(episodes_stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_episode_records(dataset_root: Path) -> list[dict[str, Any]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episode metadata file not found: {episodes_path}")
    records: list[dict[str, Any]] = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _maybe_slice_dataset_by_episode_range(
    *,
    dataset: LeRobotDataset,
    dataset_name: str,
    episode_start: int,
    episode_end: int | None,
) -> LeRobotDataset | Subset:
    dataset_root = _resolve_local_dataset_root(dataset_name)
    if dataset_root is None:
        return dataset

    try:
        episode_records = _load_episode_records(dataset_root)
    except FileNotFoundError:
        return dataset

    frame_start: int | None = None
    frame_end: int | None = None
    frame_offset = 0

    for record in episode_records:
        episode_index = int(record["episode_index"])
        episode_length = int(record["length"])

        if frame_start is None and episode_index >= episode_start:
            frame_start = frame_offset
        if episode_end is not None and episode_index >= episode_end:
            frame_end = frame_offset
            break
        frame_offset += episode_length

    if frame_start is None:
        raise ValueError(f"episode_start={episode_start} is outside the dataset episode range.")
    if frame_end is None:
        frame_end = frame_offset

    return Subset(dataset, range(frame_start, frame_end))


def _extract_episode_stat_count(value: Any) -> int:
    count_array = np.asarray(value).reshape(-1)
    if count_array.size == 0:
        raise ValueError("Encountered empty count array in episode stats.")
    return int(count_array[0])


def _aggregate_episode_feature_stats(records: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No episode stats records available for feature {feature_key!r}.")

    total_count = 0
    feature_min: np.ndarray | None = None
    feature_max: np.ndarray | None = None
    feature_sum: np.ndarray | None = None
    feature_sum_sq: np.ndarray | None = None

    for record in records:
        feature_stats = record["stats"][feature_key]
        count = _extract_episode_stat_count(feature_stats["count"])
        mean = np.asarray(feature_stats["mean"], dtype=np.float64)
        std = np.asarray(feature_stats["std"], dtype=np.float64)
        current_min = np.asarray(feature_stats["min"], dtype=np.float64)
        current_max = np.asarray(feature_stats["max"], dtype=np.float64)

        if feature_sum is None:
            feature_sum = np.zeros_like(mean, dtype=np.float64)
            feature_sum_sq = np.zeros_like(mean, dtype=np.float64)
            feature_min = current_min.copy()
            feature_max = current_max.copy()
        else:
            assert feature_sum_sq is not None and feature_min is not None and feature_max is not None
            feature_min = np.minimum(feature_min, current_min)
            feature_max = np.maximum(feature_max, current_max)

        feature_sum += mean * count
        feature_sum_sq += (np.square(std) + np.square(mean)) * count
        total_count += count

    if total_count <= 0:
        raise ValueError(f"Invalid aggregated count={total_count} for feature {feature_key!r}.")

    assert feature_sum is not None and feature_sum_sq is not None and feature_min is not None and feature_max is not None
    aggregated_mean = feature_sum / total_count
    aggregated_var = np.maximum(feature_sum_sq / total_count - np.square(aggregated_mean), 0.0)
    aggregated_std = np.sqrt(aggregated_var)

    return {
        "min": feature_min.tolist(),
        "max": feature_max.tolist(),
        "mean": aggregated_mean.tolist(),
        "std": aggregated_std.tolist(),
        "count": [total_count],
    }


def _resolve_dataset_stats(
    *,
    dataset: LeRobotDataset,
    dataset_name: str,
    dataset_schema: str,
    episode_start: int,
    num_episodes: int | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    action_stats_key, state_stats_key = _get_stats_feature_keys(dataset_schema)
    dataset_stats = dict(dataset.meta.stats)

    # For OpenPI's mixed local dataset, prefer task-subset stats over global stats.
    if dataset_schema == "openpi" and (episode_start > 0 or num_episodes is not None):
        dataset_root = _resolve_local_dataset_root(dataset_name)
        if dataset_root is not None:
            try:
                episode_start_idx, episode_end_idx = _get_selected_episode_bounds(
                    episode_start=episode_start,
                    num_episodes=num_episodes,
                )
                selected_records = [
                    record
                    for record in _load_episode_stats_records(dataset_root)
                    if _episode_is_selected(
                        episode_index=int(record["episode_index"]),
                        episode_start=episode_start_idx,
                        episode_end=episode_end_idx,
                    )
                ]
                if not selected_records:
                    raise ValueError(
                        f"No episode stats found in [{episode_start_idx}, {episode_end_idx}) for dataset {dataset_root}."
                    )
                dataset_stats[action_stats_key] = _aggregate_episode_feature_stats(selected_records, action_stats_key)
                dataset_stats[state_stats_key] = _aggregate_episode_feature_stats(selected_records, state_stats_key)
            except Exception as exc:  # noqa: BLE001
                print(f"Falling back to global dataset stats because subset aggregation failed: {exc}")

    return dataset_stats, dataset_stats[action_stats_key], dataset_stats[state_stats_key]


def _squeeze_loader_sample(sample: dict[str, Any]) -> dict[str, Any]:
    squeezed: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            squeezed[key] = value.squeeze(0).cpu()
        elif isinstance(value, list) and len(value) == 1:
            squeezed[key] = value[0]
        else:
            squeezed[key] = value
    return squeezed


def _canonicalize_offline_sample(sample: dict[str, Any], dataset_schema: str) -> dict[str, Any]:
    if dataset_schema == "resfit":
        return sample
    if dataset_schema != "openpi":
        raise ValueError(f"Unsupported offline dataset_schema={dataset_schema!r}.")

    canonical: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "state":
            canonical["observation.state"] = value
        elif key == "actions":
            canonical["action"] = value
        elif key in OPENPI_IMAGE_KEY_MAP:
            canonical[OPENPI_IMAGE_KEY_MAP[key]] = value
        else:
            canonical[key] = value
    return canonical


def _resize_image_with_pad(image: torch.Tensor, target_size: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {tuple(image.shape)}.")

    channel_first = image.shape[0] in (1, 3)
    if channel_first:
        chw = image
    elif image.shape[-1] in (1, 3):
        chw = image.permute(2, 0, 1)
    else:
        raise ValueError(f"Could not infer image layout for shape {tuple(image.shape)}.")

    if chw.shape[1] == target_size and chw.shape[2] == target_size:
        return chw if channel_first else chw.permute(1, 2, 0)

    orig_dtype = chw.dtype
    chw_float = chw.float().unsqueeze(0)
    _, _, src_h, src_w = chw_float.shape
    scale = min(float(target_size) / float(src_h), float(target_size) / float(src_w))
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    resized = F.interpolate(chw_float, size=(resized_h, resized_w), mode="bilinear", align_corners=False)

    canvas = torch.zeros(
        (1, chw_float.shape[1], target_size, target_size),
        dtype=resized.dtype,
        device=resized.device,
    )
    offset_h = (target_size - resized_h) // 2
    offset_w = (target_size - resized_w) // 2
    canvas[:, :, offset_h : offset_h + resized_h, offset_w : offset_w + resized_w] = resized
    result = canvas.squeeze(0)

    if orig_dtype == torch.uint8:
        result = result.round().clamp(0, 255).to(torch.uint8)
    else:
        result = result.to(orig_dtype)

    return result if channel_first else result.permute(1, 2, 0)


def _resize_offline_sample_images(sample: dict[str, Any], image_keys: list[str], resize_size: int) -> dict[str, Any]:
    resized = dict(sample)
    for key in image_keys:
        value = resized.get(key)
        if isinstance(value, torch.Tensor):
            resized[key] = _resize_image_with_pad(value, resize_size)
    return resized


def _load_base_policy(
    *,
    base_policy_cfg: Any,
    dataset_stats: dict[str, Any],
    default_task: str,
    device: torch.device,
) -> tuple[Any, str]:
    source = getattr(base_policy_cfg, "source", "wandb")
    load_policy_kwargs = {
        "dataset_stats": dataset_stats,
        "default_task": default_task,
    }

    if source == "wandb":
        policy_dir, _ = download_policy_from_wandb(
            base_policy_cfg.wandb_id,
            step=base_policy_cfg.wt_type,
            artifact_version=base_policy_cfg.wt_version,
        )
        policy = load_policy(policy_dir, **load_policy_kwargs)
        policy_ref = str(policy_dir)
    elif source == "local":
        if not base_policy_cfg.local_path:
            raise ValueError("base_policy.local_path must be set when base_policy.source=local.")
        policy_dir = Path(base_policy_cfg.local_path).expanduser().resolve()
        if not policy_dir.exists():
            raise FileNotFoundError(f"Local policy directory not found: {policy_dir}")
        policy = load_policy(policy_dir, **load_policy_kwargs)
        policy_ref = str(policy_dir)
    elif source == "openpi_ws":
        policy = OpenPIChunkPolicy(
            host=base_policy_cfg.openpi_host,
            port=base_policy_cfg.openpi_port,
            task_prompt=default_task,
            chunk_size=base_policy_cfg.openpi_chunk_size,
            api_key=getattr(base_policy_cfg, "openpi_api_key", None),
        )
        policy_ref = f"ws://{base_policy_cfg.openpi_host}:{base_policy_cfg.openpi_port}"
    else:
        raise ValueError(f"Unsupported base_policy.source={source!r}. Expected wandb, local, or openpi_ws.")

    policy.to(device)
    policy.eval()
    return policy, policy_ref


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
    base_policy: Any,
    device: str,
    video_key: str,
    debug: bool,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    chunk_success_threshold: float,
    camera_size: int,
    state_encoding: str,
):
    vec_env = create_vectorized_env(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        camera_size=camera_size,
        video_key=video_key,
        debug=debug,
        state_encoding=state_encoding,
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


def _chunk_cache_is_complete(cache_dir: Path) -> bool:
    required_paths = (
        cache_dir / "buffer_metadata.json",
        cache_dir / "storage" / "meta.json",
        cache_dir / "storage" / "storage_metadata.json",
    )
    return all(path.exists() for path in required_paths)


def _pad_action_chunk(actions: list[torch.Tensor], chunk_len: int, action_dim: int) -> torch.Tensor:
    if not actions:
        return torch.zeros(chunk_len, action_dim, dtype=torch.float32)
    stacked = torch.stack(actions, dim=0)
    if stacked.shape[0] >= chunk_len:
        return stacked[:chunk_len]
    pad = stacked[-1:].repeat(chunk_len - stacked.shape[0], 1)
    return torch.cat([stacked, pad], dim=0)


def _resolve_libero_task(
    *,
    task_suite_name: str,
    task_id: int,
) -> tuple[Any, Any, np.ndarray, str]:
    ensure_libero_available()

    task_suite = get_libero_task_suite(task_suite_name)
    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(f"libero task_id must be in [0, {task_suite.n_tasks - 1}], got {task_id}.")

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    task_description = str(task.language)
    return task_suite, task, initial_states, task_description


def _create_libero_chunk_env_from_base_policy(
    *,
    task: Any,
    initial_states: np.ndarray,
    task_description: str,
    num_steps_wait: int,
    env_resolution: int,
    max_steps: int | None,
    resize_size: int,
    seed: int,
    base_policy: Any,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    chunk_success_threshold: float,
    device: torch.device,
    video_key: str,
    task_suite_name: str,
) -> ResidualChunkVecEnvWrapper:
    base_chunk_size = getattr(base_policy.config, "chunk_size", None)
    if base_chunk_size is None:
        raise ValueError(f"Base policy does not expose a chunk_size: {type(base_policy.config)}")
    if base_plan_horizon > base_chunk_size:
        raise ValueError(
            f"base_plan_horizon={base_plan_horizon} exceeds LIBERO base chunk_size={base_chunk_size}."
        )

    libero_max_steps = max_steps if max_steps is not None else get_libero_max_steps(task_suite_name)
    vec_env = LiberoEvalVecEnvWrapper(
        task=task,
        initial_states=initial_states,
        task_description=task_description,
        resize_size=resize_size,
        max_steps=libero_max_steps,
        num_steps_wait=num_steps_wait,
        seed=seed,
        device=device,
        env_resolution=env_resolution,
        video_key=video_key,
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


def _create_libero_eval_env(
    *,
    task_suite_name: str,
    task_id: int,
    num_steps_wait: int,
    env_resolution: int,
    max_steps: int | None,
    resize_size: int,
    seed: int,
    base_policy_cfg: Any,
    dataset_stats: dict[str, Any],
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    chunk_success_threshold: float,
    device: torch.device,
    video_key: str,
    default_task_prompt: str | None = None,
) -> tuple[ResidualChunkVecEnvWrapper, str, str]:
    _, task, initial_states, task_description = _resolve_libero_task(
        task_suite_name=task_suite_name,
        task_id=task_id,
    )
    task_prompt = default_task_prompt or task_description

    base_policy, base_policy_ref = _load_base_policy(
        base_policy_cfg=base_policy_cfg,
        dataset_stats=dataset_stats,
        default_task=task_prompt,
        device=device,
    )
    env = _create_libero_chunk_env_from_base_policy(
        task=task,
        initial_states=initial_states,
        task_description=task_description,
        num_steps_wait=num_steps_wait,
        env_resolution=env_resolution,
        max_steps=max_steps,
        resize_size=resize_size,
        seed=seed,
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=macro_horizon,
        base_plan_horizon=base_plan_horizon,
        chunk_success_threshold=chunk_success_threshold,
        device=device,
        video_key=video_key,
        task_suite_name=task_suite_name,
    )
    return env, task_description, base_policy_ref


def _run_chunk_policy_evaluation(
    *,
    env: ResidualChunkVecEnvWrapper,
    agent: QAgent,
    num_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}.")

    successes = 0
    episodes_done = 0
    obs, _ = env.reset()

    with utils.eval_mode(agent):
        while episodes_done < num_episodes:
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                if action.dim() == 1:
                    action = action.unsqueeze(0)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            if bool(done[0].item()):
                episodes_done += 1
                successes += int(float(reward[0].item()) >= 1.0)

    success_rate = float(successes) / float(episodes_done)
    return {
        "eval/mean_return": success_rate,
        "eval/success_rate": success_rate,
        "eval/successes": float(successes),
        "eval/episodes": float(episodes_done),
    }


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
    base_policy: Any,
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
    base_policy: Any | None,
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
            episode_sample = episode_samples[step_idx + offset]
            is_explicit_terminal = bool(episode_sample.get("next.done", False))
            is_final_sample = step_idx + offset == len(episode_samples) - 1
            if is_explicit_terminal or is_final_sample:
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
    dataset_name: str,
    rb: TensorDictPrioritizedReplayBuffer,
    image_keys: list[str],
    resize_size: int,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    macro_horizon: int,
    base_plan_horizon: int,
    episode_start: int,
    num_episodes: int | None,
    dataset_schema: str,
    use_base_policy_for_base_actions: bool,
    base_policy: Any | None,
    device: torch.device,
) -> int:
    if use_base_policy_for_base_actions and base_policy is None:
        raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

    episode_start_idx, episode_end_idx = _get_selected_episode_bounds(
        episode_start=episode_start,
        num_episodes=num_episodes,
    )
    subset = _maybe_slice_dataset_by_episode_range(
        dataset=dataset,
        dataset_name=dataset_name,
        episode_start=episode_start,
        episode_end=episode_end_idx,
    )
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    current_episode: int | None = None
    episode_samples: list[dict] = []
    transitions = 0

    for sample in tqdm(loader, desc="Processing chunk offline dataset"):
        ep_idx = int(sample["episode_index"].item())
        if ep_idx < episode_start_idx:
            continue
        if episode_end_idx is not None and ep_idx >= episode_end_idx:
            break

        squeezed = _canonicalize_offline_sample(_squeeze_loader_sample(sample), dataset_schema)
        squeezed = _resize_offline_sample_images(squeezed, image_keys, resize_size)

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
    if cfg.env_backend not in {"dexmg", "libero"}:
        raise ValueError(f"env_backend must be one of ['dexmg', 'libero'], got {cfg.env_backend!r}.")
    if not 0.0 <= cfg.algo.offline_fraction <= 1.0:
        raise ValueError("algo.offline_fraction must be in [0, 1].")
    if cfg.algo.offline_fraction > 0.0 and cfg.offline_data is None:
        raise ValueError("offline_data must be provided when algo.offline_fraction > 0.")
    if cfg.libero_eval.enabled:
        if cfg.libero_eval.num_episodes <= 0:
            raise ValueError("libero_eval.num_episodes must be positive when LIBERO eval is enabled.")
        if cfg.libero_eval.interval_every_steps <= 0:
            raise ValueError("libero_eval.interval_every_steps must be positive when LIBERO eval is enabled.")

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

    dataset = LeRobotDataset(cfg.offline_data.name)
    dataset_stats, action_stats, state_stats = _resolve_dataset_stats(
        dataset=dataset,
        dataset_name=cfg.offline_data.name,
        dataset_schema=cfg.offline_data.dataset_schema,
        episode_start=cfg.offline_data.episode_start,
        num_episodes=cfg.offline_data.num_episodes,
    )
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=action_stats,
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=state_stats,
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    train_libero_task_description: str | None = None
    run_task_name = cfg.task
    if cfg.env_backend == "libero":
        _, _, _, train_libero_task_description = _resolve_libero_task(
            task_suite_name=cfg.libero_env.task_suite_name,
            task_id=cfg.libero_env.task_id,
        )
        base_task_prompt = cfg.base_policy.task_prompt or train_libero_task_description
        run_task_name = f"{cfg.libero_env.task_suite_name}_task{cfg.libero_env.task_id}"
    else:
        base_task_prompt = infer_task_prompt(
            task_name=cfg.task,
            dataset_name=cfg.offline_data.name,
            explicit_prompt=cfg.base_policy.task_prompt,
        )
    base_policy, base_policy_ref = _load_base_policy(
        base_policy_cfg=cfg.base_policy,
        dataset_stats=dataset_stats,
        default_task=base_task_prompt,
        device=device,
    )
    eval_base_policy, _ = _load_base_policy(
        base_policy_cfg=cfg.base_policy,
        dataset_stats=dataset_stats,
        default_task=base_task_prompt,
        device=device,
    )

    base_chunk_size = getattr(base_policy.config, "chunk_size", None)
    if base_chunk_size is None:
        raise ValueError(f"Base policy does not expose a chunk_size: {type(base_policy.config)}")
    if cfg.algo.base_plan_horizon > base_chunk_size:
        raise ValueError(
            f"base_plan_horizon={cfg.algo.base_plan_horizon} exceeds base chunk_size={base_chunk_size}."
        )
    offline_base_policy: Any | None = None
    if cfg.algo.offline_fraction > 0.0 and cfg.offline_data is not None and cfg.offline_data.use_base_policy_for_base_actions:
        offline_base_policy, _ = _load_base_policy(
            base_policy_cfg=cfg.base_policy,
            dataset_stats=dataset_stats,
            default_task=base_task_prompt,
            device=device,
        )

    cfg.eval_num_envs = 1
    if cfg.env_backend == "libero":
        _, train_task, train_initial_states, train_task_description = _resolve_libero_task(
            task_suite_name=cfg.libero_env.task_suite_name,
            task_id=cfg.libero_env.task_id,
        )
        env = _create_libero_chunk_env_from_base_policy(
            task=train_task,
            initial_states=train_initial_states,
            task_description=train_task_description,
            num_steps_wait=cfg.libero_env.num_steps_wait,
            env_resolution=cfg.libero_env.env_resolution,
            max_steps=cfg.libero_env.max_steps,
            resize_size=cfg.env_camera_size,
            seed=cfg.seed,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            macro_horizon=cfg.algo.macro_horizon,
            base_plan_horizon=cfg.algo.base_plan_horizon,
            chunk_success_threshold=cfg.algo.chunk_success_threshold,
            device=device,
            video_key=cfg.video_key,
            task_suite_name=cfg.libero_env.task_suite_name,
        )
        eval_env = _create_libero_chunk_env_from_base_policy(
            task=train_task,
            initial_states=train_initial_states,
            task_description=train_task_description,
            num_steps_wait=cfg.libero_env.num_steps_wait,
            env_resolution=cfg.libero_env.env_resolution,
            max_steps=cfg.libero_env.max_steps,
            resize_size=cfg.env_camera_size,
            seed=cfg.seed + 1,
            base_policy=eval_base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            macro_horizon=cfg.algo.macro_horizon,
            base_plan_horizon=cfg.algo.base_plan_horizon,
            chunk_success_threshold=cfg.algo.chunk_success_threshold,
            device=device,
            video_key=cfg.video_key,
            task_suite_name=cfg.libero_env.task_suite_name,
        )
    else:
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
            camera_size=cfg.env_camera_size,
            state_encoding=cfg.env_state_encoding,
        )
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
            camera_size=cfg.env_camera_size,
            state_encoding=cfg.env_state_encoding,
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
    libero_eval_env: ResidualChunkVecEnvWrapper | None = None
    libero_eval_task_description: str | None = None
    libero_eval_base_policy_ref: str | None = None
    if cfg.libero_eval.enabled:
        libero_eval_env, libero_eval_task_description, libero_eval_base_policy_ref = _create_libero_eval_env(
            task_suite_name=cfg.libero_eval.task_suite_name,
            task_id=cfg.libero_eval.task_id,
            num_steps_wait=cfg.libero_eval.num_steps_wait,
            env_resolution=cfg.libero_eval.env_resolution,
            max_steps=cfg.libero_eval.max_steps,
            resize_size=cfg.env_camera_size,
            seed=cfg.seed,
            base_policy_cfg=cfg.base_policy,
            dataset_stats=dataset_stats,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            macro_horizon=cfg.algo.macro_horizon,
            base_plan_horizon=cfg.algo.base_plan_horizon,
            chunk_success_threshold=cfg.algo.chunk_success_threshold,
            device=device,
            video_key=cfg.video_key,
        )
        if libero_eval_env.action_space.shape[1] != action_dim:
            raise ValueError(
                f"LIBERO eval action_dim={libero_eval_env.action_space.shape[1]} does not match training action_dim={action_dim}."
            )
        if libero_eval_env.observation_space["observation.state"].shape[1] != lowdim_dim:
            raise ValueError(
                "LIBERO eval state dim does not match training state dim: "
                f"{libero_eval_env.observation_space['observation.state'].shape[1]} vs {lowdim_dim}."
            )
        libero_obs_spaces = libero_eval_env.observation_space.spaces
        for image_key in image_keys:
            if image_key not in libero_obs_spaces:
                available_keys = sorted(libero_obs_spaces.keys())
                raise KeyError(
                    f"LIBERO eval env is missing rl_camera key {image_key!r}. Available keys: {available_keys}"
                )
        libero_img_shape = libero_eval_env.observation_space[image_keys[0]].shape[1:]
        if tuple(libero_img_shape) != (img_c, img_h, img_w):
            raise ValueError(
                "LIBERO eval image shape does not match training image shape: "
                f"{tuple(libero_img_shape)} vs {(img_c, img_h, img_w)}."
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
            "dataset_schema": cfg.offline_data.dataset_schema,
            "episode_start": cfg.offline_data.episode_start,
            "num_episodes": cfg.offline_data.num_episodes,
            "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
            "min_action_range": cfg.offline_data.min_action_range,
            "min_state_std": cfg.offline_data.min_state_std,
            "image_keys": image_keys,
            "env_camera_size": cfg.env_camera_size,
            "env_state_encoding": cfg.env_state_encoding,
            "macro_horizon": cfg.algo.macro_horizon,
            "base_plan_horizon": cfg.algo.base_plan_horizon,
            "n_step": cfg.algo.n_step,
            "gamma_macro": gamma_macro,
            "base_policy_source": cfg.base_policy.source,
            "base_policy_ref": base_policy_ref,
            "base_policy_task_prompt": base_task_prompt,
            "sampling_strategy": cfg.algo.sampling_strategy,
            "normalized_actions": True,
            "total_batch_size": cfg.algo.batch_size,
            "offline_batch_size": offline_batch_size,
            "buffer_size": cfg.algo.buffer_size,
            "torchrl_version": torchrl.__version__,
            "offline_cache_format_version": 2,
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
            if not _chunk_cache_is_complete(offline_cache_dir):
                print(
                    "Skipping incomplete chunk offline cache: "
                    f"{offline_cache_dir}. A new cache will be built."
                )
                shutil.rmtree(offline_cache_dir, ignore_errors=True)
            elif cached_storage_capacity is not None and cached_storage_capacity != cfg.algo.buffer_size:
                print(
                    "Skipping incompatible chunk offline cache: "
                    f"storage_capacity={cached_storage_capacity}, expected_buffer_size={cfg.algo.buffer_size}. "
                    "A new cache will be built."
                )
            else:
                print(f"{offline_cache_dir} found on disk. Attempting to load chunk offline buffer...")
                try:
                    offline_rb.sampler._empty()
                    optimized_replay_buffer_loads(offline_rb, offline_cache_dir)
                    loaded_offline_from_cache = True
                    print(f"Loaded chunk offline buffer from cache at {offline_cache_dir} (size={len(offline_rb)})")
                except Exception as exc:  # noqa: BLE001
                    print(
                        "Failed to load chunk offline cache, deleting and rebuilding: "
                        f"{offline_cache_dir} ({exc})"
                    )
                    shutil.rmtree(offline_cache_dir, ignore_errors=True)

        if not loaded_offline_from_cache:
            print("Populating chunk offline buffer from dataset...")
            added = populate_chunk_offline_buffer(
                dataset=dataset,
                dataset_name=cfg.offline_data.name,
                rb=offline_rb,
                image_keys=image_keys,
                resize_size=cfg.env_camera_size,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
                macro_horizon=cfg.algo.macro_horizon,
                base_plan_horizon=cfg.algo.base_plan_horizon,
                episode_start=cfg.offline_data.episode_start,
                num_episodes=cfg.offline_data.num_episodes,
                dataset_schema=cfg.offline_data.dataset_schema,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=offline_base_policy,
                device=device,
            )
            print(f"Added {added} chunk offline transitions (size={len(offline_rb)})")
            offline_cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, offline_cache_dir)
            with open(offline_cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{run_task_name}_chunk{cfg.algo.macro_horizon}_plan{cfg.algo.base_plan_horizon}__seed{cfg.seed}"
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
    if cfg.libero_eval.enabled:
        wandb.summary["libero_eval/task_suite_name"] = cfg.libero_eval.task_suite_name
        wandb.summary["libero_eval/task_id"] = cfg.libero_eval.task_id
        wandb.summary["libero_eval/task_description"] = libero_eval_task_description
        wandb.summary["libero_eval/base_policy_ref"] = libero_eval_base_policy_ref
    if cfg.env_backend == "libero":
        wandb.summary["libero_env/task_suite_name"] = cfg.libero_env.task_suite_name
        wandb.summary["libero_env/task_id"] = cfg.libero_env.task_id
        wandb.summary["libero_env/task_description"] = train_libero_task_description

    outputs_dir = _CACHE_ROOT / f"chunk_outputs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_task_name}"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_dir = outputs_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    training_timer = TrainingTimer()
    actor_updates = 0
    latest_metrics: dict[str, float] = {}

    obs, _ = env.reset()
    global_step = 0
    best_eval_success_rate = 0.0
    last_eval_success_rate: float | None = None
    best_libero_eval_success_rate = 0.0
    last_libero_eval_success_rate: float | None = None
    episode_count = 0
    training_cum_time = 0.0
    train_start_time = time.time()
    collected_warmup_steps = 0
    warmup_start_time = time.time()
    warmup_log_interval = max(1, min(cfg.log_freq * 10, max(cfg.algo.learning_starts // 10, 1)))
    next_warmup_log = min(warmup_log_interval, cfg.algo.learning_starts)
    next_checkpoint_step = cfg.checkpoint_interval if cfg.checkpoint_interval > 0 else None
    next_libero_eval_step = cfg.libero_eval.interval_every_steps if cfg.libero_eval.enabled else None

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
        nonlocal best_eval_success_rate, last_eval_success_rate
        print(f"Starting evaluation: step={step_value}, episodes={cfg.eval_num_episodes}")
        if cfg.env_backend == "libero":
            eval_metrics = _run_chunk_policy_evaluation(
                env=eval_env,
                agent=agent,
                num_episodes=cfg.eval_num_episodes,
                device=device,
            )
        else:
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
        last_eval_success_rate = current_success_rate
        if current_success_rate > best_eval_success_rate:
            best_eval_success_rate = current_success_rate
            best_ckpt_path = model_dir / "best.pt"
            save_checkpoint(
                agent=agent,
                checkpoint_path=best_ckpt_path,
                global_step=step_value,
                config=cfg,
                success_rate=current_success_rate,
            )
            if wandb.run is not None:
                wandb.save(str(best_ckpt_path))

    def _run_libero_evaluation(step_value: int) -> None:
        nonlocal best_libero_eval_success_rate, last_libero_eval_success_rate
        if libero_eval_env is None:
            return
        print(
            "Starting LIBERO evaluation: "
            f"step={step_value}, "
            f"suite={cfg.libero_eval.task_suite_name}, "
            f"task_id={cfg.libero_eval.task_id}, "
            f"episodes={cfg.libero_eval.num_episodes}"
        )
        eval_metrics = _run_chunk_policy_evaluation(
            env=libero_eval_env,
            agent=agent,
            num_episodes=cfg.libero_eval.num_episodes,
            device=device,
        )
        current_success_rate = eval_metrics["eval/success_rate"]
        log_metrics = {
            "libero_eval/success_rate": current_success_rate,
            "libero_eval/successes": eval_metrics["eval/successes"],
            "libero_eval/episodes": eval_metrics["eval/episodes"],
        }
        wandb.log(log_metrics, step=step_value)
        print(
            "LIBERO evaluation complete: "
            f"step={step_value}, "
            f"success_rate={current_success_rate:.4f}, "
            f"task={libero_eval_task_description}"
        )
        last_libero_eval_success_rate = current_success_rate
        if cfg.libero_eval.save_best_checkpoint and current_success_rate > best_libero_eval_success_rate:
            best_libero_eval_success_rate = current_success_rate
            best_libero_ckpt_path = model_dir / "best_libero.pt"
            save_checkpoint(
                agent=agent,
                checkpoint_path=best_libero_ckpt_path,
                global_step=step_value,
                config=cfg,
                success_rate=current_success_rate,
                libero_eval_task_suite=cfg.libero_eval.task_suite_name,
                libero_eval_task_id=cfg.libero_eval.task_id,
                libero_eval_task_description=libero_eval_task_description,
            )
            if wandb.run is not None:
                wandb.save(str(best_libero_ckpt_path))

    if cfg.eval_first:
        with training_timer.time("evaluation"):
            _run_evaluation(0)
    if cfg.libero_eval.enabled and cfg.libero_eval.eval_first:
        with training_timer.time("evaluation"):
            _run_libero_evaluation(0)
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

        if next_libero_eval_step is not None and global_step >= next_libero_eval_step:
            with training_timer.time("evaluation"):
                _run_libero_evaluation(global_step)
            while next_libero_eval_step <= global_step:
                next_libero_eval_step += cfg.libero_eval.interval_every_steps

        if next_checkpoint_step is not None and global_step >= next_checkpoint_step:
            latest_ckpt_path = model_dir / "latest.pt"
            save_checkpoint(
                agent=agent,
                checkpoint_path=latest_ckpt_path,
                global_step=global_step,
                config=cfg,
                success_rate=last_eval_success_rate,
            )
            step_ckpt_path = model_dir / f"step_{global_step}.pt"
            save_checkpoint(
                agent=agent,
                checkpoint_path=step_ckpt_path,
                global_step=global_step,
                config=cfg,
                success_rate=last_eval_success_rate,
            )
            while next_checkpoint_step <= global_step:
                next_checkpoint_step += cfg.checkpoint_interval

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

    latest_ckpt_path = model_dir / "latest.pt"
    save_checkpoint(
        agent=agent,
        checkpoint_path=latest_ckpt_path,
        global_step=global_step,
        config=cfg,
        success_rate=last_eval_success_rate,
    )
    final_step_ckpt_path = model_dir / f"step_{global_step}.pt"
    save_checkpoint(
        agent=agent,
        checkpoint_path=final_step_ckpt_path,
        global_step=global_step,
        config=cfg,
        success_rate=last_eval_success_rate,
    )
    if libero_eval_env is not None:
        libero_eval_env.close()

    print(f"Chunk residual TD3 training finished in {time.time() - train_start_time:.2f} seconds.")


@hydra.main(version_base=None, config_name="residual_td3_chunk_dexmg_config")
def hydra_entry(cfg: ResidualTD3ChunkDexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
