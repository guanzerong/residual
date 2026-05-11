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
from robosuite.utils.camera_utils import get_camera_transform_matrix
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper, create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
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
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        if depth_cls_cache_fn is not None:
            curr_obs_i.update(depth_cls_cache_fn(curr_obs_i))
            next_obs_i.update(depth_cls_cache_fn(next_obs_i))
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


def _build_fixed_camera_world_to_camera_transform(
    *,
    task: str,
    camera_key: str,
    camera_height: int,
    camera_width: int,
    fixed_camera_only: bool,
) -> torch.Tensor:
    camera_name = camera_key.replace("observation.images.", "", 1)
    lowered_camera_name = camera_name.lower()
    dynamic_camera_markers = ("eye_in_hand", "eye_in_left_hand", "eye_in_right_hand")

    if fixed_camera_only and any(marker in lowered_camera_name for marker in dynamic_camera_markers):
        raise ValueError(
            "Macro local depth gating currently supports fixed cameras only. "
            f"Got camera_key={camera_key!r}."
        )
    if camera_height != camera_width:
        raise ValueError(
            "Fixed-camera projection helper currently expects square RL images because RobosuiteGymWrapper "
            f"only exposes a single camera_size argument. Got HxW={camera_height}x{camera_width}."
        )

    temp_env = RobosuiteGymWrapper(
        env_name=task,
        num_envs=1,
        camera_size=camera_height,
    )
    try:
        world_to_camera = get_camera_transform_matrix(
            sim=temp_env.env.sim,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        )
    finally:
        temp_env.close()

    return torch.tensor(world_to_camera, dtype=torch.float32)


def _natural_demo_sort_key(name: str):
    try:
        return int(str(name).split("_")[-1])
    except ValueError:
        return str(name)


def _load_robomimic_hdf5_stats(dataset_path: str, num_episodes: int | None):
    import h5py

    low_dim_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    action_batches: list[np.ndarray] = []
    state_batches: list[np.ndarray] = []
    episode_lengths: list[int] = []

    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=_natural_demo_sort_key)
        if num_episodes is not None:
            demo_keys = demo_keys[: int(num_episodes)]
        for demo_key in demo_keys:
            demo_grp = f[f"data/{demo_key}"]
            actions = np.asarray(demo_grp["actions"][()], dtype=np.float32)
            obs_grp = demo_grp["obs"]
            state_parts = [np.asarray(obs_grp[key][()], dtype=np.float32) for key in low_dim_keys if key in obs_grp]
            if not state_parts:
                raise ValueError(f"No supported low-dimensional state keys found in {dataset_path}:{demo_key}")
            state = np.concatenate(state_parts, axis=-1).astype(np.float32)
            action_batches.append(actions)
            state_batches.append(state)
            episode_lengths.append(int(actions.shape[0]))

    if not action_batches:
        raise ValueError(f"No demos found in robomimic HDF5 dataset: {dataset_path}")

    actions_all = np.concatenate(action_batches, axis=0)
    states_all = np.concatenate(state_batches, axis=0)

    def _stats(values: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
        }

    return {
        "action": _stats(actions_all),
        "observation.state": _stats(states_all),
    }, episode_lengths


_GROOT_LEROBOT_TASK_INDEX_BY_TASK = {
    "lift": 0,
    "can": 1,
    "square": 2,
    "toolhang": 3,
    "tool_hang": 3,
}


def _task_key(task_name: str) -> str:
    return str(task_name).replace("-", "_").replace(" ", "_").lower()


def _infer_groot_lerobot_task_index(task_name: str) -> int:
    task_key = _task_key(task_name)
    if task_key not in _GROOT_LEROBOT_TASK_INDEX_BY_TASK:
        raise ValueError(
            "offline_data.task_index must be set when using a GR00T/OpenPI-style multi-task dataset "
            f"for task={task_name!r}."
        )
    return int(_GROOT_LEROBOT_TASK_INDEX_BY_TASK[task_key])


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def _groot_rpy_state_to_residual_quat_state(state: np.ndarray) -> np.ndarray:
    """Map GR00T dataset state [xyz, rpy, gripper2] to residual env state [xyz, quat_xyzw, gripper2]."""
    state = np.asarray(state, dtype=np.float32)
    single = state.ndim == 1
    if single:
        state = state[None]
    if state.ndim != 2 or state.shape[-1] != 8:
        raise ValueError(f"Expected GR00T/OpenPI state shape [B, 8], got {tuple(state.shape)}")

    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_euler("XYZ", state[:, 3:6], degrees=False).as_quat().astype(np.float32)
    converted = np.concatenate([state[:, :3], quat_xyzw, state[:, 6:8]], axis=-1).astype(np.float32)
    return converted[0] if single else converted


def _is_groot_lerobot_dataset(dataset: LeRobotDataset, requested_format: str) -> bool:
    if requested_format == "groot_lerobot":
        return True
    if requested_format not in {"auto", "lerobot"}:
        raise ValueError(f"Unsupported offline_data.format={requested_format!r}")
    if requested_format == "lerobot":
        return False
    feature_keys = set(getattr(dataset, "features", {}).keys())
    stats_keys = set(getattr(dataset.meta, "stats", {}).keys())
    return {"image", "wrist_image", "state", "actions", "task_index"}.issubset(feature_keys) and {
        "state",
        "actions",
    }.issubset(stats_keys)


class _SimpleDatasetMeta:
    def __init__(
        self,
        *,
        episodes: dict[int, dict],
        total_frames: int,
        stats: dict[str, dict[str, list[float]]],
        source_meta,
    ) -> None:
        self.episodes = episodes
        self.total_episodes = len(episodes)
        self.total_frames = int(total_frames)
        self.stats = stats
        self.features = getattr(source_meta, "features", {})
        self.fps = getattr(source_meta, "fps", None)
        self.total_tasks = 1


class GROOTLeRobotResidualAdapter:
    """Adapter from GR00T/OpenPI LeRobot fields to this residual TD3 trainer's expected fields."""

    state_key = "state"
    action_key = "actions"
    image_key = "image"
    wrist_image_key = "wrist_image"

    def __init__(
        self,
        source: LeRobotDataset,
        *,
        task_name: str,
        task_index: int | None,
        num_episodes: int | None,
        output_image_keys: list[str],
    ) -> None:
        self.source = source
        self.output_image_keys = list(output_image_keys)
        if not self.output_image_keys:
            raise ValueError("At least one output image key is required for the GR00T LeRobot adapter.")
        self.output_base_image_key = self.output_image_keys[0]
        self.output_wrist_image_key = (
            self.output_image_keys[1] if len(self.output_image_keys) > 1 else "observation.images.robot0_eye_in_hand"
        )

        selected_task_index = int(task_index) if task_index is not None else _infer_groot_lerobot_task_index(task_name)
        self.task_index = selected_task_index

        episode_ids = self._select_episode_ids(selected_task_index)
        if num_episodes is not None:
            episode_ids = episode_ids[: int(num_episodes)]
        if not episode_ids:
            raise ValueError(
                f"No episodes with task_index={selected_task_index} found in {getattr(source, 'root', '<dataset>')}"
            )

        self.source_episode_ids = episode_ids
        self._episode_id_to_local = {source_ep: local_ep for local_ep, source_ep in enumerate(episode_ids)}
        self._source_episode_lengths = {
            source_ep: int(source.meta.episodes[source_ep]["length"]) for source_ep in episode_ids
        }

        starts = source.episode_data_index["from"]
        ends = source.episode_data_index["to"]
        frame_indices: list[int] = []
        episodes_meta: dict[int, dict] = {}
        for local_ep, source_ep in enumerate(episode_ids):
            start = int(starts[source_ep].item())
            end = int(ends[source_ep].item())
            frame_indices.extend(range(start, end))
            source_episode_meta = source.meta.episodes[source_ep]
            episodes_meta[local_ep] = {
                "episode_index": local_ep,
                "source_episode_index": source_ep,
                "tasks": source_episode_meta.get("tasks", []),
                "length": int(source_episode_meta["length"]),
            }

        self._frame_indices = frame_indices
        self._stats, self.episode_lengths = self._compute_stats_from_parquet(episode_ids)
        self.meta = _SimpleDatasetMeta(
            episodes=episodes_meta,
            total_frames=len(frame_indices),
            stats=self._stats,
            source_meta=source.meta,
        )
        self.features = {
            "action": {"shape": (7,)},
            "observation.state": {"shape": (9,)},
            self.output_base_image_key: source.features[self.image_key],
            self.output_wrist_image_key: source.features[self.wrist_image_key],
            "next.done": {"shape": (1,)},
        }

        print(
            "Using GR00T/OpenPI LeRobot adapter: "
            f"task={task_name}, task_index={self.task_index}, episodes={len(episode_ids)}, frames={len(frame_indices)}"
        )

    @property
    def stats(self) -> dict[str, dict[str, list[float]]]:
        return self._stats

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self._frame_indices[int(index)]
        sample = self.source[source_index]

        source_ep = int(torch.as_tensor(sample["episode_index"]).item())
        local_ep = self._episode_id_to_local[source_ep]
        frame_idx = int(torch.as_tensor(sample["frame_index"]).item())
        done = frame_idx >= self._source_episode_lengths[source_ep] - 1

        state = torch.as_tensor(
            _groot_rpy_state_to_residual_quat_state(torch.as_tensor(sample[self.state_key]).cpu().numpy()),
            dtype=torch.float32,
        )
        action = torch.as_tensor(sample[self.action_key], dtype=torch.float32)

        return {
            self.output_base_image_key: torch.as_tensor(sample[self.image_key]),
            self.output_wrist_image_key: torch.as_tensor(sample[self.wrist_image_key]),
            "observation.state": state,
            "action": action,
            "next.done": torch.tensor(done, dtype=torch.bool),
            "episode_index": torch.tensor(local_ep, dtype=torch.long),
            "frame_index": torch.tensor(frame_idx, dtype=torch.long),
            "source_episode_index": torch.tensor(source_ep, dtype=torch.long),
            "task_index": torch.tensor(self.task_index, dtype=torch.long),
        }

    def _select_episode_ids(self, task_index: int) -> list[int]:
        task_column = self.source.hf_dataset.select_columns(["task_index"])
        starts = self.source.episode_data_index["from"]
        episode_ids: list[int] = []
        for ep_idx in range(int(self.source.meta.total_episodes)):
            start = int(starts[ep_idx].item())
            ep_task_index = int(torch.as_tensor(task_column[start]["task_index"]).item())
            if ep_task_index == int(task_index):
                episode_ids.append(ep_idx)
        return episode_ids

    def _compute_stats_from_parquet(self, episode_ids: list[int]) -> tuple[dict[str, dict[str, list[float]]], list[int]]:
        import pandas as pd

        action_batches: list[np.ndarray] = []
        state_batches: list[np.ndarray] = []
        episode_lengths: list[int] = []
        root = Path(self.source.root)

        for source_ep in tqdm(episode_ids, desc="Computing GR00T LeRobot stats"):
            parquet_path = root / self.source.meta.get_data_file_path(source_ep)
            frame_df = pd.read_parquet(parquet_path, columns=[self.state_key, self.action_key])
            state = np.stack(frame_df[self.state_key].to_numpy()).astype(np.float32)
            action = np.stack(frame_df[self.action_key].to_numpy()).astype(np.float32)
            state_batches.append(_groot_rpy_state_to_residual_quat_state(state))
            action_batches.append(action)
            episode_lengths.append(int(action.shape[0]))

        states_all = np.concatenate(state_batches, axis=0)
        actions_all = np.concatenate(action_batches, axis=0)
        return {
            "action": _stats(actions_all),
            "observation.state": _stats(states_all),
        }, episode_lengths


def _prepare_lerobot_dataset_for_residual(
    dataset: LeRobotDataset,
    cfg: ResidualTD3DexmgConfig,
    *,
    image_keys: list[str],
) -> tuple[object, dict[str, dict[str, list[float]]], list[int] | None]:
    dataset_format = str(getattr(cfg.offline_data, "format", "auto"))
    if _is_groot_lerobot_dataset(dataset, dataset_format):
        adapted = GROOTLeRobotResidualAdapter(
            dataset,
            task_name=cfg.task,
            task_index=getattr(cfg.offline_data, "task_index", None),
            num_episodes=cfg.offline_data.num_episodes,
            output_image_keys=image_keys,
        )
        return adapted, adapted.stats, adapted.episode_lengths

    dataset_stats = getattr(dataset, "stats", dataset.meta.stats)
    return dataset, dataset_stats, None


def _load_base_policies(
    cfg: ResidualTD3DexmgConfig,
    *,
    task_name: str,
    state_dim: int,
    device: torch.device,
    image_size: int,
):
    provider = str(getattr(cfg.base_policy, "provider", "wandb_act")).lower()
    if provider == "wandb_act":
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
        identity = {
            "provider": provider,
            "wandb_id": cfg.base_policy.wandb_id,
            "wt_type": cfg.base_policy.wt_type,
            "wt_version": cfg.base_policy.wt_version,
        }
        return base_policy, eval_base_policy, identity

    if provider == "groot_remote":
        from resfit.rl_finetuning.utils.groot_adapter import GROOTRemoteBasePolicyAdapter

        adapter_kwargs = dict(
            groot_root=cfg.base_policy.groot_root,
            host=cfg.base_policy.groot_remote_host,
            port=int(cfg.base_policy.groot_remote_port),
            state_dim=state_dim,
            base_image_key=cfg.base_policy.groot_base_image_key,
            wrist_image_key=cfg.base_policy.groot_wrist_image_key,
            image_size=image_size,
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
            "base_image_key": cfg.base_policy.groot_base_image_key,
            "wrist_image_key": cfg.base_policy.groot_wrist_image_key,
        }
        return base_policy, eval_base_policy, identity

    raise ValueError(f"Unsupported base_policy.provider={provider!r}")


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

    # Determine camera keys early because GR00T/OpenPI-style offline datasets
    # need to be adapted into the same observation key names as the online env.
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)

    base_policy_provider = str(getattr(cfg.base_policy, "provider", "wandb_act")).lower()
    if cfg.camera_size is not None:
        camera_size = int(cfg.camera_size)
    elif base_policy_provider == "groot_remote":
        camera_size = int(cfg.auto_groot_camera_size)
    else:
        camera_size = 84
    if camera_size <= 0:
        raise ValueError(f"camera_size must be positive, got {camera_size}.")
    print(f"Residual RL observation camera size: {camera_size}x{camera_size} (provider={base_policy_provider})")

    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    dataset = None
    selected_episode_lengths: list[int] | None = None
    offline_name = str(cfg.offline_data.name)
    if offline_name.endswith((".hdf5", ".h5")):
        if cfg.algo.offline_fraction > 0.0:
            raise ValueError("Robomimic HDF5 stats-only loading is only supported with algo.offline_fraction=0.0.")
        dataset_stats, selected_episode_lengths = _load_robomimic_hdf5_stats(
            offline_name,
            cfg.offline_data.num_episodes,
        )
    else:
        dataset = LeRobotDataset(cfg.offline_data.name)
        dataset, dataset_stats, selected_episode_lengths = _prepare_lerobot_dataset_for_residual(
            dataset,
            cfg,
            image_keys=image_keys,
        )

    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset_stats["action"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )

    # Create state standardizer from dataset statistics
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset_stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    state_dim = int(torch.as_tensor(dataset_stats["observation.state"]["mean"]).numel())

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
        image_size=camera_size,
    )

    # Extract the configuration from base policy
    base_cfg = base_policy.config
    use_groot_features = bool(getattr(cfg.agent, "use_groot_features", False))
    groot_feature_key = str(getattr(cfg.agent, "groot_feature_key", "observation.groot_features"))
    if use_groot_features:
        if base_policy_provider != "groot_remote":
            raise ValueError("agent.use_groot_features=True currently requires base_policy.provider=groot_remote.")
        if not hasattr(base_policy, "encode_observation_features"):
            raise ValueError("GR00T feature mode requires base_policy.encode_observation_features().")
        if not hasattr(base_policy, "observation_feature_shape"):
            raise ValueError("GR00T feature mode requires base_policy.observation_feature_shape.")
        groot_token_count, groot_token_dim = base_policy.observation_feature_shape
        cfg.agent.groot_token_count = int(groot_token_count)
        cfg.agent.groot_token_dim = int(groot_token_dim)
        print(
            "GR00T feature mode enabled: "
            f"key={groot_feature_key} tokens={cfg.agent.groot_token_count} dim={cfg.agent.groot_token_dim}"
        )

    if isinstance(base_cfg, ACTConfig) or str(getattr(cfg.base_policy, "provider", "wandb_act")).lower() in {
        "groot_remote",
    }:
        cfg.actor_name = "residual_act"
    else:
        raise ValueError(f"Unknown base policy type: {type(base_cfg)}")

    macro_action_horizon = int(cfg.algo.macro_action_horizon)
    use_macro_actions = macro_action_horizon > 1
    adaptive_horizons = tuple(sorted(int(h) for h in cfg.algo.adaptive_macro_horizons)) if cfg.algo.adaptive_macro_enabled else ()
    use_adaptive_macro_actions = use_macro_actions and bool(adaptive_horizons)
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
        print(f"Adaptive macro horizon choices enabled: {adaptive_horizons}")

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
            camera_size=camera_size,
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
            include_base_observation_features=use_groot_features,
            base_observation_feature_key=groot_feature_key,
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
    if use_macro_actions:
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
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    base_action_dim = env.base_action_dim if hasattr(env, "base_action_dim") else action_dim

    lowdim_keys = ["observation.state", "observation.base_action"]
    if use_groot_features:
        lowdim_keys.append(groot_feature_key)

    macro_local_depth_world_to_camera = None
    macro_local_depth_cfg = getattr(cfg.agent, "macro_local_depth_gating", None)
    if macro_local_depth_cfg is not None and macro_local_depth_cfg.enabled:
        macro_local_depth_world_to_camera = _build_fixed_camera_world_to_camera_transform(
            task=cfg.task,
            camera_key=macro_local_depth_cfg.camera_key,
            camera_height=img_h,
            camera_width=img_w,
            fixed_camera_only=macro_local_depth_cfg.fixed_camera_only,
        )
        print(
            "Macro local depth projection enabled: "
            f"camera={macro_local_depth_cfg.camera_key} image={img_h}x{img_w}"
        )

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
        adaptive_horizons=adaptive_horizons if use_adaptive_macro_actions else None,
        adaptive_horizon_entropy_reg=cfg.algo.adaptive_macro_horizon_entropy_reg,
        adaptive_horizon_value_ce_coef=cfg.algo.adaptive_macro_horizon_value_ce_coef,
        adaptive_horizon_value_temperature=cfg.algo.adaptive_macro_horizon_value_temperature,
        adaptive_horizon_length_penalty=cfg.algo.adaptive_macro_horizon_length_penalty,
        adaptive_horizon_residual_penalty=cfg.algo.adaptive_macro_residual_horizon_penalty,
        adaptive_horizon_conditioned_actions=cfg.algo.adaptive_macro_horizon_conditioned_actions,
        macro_action_horizon=macro_action_horizon,
        action_scaler_limits=(action_scaler.limits.min, action_scaler.limits.max),
        state_standardizer_stats=(state_standardizer._mean, state_standardizer._std),
        world_to_camera_transform=macro_local_depth_world_to_camera,
    )
    depth_cls_cache_fn = None
    offline_depth_cls_cache_fn = None
    if agent.uses_depth_anything_v2_conditioning:
        print("Depth CLS replay caching enabled.")

        def depth_cls_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return agent.compute_depth_cls_cache(obs_dict, cpu=False)

        def offline_depth_cls_cache_fn(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return agent.compute_depth_cls_cache(obs_dict, cpu=True)

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
        "image_keys": image_keys,
        "env_camera_size": camera_size,
        "n_step": replay_n_step,
        "gamma": replay_gamma,
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        "macro_action_horizon": macro_action_horizon,
        "adaptive_macro_horizons": adaptive_horizons,
        "uses_sampled_gamma": use_adaptive_macro_actions,
        "adaptive_macro_buffer_format_version": 2 if use_adaptive_macro_actions else 1,
        # Include random action noise scale to prevent mixing data from different noise levels
        "random_action_noise_scale": cfg.algo.random_action_noise_scale,
        # Normalization parameters for consistency
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "normalized_actions": True,
        "agent_vit_depth": cfg.agent.vit.depth,
        "depth_anything_v2_conditioning_enabled": agent.uses_depth_anything_v2_conditioning,
        "cached_depth_cls": agent.uses_depth_anything_v2_conditioning,
        "groot_observation_features_enabled": use_groot_features,
        "groot_feature_key": groot_feature_key if use_groot_features else None,
        "groot_token_count": int(cfg.agent.groot_token_count),
        "groot_token_dim": int(cfg.agent.groot_token_dim),
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    online_cache_meta["base_policy_provider"] = base_policy_identity.get("provider")
    if base_policy_identity.get("provider") == "wandb_act":
        online_cache_meta["base_policy_wandb_id"] = base_policy_identity.get("wandb_id")
        online_cache_meta["base_policy_wt_type"] = base_policy_identity.get("wt_type")
        online_cache_meta["base_policy_wt_version"] = base_policy_identity.get("wt_version")
    else:
        online_cache_meta["base_policy_model_path"] = base_policy_identity.get("model_path")
        online_cache_meta["base_policy_remote_host"] = base_policy_identity.get("host")
        online_cache_meta["base_policy_remote_port"] = base_policy_identity.get("port")
        online_cache_meta["base_policy_token_target_count"] = base_policy_identity.get("token_target_count")
        online_cache_meta["base_policy_base_image_key"] = base_policy_identity.get("base_image_key")
        online_cache_meta["base_policy_wrist_image_key"] = base_policy_identity.get("wrist_image_key")
    if agent.uses_depth_anything_v2_conditioning:
        online_cache_meta["depth_anything_v2_encoder"] = cfg.agent.depth_anything_v2_conditioning.encoder
        online_cache_meta["depth_anything_v2_num_conditioned_layers"] = (
            cfg.agent.depth_anything_v2_conditioning.num_conditioned_layers
        )
    if cfg.algo.sampling_strategy == "prioritized_replay":
        online_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        online_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(online_cache_meta)
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
    if selected_episode_lengths is not None:
        total_frames = sum(selected_episode_lengths)
        num_episodes = len(selected_episode_lengths)
    elif cfg.offline_data.num_episodes is not None:
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
        dataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        use_base_policy_for_base_actions: bool = False,
        base_policy: ACTPolicy | None = None,
        macro_action_horizon: int = 1,
        macro_discount_gamma: float = 0.99,
        adaptive_horizons: tuple[int, ...] = (),
        adaptive_macro_offline_stride: int = 1,
        base_policy_batch_size: int = 1,
        depth_cls_cache_fn=None,
        cache_groot_features: bool = False,
        groot_feature_key: str = "observation.groot_features",
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
        if cache_groot_features and base_policy is None:
            raise ValueError("base_policy must be provided when cache_groot_features=True")
        if cache_groot_features and not hasattr(base_policy, "encode_observation_features"):
            raise ValueError("cache_groot_features=True requires base_policy.encode_observation_features().")
        if macro_action_horizon > 1 and not use_base_policy_for_base_actions:
            raise ValueError(
                "Macro offline buffer construction currently requires "
                "offline_data.use_base_policy_for_base_actions=True."
            )

        # Populate buffer from pre-loaded dataset
        print("Populating offline buffer from dataset...")
        loader_batch_size = max(1, int(base_policy_batch_size if use_base_policy_for_base_actions else 1))
        print(f"Offline base-policy inference batch size: {loader_batch_size}")
        loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False, num_workers=0)

        episode_cache: dict[int, dict] = {}
        episode_macro_cache: dict[int, list[TensorDict]] = {}
        transitions = 0
        use_adaptive_macro = macro_action_horizon > 1 and len(adaptive_horizons) > 0
        offline_stride = max(1, int(adaptive_macro_offline_stride))
        min_adaptive_horizon = min(adaptive_horizons) if use_adaptive_macro else 1

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
            # ------------------------------------------------------------------
            # Build observation and action directly for replay buffer ----------
            # ------------------------------------------------------------------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            _gt_action_batch: torch.Tensor = sample["action"].float()
            if _gt_action_batch.dim() == 1:
                _gt_action_batch = _gt_action_batch.unsqueeze(0)
            batch_size = int(_gt_action_batch.shape[0])
            gt_action_scaled_batch = action_scaler.scale(_gt_action_batch)

            done_batch = sample["next.done"].bool().view(batch_size)

            groot_features_batch = None
            needs_base_policy_obs = use_base_policy_for_base_actions or cache_groot_features
            if needs_base_policy_obs:
                # Use base policy to generate base action from current observation
                # Build raw observation first for base policy inference.
                raw_obs = {}
                for k in sample:
                    if "observation" in k:
                        raw_obs[k] = sample[k].to(device)  # Keep batch dimension for base policy

                with torch.no_grad():
                    if use_base_policy_for_base_actions:
                        if cache_groot_features and hasattr(base_policy, "infer_action_chunk_and_observation_features"):
                            requested_steps = max(1, macro_action_horizon)
                            base_action_chunk, groot_features_batch = base_policy.infer_action_chunk_and_observation_features(
                                raw_obs,
                                n_action_steps=requested_steps,
                            )
                            if macro_action_horizon > 1:
                                base_action_scaled_batch = action_scaler.scale(base_action_chunk).cpu().reshape(batch_size, -1)
                            else:
                                base_action_scaled_batch = action_scaler.scale(base_action_chunk[:, 0].cpu())
                        elif macro_action_horizon > 1:
                            base_action = base_policy.select_action_chunk(
                                raw_obs,
                                n_action_steps=macro_action_horizon,
                            )
                            base_action_scaled_batch = action_scaler.scale(base_action).cpu().reshape(batch_size, -1)
                        else:
                            base_action = base_policy.select_action(raw_obs)
                            base_action_scaled_batch = action_scaler.scale(base_action.cpu())
                    else:
                        # Use GT action as base action (original behavior)
                        base_action_scaled_batch = gt_action_scaled_batch

                    if cache_groot_features and groot_features_batch is None:
                        groot_features_batch = base_policy.encode_observation_features(raw_obs)
                    if groot_features_batch is not None:
                        groot_features_batch = groot_features_batch.detach().cpu()
            else:
                # Use GT action as base action (original behavior)
                base_action_scaled_batch = gt_action_scaled_batch

            for batch_idx in range(batch_size):
                ep_idx = int(sample["episode_index"][batch_idx].item())
                if num_episodes is not None and ep_idx == num_episodes:
                    break

                gt_action_scaled = gt_action_scaled_batch[batch_idx].cpu()
                base_action_scaled = base_action_scaled_batch[batch_idx].cpu()
                done_flag = bool(done_batch[batch_idx].item())

                # Build observation dict directly in target format
                curr_obs = {
                    "observation.state": state_standardizer.standardize(
                        sample["observation.state"][batch_idx].float()
                    ),
                    "observation.base_action": base_action_scaled,
                }
                for k in image_keys:
                    curr_obs[k] = sample[k][batch_idx]
                if cache_groot_features:
                    if groot_features_batch is None:
                        raise RuntimeError("GR00T feature cache was requested but no features were computed.")
                    curr_obs[groot_feature_key] = groot_features_batch[batch_idx].float()
                if depth_cls_cache_fn is not None:
                    curr_obs.update(depth_cls_cache_fn(curr_obs))

                # Convert images to uint8 for memory-efficient storage
                to_uint8(curr_obs, image_keys)

                # ------------------------------------------------------------------
                # If we already cached the *previous* frame for this episode we can
                # create transitions now.
                # ------------------------------------------------------------------
                if ep_idx in episode_cache:
                    # Create transitions for each combination of prev and current variants
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
                            "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
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

                # Cache current frame for pairing with the next one ---------------
                episode_cache[ep_idx] = {
                    "obs": curr_obs,
                    "action": gt_action_scaled,
                }

                if done_flag:
                    episode_cache.pop(ep_idx, None)

            if num_episodes is not None and batch_size > 0 and int(sample["episode_index"][-1].item()) == num_episodes:
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
        "dataset_format": getattr(cfg.offline_data, "format", "auto"),
        "dataset_task_index": getattr(cfg.offline_data, "task_index", None),
        "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
        "base_policy_batch_size": getattr(cfg.offline_data, "base_policy_batch_size", 1),
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "image_keys": image_keys,
        "n_step": replay_n_step,
        "gamma": replay_gamma,
        "macro_action_horizon": macro_action_horizon,
        "adaptive_macro_horizons": adaptive_horizons,
        "uses_sampled_gamma": use_adaptive_macro_actions,
        "adaptive_macro_buffer_format_version": 2 if use_adaptive_macro_actions else 1,
        "base_policy_provider": base_policy_identity.get("provider"),
        "sampling_strategy": cfg.algo.sampling_strategy,
        "normalized_actions": True,
        "batch_size": offline_batch_size,
        "adaptive_macro_offline_stride": cfg.algo.adaptive_macro_offline_stride,
        "agent_vit_depth": cfg.agent.vit.depth,
        "depth_anything_v2_conditioning_enabled": agent.uses_depth_anything_v2_conditioning,
        "cached_depth_cls": agent.uses_depth_anything_v2_conditioning,
        "groot_observation_features_enabled": use_groot_features,
        "groot_feature_key": groot_feature_key if use_groot_features else None,
        "groot_token_count": int(cfg.agent.groot_token_count),
        "groot_token_dim": int(cfg.agent.groot_token_dim),
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if base_policy_identity.get("provider") == "wandb_act":
        offline_cache_meta["base_policy_wandb_id"] = base_policy_identity.get("wandb_id")
        offline_cache_meta["base_policy_wt_type"] = base_policy_identity.get("wt_type")
        offline_cache_meta["base_policy_wt_version"] = base_policy_identity.get("wt_version")
    else:
        offline_cache_meta["base_policy_model_path"] = base_policy_identity.get("model_path")
        offline_cache_meta["base_policy_remote_host"] = base_policy_identity.get("host")
        offline_cache_meta["base_policy_remote_port"] = base_policy_identity.get("port")
        offline_cache_meta["base_policy_token_target_count"] = base_policy_identity.get("token_target_count")
        offline_cache_meta["base_policy_base_image_key"] = base_policy_identity.get("base_image_key")
        offline_cache_meta["base_policy_wrist_image_key"] = base_policy_identity.get("wrist_image_key")
    if agent.uses_depth_anything_v2_conditioning:
        offline_cache_meta["depth_anything_v2_encoder"] = cfg.agent.depth_anything_v2_conditioning.encoder
        offline_cache_meta["depth_anything_v2_num_conditioned_layers"] = (
            cfg.agent.depth_anything_v2_conditioning.num_conditioned_layers
        )
    if cfg.algo.sampling_strategy == "prioritized_replay":
        offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(offline_cache_meta)

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(offline_cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer ---------------------------------------
    cache_dir = OFFLINE_CACHE_DIR / cache_hash

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
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if (cfg.offline_data.use_base_policy_for_base_actions or use_groot_features) else None,
                macro_action_horizon=macro_action_horizon,
                macro_discount_gamma=cfg.algo.gamma,
                adaptive_horizons=adaptive_horizons if use_adaptive_macro_actions else (),
                adaptive_macro_offline_stride=cfg.algo.adaptive_macro_offline_stride,
                base_policy_batch_size=getattr(cfg.offline_data, "base_policy_batch_size", 1),
                depth_cls_cache_fn=offline_depth_cls_cache_fn,
                cache_groot_features=use_groot_features,
                groot_feature_key=groot_feature_key,
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

        # Persist freshly-collected buffer (local + HF) --------------------
        if _env_flag("RESFIT_SKIP_ONLINE_CACHE_DUMP"):
            print(
                "Skipping online replay buffer cache dump because "
                "RESFIT_SKIP_ONLINE_CACHE_DUMP=1"
            )
        else:
            online_cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(online_rb, online_cache_dir)
            with open(online_cache_dir / "user_metadata.json", "w") as f:
                json.dump(online_cache_meta, f, indent=2)
            if ONLINE_HF_REPO is not None:
                _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        loaded_online_from_cache = True  # treat as cached going forward

    _hp_parts: list[str] = [
        cfg.task,  # e.g. "TwoArmBoxCleanup"
        f"n{replay_n_step}",  # replay-buffer n-step horizon
        f"utd{cfg.algo.num_updates_per_iteration}",  # updates-to-data ratio
        f"buf{cfg.algo.buffer_size}",  # replay buffer size
    ]

    if use_macro_actions:
        _hp_parts.append(f"macro{macro_action_horizon}")
    if use_adaptive_macro_actions:
        _hp_parts.append("adaptive" + "-".join(str(h) for h in adaptive_horizons))

    # Offline dataset statistics (if any)
    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None and cfg.algo.offline_fraction > 0.0:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")
    elif cfg.algo.offline_fraction == 0.0:
        _hp_parts.append("online_only")

    # Learning-rate, expressed in scientific notation for brevity (e.g. 1e-4 → 1e-04)
    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")
    if camera_size != 84:
        _hp_parts.append(f"img{camera_size}")
    if use_groot_features:
        _hp_parts.append("grootfeat")

    # Additional flags ---------------------------------------------------------
    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
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

    # Log horizon to wandb summary
    wandb.summary["environment/horizon"] = env.vec_env.metadata["horizon"]

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

    def _run_evaluation(step_value: int) -> None:
        nonlocal best_eval_success_rate
        with training_timer.time("evaluation"):
            eval_metrics = run_dexmg_evaluation(
                env=eval_env,
                agent=agent,
                num_episodes=cfg.eval_num_episodes,
                device=device,
                global_step=step_value,
                save_video=cfg.save_video,
                save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                run_name=run_name,
                output_dir=outputs_dir,
            )

        current_success_rate = eval_metrics["eval/success_rate"]
        wandb.summary["paper/latest_task_success_rate"] = current_success_rate
        if current_success_rate > best_eval_success_rate:
            print(f"🎉 New best success rate: {current_success_rate:.4f} (prev: {best_eval_success_rate:.4f})")
            best_eval_success_rate = current_success_rate
            wandb.summary["paper/best_task_success_rate"] = best_eval_success_rate

    if cfg.eval_first:
        _run_evaluation(0)

    next_eval_step = cfg.eval_interval_every_steps
    next_log_step = cfg.log_freq

    def adaptive_horizon_exploration_epsilon(step: int) -> float:
        if not use_adaptive_macro_actions:
            return 0.0
        start = float(cfg.algo.adaptive_macro_horizon_exploration_initial)
        final = float(cfg.algo.adaptive_macro_horizon_exploration_final)
        decay_steps = max(1, int(cfg.algo.adaptive_macro_horizon_exploration_steps))
        mix = min(1.0, max(0.0, step / decay_steps))
        return start + (final - start) * mix

    while global_step < cfg.algo.total_timesteps:
        iter_start = time.time()
        # ------------------------------------------------------------------
        # (1) Collect action + Environment step ---------------------------
        # ------------------------------------------------------------------
        with training_timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                horizon_epsilon = adaptive_horizon_exploration_epsilon(global_step)
                action = agent.act(
                    obs,
                    eval_mode=False,
                    stddev=stddev,
                    cpu=False,
                    horizon_epsilon=horizon_epsilon,
                )

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
        if global_step % cfg.algo.update_every_n_steps == 0 or global_step == step_increment:
            i = 0
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            # Normal training loop - critic is already warmed up
            while i < cfg.algo.num_updates_per_iteration:
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
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        # Online-only training
                        batch = online_batch

                # Update actor on the last iteration of each update cycle
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

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
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
                "training/horizon_exploration_epsilon": adaptive_horizon_exploration_epsilon(global_step),
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

    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if os.environ.get("RESFIT_KEEP_RUN_DIR") == "1":
        print(f"Keeping run directory because RESFIT_KEEP_RUN_DIR=1: {run_cache_dir}")
    elif run_cache_dir.exists():
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
