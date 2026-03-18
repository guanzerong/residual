from __future__ import annotations

import argparse
from collections import deque
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.lerobot.utils.openpi_chunk_policy import OpenPIChunkPolicy
from resfit.rl_finetuning.utils.libero_eval_env import (
    LiberoEvalVecEnvWrapper,
    ensure_libero_available,
    get_libero_max_steps,
    get_libero_task_suite,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.wrappers.residual_chunk_env_wrapper import ResidualChunkVecEnvWrapper

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class PreparedBasePolicy:
    source: str
    policy_ref: str
    policy_dir: Path | None = None
    openpi_host: str | None = None
    openpi_port: int | None = None
    openpi_chunk_size: int | None = None
    openpi_api_key: str | None = None


def _deep_get(mapping: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _unflatten_dict(flat_mapping: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in flat_mapping.items():
        if not isinstance(key, str) or "." not in key:
            nested[key] = value
            continue
        current = nested
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def _normalize_checkpoint_config(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "_metadata"):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Unsupported checkpoint config type: {type(config)}")
    if "agent" in config or "algo" in config or "base_policy" in config:
        return config
    if any(isinstance(key, str) and "." in key for key in config):
        return _unflatten_dict(config)
    return config


def _get_stats_feature_keys(dataset_schema: str) -> tuple[str, str]:
    if dataset_schema == "resfit":
        return "action", "observation.state"
    if dataset_schema == "openpi":
        return "actions", "state"
    raise ValueError(f"Unsupported dataset_schema={dataset_schema!r}.")


def _resolve_local_dataset_root(dataset_name: str) -> Path | None:
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()
    return None


def _load_episode_stats_records(dataset_root: Path) -> list[dict[str, Any]]:
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    records: list[dict[str, Any]] = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_episode_stat_count(value: Any) -> int:
    count_array = np.asarray(value).reshape(-1)
    if count_array.size == 0:
        raise ValueError("Encountered empty count array in episode stats.")
    return int(count_array[0])


def _aggregate_episode_feature_stats(records: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
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
            assert feature_min is not None and feature_max is not None
            feature_min = np.minimum(feature_min, current_min)
            feature_max = np.maximum(feature_max, current_max)

        feature_sum += mean * count
        feature_sum_sq += (np.square(std) + np.square(mean)) * count
        total_count += count

    if total_count <= 0 or feature_sum is None or feature_sum_sq is None or feature_min is None or feature_max is None:
        raise ValueError(f"Invalid aggregated stats for feature {feature_key!r}.")

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

    dataset_root = _resolve_local_dataset_root(dataset_name)
    if dataset_schema == "openpi" and dataset_root is not None and (episode_start > 0 or num_episodes is not None):
        try:
            all_records = _load_episode_stats_records(dataset_root)
            episode_end = None if num_episodes is None else episode_start + num_episodes
            selected_records = [
                record
                for record in all_records
                if int(record["episode_index"]) >= episode_start
                and (episode_end is None or int(record["episode_index"]) < episode_end)
            ]
            if selected_records:
                dataset_stats[action_stats_key] = _aggregate_episode_feature_stats(selected_records, action_stats_key)
                dataset_stats[state_stats_key] = _aggregate_episode_feature_stats(selected_records, state_stats_key)
        except Exception as exc:  # noqa: BLE001
            print(f"Falling back to global dataset stats because subset aggregation failed: {exc}")

    return dataset_stats, dataset_stats[action_stats_key], dataset_stats[state_stats_key]


def _prepare_base_policy(args: argparse.Namespace) -> PreparedBasePolicy:
    if args.base_policy_source == "wandb":
        if not args.base_wandb_id:
            raise ValueError("--base-wandb-id is required when --base-policy-source=wandb")
        policy_dir, _ = download_policy_from_wandb(
            args.base_wandb_id,
            step=args.base_wt_type,
            artifact_version=args.base_wt_version,
        )
        return PreparedBasePolicy(
            source="wandb",
            policy_ref=str(policy_dir),
            policy_dir=policy_dir,
        )

    if args.base_policy_source == "local":
        if not args.base_local_path:
            raise ValueError("--base-local-path is required when --base-policy-source=local")
        policy_dir = Path(args.base_local_path).expanduser().resolve()
        if not policy_dir.exists():
            raise FileNotFoundError(f"Local base policy directory not found: {policy_dir}")
        return PreparedBasePolicy(
            source="local",
            policy_ref=str(policy_dir),
            policy_dir=policy_dir,
        )

    if args.base_policy_source == "openpi_ws":
        return PreparedBasePolicy(
            source="openpi_ws",
            policy_ref=f"ws://{args.openpi_host}:{args.openpi_port}",
            openpi_host=args.openpi_host,
            openpi_port=args.openpi_port,
            openpi_chunk_size=args.openpi_chunk_size,
            openpi_api_key=args.openpi_api_key,
        )

    raise ValueError(f"Unsupported base policy source: {args.base_policy_source!r}")


def _load_base_policy_for_task(
    *,
    prepared: PreparedBasePolicy,
    dataset_stats: dict[str, Any],
    task_prompt: str,
    device: torch.device,
):
    if prepared.source in {"wandb", "local"}:
        assert prepared.policy_dir is not None
        policy = load_policy(prepared.policy_dir, dataset_stats=dataset_stats, default_task=task_prompt)
    elif prepared.source == "openpi_ws":
        policy = OpenPIChunkPolicy(
            host=prepared.openpi_host or "127.0.0.1",
            port=prepared.openpi_port or 8000,
            task_prompt=task_prompt,
            chunk_size=prepared.openpi_chunk_size or 20,
            api_key=prepared.openpi_api_key,
        )
    else:
        raise ValueError(f"Unsupported prepared base policy source: {prepared.source!r}")

    policy.to(device)
    policy.eval()
    return policy


def _build_residual_agent(
    *,
    checkpoint: dict[str, Any],
    checkpoint_config: dict[str, Any],
    env: ResidualChunkVecEnvWrapper,
    device: torch.device,
    rl_cameras: list[str],
):
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    if "agent_state_dict" not in checkpoint:
        raise KeyError("Residual checkpoint is missing 'agent_state_dict'.")

    agent_cfg_dict = _deep_get(checkpoint_config, "agent")
    if not isinstance(agent_cfg_dict, dict):
        raise ValueError("Residual checkpoint config is missing nested 'agent' settings.")

    for camera_name in rl_cameras:
        if camera_name not in env.observation_space.spaces:
            available = sorted(env.observation_space.spaces.keys())
            raise KeyError(f"Checkpoint expects camera {camera_name!r}, available observations are {available}")

    agent_cfg_payload = dict(agent_cfg_dict)
    agent_cfg_payload["device"] = str(device)
    agent_cfg = OmegaConf.create(agent_cfg_payload)

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[rl_cameras[0]].shape[1:]
    action_dim = env.action_space.shape[1]

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=rl_cameras,
        cfg=agent_cfg,
        residual_actor=True,
    )
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    agent.eval()
    agent.to(device)
    return agent


def _evaluate_policy(
    *,
    env: ResidualChunkVecEnvWrapper,
    agent: QAgent | None,
    num_episodes: int,
    device: torch.device,
) -> tuple[int, int]:
    successes = 0
    episodes_done = 0
    obs, _ = env.reset()

    while episodes_done < num_episodes:
        with torch.no_grad():
            if agent is None:
                residual_action = torch.zeros(
                    (1, env.action_space.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                if residual_action.dim() == 1:
                    residual_action = residual_action.unsqueeze(0)

        obs, reward, terminated, truncated, _ = env.step(residual_action)
        done = terminated | truncated
        if bool(done[0].item()):
            episodes_done += 1
            successes += int(float(reward[0].item()) >= 1.0)

    return successes, episodes_done


def _evaluate_base_policy_direct(
    *,
    raw_env: LiberoEvalVecEnvWrapper,
    base_policy,
    num_episodes: int,
    replan_steps: int,
) -> tuple[int, int]:
    if replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {replan_steps}.")

    successes = 0
    episodes_done = 0
    obs, _ = raw_env.reset()
    action_plan: deque[torch.Tensor] = deque()
    base_policy.reset()

    while episodes_done < num_episodes:
        if not action_plan:
            with torch.no_grad():
                action_chunk = base_policy.select_action_chunk(obs, n_steps=replan_steps)
            if action_chunk.dim() == 2:
                action_chunk = action_chunk.unsqueeze(0)
            if action_chunk.shape[0] != 1:
                raise ValueError(
                    "Base-only direct eval currently supports batch size 1. "
                    f"Got action chunk shape {tuple(action_chunk.shape)}."
                )
            action_plan.extend(action_chunk[:, step_idx, :] for step_idx in range(action_chunk.shape[1]))

        obs, _, terminated, truncated, _ = raw_env.step(action_plan.popleft())
        done = terminated | truncated
        if bool(done[0].item()):
            episodes_done += 1
            successes += int(bool(terminated[0].item()))
            base_policy.reset()
            action_plan.clear()

    return successes, episodes_done


def _populate_defaults_from_checkpoint(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any] | None,
    default_device: str,
) -> argparse.Namespace:
    if args.device is None:
        args.device = default_device

    if args.base_policy_source is None:
        args.base_policy_source = str(_deep_get(checkpoint_config, "base_policy.source", "openpi_ws"))

    if args.base_wandb_id is None:
        args.base_wandb_id = _deep_get(checkpoint_config, "base_policy.wandb_id")
    if args.base_wt_type is None:
        args.base_wt_type = str(_deep_get(checkpoint_config, "base_policy.wt_type", "best"))
    if args.base_wt_version is None:
        args.base_wt_version = str(_deep_get(checkpoint_config, "base_policy.wt_version", "latest"))
    if args.base_local_path is None:
        args.base_local_path = _deep_get(checkpoint_config, "base_policy.local_path")

    if args.openpi_host is None:
        args.openpi_host = str(_deep_get(checkpoint_config, "base_policy.openpi_host", "127.0.0.1"))
    if args.openpi_port is None:
        args.openpi_port = int(_deep_get(checkpoint_config, "base_policy.openpi_port", 8000))
    if args.openpi_chunk_size is None:
        args.openpi_chunk_size = int(_deep_get(checkpoint_config, "base_policy.openpi_chunk_size", 20))
    if args.openpi_api_key is None:
        args.openpi_api_key = _deep_get(checkpoint_config, "base_policy.openpi_api_key")

    if args.resize_size is None:
        args.resize_size = int(_deep_get(checkpoint_config, "env_camera_size", 224))
    if args.macro_horizon is None:
        args.macro_horizon = int(_deep_get(checkpoint_config, "algo.macro_horizon", 2))
    if args.base_plan_horizon is None:
        base_plan_horizon = _deep_get(checkpoint_config, "algo.base_plan_horizon")
        args.base_plan_horizon = None if base_plan_horizon is None else int(base_plan_horizon)
    if args.chunk_success_threshold is None:
        args.chunk_success_threshold = float(_deep_get(checkpoint_config, "algo.chunk_success_threshold", 0.5))
    if args.action_scale is None:
        args.action_scale = float(_deep_get(checkpoint_config, "agent.actor.action_scale", 1.0))
    if args.max_steps is None:
        args.max_steps = get_libero_max_steps(args.task_suite_name)

    if args.rl_camera is None:
        rl_camera = _deep_get(checkpoint_config, "rl_camera")
        if rl_camera is None:
            args.rl_camera = ["observation.images.agentview"]
        elif isinstance(rl_camera, str):
            args.rl_camera = [rl_camera]
        else:
            args.rl_camera = list(rl_camera)

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base-only or base+residual chunk policy on LIBERO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=None, help="If omitted, evaluate the whole suite.")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--env-resolution", type=int, default=256)

    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-schema", choices=["resfit", "openpi"], default="openpi")
    parser.add_argument("--dataset-episode-start", type=int, default=0)
    parser.add_argument("--dataset-num-episodes", type=int, default=None)
    parser.add_argument("--min-action-range", type=float, default=1e-1)
    parser.add_argument("--min-state-std", type=float, default=1e-1)
    parser.add_argument("--action-scale", type=float, default=None)

    parser.add_argument("--base-policy-source", choices=["wandb", "local", "openpi_ws"], default=None)
    parser.add_argument("--base-wandb-id", type=str, default=None)
    parser.add_argument("--base-wt-type", type=str, default=None)
    parser.add_argument("--base-wt-version", type=str, default=None)
    parser.add_argument("--base-local-path", type=str, default=None)
    parser.add_argument("--openpi-host", type=str, default=None)
    parser.add_argument("--openpi-port", type=int, default=None)
    parser.add_argument("--openpi-chunk-size", type=int, default=None)
    parser.add_argument("--openpi-api-key", type=str, default=None)

    parser.add_argument("--residual-checkpoint", type=str, default=None)
    parser.add_argument("--macro-horizon", type=int, default=None)
    parser.add_argument("--base-plan-horizon", type=int, default=None)
    parser.add_argument("--base-replan-steps", type=int, default=None)
    parser.add_argument("--chunk-success-threshold", type=float, default=None)
    parser.add_argument("--rl-camera", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_libero_available()

    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive.")
    if args.dataset_episode_start < 0:
        raise ValueError("--dataset-episode-start must be non-negative.")
    if args.base_replan_steps is not None and args.base_replan_steps <= 0:
        raise ValueError("--base-replan-steps must be positive when provided.")

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint: dict[str, Any] | None = None
    checkpoint_config: dict[str, Any] | None = None
    if args.residual_checkpoint:
        from resfit.rl_finetuning.utils.checkpoint import load_checkpoint

        checkpoint = load_checkpoint(args.residual_checkpoint, map_location="cpu")
        checkpoint_config = _normalize_checkpoint_config(checkpoint.get("config"))

    args = _populate_defaults_from_checkpoint(args, checkpoint_config, default_device=default_device)
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = LeRobotDataset(args.dataset_name)
    dataset_stats, action_stats, state_stats = _resolve_dataset_stats(
        dataset=dataset,
        dataset_name=args.dataset_name,
        dataset_schema=args.dataset_schema,
        episode_start=args.dataset_episode_start,
        num_episodes=args.dataset_num_episodes,
    )
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=action_stats,
        action_scale=args.action_scale,
        min_range_per_dim=args.min_action_range,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=state_stats,
        min_std=args.min_state_std,
        device=device,
    )

    prepared_base_policy = _prepare_base_policy(args)
    task_suite = get_libero_task_suite(args.task_suite_name)

    if args.task_id is None:
        task_ids = list(range(task_suite.n_tasks))
    else:
        if args.task_id < 0 or args.task_id >= task_suite.n_tasks:
            raise ValueError(f"--task-id must be in [0, {task_suite.n_tasks - 1}], got {args.task_id}.")
        task_ids = [args.task_id]

    agent: QAgent | None = None
    total_successes = 0
    total_episodes = 0
    resolved_base_plan_horizon: int | None = args.base_plan_horizon

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_description = str(task.language)
        initial_states = task_suite.get_task_init_states(task_id)
        base_policy = _load_base_policy_for_task(
            prepared=prepared_base_policy,
            dataset_stats=dataset_stats,
            task_prompt=task_description,
            device=device,
        )

        base_chunk_size = getattr(base_policy.config, "chunk_size", None)
        if base_chunk_size is None:
            raise ValueError(f"Base policy does not expose config.chunk_size: {type(base_policy)}")

        if resolved_base_plan_horizon is None:
            resolved_base_plan_horizon = int(base_chunk_size)
        if resolved_base_plan_horizon > int(base_chunk_size):
            raise ValueError(
                f"base_plan_horizon={resolved_base_plan_horizon} exceeds base policy chunk_size={base_chunk_size}."
            )
        if resolved_base_plan_horizon < args.macro_horizon:
            raise ValueError(
                f"base_plan_horizon={resolved_base_plan_horizon} must be >= macro_horizon={args.macro_horizon}."
            )
        if args.base_replan_steps is not None and args.base_replan_steps > int(base_chunk_size):
            raise ValueError(
                f"base_replan_steps={args.base_replan_steps} exceeds base policy chunk_size={base_chunk_size}."
            )

        raw_env = LiberoEvalVecEnvWrapper(
            task=task,
            initial_states=initial_states,
            task_description=task_description,
            resize_size=args.resize_size,
            max_steps=args.max_steps,
            num_steps_wait=args.num_steps_wait,
            seed=args.seed,
            device=device,
            env_resolution=args.env_resolution,
        )
        use_direct_base_eval = checkpoint is None and args.base_replan_steps is not None
        if use_direct_base_eval:
            try:
                task_successes, task_episodes = _evaluate_base_policy_direct(
                    raw_env=raw_env,
                    base_policy=base_policy,
                    num_episodes=args.episodes_per_task,
                    replan_steps=args.base_replan_steps,
                )
            finally:
                raw_env.close()
        else:
            env = ResidualChunkVecEnvWrapper(
                vec_env=raw_env,
                base_policy=base_policy,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
                macro_horizon=args.macro_horizon,
                base_plan_horizon=resolved_base_plan_horizon,
                chunk_success_threshold=args.chunk_success_threshold,
            )

            try:
                if checkpoint is not None and agent is None:
                    agent = _build_residual_agent(
                        checkpoint=checkpoint,
                        checkpoint_config=checkpoint_config or {},
                        env=env,
                        device=device,
                        rl_cameras=list(args.rl_camera),
                    )

                task_successes, task_episodes = _evaluate_policy(
                    env=env,
                    agent=agent,
                    num_episodes=args.episodes_per_task,
                    device=device,
                )
            finally:
                env.close()

        total_successes += task_successes
        total_episodes += task_episodes
        task_success_rate = float(task_successes) / float(task_episodes)
        if checkpoint is not None:
            mode = "base+residual"
        elif use_direct_base_eval:
            mode = f"base-only-replan{args.base_replan_steps}"
        else:
            mode = "base-only"
        print(
            f"[task {task_id:02d}] {mode} success_rate={task_success_rate:.4f} "
            f"({task_successes}/{task_episodes}) prompt={task_description}"
        )

    overall_success_rate = float(total_successes) / float(total_episodes)
    print(
        f"[overall] success_rate={overall_success_rate:.4f} "
        f"({total_successes}/{total_episodes}) suite={args.task_suite_name} tasks={len(task_ids)}"
    )


if __name__ == "__main__":
    main()
