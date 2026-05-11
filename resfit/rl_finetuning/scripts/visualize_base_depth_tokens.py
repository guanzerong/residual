from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data_all/gzr1/.mplconfig")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from robosuite import load_composite_controller_config
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

import robosuite
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import download_policy_from_wandb
from resfit.rl_finetuning.config.rlpd import DepthAnythingV2ConditioningConfig
from resfit.rl_finetuning.off_policy.networks.encoder import DepthAnythingV2TokenEncoder


ALIAS_TO_CANONICAL_TASK = {
    "Can": "PickPlaceCan",
    "Square": "NutAssemblySquare",
    "Transport": "TwoArmTransport",
}

ENV_ROBOTS = {
    "Lift": ["Panda"],
    "PickPlaceCan": ["Panda"],
    "NutAssemblySquare": ["Panda"],
    "Threading": ["Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmCanSortRandom": ["GR1FixedLowerBody"],
}

DEFAULT_DATASET_SEARCH_ROOTS = (
    Path("/data_all/gzr1/.hf_home/lerobot"),
    Path.home() / ".cache" / "huggingface" / "lerobot",
)

DEFAULT_WANDB_ARTIFACT_ROOTS = (
    Path("/data_all/gzr1/.wandb/artifacts"),
    Path.home() / ".wandb" / "artifacts",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize which local DepthAnything patch tokens are traversed by the projected base trajectory."
    )
    parser.add_argument("--dataset-repo-id", type=str, default="ankile/robomimic-ph-square-image")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--task", type=str, default="Square")
    parser.add_argument("--camera-key", type=str, default="observation.images.agentview")
    parser.add_argument("--base-policy-dir", type=str, default=None)
    parser.add_argument("--base-policy-wandb-id", type=str, default="square-ph-bc/i9tt1t4a")
    parser.add_argument("--base-policy-step", type=str, default="latest")
    parser.add_argument("--chunk-horizon", type=int, default=20)
    parser.add_argument("--traj-samples-per-segment", type=int, default=6)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="core_expand",
        choices=["core_expand", "point_quad_union"],
    )
    parser.add_argument("--patch-neighborhood", type=int, default=0)
    parser.add_argument("--head-core-count", type=int, default=0)
    parser.add_argument("--tail-core-count", type=int, default=0)
    parser.add_argument("--selection-point-count", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth-device", type=str, default=None)
    parser.add_argument("--depth-encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth-source-root", type=str, default=None)
    parser.add_argument("--depth-weights", type=str, default=None)
    parser.add_argument("--depth-resize-to", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data_all/gzr1/visualizations/base_depth_tokens",
    )
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def _canonical_task_name(task_name: str) -> str:
    return ALIAS_TO_CANONICAL_TASK.get(task_name, task_name)


def _ensure_hf_dataset_cache() -> None:
    default_cache = Path("/data_all/gzr1/.hf_datasets_cache")
    default_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(default_cache))


def _find_local_dataset_root(repo_id: str, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        root = Path(explicit_root).expanduser()
        if (root / "meta" / "info.json").is_file():
            return root
        raise FileNotFoundError(f"Dataset root does not contain meta/info.json: {root}")

    for parent in DEFAULT_DATASET_SEARCH_ROOTS:
        candidate = parent / repo_id
        if (candidate / "meta" / "info.json").is_file():
            return candidate

    searched = [str(parent / repo_id) for parent in DEFAULT_DATASET_SEARCH_ROOTS]
    raise FileNotFoundError(
        "Could not find a local dataset root. "
        f"Searched: {searched}. Pass --dataset-root explicitly."
    )


def _find_local_policy_dir(run_id: str, step: str) -> Path | None:
    _, run_name = run_id.split("/", maxsplit=1)
    if step.lower() in {"latest", "best"}:
        expected_middle = step.lower()
        pattern = re.compile(rf"^run_{re.escape(run_name)}_{expected_middle}:v(\d+)$")
    else:
        expected_middle = f"model_step_{step}"
        pattern = re.compile(rf"^run_{re.escape(run_name)}_{re.escape(expected_middle)}:v(\d+)$")

    matches: list[tuple[int, Path]] = []
    for artifacts_root in DEFAULT_WANDB_ARTIFACT_ROOTS:
        if not artifacts_root.is_dir():
            continue
        for child in artifacts_root.iterdir():
            match = pattern.match(child.name)
            if match is None:
                continue
            policy_dir = child / "policy"
            if (policy_dir / "config.json").is_file() and (policy_dir / "model.safetensors").is_file():
                matches.append((int(match.group(1)), policy_dir))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _resolve_policy_dir(
    explicit_policy_dir: str | None,
    wandb_run_id: str,
    step: str,
) -> Path:
    if explicit_policy_dir is not None:
        policy_dir = Path(explicit_policy_dir).expanduser()
        if not (policy_dir / "config.json").is_file():
            raise FileNotFoundError(f"Policy directory missing config.json: {policy_dir}")
        return policy_dir

    local_policy_dir = _find_local_policy_dir(wandb_run_id, step)
    if local_policy_dir is not None:
        return local_policy_dir

    policy_dir, _ = download_policy_from_wandb(
        wandb_run_id,
        step=step,
        artifact_version="latest",
    )
    return policy_dir


def _load_policy(policy_dir: Path, device: str) -> ACTPolicy:
    cfg = ACTConfig.from_pretrained(policy_dir, local_files_only=True)
    cfg.device = device
    policy = ACTPolicy.from_pretrained(
        policy_dir,
        config=cfg,
        local_files_only=True,
    )
    policy.to(device)
    policy.eval()
    return policy


def _resolve_sample_index(
    dataset: LeRobotDataset,
    sample_index: int | None,
    episode_index: int | None,
    frame_index: int | None,
) -> int:
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample_index {sample_index} is out of range for dataset of size {len(dataset)}")
        return sample_index

    target_episode = 0 if episode_index is None else episode_index
    target_frame = 0 if frame_index is None else frame_index
    episode_column = dataset.hf_dataset["episode_index"]
    frame_column = dataset.hf_dataset["frame_index"]
    for idx, (ep, frame) in enumerate(zip(episode_column, frame_column, strict=True)):
        ep_int = int(ep.item() if torch.is_tensor(ep) else ep)
        frame_int = int(frame.item() if torch.is_tensor(frame) else frame)
        if ep_int == target_episode and frame_int == target_frame:
            return idx

    raise ValueError(
        f"Could not find sample for episode_index={target_episode}, frame_index={target_frame}. "
        "Pass --sample-index explicitly."
    )


def _build_policy_batch(sample: dict[str, object], policy: ACTPolicy, device: str) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in policy.config.input_features:
        value = sample[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Expected tensor input for {key}, got {type(value)}")
        batch[key] = value.unsqueeze(0).to(device)
    return batch


def _predict_base_action_chunk(sample: dict[str, object], policy: ACTPolicy, device: str, horizon: int) -> torch.Tensor:
    if horizon > policy.config.n_action_steps:
        raise ValueError(
            f"Requested horizon={horizon}, but base policy only supports n_action_steps={policy.config.n_action_steps}."
        )

    batch = _build_policy_batch(sample, policy, device)
    with torch.no_grad():
        action_chunk = policy.select_action_chunk(batch, n_action_steps=horizon)
    return action_chunk.squeeze(0).detach().cpu()


def _extract_rgb_image(sample: dict[str, object], camera_key: str) -> np.ndarray:
    image = sample[camera_key]
    if not torch.is_tensor(image):
        raise TypeError(f"Expected tensor image for {camera_key}, got {type(image)}")
    image_np = image.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    return np.clip(image_np, 0.0, 1.0)


def _current_eef_position(sample: dict[str, object]) -> np.ndarray:
    state = sample["observation.state"]
    if not torch.is_tensor(state):
        raise TypeError(f"Expected tensor state, got {type(state)}")
    return state[:3].detach().cpu().numpy()


def _single_arm_controller_pos_bounds(task_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical_task = _canonical_task_name(task_name)
    if canonical_task not in ENV_ROBOTS:
        raise ValueError(f"Unknown task {task_name!r} -> canonical {canonical_task!r}")

    robots = ENV_ROBOTS[canonical_task]
    controller_cfg = load_composite_controller_config(robot=robots[0])
    body_parts = controller_cfg.get("body_parts", {})
    arm_cfg = None

    if "arms" in body_parts and isinstance(body_parts["arms"], dict):
        arm_cfg = body_parts["arms"].get("right")
    if arm_cfg is None:
        arm_cfg = body_parts.get("right")
    if arm_cfg is None:
        raise KeyError(f"Could not find single-arm controller config for task {task_name!r}")

    input_min = np.broadcast_to(np.asarray(arm_cfg["input_min"], dtype=np.float32), (6,))
    input_max = np.broadcast_to(np.asarray(arm_cfg["input_max"], dtype=np.float32), (6,))
    output_min = np.asarray(arm_cfg["output_min"], dtype=np.float32)
    output_max = np.asarray(arm_cfg["output_max"], dtype=np.float32)
    return input_min[:3], input_max[:3], output_min[:3], output_max[:3]


def _scale_normalized_delta_pos(task_name: str, raw_delta_pos: np.ndarray) -> np.ndarray:
    input_min, input_max, output_min, output_max = _single_arm_controller_pos_bounds(task_name)
    raw_clipped = np.clip(raw_delta_pos, input_min, input_max)
    input_mid = (input_max + input_min) / 2.0
    output_mid = (output_max + output_min) / 2.0
    scaled = (raw_clipped - input_mid) * (output_max - output_min) / np.maximum(input_max - input_min, 1e-8)
    return scaled + output_mid


def _approximate_future_eef_positions(
    start_eef_pos: np.ndarray,
    action_chunk: torch.Tensor,
    task_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_delta_pos = action_chunk[:, :3].detach().cpu().numpy()
    delta_pos = _scale_normalized_delta_pos(task_name, raw_delta_pos)
    positions = [start_eef_pos.astype(np.float32)]
    running = start_eef_pos.astype(np.float32).copy()
    for delta in delta_pos:
        running = running + delta.astype(np.float32)
        positions.append(running.copy())
    return np.stack(positions, axis=0), delta_pos


def _densify_polyline_world(points_world: np.ndarray, samples_per_segment: int) -> np.ndarray:
    dense_points = []
    for idx in range(len(points_world) - 1):
        start = points_world[idx]
        end = points_world[idx + 1]
        for step_idx in range(samples_per_segment):
            t = step_idx / samples_per_segment
            dense_points.append((1.0 - t) * start + t * end)
    dense_points.append(points_world[-1])
    return np.stack(dense_points, axis=0)


def _camera_name_from_key(camera_key: str) -> str:
    prefix = "observation.images."
    if not camera_key.startswith(prefix):
        raise ValueError(f"camera_key must start with {prefix!r}, got {camera_key!r}")
    return camera_key[len(prefix) :]


def _make_projection_env(task_name: str, camera_name: str):
    canonical_task = _canonical_task_name(task_name)
    if canonical_task not in ENV_ROBOTS:
        raise ValueError(f"Unknown task {task_name!r} -> canonical {canonical_task!r}")

    robots = ENV_ROBOTS[canonical_task]
    controller_configs = load_composite_controller_config(robot=robots[0])
    if "composite_controller_specific_configs" in controller_configs:
        controller_configs["composite_controller_specific_configs"]["ik_input_ref_frame"] = "world"

    env_kwargs = {
        "env_name": canonical_task,
        "robots": robots,
        "controller_configs": controller_configs,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": False,
        "use_camera_obs": False,
        "camera_names": [camera_name],
        "control_freq": 20,
    }
    env = robosuite.make(**env_kwargs)
    env.reset()
    return env


def _project_world_points_to_pixels(
    task_name: str,
    camera_key: str,
    world_points: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    camera_name = _camera_name_from_key(camera_key)
    env = _make_projection_env(task_name, camera_name)
    try:
        world_to_camera = get_camera_transform_matrix(
            env.sim,
            camera_name=camera_name,
            camera_height=image_height,
            camera_width=image_width,
        )
        pixels_rc = project_points_from_world_to_camera(
            world_points,
            world_to_camera_transform=world_to_camera,
            camera_height=image_height,
            camera_width=image_width,
        )
    finally:
        env.close()
    return pixels_rc


def _build_depth_encoder(
    image_shape_chw: tuple[int, int, int],
    source_root: str | None,
    weights: str | None,
    depth_encoder_name: str,
    resize_to: int,
    device: str,
) -> DepthAnythingV2TokenEncoder:
    cfg = DepthAnythingV2ConditioningConfig(
        enabled=True,
        encoder=depth_encoder_name,
        source_root=source_root,
        weights=weights,
        freeze_encoder=True,
        num_intermediate_layers=1,
        feature_layer=-1,
        num_conditioned_layers=0,
        resize_to=resize_to,
    )
    encoder = DepthAnythingV2TokenEncoder(image_shape_chw, cfg).to(device)
    encoder.eval()
    return encoder


def _compute_depth_outputs(
    image_chw: np.ndarray,
    encoder: DepthAnythingV2TokenEncoder,
    device: str,
) -> tuple[np.ndarray, torch.Tensor, tuple[int, int], int, tuple[int, int]]:
    image_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        preprocessed = encoder._preprocess(image_tensor)
        patch_tokens, _ = encoder.forward_patches_and_cls(image_tensor, flatten_patches=False)
        depth_map = encoder.model(preprocessed).detach().cpu().numpy()[0]

    proc_h, proc_w = int(preprocessed.shape[-2]), int(preprocessed.shape[-1])
    patch_size = int(encoder.patch_size)
    patch_rows = proc_h // patch_size
    patch_cols = proc_w // patch_size
    if patch_rows * patch_cols != int(patch_tokens.shape[1]):
        raise RuntimeError(
            "Patch grid size does not match token count: "
            f"patch_rows={patch_rows}, patch_cols={patch_cols}, num_tokens={patch_tokens.shape[1]}"
        )

    return depth_map, patch_tokens.squeeze(0).detach().cpu(), (patch_rows, patch_cols), patch_size, (proc_h, proc_w)


def _ordered_core_patches(
    pixels_rc: np.ndarray,
    image_hw: tuple[int, int],
    proc_hw: tuple[int, int],
    patch_hw: tuple[int, int],
) -> list[tuple[int, int]]:
    image_h, image_w = image_hw
    proc_h, proc_w = proc_hw
    patch_rows, patch_cols = patch_hw
    row_scale = proc_h / image_h
    col_scale = proc_w / image_w
    patch_h_px = max(proc_h // patch_rows, 1)
    patch_w_px = max(proc_w // patch_cols, 1)

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row_px, col_px in pixels_rc:
        proc_row = int(np.clip(np.floor(row_px * row_scale), 0, proc_h - 1))
        proc_col = int(np.clip(np.floor(col_px * col_scale), 0, proc_w - 1))
        patch_row = min(proc_row // patch_h_px, patch_rows - 1)
        patch_col = min(proc_col // patch_w_px, patch_cols - 1)
        core = (patch_row, patch_col)
        if core in seen:
            continue
        seen.add(core)
        ordered.append(core)

    return ordered


def _sample_point_indices(total_points: int, sample_count: int) -> list[int]:
    if total_points <= 0:
        return []
    if sample_count <= 0 or sample_count >= total_points:
        return list(range(total_points))
    sampled = np.linspace(0, total_points - 1, num=sample_count)
    return np.unique(np.round(sampled).astype(int)).tolist()


def _bounded_pair_indices(base_idx: int, frac: float, upper_bound: int) -> list[int]:
    if upper_bound <= 1:
        return [0]

    if frac < 0.5:
        low, high = base_idx - 1, base_idx
    else:
        low, high = base_idx, base_idx + 1

    if low < 0:
        low, high = 0, 1
    if high >= upper_bound:
        low, high = upper_bound - 2, upper_bound - 1

    return [low, high]


def _point_quad_union_patches(
    pixels_rc: np.ndarray,
    image_hw: tuple[int, int],
    proc_hw: tuple[int, int],
    patch_hw: tuple[int, int],
    sample_count: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    image_h, image_w = image_hw
    proc_h, proc_w = proc_hw
    patch_rows, patch_cols = patch_hw
    patch_h_px = proc_h / patch_rows
    patch_w_px = proc_w / patch_cols
    row_scale = proc_h / image_h
    col_scale = proc_w / image_w

    sampled_indices = _sample_point_indices(len(pixels_rc), sample_count)
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for point_idx in sampled_indices:
        row_px, col_px = pixels_rc[point_idx]
        proc_row = np.clip(row_px * row_scale, 0.0, max(proc_h - 1, 0))
        proc_col = np.clip(col_px * col_scale, 0.0, max(proc_w - 1, 0))

        row_float = proc_row / patch_h_px
        col_float = proc_col / patch_w_px
        base_row = int(np.clip(np.floor(row_float), 0, patch_rows - 1))
        base_col = int(np.clip(np.floor(col_float), 0, patch_cols - 1))
        row_frac = float(row_float - np.floor(row_float))
        col_frac = float(col_float - np.floor(col_float))
        row_candidates = _bounded_pair_indices(base_row, row_frac, patch_rows)
        col_candidates = _bounded_pair_indices(base_col, col_frac, patch_cols)

        for patch_row in row_candidates:
            for patch_col in col_candidates:
                if patch_row < 0 or patch_row >= patch_rows or patch_col < 0 or patch_col >= patch_cols:
                    continue
                coord = (patch_row, patch_col)
                if coord in seen:
                    continue
                seen.add(coord)
                ordered.append(coord)

    return ordered, sampled_indices


def _expand_patch_coords(
    core_patch_coords_rc: list[tuple[int, int]],
    patch_hw: tuple[int, int],
    neighborhood: int,
    head_core_count: int,
    tail_core_count: int,
) -> list[tuple[int, int]]:
    patch_rows, patch_cols = patch_hw
    if neighborhood <= 0:
        return list(core_patch_coords_rc)

    if head_core_count <= 0 and tail_core_count <= 0:
        anchor_coords = core_patch_coords_rc
    else:
        anchor_coords = []
        if head_core_count > 0:
            anchor_coords.extend(core_patch_coords_rc[:head_core_count])
        if tail_core_count > 0:
            anchor_coords.extend(core_patch_coords_rc[-tail_core_count:])

    offsets: list[tuple[int, int]] = [(0, 0)]
    for radius in range(1, neighborhood + 1):
        radius_offsets = []
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                if max(abs(d_row), abs(d_col)) != radius:
                    continue
                radius_offsets.append((d_row, d_col))
        radius_offsets.sort(key=lambda item: (abs(item[0]) + abs(item[1]), item[0], item[1]))
        offsets.extend(radius_offsets)

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for patch_row, patch_col in anchor_coords:
        for d_row, d_col in offsets:
            nbr = (patch_row + d_row, patch_col + d_col)
            if nbr[0] < 0 or nbr[0] >= patch_rows or nbr[1] < 0 or nbr[1] >= patch_cols:
                continue
            if nbr in seen:
                continue
            seen.add(nbr)
            ordered.append(nbr)

    return ordered


def _patch_indices_linear(patch_coords_rc: list[tuple[int, int]], patch_cols: int) -> list[int]:
    return [row * patch_cols + col for row, col in patch_coords_rc]


def _selected_patch_tokens(
    patch_tokens: torch.Tensor,
    selected_patch_indices: list[int],
) -> torch.Tensor:
    if not selected_patch_indices:
        return patch_tokens[:0]
    index_tensor = torch.tensor(selected_patch_indices, dtype=torch.long)
    return patch_tokens.index_select(0, index_tensor)


def _draw_patch_rectangles(ax, canvas_h: int, canvas_w: int, patch_rows: int, patch_cols: int, patch_coords_rc):
    patch_h_px = canvas_h / patch_rows
    patch_w_px = canvas_w / patch_cols
    for row, col in patch_coords_rc:
        ax.add_patch(
            Rectangle(
                (col * patch_w_px, row * patch_h_px),
                patch_w_px,
                patch_h_px,
                facecolor=(1.0, 0.85, 0.1, 0.22),
                edgecolor=(1.0, 0.75, 0.0, 0.9),
                linewidth=1.0,
            )
        )


def _plot_projected_trajectory(ax, pixels_rc: np.ndarray):
    xs = pixels_rc[:, 1]
    ys = pixels_rc[:, 0]
    colors = np.linspace(0.0, 1.0, len(xs))
    ax.plot(xs, ys, color="cyan", linewidth=1.2, alpha=0.9)
    ax.scatter(xs, ys, c=colors, cmap="plasma", s=10, edgecolors="none")
    if len(xs) > 0:
        ax.scatter([xs[0]], [ys[0]], c=["lime"], s=40, marker="o")
        ax.scatter([xs[-1]], [ys[-1]], c=["red"], s=40, marker="x")


def _rescale_pixels_rc(
    pixels_rc: np.ndarray,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = pixels_rc.astype(np.float32).copy()
    scaled[:, 0] *= dst_h / src_h
    scaled[:, 1] *= dst_w / src_w
    return scaled


def _visualize(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    dense_pixels_rc: np.ndarray,
    selected_patch_coords_rc: list[tuple[int, int]],
    patch_grid_hw: tuple[int, int],
    output_png: Path,
    title: str,
) -> None:
    image_h, image_w = rgb_image.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]
    patch_rows, patch_cols = patch_grid_hw
    depth_pixels_rc = _rescale_pixels_rc(dense_pixels_rc, (image_h, image_w), (depth_h, depth_w))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB + Projected Base Trajectory")
    _plot_projected_trajectory(axes[0], dense_pixels_rc)
    if patch_rows > 0 and patch_cols > 0:
        _draw_patch_rectangles(axes[0], image_h, image_w, patch_rows, patch_cols, selected_patch_coords_rc)
    axes[0].set_axis_off()

    axes[1].imshow(depth_map, cmap="viridis")
    axes[1].set_title("DepthAnything Depth + Selected Patches")
    _plot_projected_trajectory(axes[1], depth_pixels_rc)
    if patch_rows > 0 and patch_cols > 0:
        _draw_patch_rectangles(axes[1], depth_h, depth_w, patch_rows, patch_cols, selected_patch_coords_rc)
    axes[1].set_axis_off()

    mask = np.zeros((patch_rows, patch_cols), dtype=np.float32)
    order_lookup = {coord: idx + 1 for idx, coord in enumerate(selected_patch_coords_rc)}
    for row, col in selected_patch_coords_rc:
        mask[row, col] = 1.0
    axes[2].imshow(mask, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[2].set_title("Patch Grid Selection Order")
    axes[2].set_xticks(np.arange(patch_cols))
    axes[2].set_yticks(np.arange(patch_rows))
    axes[2].set_xlim(-0.5, patch_cols - 0.5)
    axes[2].set_ylim(patch_rows - 0.5, -0.5)
    axes[2].grid(color="lightgray", linewidth=0.5)
    for (row, col), order in order_lookup.items():
        axes[2].text(col, row, str(order), ha="center", va="center", fontsize=8, color="tab:red")

    fig.suptitle(title, fontsize=11)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _ensure_hf_dataset_cache()

    dataset_root = _find_local_dataset_root(args.dataset_repo_id, args.dataset_root)
    dataset = LeRobotDataset(args.dataset_repo_id, root=str(dataset_root))
    sample_index = _resolve_sample_index(dataset, args.sample_index, args.episode_index, args.frame_index)
    sample = dataset[sample_index]

    policy_dir = _resolve_policy_dir(
        explicit_policy_dir=args.base_policy_dir,
        wandb_run_id=args.base_policy_wandb_id,
        step=args.base_policy_step,
    )
    policy = _load_policy(policy_dir, device=args.device)

    rgb_image = _extract_rgb_image(sample, args.camera_key)
    image_h, image_w = rgb_image.shape[:2]
    image_chw = np.transpose(rgb_image, (2, 0, 1)).astype(np.float32)

    action_chunk = _predict_base_action_chunk(sample, policy, args.device, args.chunk_horizon)
    current_eef_pos = _current_eef_position(sample)
    future_points_world, scaled_delta_pos = _approximate_future_eef_positions(current_eef_pos, action_chunk, args.task)
    dense_points_world = _densify_polyline_world(future_points_world, args.traj_samples_per_segment)
    dense_pixels_rc = _project_world_points_to_pixels(
        task_name=args.task,
        camera_key=args.camera_key,
        world_points=dense_points_world,
        image_height=image_h,
        image_width=image_w,
    )

    depth_device = args.depth_device or args.device
    depth_source_root = args.depth_source_root or str(
        Path(__file__).resolve().parents[3] / "third_party" / "Depth-Anything-V2"
    )
    depth_encoder = _build_depth_encoder(
        image_shape_chw=tuple(image_chw.shape),
        source_root=depth_source_root,
        weights=args.depth_weights,
        depth_encoder_name=args.depth_encoder,
        resize_to=args.depth_resize_to,
        device=depth_device,
    )

    depth_map, patch_tokens, patch_grid_hw, _, proc_hw = _compute_depth_outputs(
        image_chw=image_chw,
        encoder=depth_encoder,
        device=depth_device,
    )
    core_patch_coords_rc = _ordered_core_patches(
        pixels_rc=dense_pixels_rc,
        image_hw=(image_h, image_w),
        proc_hw=proc_hw,
        patch_hw=patch_grid_hw,
    )
    sampled_point_indices: list[int] = []
    if args.selection_mode == "point_quad_union":
        selected_patch_coords_rc, sampled_point_indices = _point_quad_union_patches(
            pixels_rc=dense_pixels_rc,
            image_hw=(image_h, image_w),
            proc_hw=proc_hw,
            patch_hw=patch_grid_hw,
            sample_count=args.selection_point_count,
        )
    else:
        selected_patch_coords_rc = _expand_patch_coords(
            core_patch_coords_rc=core_patch_coords_rc,
            patch_hw=patch_grid_hw,
            neighborhood=args.patch_neighborhood,
            head_core_count=args.head_core_count,
            tail_core_count=args.tail_core_count,
        )
    patch_rows, patch_cols = patch_grid_hw
    selected_patch_indices = _patch_indices_linear(selected_patch_coords_rc, patch_cols)
    selected_patch_vectors = _selected_patch_tokens(patch_tokens, selected_patch_indices)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    if prefix is None:
        episode_index = int(sample["episode_index"].item())
        frame_index = int(sample["frame_index"].item())
        prefix = f"sample{sample_index:05d}_ep{episode_index:03d}_frame{frame_index:04d}_{_camera_name_from_key(args.camera_key)}"

    output_png = output_dir / f"{prefix}.png"
    output_json = output_dir / f"{prefix}.json"
    output_tokens = output_dir / f"{prefix}.pt"

    title = (
        f"sample={sample_index}  task={sample['task']}  camera={args.camera_key}  "
        f"horizon={args.chunk_horizon}  selected_patches={len(selected_patch_indices)}"
    )
    _visualize(
        rgb_image=rgb_image,
        depth_map=depth_map,
        dense_pixels_rc=dense_pixels_rc,
        selected_patch_coords_rc=selected_patch_coords_rc,
        patch_grid_hw=patch_grid_hw,
        output_png=output_png,
        title=title,
    )

    metadata = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "policy_dir": str(policy_dir),
        "sample_index": int(sample_index),
        "episode_index": int(sample["episode_index"].item()),
        "frame_index": int(sample["frame_index"].item()),
        "task": sample["task"],
        "camera_key": args.camera_key,
        "chunk_horizon": int(args.chunk_horizon),
        "traj_samples_per_segment": int(args.traj_samples_per_segment),
        "selection_mode": args.selection_mode,
        "patch_neighborhood": int(args.patch_neighborhood),
        "head_core_count": int(args.head_core_count),
        "tail_core_count": int(args.tail_core_count),
        "selection_point_count": int(args.selection_point_count),
        "image_hw": [int(image_h), int(image_w)],
        "depth_preprocessed_hw": [int(proc_hw[0]), int(proc_hw[1])],
        "patch_grid_hw": [int(patch_rows), int(patch_cols)],
        "core_patch_coords_rc": [[int(row), int(col)] for row, col in core_patch_coords_rc],
        "core_patch_indices_linear": _patch_indices_linear(core_patch_coords_rc, patch_cols),
        "sampled_point_indices": [int(idx) for idx in sampled_point_indices],
        "selected_patch_coords_rc": [[int(row), int(col)] for row, col in selected_patch_coords_rc],
        "selected_patch_indices_linear": [int(idx) for idx in selected_patch_indices],
        "dense_projected_pixels_rc": dense_pixels_rc.astype(int).tolist(),
        "predicted_action_chunk": action_chunk.numpy().tolist(),
        "scaled_delta_pos_m_approx": scaled_delta_pos.tolist(),
        "future_eef_positions_world": future_points_world.tolist(),
    }
    output_json.write_text(json.dumps(metadata, indent=2))

    torch.save(
        {
            "patch_tokens": patch_tokens,
            "selected_patch_tokens": selected_patch_vectors,
            "selected_patch_indices": torch.tensor(selected_patch_indices, dtype=torch.long),
            "selected_patch_coords_rc": torch.tensor(selected_patch_coords_rc, dtype=torch.long)
            if selected_patch_coords_rc
            else torch.empty((0, 2), dtype=torch.long),
            "metadata": metadata,
            "depth_encoder_cfg": asdict(depth_encoder.cfg),
        },
        output_tokens,
    )

    print(f"Saved visualization to {output_png}")
    print(f"Saved metadata to {output_json}")
    print(f"Saved patch tokens to {output_tokens}")
    print(f"Selected patch indices: {selected_patch_indices}")


if __name__ == "__main__":
    main()
