from __future__ import annotations

import numpy as np
import torch


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


def _camera_name_from_key(camera_key: str) -> str:
    prefix = "observation.images."
    if not camera_key.startswith(prefix):
        raise ValueError(f"camera_key must start with {prefix!r}, got {camera_key!r}")
    return camera_key[len(prefix) :]


def _canonical_task_name(task_name: str) -> str:
    return ALIAS_TO_CANONICAL_TASK.get(task_name, task_name)


def gather_selected_patch_tokens(patch_tokens: torch.Tensor, patch_indices: torch.Tensor) -> torch.Tensor:
    """Gather a fixed number of patch tokens, leaving padded slots as zeros."""
    if patch_indices.dim() != 2:
        raise ValueError(f"Expected patch_indices with shape [B, M], got {tuple(patch_indices.shape)}")

    safe_indices = patch_indices.clamp(min=0)
    expanded_index = safe_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
    gathered = torch.gather(patch_tokens, dim=1, index=expanded_index)
    valid_mask = patch_indices.ge(0).unsqueeze(-1)
    return gathered * valid_mask.to(gathered.dtype)


class BaseTrajectoryPatchSelector:
    """Selects pre-decoder DepthAnything patches along the base action trajectory."""

    def __init__(
        self,
        *,
        task_name: str,
        camera_keys: tuple[str, ...],
        image_hw: tuple[int, int],
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        primitive_action_dim: int,
    ):
        self.task_name = _canonical_task_name(task_name)
        self.camera_keys = tuple(camera_keys)
        self.image_hw = tuple(int(v) for v in image_hw)
        self.primitive_action_dim = int(primitive_action_dim)
        self.action_min = action_min.detach().cpu().float()
        self.action_max = action_max.detach().cpu().float()
        self.state_mean = state_mean.detach().cpu().float()
        self.state_std = state_std.detach().cpu().float()
        (
            self.ctrl_input_min,
            self.ctrl_input_max,
            self.ctrl_output_min,
            self.ctrl_output_max,
        ) = self._single_arm_controller_pos_bounds(self.task_name)
        self.world_to_camera_by_key = self._build_world_to_camera_matrices()

    def _single_arm_controller_pos_bounds(
        self,
        task_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from robosuite import load_composite_controller_config

        if task_name not in ENV_ROBOTS:
            raise ValueError(f"Unknown task {task_name!r}")

        robots = ENV_ROBOTS[task_name]
        controller_cfg = load_composite_controller_config(robot=robots[0])
        body_parts = controller_cfg.get("body_parts", {})
        arm_cfg = None

        if "arms" in body_parts and isinstance(body_parts["arms"], dict):
            arm_cfg = body_parts["arms"].get("right")
        if arm_cfg is None:
            arm_cfg = body_parts.get("right")
        if arm_cfg is None:
            raise KeyError(f"Could not find single-arm controller config for task {task_name!r}")

        input_min = np.broadcast_to(np.asarray(arm_cfg["input_min"], dtype=np.float32), (6,)).copy()
        input_max = np.broadcast_to(np.asarray(arm_cfg["input_max"], dtype=np.float32), (6,)).copy()
        output_min = np.asarray(arm_cfg["output_min"], dtype=np.float32).copy()
        output_max = np.asarray(arm_cfg["output_max"], dtype=np.float32).copy()
        return (
            torch.from_numpy(input_min[:3]).float(),
            torch.from_numpy(input_max[:3]).float(),
            torch.from_numpy(output_min[:3]).float(),
            torch.from_numpy(output_max[:3]).float(),
        )

    def _build_world_to_camera_matrices(self) -> dict[str, np.ndarray]:
        import robosuite
        from robosuite import load_composite_controller_config
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        if self.task_name not in ENV_ROBOTS:
            raise ValueError(f"Unknown task {self.task_name!r}")

        robots = ENV_ROBOTS[self.task_name]
        controller_configs = load_composite_controller_config(robot=robots[0])
        if "composite_controller_specific_configs" in controller_configs:
            controller_configs["composite_controller_specific_configs"]["ik_input_ref_frame"] = "world"

        camera_names = [_camera_name_from_key(camera_key) for camera_key in self.camera_keys]
        env_kwargs = {
            "env_name": self.task_name,
            "robots": robots,
            "controller_configs": controller_configs,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": False,
            "use_camera_obs": False,
            "camera_names": camera_names,
            "control_freq": 20,
        }
        env = robosuite.make(**env_kwargs)
        env.reset()
        try:
            image_h, image_w = self.image_hw
            matrices = {}
            for camera_key, camera_name in zip(self.camera_keys, camera_names, strict=True):
                matrices[camera_key] = get_camera_transform_matrix(
                    env.sim,
                    camera_name=camera_name,
                    camera_height=image_h,
                    camera_width=image_w,
                )
        finally:
            env.close()
        return matrices

    def _unscale_base_action(self, base_action: torch.Tensor) -> torch.Tensor:
        base_action = base_action.float().detach().cpu()
        if base_action.dim() == 1:
            base_action = base_action.unsqueeze(0)
        if base_action.shape[-1] % self.primitive_action_dim != 0:
            raise ValueError(
                "Base action does not align with primitive_action_dim. "
                f"Got shape {tuple(base_action.shape)} and primitive_action_dim={self.primitive_action_dim}."
            )

        horizon = base_action.shape[-1] // self.primitive_action_dim
        chunk = base_action.reshape(base_action.shape[0], horizon, self.primitive_action_dim).clamp(-1.0, 1.0)
        action_min = self.action_min.view(1, 1, -1)
        action_max = self.action_max.view(1, 1, -1)
        return action_min + (chunk + 1.0) * (action_max - action_min) / 2.0

    def _unstandardize_eef_position(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float().detach().cpu()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_std = torch.maximum(self.state_std, torch.tensor(1e-8))
        raw_state = state * state_std.view(1, -1) + self.state_mean.view(1, -1)
        return raw_state[:, :3]

    def _scale_delta_pos(self, raw_delta_pos: torch.Tensor) -> torch.Tensor:
        raw_clipped = torch.clamp(raw_delta_pos, self.ctrl_input_min.view(1, 1, 3), self.ctrl_input_max.view(1, 1, 3))
        input_mid = (self.ctrl_input_max + self.ctrl_input_min) / 2.0
        output_mid = (self.ctrl_output_max + self.ctrl_output_min) / 2.0
        scale = (self.ctrl_output_max - self.ctrl_output_min) / torch.clamp(
            self.ctrl_input_max - self.ctrl_input_min,
            min=1e-8,
        )
        return (raw_clipped - input_mid.view(1, 1, 3)) * scale.view(1, 1, 3) + output_mid.view(1, 1, 3)

    def _future_points_world(self, obs_state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        current_eef = self._unstandardize_eef_position(obs_state)
        raw_action_chunk = self._unscale_base_action(base_action)
        delta_pos = self._scale_delta_pos(raw_action_chunk[:, :, :3])
        future_points = current_eef.unsqueeze(1) + torch.cumsum(delta_pos, dim=1)
        return future_points

    def _project_world_points_to_pixels(self, camera_key: str, world_points: np.ndarray) -> np.ndarray:
        from robosuite.utils.camera_utils import project_points_from_world_to_camera

        image_h, image_w = self.image_hw
        world_to_camera = self.world_to_camera_by_key[camera_key]
        return project_points_from_world_to_camera(
            world_points,
            world_to_camera_transform=world_to_camera,
            camera_height=image_h,
            camera_width=image_w,
        )

    @staticmethod
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

    def _point_quad_union_indices(
        self,
        *,
        pixels_rc: np.ndarray,
        proc_hw: tuple[int, int],
        patch_hw: tuple[int, int],
        max_patches: int,
    ) -> list[int]:
        image_h, image_w = self.image_hw
        proc_h, proc_w = proc_hw
        patch_rows, patch_cols = patch_hw
        patch_h_px = proc_h / patch_rows
        patch_w_px = proc_w / patch_cols
        row_scale = proc_h / image_h
        col_scale = proc_w / image_w

        ordered: list[int] = []
        seen: set[int] = set()
        for row_px, col_px in pixels_rc:
            proc_row = np.clip(row_px * row_scale, 0.0, max(proc_h - 1, 0))
            proc_col = np.clip(col_px * col_scale, 0.0, max(proc_w - 1, 0))

            row_float = proc_row / patch_h_px
            col_float = proc_col / patch_w_px
            base_row = int(np.clip(np.floor(row_float), 0, patch_rows - 1))
            base_col = int(np.clip(np.floor(col_float), 0, patch_cols - 1))
            row_frac = float(row_float - np.floor(row_float))
            col_frac = float(col_float - np.floor(col_float))
            row_candidates = self._bounded_pair_indices(base_row, row_frac, patch_rows)
            col_candidates = self._bounded_pair_indices(base_col, col_frac, patch_cols)

            for patch_row in row_candidates:
                for patch_col in col_candidates:
                    linear_idx = patch_row * patch_cols + patch_col
                    if linear_idx in seen:
                        continue
                    seen.add(linear_idx)
                    ordered.append(linear_idx)
                    if len(ordered) >= max_patches:
                        return ordered
        return ordered

    def select_patch_indices(
        self,
        *,
        obs_state: torch.Tensor,
        base_action: torch.Tensor,
        camera_key: str,
        proc_hw: tuple[int, int],
        patch_hw: tuple[int, int],
        max_patches: int,
    ) -> torch.Tensor:
        future_points = self._future_points_world(obs_state, base_action)
        device = base_action.device
        batch_size = future_points.shape[0]
        patch_indices = torch.full((batch_size, max_patches), -1, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            world_points = future_points[batch_idx].numpy()
            pixels_rc = self._project_world_points_to_pixels(camera_key, world_points)
            linear_indices = self._point_quad_union_indices(
                pixels_rc=pixels_rc,
                proc_hw=proc_hw,
                patch_hw=patch_hw,
                max_patches=max_patches,
            )
            if not linear_indices:
                continue
            count = min(len(linear_indices), max_patches)
            patch_indices[batch_idx, :count] = torch.tensor(linear_indices[:count], dtype=torch.long, device=device)

        return patch_indices
