# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy import common_utils
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.networks.depth_patch_state import (
    BaseTrajectoryPatchSelector,
    gather_selected_patch_tokens,
)
from resfit.rl_finetuning.off_policy.networks.encoder import DepthAnythingV2TokenEncoder, VitEncoder
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.critic import Critic


class QAgent(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        cfg: QAgentConfig,
        residual_actor: bool = False,
        base_action_dim: int | None = None,
        primitive_action_dim: int | None = None,
        adaptive_horizons: tuple[int, ...] | None = None,
        adaptive_horizon_entropy_reg: float = 0.0,
        task_name: str | None = None,
        action_scaler_min: torch.Tensor | None = None,
        action_scaler_max: torch.Tensor | None = None,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        base_policy: nn.Module | None = None,
    ):
        """Initialize the Q-agent.

        Parameters
        ----------
        obs_shape : tuple[int, int, int]
            Shape (C, H, W) for **a single camera** image.  When multiple
            cameras are used the same shape is assumed for every view.
        prop_shape : tuple[int]
            Shape of the proprioceptive (low-dimensional) observation vector.
        action_dim : int
            Number of action dimensions.
        rl_cameras : list[str] | str
            Name(s) of the camera images to be used by the RL policy.
            These are keys into the env's observation. A single string
            is accepted for backwards-compatibility but the preferred interface
            is to pass a list of camera names.
        cfg : QAgentConfig
            Hyper-parameter configuration dataclass.
        """
        super().__init__()
        # Normalise *rl_cameras* to a list for unified processing
        if isinstance(rl_cameras, str):
            rl_cameras = [rl_cameras]
        assert len(rl_cameras) > 0, "At least one camera must be provided"

        self.rl_cameras = rl_cameras
        self.cfg = cfg
        self.residual_actor = residual_actor
        self.use_residual_image_encoder = bool(getattr(self.cfg, "use_residual_image_encoder", True))
        self.uses_base_act_encoder_state = bool(getattr(self.cfg, "use_base_act_encoder_state", False))
        self.state_mean = None if state_mean is None else state_mean.detach().cpu().float()
        self.state_std = None if state_std is None else state_std.detach().cpu().float()
        self.critic_action_dim = int(action_dim)
        self.base_action_dim = int(base_action_dim if base_action_dim is not None else action_dim)
        self.primitive_action_dim = int(primitive_action_dim if primitive_action_dim is not None else self.base_action_dim)
        if self.base_action_dim % self.primitive_action_dim != 0:
            raise ValueError(
                "base_action_dim must be divisible by primitive_action_dim. "
                f"Got base_action_dim={self.base_action_dim}, primitive_action_dim={self.primitive_action_dim}."
            )
        self.base_chunk_horizon = self.base_action_dim // self.primitive_action_dim
        self.adaptive_horizons = tuple(sorted(adaptive_horizons or ()))
        self.num_adaptive_horizons = len(self.adaptive_horizons)
        self.uses_adaptive_horizons = self.num_adaptive_horizons > 0
        self.adaptive_horizon_entropy_reg = float(adaptive_horizon_entropy_reg)
        self.max_action_horizon = max(self.adaptive_horizons) if self.adaptive_horizons else 1
        if self.uses_adaptive_horizons:
            if not residual_actor:
                raise ValueError("Adaptive macro horizons are only implemented for residual actors.")
            if self.base_action_dim % self.max_action_horizon != 0:
                raise ValueError(
                    "base_action_dim must be divisible by the maximum adaptive horizon. "
                    f"Got base_action_dim={self.base_action_dim}, max_horizon={self.max_action_horizon}."
                )
            expected_primitive_action_dim = self.base_action_dim // self.max_action_horizon
            if self.primitive_action_dim != expected_primitive_action_dim:
                raise ValueError(
                    "primitive_action_dim does not match the adaptive horizon layout. "
                    f"Expected {expected_primitive_action_dim}, got {self.primitive_action_dim}."
                )
            expected_critic_dim = self.base_action_dim + self.num_adaptive_horizons
            if self.critic_action_dim != expected_critic_dim:
                raise ValueError(
                    "Adaptive horizon critic action dimension mismatch. "
                    f"Expected {expected_critic_dim}, got {self.critic_action_dim}."
                )
        self.depth_cache_keys_by_camera = {
            cam_name: cam_name.replace("observation.images.", "observation.depth_cls.", 1) for cam_name in self.rl_cameras
        }
        self.depth_patch_cache_keys_by_camera = {
            cam_name: cam_name.replace("observation.images.", "observation.depth_patch_tokens.", 1)
            for cam_name in self.rl_cameras
        }
        self.base_act_encoder_cache_key = "observation.base_act_encoder_tokens"

        # Build the per-camera encoders *after* `self.rl_cameras` is defined so
        # that the helper function can iterate over them.
        self.encoders: nn.ModuleList = (
            self._build_encoders(obs_shape) if self.use_residual_image_encoder else nn.ModuleList()
        )

        # All encoders share the same architecture ⇒ repr / patch dim are identical.
        if self.use_residual_image_encoder:
            sample_encoder = self.encoders[0]
            repr_dim_single = int(sample_encoder.repr_dim)  # type: ignore[attr-defined]
            patch_repr_dim = int(sample_encoder.patch_repr_dim)  # type: ignore[attr-defined]
        else:
            sample_encoder = None
            repr_dim_single = 0
            patch_repr_dim = int(self.cfg.vit.embed_dim)

        self.depth_anything_v2_encoder: DepthAnythingV2TokenEncoder | None = None
        self.depth_cls_projectors: nn.ModuleList | None = None
        self.depth_patch_encoder: DepthAnythingV2TokenEncoder | None = None
        self.depth_patch_projectors: nn.ModuleList | None = None
        self.depth_patch_selector: BaseTrajectoryPatchSelector | None = None
        self.depth_patch_camera_keys: tuple[str, ...] = ()
        self.depth_patch_camera_to_projector_idx: dict[str, int] = {}
        self.depth_patch_max_tokens_per_camera = 0
        self.depth_patch_selection_mode = "trajectory"
        self.depth_patch_token_scale_raw: nn.Parameter | None = None
        self.depth_patch_token_dropout_enabled = True
        self.depth_patch_update_step = 0
        self.base_act_encoder_projector: nn.Module | None = None
        self.base_act_encoder_num_tokens = 0
        depth_cfg = getattr(self.cfg, "depth_anything_v2_conditioning", None)
        depth_patch_cfg = getattr(self.cfg, "depth_anything_v2_patch_state", None)
        self.uses_depth_anything_v2_conditioning = bool(depth_cfg is not None and depth_cfg.enabled)
        self.uses_depth_patch_state = bool(depth_patch_cfg is not None and depth_patch_cfg.enabled)
        # Depth patch tokens are selected in the original image coordinate system.
        # Disable geometric image augmentation so the residual RGB tokens stay aligned
        # with the projected depth patch locations during training.
        self.image_geom_augmentation_enabled = not self.uses_depth_patch_state
        self.num_depth_conditioned_layers = 0

        if self.uses_depth_anything_v2_conditioning:
            if not self.use_residual_image_encoder:
                raise ValueError(
                    "DepthAnythingV2 CLS conditioning injects tokens into the residual MinViT encoder, "
                    "so it requires agent.use_residual_image_encoder=True."
                )
            if self.cfg.enc_type != "vit":
                raise ValueError(
                    "DepthAnythingV2 CLS conditioning is currently only implemented for enc_type='vit'. "
                    f"Got enc_type={self.cfg.enc_type!r}."
                )
            assert sample_encoder is not None
            if not isinstance(sample_encoder, VitEncoder):
                raise TypeError(
                    "DepthAnythingV2 CLS conditioning expects VitEncoder instances. "
                    f"Got {type(sample_encoder).__name__}."
                )

            total_vit_layers = int(sample_encoder.vit.depth)
            requested_layers = int(depth_cfg.num_conditioned_layers) or total_vit_layers
            if requested_layers > total_vit_layers:
                raise ValueError(
                    "DepthAnythingV2 conditioning requested more layers than the MinViT depth. "
                    f"Requested {requested_layers}, but MinViT depth is {total_vit_layers}."
                )

            self.depth_anything_v2_encoder = DepthAnythingV2TokenEncoder(obs_shape, depth_cfg).to(self.cfg.device)
            self.depth_cls_projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.depth_anything_v2_encoder.patch_repr_dim, patch_repr_dim),
                        nn.LayerNorm(patch_repr_dim),
                    )
                    for _ in self.rl_cameras
                ]
            )
            self.num_depth_conditioned_layers = requested_layers

            print(common_utils.wrap_ruler("depth-anything-v2 cls conditioning"))
            print(
                f"enabled=True encoder={depth_cfg.encoder} freeze_encoder={depth_cfg.freeze_encoder} "
                f"conditioned_layers={self.num_depth_conditioned_layers}"
            )

        if self.uses_depth_patch_state:
            if depth_patch_cfg is None:
                raise RuntimeError("DepthAnything patch state is enabled but its config is missing.")
            if task_name is None:
                raise ValueError("task_name is required when depth_anything_v2_patch_state.enabled=True.")
            if action_scaler_min is None or action_scaler_max is None:
                raise ValueError("Action scaler limits are required when depth_anything_v2_patch_state.enabled=True.")
            if state_mean is None or state_std is None:
                raise ValueError("State normalization statistics are required when depth_anything_v2_patch_state.enabled=True.")

            requested_cameras = tuple(depth_patch_cfg.camera_keys) if depth_patch_cfg.camera_keys else (self.rl_cameras[0],)
            missing_cameras = [camera_key for camera_key in requested_cameras if camera_key not in self.rl_cameras]
            if missing_cameras:
                raise ValueError(
                    "Depth patch state cameras must be a subset of rl_cameras. "
                    f"Missing: {missing_cameras}, rl_cameras={self.rl_cameras}."
                )

            self.depth_patch_camera_keys = requested_cameras
            self.depth_patch_camera_to_projector_idx = {
                camera_key: camera_idx for camera_idx, camera_key in enumerate(self.depth_patch_camera_keys)
            }
            self.depth_patch_selection_mode = str(getattr(depth_patch_cfg, "selection_mode", "trajectory"))
            depth_patch_token_budget = int(depth_patch_cfg.max_patches_per_camera) or (self.base_chunk_horizon * 4)

            self.depth_patch_encoder = DepthAnythingV2TokenEncoder(obs_shape, depth_patch_cfg).to(self.cfg.device)
            self.depth_patch_max_tokens_per_camera = min(
                depth_patch_token_budget,
                int(self.depth_patch_encoder.num_patch),
            )
            self.depth_patch_projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.depth_patch_encoder.patch_repr_dim, patch_repr_dim),
                        nn.LayerNorm(patch_repr_dim),
                    )
                    for _ in self.depth_patch_camera_keys
                ]
            )
            if bool(getattr(depth_patch_cfg, "zero_init_projector", False)):
                for projector in self.depth_patch_projectors:
                    linear = projector[0]
                    if isinstance(linear, nn.Linear):
                        nn.init.zeros_(linear.weight)
                        if linear.bias is not None:
                            nn.init.zeros_(linear.bias)
            self.depth_patch_token_scale_raw = nn.Parameter(
                self._inverse_softplus_scalar(float(depth_patch_cfg.token_scale_init))
            )
            self.depth_patch_selector = BaseTrajectoryPatchSelector(
                task_name=task_name,
                camera_keys=self.depth_patch_camera_keys,
                image_hw=(obs_shape[1], obs_shape[2]),
                action_min=action_scaler_min,
                action_max=action_scaler_max,
                state_mean=state_mean,
                state_std=state_std,
                primitive_action_dim=self.primitive_action_dim,
            )

            print(common_utils.wrap_ruler("depth-anything-v2 patch state"))
            print(
                f"enabled=True encoder={depth_patch_cfg.encoder} freeze_encoder={depth_patch_cfg.freeze_encoder} "
                f"cameras={self.depth_patch_camera_keys} max_tokens_per_camera={self.depth_patch_max_tokens_per_camera} "
                f"selection_mode={self.depth_patch_selection_mode} "
                f"token_scale_init={depth_patch_cfg.token_scale_init} "
                f"token_scale_warmup_steps={depth_patch_cfg.token_scale_warmup_steps} "
                f"zero_init_projector={depth_patch_cfg.zero_init_projector} "
                f"token_dropout={depth_patch_cfg.token_dropout}"
            )

        if self.uses_base_act_encoder_state:
            if base_policy is None:
                raise ValueError("base_policy is required when agent.use_base_act_encoder_state=True.")
            if self.state_mean is None or self.state_std is None:
                raise ValueError("State normalization statistics are required for base ACT encoder features.")

            base_policy.to(self.cfg.device)
            base_policy.eval()
            for param in base_policy.parameters():
                param.requires_grad = False
            # Keep the frozen base policy out of QAgent.state_dict() and optimizers.
            self.__dict__["base_act_policy"] = base_policy
            base_act_image_keys = tuple(base_policy.config.image_features.keys())
            missing_base_act_images = [camera_key for camera_key in base_act_image_keys if camera_key not in self.rl_cameras]
            if missing_base_act_images:
                raise ValueError(
                    "Base ACT encoder features require replay observations for every base-policy camera. "
                    f"Missing from rl_cameras: {missing_base_act_images}."
                )

            self.base_act_encoder_num_tokens, base_act_encoder_dim = self._infer_base_act_encoder_shape(
                obs_shape=obs_shape,
                prop_shape=prop_shape,
            )
            self.base_act_encoder_projector = nn.Sequential(
                nn.Linear(base_act_encoder_dim, patch_repr_dim),
                nn.LayerNorm(patch_repr_dim),
            )
            print(common_utils.wrap_ruler("base ACT encoder state"))
            print(
                "enabled=True frozen=True "
                f"tokens={self.base_act_encoder_num_tokens} dim={base_act_encoder_dim}->{patch_repr_dim}"
            )

        # Concatenate the patch dimension from every camera (dim=1) → overall
        # representation dimension scales linearly with #cameras.
        repr_dim = repr_dim_single * len(self.rl_cameras)
        if self.uses_base_act_encoder_state:
            repr_dim += patch_repr_dim * self.base_act_encoder_num_tokens
        if self.uses_depth_patch_state:
            repr_dim += patch_repr_dim * self.depth_patch_max_tokens_per_camera * len(self.depth_patch_camera_keys)
        if repr_dim <= 0:
            raise ValueError(
                "QAgent has no feature tokens. Enable at least one of "
                "use_residual_image_encoder, use_base_act_encoder_state, or depth_anything_v2_patch_state."
            )
        print("encoder output dim: ", repr_dim)
        print("patch output dim: ", patch_repr_dim)

        assert len(prop_shape) == 1
        prop_dim = prop_shape[0] if cfg.use_prop else 0

        # create critics & actor
        self.critic = Critic(
            repr_dim=repr_dim,
            patch_repr_dim=patch_repr_dim,
            prop_dim=prop_dim,
            action_dim=self.critic_action_dim,
            cfg=self.cfg.critic,
        )
        self.actor = Actor(
            repr_dim,
            patch_repr_dim,
            prop_dim,
            self.base_action_dim,
            cfg.actor,
            residual_actor=residual_actor,
            horizon_choices=self.adaptive_horizons,
        )

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        print(common_utils.wrap_ruler("encoder weights"))
        print(self.encoders)
        common_utils.count_parameters(self.encoders)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        # Freeze encoder parameters if requested
        if getattr(self.cfg, "freeze_encoder", False):
            for param in self.encoders.parameters():
                param.requires_grad = False
            if self.depth_cls_projectors is not None:
                for param in self.depth_cls_projectors.parameters():
                    param.requires_grad = False
            if self.depth_anything_v2_encoder is not None:
                for param in self.depth_anything_v2_encoder.parameters():
                    param.requires_grad = False
            if self.depth_patch_projectors is not None:
                for param in self.depth_patch_projectors.parameters():
                    param.requires_grad = False
            if self.depth_patch_encoder is not None:
                for param in self.depth_patch_encoder.parameters():
                    param.requires_grad = False
            if self.depth_patch_token_scale_raw is not None:
                self.depth_patch_token_scale_raw.requires_grad = False
            if self.base_act_encoder_projector is not None:
                for param in self.base_act_encoder_projector.parameters():
                    param.requires_grad = False
            print("🧊 Encoder parameters frozen - no gradient updates will be performed")

        # Create optimizers (PyTorch will ignore frozen parameters)
        self.encoder_opt = torch.optim.AdamW(self._encoder_parameters(), lr=self.cfg.critic_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_lr)

        # LR schedulers for warmup (if warmup is enabled)
        self.encoder_scheduler = None
        self.critic_scheduler = None
        self.actor_scheduler = None

        if self.cfg.lr_warmup_steps > 0:
            # LinearLR scheduler that linearly ramps from start_factor to 1.0 over total_iters steps
            # Note: start_factor must be > 0 for LinearLR scheduler
            warmup_start = self.cfg.lr_warmup_start

            # Calculate start factors for each optimizer
            critic_start_factor = warmup_start / self.cfg.critic_lr if self.cfg.critic_lr > 0 else 1e-8
            critic_start_factor = max(critic_start_factor, 1e-8)

            actor_start_factor = warmup_start / self.cfg.actor_lr if self.cfg.actor_lr > 0 else 1e-8
            actor_start_factor = max(actor_start_factor, 1e-8)

            # Create schedulers with appropriate start factors
            self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.encoder_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=actor_start_factor, total_iters=self.cfg.lr_warmup_steps
            )

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)
        if not self.image_geom_augmentation_enabled:
            print("Image geometric augmentation disabled because depth patch state is enabled.")

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: common_utils.MultiCounter | None = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    def _build_encoders(self, obs_shape):
        """Constructs and returns an ``nn.ModuleList`` with one encoder per
        camera based on ``self.cfg.enc_type``.  All encoders share the same
        architecture and therefore yield feature tensors with identical
        dimensions which simplifies feature fusion downstream.
        """

        encoders = nn.ModuleList()

        for _ in self.rl_cameras:
            if self.cfg.enc_type == "vit":
                enc = VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
            else:
                raise AssertionError(f"Unknown encoder type {self.cfg.enc_type}.")

            encoders.append(enc)

        return encoders

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        self.encoders.train(training)
        if self.depth_cls_projectors is not None:
            self.depth_cls_projectors.train(training)
        if self.depth_anything_v2_encoder is not None:
            self.depth_anything_v2_encoder.train(training)
        if self.depth_patch_projectors is not None:
            self.depth_patch_projectors.train(training)
        if self.depth_patch_encoder is not None:
            self.depth_patch_encoder.train(training)
        if self.base_act_encoder_projector is not None:
            self.base_act_encoder_projector.train(training)
        base_act_policy = self.__dict__.get("base_act_policy")
        if base_act_policy is not None:
            base_act_policy.eval()
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    def _encoder_parameters(self):
        params = list(self.encoders.parameters())
        if self.depth_cls_projectors is not None:
            params.extend(self.depth_cls_projectors.parameters())
        if self.depth_anything_v2_encoder is not None:
            params.extend(self.depth_anything_v2_encoder.parameters())
        if self.depth_patch_projectors is not None:
            params.extend(self.depth_patch_projectors.parameters())
        if self.depth_patch_encoder is not None:
            params.extend(self.depth_patch_encoder.parameters())
        if self.depth_patch_token_scale_raw is not None:
            params.append(self.depth_patch_token_scale_raw)
        if self.base_act_encoder_projector is not None:
            params.extend(self.base_act_encoder_projector.parameters())
        return params

    @staticmethod
    def _inverse_softplus_scalar(value: float) -> torch.Tensor:
        value_tensor = torch.tensor(max(float(value), 1e-8), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def _depth_patch_token_scale(self) -> torch.Tensor | None:
        if self.depth_patch_token_scale_raw is None:
            return None
        scale = torch.nn.functional.softplus(self.depth_patch_token_scale_raw)
        depth_patch_cfg = getattr(self.cfg, "depth_anything_v2_patch_state", None)
        warmup_steps = int(getattr(depth_patch_cfg, "token_scale_warmup_steps", 0))
        if warmup_steps > 0:
            multiplier = min(1.0, float(self.depth_patch_update_step) / float(warmup_steps))
            scale = scale * scale.new_tensor(multiplier)
        return scale

    def _depth_patch_token_scale_multiplier(self) -> float:
        depth_patch_cfg = getattr(self.cfg, "depth_anything_v2_patch_state", None)
        warmup_steps = int(getattr(depth_patch_cfg, "token_scale_warmup_steps", 0))
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.depth_patch_update_step) / float(warmup_steps))

    def _apply_depth_patch_token_controls(
        self,
        projected_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask.to(device=projected_tokens.device, dtype=projected_tokens.dtype)
        controlled = projected_tokens * valid

        depth_patch_cfg = getattr(self.cfg, "depth_anything_v2_patch_state", None)
        token_dropout = float(getattr(depth_patch_cfg, "token_dropout", 0.0))
        if self.training and self.depth_patch_token_dropout_enabled and token_dropout > 0.0:
            keep = torch.rand(valid.shape, device=projected_tokens.device).ge(token_dropout)
            controlled = controlled * keep.to(projected_tokens.dtype) / (1.0 - token_dropout)

        token_scale = self._depth_patch_token_scale()
        if token_scale is not None:
            controlled = controlled * token_scale.to(device=projected_tokens.device, dtype=projected_tokens.dtype)
        return controlled

    def _base_act_policy(self) -> nn.Module:
        base_act_policy = self.__dict__.get("base_act_policy")
        if base_act_policy is None:
            raise RuntimeError("Base ACT encoder state is enabled but no base ACT policy is attached.")
        return base_act_policy

    def _base_act_device(self) -> torch.device:
        base_act_policy = self._base_act_policy()
        try:
            return next(base_act_policy.parameters()).device
        except StopIteration:
            return torch.device(self.cfg.device)

    def _unstandardize_state_for_base_act(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_mean is None or self.state_std is None:
            raise RuntimeError("State normalization statistics are required for base ACT encoder features.")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.float()
        mean = self.state_mean.to(state.device).view(1, -1)
        std = torch.maximum(self.state_std.to(state.device), torch.tensor(1e-8, device=state.device)).view(1, -1)
        return state * std + mean

    def _build_base_act_encoder_batch(self, obs_group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base_act_policy = self._base_act_policy()
        device = self._base_act_device()
        batch: dict[str, torch.Tensor] = {}

        for camera_key in base_act_policy.config.image_features:
            if camera_key not in obs_group:
                raise KeyError(f"Base ACT encoder feature extraction requires {camera_key!r} in the observation.")
            batch[camera_key] = self._prepare_image_batch(obs_group[camera_key], augment=False).to(device)

        if base_act_policy.config.robot_state_feature:
            if "observation.state" not in obs_group:
                raise KeyError("Base ACT encoder feature extraction requires 'observation.state'.")
            raw_state = self._unstandardize_state_for_base_act(obs_group["observation.state"])
            batch["observation.state"] = raw_state.to(device)

        if base_act_policy.config.env_state_feature:
            if "observation.environment_state" not in obs_group:
                raise KeyError("Base ACT encoder feature extraction requires 'observation.environment_state'.")
            env_state = obs_group["observation.environment_state"]
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            batch["observation.environment_state"] = env_state.float().to(device)

        return batch

    @torch.no_grad()
    def _infer_base_act_encoder_shape(
        self,
        *,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
    ) -> tuple[int, int]:
        base_act_policy = self._base_act_policy()
        device = self._base_act_device()
        dummy: dict[str, torch.Tensor] = {}

        for camera_key in base_act_policy.config.image_features:
            dummy[camera_key] = torch.zeros((1, *obs_shape), dtype=torch.float32, device=device)

        if base_act_policy.config.robot_state_feature:
            state_dim = int(base_act_policy.config.robot_state_feature.shape[0])
            if prop_shape and int(prop_shape[0]) != state_dim:
                raise ValueError(
                    "QAgent prop_shape does not match the base ACT robot state dimension. "
                    f"Got prop_shape={prop_shape}, base ACT state_dim={state_dim}."
                )
            dummy["observation.state"] = torch.zeros((1, state_dim), dtype=torch.float32, device=device)

        if base_act_policy.config.env_state_feature:
            env_dim = int(base_act_policy.config.env_state_feature.shape[0])
            dummy["observation.environment_state"] = torch.zeros((1, env_dim), dtype=torch.float32, device=device)

        tokens = base_act_policy.encode_observation_features(dummy)
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected base ACT encoder tokens with shape [B, S, D], got {tuple(tokens.shape)}")
        return int(tokens.shape[1]), int(tokens.shape[2])

    def _project_cached_base_act_encoder_tokens(self, cached_tokens: torch.Tensor) -> torch.Tensor:
        if self.base_act_encoder_projector is None:
            raise RuntimeError("Base ACT encoder state is enabled but no projector is available.")

        if cached_tokens.dim() == 2:
            cached_tokens = cached_tokens.unsqueeze(0)
        if cached_tokens.dim() != 3:
            raise ValueError(
                "Cached base ACT encoder tokens must have shape [S, D] or [B, S, D]. "
                f"Got {tuple(cached_tokens.shape)}."
            )
        if int(cached_tokens.shape[1]) != self.base_act_encoder_num_tokens:
            raise ValueError(
                "Cached base ACT encoder token count mismatch. "
                f"Expected {self.base_act_encoder_num_tokens}, got {cached_tokens.shape[1]}."
            )
        expected_dim = int(self.base_act_encoder_projector[0].in_features)
        if int(cached_tokens.shape[-1]) != expected_dim:
            raise ValueError(
                "Cached base ACT encoder token dim mismatch. "
                f"Expected {expected_dim}, got {cached_tokens.shape[-1]}."
            )

        projector_device = next(self.base_act_encoder_projector.parameters()).device
        cached_tokens = cached_tokens.to(device=projector_device, dtype=torch.float32)
        return self.base_act_encoder_projector(cached_tokens)

    @torch.no_grad()
    def compute_base_act_encoder_cache(
        self,
        obs: dict[str, torch.Tensor],
        *,
        cpu: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        if not self.uses_base_act_encoder_state:
            return {}

        base_act_policy = self._base_act_policy()
        base_act_policy.eval()
        try:
            batch = self._build_base_act_encoder_batch(obs)
        except KeyError:
            return {}
        tokens = base_act_policy.encode_observation_features(batch)
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected base ACT encoder tokens with shape [B, S, D], got {tuple(tokens.shape)}")
        if int(tokens.shape[1]) != self.base_act_encoder_num_tokens:
            raise RuntimeError(
                "Base ACT encoder token count changed between initialization and cache creation. "
                f"Expected {self.base_act_encoder_num_tokens}, got {tokens.shape[1]}."
            )
        tokens = tokens.detach()
        if int(tokens.shape[0]) == 1:
            tokens = tokens.squeeze(0)
        if dtype is not None:
            tokens = tokens.to(dtype=dtype)
        if cpu:
            tokens = tokens.cpu()
        return {self.base_act_encoder_cache_key: tokens}

    def _encode_base_act_encoder_tokens(self, obs_group: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.base_act_encoder_projector is None:
            raise RuntimeError("Base ACT encoder state is enabled but no projector is available.")

        if self.base_act_encoder_cache_key in obs_group:
            return self._project_cached_base_act_encoder_tokens(obs_group[self.base_act_encoder_cache_key])

        base_act_policy = self._base_act_policy()
        base_act_policy.eval()
        batch = self._build_base_act_encoder_batch(obs_group)
        with torch.no_grad():
            tokens = base_act_policy.encode_observation_features(batch)
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected base ACT encoder tokens with shape [B, S, D], got {tuple(tokens.shape)}")
        if int(tokens.shape[1]) != self.base_act_encoder_num_tokens:
            raise RuntimeError(
                "Base ACT encoder token count changed between initialization and runtime. "
                f"Expected {self.base_act_encoder_num_tokens}, got {tokens.shape[1]}."
            )
        return self.base_act_encoder_projector(tokens)

    def _min_over_random_two(self, q_values: torch.Tensor) -> torch.Tensor:
        """Compute min over a random subset of 2 heads from q_values [K, B, 1]."""
        assert q_values.dim() == 3 and q_values.size(-1) == 1
        num_heads = q_values.size(0)
        if num_heads <= 2:
            return torch.min(q_values[0], q_values[1])
        # Sample two unique head indices uniformly at random
        idx = torch.randperm(num_heads, device=q_values.device)[:2]
        subset = q_values.index_select(dim=0, index=idx)  # [2, B, 1]
        return subset.min(dim=0).values

    def _forward_actor_policy(self, obs: dict[str, torch.Tensor], stddev: float, *, use_target: bool):
        actor = self.actor_target if use_target else self.actor
        return actor.forward(obs, stddev)

    @staticmethod
    def _sample_action_from_policy_output(policy_output, *, eval_mode: bool, clip: float | None) -> torch.Tensor:
        if eval_mode:
            return policy_output.action_dist.mean
        return policy_output.action_dist.sample(clip=clip)

    def _horizon_probs_from_logits(self, horizon_logits: torch.Tensor | None) -> torch.Tensor | None:
        if horizon_logits is None:
            return None
        return torch.softmax(horizon_logits, dim=-1)

    def _sample_horizon_onehot(self, horizon_logits: torch.Tensor, *, eval_mode: bool) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(horizon_logits, dim=-1)
        if eval_mode:
            horizon_idx = torch.argmax(probs, dim=-1)
        else:
            horizon_idx = torch.distributions.Categorical(probs=probs).sample()
        horizon_onehot = torch.nn.functional.one_hot(horizon_idx, num_classes=self.num_adaptive_horizons).to(
            probs.dtype
        )
        return horizon_onehot, probs

    def _combine_with_base_action(self, obs: dict[str, torch.Tensor], residual_action: torch.Tensor) -> torch.Tensor:
        if self.residual_actor:
            return torch.clamp(obs["observation.base_action"] + residual_action, -1.0, 1.0)
        return residual_action

    def _build_adaptive_critic_actions(self, combined_action: torch.Tensor) -> torch.Tensor:
        batch_size = combined_action.shape[0]
        combined_chunk = combined_action.view(batch_size, self.max_action_horizon, self.primitive_action_dim)
        candidate_actions = []
        for horizon_idx, horizon in enumerate(self.adaptive_horizons):
            executed_chunk = torch.zeros_like(combined_chunk)
            executed_chunk[:, :horizon] = combined_chunk[:, :horizon]
            horizon_onehot = torch.zeros(
                batch_size,
                self.num_adaptive_horizons,
                device=combined_action.device,
                dtype=combined_action.dtype,
            )
            horizon_onehot[:, horizon_idx] = 1.0
            candidate_actions.append(torch.cat([executed_chunk.reshape(batch_size, -1), horizon_onehot], dim=-1))
        return torch.stack(candidate_actions, dim=1)

    def _build_critic_actions(
        self,
        obs: dict[str, torch.Tensor],
        residual_action: torch.Tensor,
        *,
        horizon_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        combined_action = self._combine_with_base_action(obs, residual_action)
        if not self.uses_adaptive_horizons:
            return combined_action

        candidate_actions = self._build_adaptive_critic_actions(combined_action)
        if horizon_onehot is None:
            return candidate_actions
        horizon_index = torch.argmax(horizon_onehot, dim=-1)
        batch_index = torch.arange(candidate_actions.shape[0], device=candidate_actions.device)
        return candidate_actions[batch_index, horizon_index]

    def _build_env_action(self, residual_action: torch.Tensor, horizon_onehot: torch.Tensor | None = None) -> torch.Tensor:
        if not self.uses_adaptive_horizons:
            return residual_action
        if horizon_onehot is None:
            raise ValueError("Adaptive horizon policies require a horizon_onehot when building env actions.")
        return torch.cat([residual_action, horizon_onehot], dim=-1)

    def _evaluate_critic_actions(
        self,
        critic: Critic,
        feat: torch.Tensor,
        prop: torch.Tensor,
        critic_actions: torch.Tensor,
        *,
        for_policy: bool,
    ) -> torch.Tensor:
        if critic_actions.dim() == 2:
            q_fn = critic.q_value_for_policy if for_policy else critic.q_value
            return q_fn(feat, prop, critic_actions).squeeze(-1)

        batch_size, num_horizons, _ = critic_actions.shape
        feat_rep = feat.repeat_interleave(num_horizons, dim=0)
        prop_rep = prop.repeat_interleave(num_horizons, dim=0)
        action_rep = critic_actions.reshape(batch_size * num_horizons, -1)
        q_fn = critic.q_value_for_policy if for_policy else critic.q_value
        q_values = q_fn(feat_rep, prop_rep, action_rep).squeeze(-1)
        return q_values.reshape(batch_size, num_horizons)

    def _adaptive_prefix_l2_penalty(self, residual_action: torch.Tensor, horizon_probs: torch.Tensor) -> torch.Tensor:
        residual_chunk = residual_action.view(residual_action.shape[0], self.max_action_horizon, self.primitive_action_dim)
        per_step_sq = torch.sum(residual_chunk**2, dim=-1)
        prefix_sq = torch.cumsum(per_step_sq, dim=-1)
        horizon_penalties = torch.stack([prefix_sq[:, horizon - 1] for horizon in self.adaptive_horizons], dim=-1)
        expected_penalty = torch.sum(horizon_probs * horizon_penalties, dim=-1)
        return self.cfg.actor.action_l2_reg_weight * expected_penalty.mean()

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method

    def _prepare_image_batch(self, data: torch.Tensor, *, augment: bool) -> torch.Tensor:
        if data.dim() == 3:
            data = data.unsqueeze(0)
        elif data.dim() > 4:
            extra_dims = data.shape[1:-3]
            if not all(dim == 1 for dim in extra_dims):
                raise ValueError(
                    "Expected image tensors to be BCHW or have only singleton extra dims after the batch dimension, "
                    f"got shape {tuple(data.shape)}."
                )
            data = data.reshape(data.shape[0], *data.shape[-3:])

        if data.dtype == torch.uint8:
            data = data.float().div_(255.0)
        else:
            data = data.float()

        if augment and self.image_geom_augmentation_enabled:
            data = self.aug(data)
        return data

    def _prepare_observation_images(self, obs: dict[str, torch.Tensor], *, augment: bool) -> dict[str, torch.Tensor]:
        return {cam_name: self._prepare_image_batch(obs[cam_name], augment=augment) for cam_name in self.rl_cameras}

    def get_depth_cache_keys(self) -> list[str]:
        return [self.depth_cache_keys_by_camera[cam_name] for cam_name in self.rl_cameras]

    def get_depth_patch_cache_keys(self) -> list[str]:
        return [self.depth_patch_cache_keys_by_camera[cam_name] for cam_name in self.depth_patch_camera_keys]

    def _extract_cached_depth_cls_groups(
        self,
        obs_groups: list[dict[str, torch.Tensor]],
    ) -> list[list[torch.Tensor]] | None:
        if not self.uses_depth_anything_v2_conditioning:
            return None

        cache_keys = self.get_depth_cache_keys()
        if not all(all(cache_key in obs for cache_key in cache_keys) for obs in obs_groups):
            return None

        cached_groups: list[list[torch.Tensor]] = []
        for obs in obs_groups:
            cls_by_cam: list[torch.Tensor] = []
            for cam_name in self.rl_cameras:
                cache_key = self.depth_cache_keys_by_camera[cam_name]
                cls = obs[cache_key].float()
                if cls.dim() == 1:
                    cls = cls.unsqueeze(0)
                cls_by_cam.append(cls)
            cached_groups.append(cls_by_cam)
        return cached_groups

    @torch.no_grad()
    def compute_depth_cls_cache(
        self,
        obs: dict[str, torch.Tensor],
        *,
        cpu: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not self.uses_depth_anything_v2_conditioning:
            return {}

        if self.depth_anything_v2_encoder is None:
            raise RuntimeError("DepthAnythingV2 conditioning is disabled.")
        if not all(cam_name in obs for cam_name in self.rl_cameras):
            return {}
        encoder_device = self.depth_anything_v2_encoder.pixel_mean.device
        prepared = {}
        for cam_name in self.rl_cameras:
            data = obs[cam_name]
            if data.dim() == 3:
                data = data.unsqueeze(0)
            prepared[cam_name] = self._prepare_image_batch(data, augment=False).to(encoder_device)

        cls_by_cam = self._encode_depth_anything_v2_cls_groups([prepared])[0]

        cached = {}
        for cam_name, cls in zip(self.rl_cameras, cls_by_cam, strict=True):
            cls = cls.squeeze(0).detach()
            if cpu:
                cls = cls.cpu()
            cached[self.depth_cache_keys_by_camera[cam_name]] = cls
        return cached

    def _compute_raw_depth_patch_tokens(
        self,
        obs_group: dict[str, torch.Tensor],
        image_batch: torch.Tensor,
        *,
        camera_key: str,
    ) -> torch.Tensor:
        if self.depth_patch_encoder is None or self.depth_patch_selector is None:
            raise RuntimeError("Depth patch state is enabled but not fully initialized.")

        if "observation.base_action" not in obs_group:
            raise KeyError("Depth patch state requires observation.base_action in the observation dict.")
        if "observation.state" not in obs_group:
            raise KeyError("Depth patch state requires observation.state in the observation dict.")

        patch_tokens, _ = self.depth_patch_encoder.forward_patches_and_cls(image_batch, flatten_patches=False)
        preprocessed = self.depth_patch_encoder._preprocess(image_batch)
        proc_hw = (int(preprocessed.shape[-2]), int(preprocessed.shape[-1]))
        patch_rows = proc_hw[0] // int(self.depth_patch_encoder.patch_size)
        patch_cols = proc_hw[1] // int(self.depth_patch_encoder.patch_size)
        if patch_rows * patch_cols != int(patch_tokens.shape[1]):
            raise RuntimeError(
                "Depth patch grid size does not match token count: "
                f"patch_rows={patch_rows}, patch_cols={patch_cols}, num_tokens={patch_tokens.shape[1]}"
            )

        if self.depth_patch_selection_mode == "all":
            num_tokens = int(patch_tokens.shape[1])
            if self.depth_patch_max_tokens_per_camera >= num_tokens:
                return patch_tokens
            patch_indices = torch.linspace(
                0,
                num_tokens - 1,
                steps=self.depth_patch_max_tokens_per_camera,
                device=patch_tokens.device,
            ).long()
            return patch_tokens.index_select(dim=1, index=patch_indices)

        patch_device = patch_tokens.device
        patch_indices = self.depth_patch_selector.select_patch_indices(
            obs_state=obs_group["observation.state"].to(patch_device),
            base_action=obs_group["observation.base_action"].to(patch_device),
            camera_key=camera_key,
            proc_hw=proc_hw,
            patch_hw=(patch_rows, patch_cols),
            max_patches=self.depth_patch_max_tokens_per_camera,
        )
        return gather_selected_patch_tokens(patch_tokens, patch_indices)

    @staticmethod
    def _depth_patch_valid_mask(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.float().abs().sum(dim=-1, keepdim=True).gt(0)

    def _project_cached_depth_patch_tokens(
        self,
        cached_tokens: torch.Tensor,
        *,
        camera_key: str,
    ) -> torch.Tensor:
        if self.depth_patch_encoder is None or self.depth_patch_projectors is None:
            raise RuntimeError("Depth patch state is enabled but not fully initialized.")

        if cached_tokens.dim() == 2:
            cached_tokens = cached_tokens.unsqueeze(0)
        if cached_tokens.dim() != 3:
            raise ValueError(
                "Cached depth patch tokens must have shape [M, D] or [B, M, D]. "
                f"Got {tuple(cached_tokens.shape)}."
            )
        if int(cached_tokens.shape[1]) != self.depth_patch_max_tokens_per_camera:
            raise ValueError(
                "Cached depth patch token count mismatch. "
                f"Expected {self.depth_patch_max_tokens_per_camera}, got {cached_tokens.shape[1]}."
            )
        expected_dim = int(self.depth_patch_encoder.patch_repr_dim)
        if int(cached_tokens.shape[-1]) != expected_dim:
            raise ValueError(
                "Cached depth patch token dim mismatch. "
                f"Expected {expected_dim}, got {cached_tokens.shape[-1]}."
            )

        projector_idx = self.depth_patch_camera_to_projector_idx[camera_key]
        projector = self.depth_patch_projectors[projector_idx]
        projector_device = next(projector.parameters()).device
        cached_tokens = cached_tokens.to(device=projector_device, dtype=torch.float32)
        valid_mask = self._depth_patch_valid_mask(cached_tokens).to(device=projector_device)
        projected_tokens = projector(cached_tokens)
        return self._apply_depth_patch_token_controls(projected_tokens, valid_mask)

    @torch.no_grad()
    def compute_depth_patch_cache(
        self,
        obs: dict[str, torch.Tensor],
        *,
        cpu: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        if not self.uses_depth_patch_state:
            return {}

        if self.depth_patch_encoder is None:
            raise RuntimeError("Depth patch state is disabled.")
        if not all(camera_key in obs for camera_key in self.depth_patch_camera_keys):
            return {}
        if "observation.base_action" not in obs or "observation.state" not in obs:
            return {}

        encoder_device = self.depth_patch_encoder.pixel_mean.device
        cached: dict[str, torch.Tensor] = {}
        for camera_key in self.depth_patch_camera_keys:
            image = obs[camera_key]
            squeeze_batch = image.dim() == 3
            if squeeze_batch:
                image = image.unsqueeze(0)

            image_batch = self._prepare_image_batch(image, augment=False).to(encoder_device)
            selected_tokens = self._compute_raw_depth_patch_tokens(
                obs,
                image_batch,
                camera_key=camera_key,
            ).detach()
            if squeeze_batch:
                selected_tokens = selected_tokens.squeeze(0)
            if dtype is not None:
                selected_tokens = selected_tokens.to(dtype=dtype)
            if cpu:
                selected_tokens = selected_tokens.cpu()
            cached[self.depth_patch_cache_keys_by_camera[camera_key]] = selected_tokens
        return cached

    def _encode_depth_anything_v2_cls_groups(
        self,
        prepared_groups: list[dict[str, torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        if self.depth_anything_v2_encoder is None:
            raise RuntimeError("DepthAnythingV2 conditioning is disabled.")

        packed_batches: list[torch.Tensor] = []
        group_batch_sizes: list[int] = []
        for prepared in prepared_groups:
            batch_size = None
            for cam_name in self.rl_cameras:
                data = prepared[cam_name]
                packed_batches.append(data)
                if batch_size is None:
                    batch_size = int(data.shape[0])
            group_batch_sizes.append(batch_size or 0)

        packed = torch.cat(packed_batches, dim=0)
        packed_cls = self.depth_anything_v2_encoder.forward_cls(packed)

        cls_groups: list[list[torch.Tensor]] = []
        offset = 0
        for batch_size in group_batch_sizes:
            cls_by_cam: list[torch.Tensor] = []
            for _ in self.rl_cameras:
                next_offset = offset + batch_size
                cls_by_cam.append(packed_cls[offset:next_offset])
                offset = next_offset
            cls_groups.append(cls_by_cam)
        return cls_groups

    def _encode_depth_patch_tokens(
        self,
        obs_group: dict[str, torch.Tensor],
        prepared: dict[str, torch.Tensor],
        *,
        camera_key: str,
    ) -> torch.Tensor:
        if self.depth_patch_encoder is None or self.depth_patch_projectors is None:
            raise RuntimeError("Depth patch state is enabled but not fully initialized.")

        cache_key = self.depth_patch_cache_keys_by_camera[camera_key]
        if cache_key in obs_group:
            return self._project_cached_depth_patch_tokens(obs_group[cache_key], camera_key=camera_key)

        image_batch = self._prepare_image_batch(obs_group[camera_key], augment=False).to(prepared[camera_key].device)
        selected_tokens = self._compute_raw_depth_patch_tokens(obs_group, image_batch, camera_key=camera_key)
        valid_mask = self._depth_patch_valid_mask(selected_tokens).to(device=selected_tokens.device)
        projector_idx = self.depth_patch_camera_to_projector_idx[camera_key]
        projected_tokens = self.depth_patch_projectors[projector_idx](selected_tokens)
        return self._apply_depth_patch_token_controls(projected_tokens, valid_mask)

    def _encode_prepared_group(
        self,
        prepared: dict[str, torch.Tensor],
        *,
        obs_group: dict[str, torch.Tensor],
        depth_cls_by_cam: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        feats = []
        for cam_idx, cam_name in enumerate(self.rl_cameras):
            layer_prefix_tokens = None
            if depth_cls_by_cam is not None:
                if self.depth_cls_projectors is None:
                    raise RuntimeError("DepthAnythingV2 conditioning is enabled but no projector is available.")
                projected_cls = self.depth_cls_projectors[cam_idx](depth_cls_by_cam[cam_idx])
                vit_depth = int(self.encoders[cam_idx].vit.depth)
                layer_prefix_tokens = [
                    projected_cls if layer_idx >= vit_depth - self.num_depth_conditioned_layers else None
                    for layer_idx in range(vit_depth)
                ]

            if self.use_residual_image_encoder:
                feat_cam = self.encoders[cam_idx].forward(
                    prepared[cam_name],
                    flatten=False,
                    layer_prefix_tokens=layer_prefix_tokens,
                )
                feats.append(feat_cam)
            if cam_name in self.depth_patch_camera_to_projector_idx:
                feats.append(self._encode_depth_patch_tokens(obs_group, prepared, camera_key=cam_name))

        if self.uses_base_act_encoder_state:
            feats.append(self._encode_base_act_encoder_tokens(obs_group))
        if not feats:
            raise RuntimeError("No feature tokens were encoded for this observation.")
        return torch.cat(feats, dim=1)

    def _encode_observation_groups(
        self,
        obs_groups: list[dict[str, torch.Tensor]],
        *,
        augment: bool,
    ) -> list[torch.Tensor]:
        prepared_groups = [self._prepare_observation_images(obs, augment=augment) for obs in obs_groups]
        depth_cls_groups: list[list[torch.Tensor]] | None = None
        if self.uses_depth_anything_v2_conditioning:
            depth_cls_groups = self._extract_cached_depth_cls_groups(obs_groups)
            if depth_cls_groups is None:
                depth_cls_groups = self._encode_depth_anything_v2_cls_groups(prepared_groups)

        encoded_groups = []
        for group_idx, prepared in enumerate(prepared_groups):
            cls_by_cam = None if depth_cls_groups is None else depth_cls_groups[group_idx]
            encoded_groups.append(
                self._encode_prepared_group(
                    prepared,
                    obs_group=obs_groups[group_idx],
                    depth_cls_by_cam=cls_by_cam,
                )
            )
        return encoded_groups

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        r"""This function encodes the observation into feature tensor.

        Images may be stored in the replay buffers as uint8 to save GPU memory.  In
        that case we convert them to float32 in \[0,1] before feeding them to the
        encoders.  If the image is already a float tensor (offline dataset or
        direct env observations during evaluation) we assume it is properly
        normalised.
        """
        return self._encode_observation_groups([obs], augment=augment)[0]

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        # Make a shallow copy of the observation dict
        obs = copy.copy(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        assert "feat" not in obs
        obs["feat"] = self._encode(obs, augment=False)

        policy_output = self._forward_actor_policy(obs, stddev, use_target=False)
        residual_action = self._sample_action_from_policy_output(policy_output, eval_mode=eval_mode, clip=None)
        if self.uses_adaptive_horizons:
            if policy_output.horizon_logits is None:
                raise RuntimeError("Adaptive horizon policy is enabled but the actor did not return horizon logits.")
            horizon_onehot, _ = self._sample_horizon_onehot(policy_output.horizon_logits, eval_mode=eval_mode)
            action = self._build_env_action(residual_action, horizon_onehot)
        else:
            action = residual_action

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: float | None,
        use_target: bool,
    ) -> torch.Tensor:
        policy_output = self._forward_actor_policy(obs, stddev, use_target=use_target)

        # Only assert not training when this is called from the public act() method
        # (which is used for actual evaluation), not when called internally during training
        if eval_mode and not use_target:
            assert not self.training

        return self._sample_action_from_policy_output(policy_output, eval_mode=eval_mode, clip=clip)

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        importance_weights: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            next_policy_output = self._forward_actor_policy(next_obs, stddev, use_target=True)
            next_residual_action = self._sample_action_from_policy_output(
                next_policy_output,
                eval_mode=not self.cfg.target_action_noise,
                clip=self.cfg.stddev_clip,
            )

            if self.uses_adaptive_horizons:
                if next_policy_output.horizon_logits is None:
                    raise RuntimeError("Adaptive horizon policy is enabled but the actor target did not return logits.")
                next_horizon_probs = self._horizon_probs_from_logits(next_policy_output.horizon_logits)
                assert next_horizon_probs is not None
                next_critic_actions = self._build_critic_actions(next_obs, next_residual_action)
                next_q = self._evaluate_critic_actions(
                    self.critic_target,
                    next_obs["feat"],
                    next_obs["observation.state"],
                    next_critic_actions,
                    for_policy=False,
                )
                target_q_min = torch.sum(next_horizon_probs * next_q, dim=-1)
            else:
                next_critic_actions = self._build_critic_actions(next_obs, next_residual_action)
                next_action = next_critic_actions
                target_all = self.critic_target.q_value(
                    next_obs["feat"], next_obs["observation.state"], next_critic_actions
                )
                target_q_min = target_all.squeeze(-1)  # [B]
            target_q = (reward + (discount * target_q_min)).detach()

        if self.cfg.clip_q_target_to_reward_range:
            target_q = torch.clamp(target_q, min=0, max=1)  # Sparse rewards are in {0, 1}

        td_errors = None

        if self.critic.loss_cfg.type == "hl_gauss":
            # Compute logits for current Q heads and average HL-Gauss loss across heads
            q_per_head, logits_per_head = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)
            K = logits_per_head.shape[0]
            losses = [self.critic.hl_loss(logits_per_head[i], target_q) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        elif self.critic.loss_cfg.type == "c51":
            if self.uses_adaptive_horizons:
                raise NotImplementedError("Adaptive horizon residual TD3 currently supports only scalar-value critics.")
            # Compute logits for current Q heads and C51 distributional loss
            q_per_head, logits_per_head = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)

            # Get next state distribution for C51 target computation
            with torch.no_grad():
                _, next_logits = self.critic_target(
                    next_obs["feat"], next_obs["observation.state"], next_action, return_logits=True
                )
                # Take min over random subset of heads for next distribution (configurable via min_q_heads)
                num_heads = min(self.critic.cfg.min_q_heads, next_logits.shape[0])
                idx = torch.randperm(next_logits.shape[0], device=next_logits.device)[:num_heads]
                next_logits_min = torch.min(next_logits.index_select(0, idx), dim=0).values
                next_distribution = torch.softmax(next_logits_min, dim=-1)

                # Project the target distribution
                # The discount factor passed to this function is already discount = gamma * (1 - done)
                # For C51, we need to extract the done mask and gamma separately
                # We'll use a simple heuristic: if discount is 0, then done=1, otherwise done=0
                dones = (discount == 0.0).float()
                gamma = 0.99  # Assume standard gamma value
                target_distribution = self.critic.c51_loss.project_distribution(next_distribution, reward, dones, gamma)

            # Compute C51 loss for each head
            K = logits_per_head.shape[0]
            losses = [self.critic.c51_loss(logits_per_head[i], target_distribution) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        else:
            q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [K,B]
            # Compute TD errors for prioritized experience replay (before taking mean)
            td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads

            # Apply importance sampling weights if provided (for prioritized experience replay)
            if importance_weights is not None:
                # Weight the squared TD errors by importance sampling weights
                weighted_td_errors = td_errors**2 * importance_weights
                critic_loss = weighted_td_errors.mean()
            else:
                # Mean squared error across heads and batch (uniform sampling)
                critic_loss = (td_errors**2).mean()

        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()
        if self.depth_patch_token_scale_raw is not None:
            metrics["train/depth_patch_token_scale"] = self._depth_patch_token_scale().detach().item()
            metrics["train/depth_patch_token_scale_multiplier"] = self._depth_patch_token_scale_multiplier()
        # Store target_q for potential logging (calculated only when needed)
        metrics["_target_q"] = target_q.detach().cpu()
        # Store TD errors for prioritized experience replay
        if td_errors is not None:
            metrics["_td_errors"] = td_errors.detach().cpu()
        # Log importance sampling weights for monitoring PER behavior
        if importance_weights is not None:
            metrics["train/importance_weights_mean"] = importance_weights.mean().item()
            metrics["train/importance_weights_std"] = importance_weights.std().item()
            metrics["train/importance_weights_min"] = importance_weights.min().item()
            metrics["train/importance_weights_max"] = importance_weights.max().item()

        # Zero gradients
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        # Gradient clipping
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self._encoder_parameters(), self.cfg.critic_grad_clip_norm)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)

        # Store gradient norms for logging
        metrics["train/encoder_grad_norm"] = encoder_grad_norm.item()
        metrics["train/critic_grad_norm"] = critic_grad_norm.item()

        self.encoder_opt.step()
        self.critic_opt.step()

        return metrics

    def _residual_l1_actor_penalty(self, residual_action: torch.Tensor):
        residual_l1 = residual_action.abs().mean()
        coef = float(getattr(self.cfg, "residual_l1_penalty_coef", 0.0))
        target = float(getattr(self.cfg, "residual_l1_penalty_target", 0.0))
        if coef <= 0.0:
            return residual_l1.new_zeros(()), residual_l1
        return coef * torch.relu(residual_l1 - target), residual_l1

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        assert "feat" in obs, "safety check"

        policy_output = self._forward_actor_policy(obs, 0.0, use_target=False)
        action_pred = self._sample_action_from_policy_output(policy_output, eval_mode=True, clip=self.cfg.stddev_clip)
        residual_l1_penalty, residual_l1 = self._residual_l1_actor_penalty(action_pred)

        if self.uses_adaptive_horizons:
            if policy_output.horizon_logits is None:
                raise RuntimeError("Adaptive horizon policy is enabled but the actor did not return logits.")
            horizon_probs = self._horizon_probs_from_logits(policy_output.horizon_logits)
            assert horizon_probs is not None
            critic_actions = self._build_critic_actions(obs, action_pred)
            q_by_horizon = self._evaluate_critic_actions(
                self.critic,
                obs["feat"],
                obs["observation.state"],
                critic_actions,
                for_policy=True,
            )
            expected_q = torch.sum(horizon_probs * q_by_horizon, dim=-1)
            actor_loss_base = -expected_q.mean()
            action_l2_penalty = self._adaptive_prefix_l2_penalty(action_pred, horizon_probs)
            entropy = -(horizon_probs * torch.log(horizon_probs.clamp_min(1e-8))).sum(dim=-1).mean()
            actor_loss_total = (
                actor_loss_base
                + action_l2_penalty
                + residual_l1_penalty
                - (self.adaptive_horizon_entropy_reg * entropy)
            )

            greedy_horizon_onehot = torch.nn.functional.one_hot(
                torch.argmax(horizon_probs, dim=-1), num_classes=self.num_adaptive_horizons
            ).to(action_pred.dtype)
            combined_action = self._build_critic_actions(obs, action_pred, horizon_onehot=greedy_horizon_onehot)
            return (
                actor_loss_total,
                actor_loss_base,
                combined_action,
                action_pred,
                action_l2_penalty,
                {
                    "horizon_probs": horizon_probs,
                    "q_by_horizon": q_by_horizon,
                    "horizon_entropy": entropy,
                    "residual_l1_penalty": residual_l1_penalty,
                    "actor_residual_l1": residual_l1,
                },
            )

        action_l2_penalty = self.cfg.actor.action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))
        combined_action = self._build_critic_actions(obs, action_pred)
        q = self.critic.q_value_for_policy(obs["feat"], obs["observation.state"], combined_action)
        actor_loss_base = -q.mean()
        actor_loss_total = actor_loss_base + action_l2_penalty + residual_l1_penalty

        return (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
            {
                "residual_l1_penalty": residual_l1_penalty,
                "actor_residual_l1": residual_l1,
            },
        )

    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        assert not self.residual_actor, "Not implemented"
        obs: dict[str, torch.Tensor] = batch["obs"]

        assert "feat" not in obs, "safety check"
        obs["feat"] = self._encode(obs, augment=True)

        if not backprop_encoder:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        action: torch.Tensor = batch["action"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss  # noqa: RET504

    def update_actor(self, obs: dict[str, torch.Tensor], stddev: float):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
            actor_aux,
        ) = self._compute_actor_loss(obs, stddev)

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        metrics["train/actor_residual_l1"] = actor_aux["actor_residual_l1"].item()
        if self.cfg.residual_l1_penalty_coef > 0:
            metrics["train/actor_residual_l1_penalty"] = actor_aux["residual_l1_penalty"].item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()
        if self.uses_adaptive_horizons:
            horizon_probs = actor_aux["horizon_probs"].detach()
            q_by_horizon = actor_aux["q_by_horizon"].detach()
            metrics["train/horizon_entropy"] = actor_aux["horizon_entropy"].item()
            for horizon_idx, horizon in enumerate(self.adaptive_horizons):
                metrics[f"train/horizon_prob_{horizon}"] = horizon_probs[:, horizon_idx].mean().item()
                metrics[f"train/actor_q_h{horizon}"] = q_by_horizon[:, horizon_idx].mean().item()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_total.backward()

        # Gradient clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)

        # Store gradient norm for logging
        metrics["train/actor_grad_norm"] = actor_grad_norm.item()

        self.actor_opt.step()

        return metrics

    def update_actor_rft(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: QAgent,
    ):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
            actor_aux,
        ) = self._compute_actor_loss(obs, stddev)

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        metrics["train/actor_residual_l1"] = actor_aux["actor_residual_l1"].item()
        if self.cfg.residual_l1_penalty_coef > 0:
            metrics["train/actor_residual_l1_penalty"] = actor_aux["residual_l1_penalty"].item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()
        if self.uses_adaptive_horizons:
            horizon_probs = actor_aux["horizon_probs"].detach()
            q_by_horizon = actor_aux["q_by_horizon"].detach()
            metrics["train/horizon_entropy"] = actor_aux["horizon_entropy"].item()
            for horizon_idx, horizon in enumerate(self.adaptive_horizons):
                metrics[f"train/horizon_prob_{horizon}"] = horizon_probs[:, horizon_idx].mean().item()
                metrics[f"train/actor_q_h{horizon}"] = q_by_horizon[:, horizon_idx].mean().item()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        # Use config option to control whether BC loss updates encoder
        bc_backprop_encoder = self.cfg.bc_backprop_encoder
        bc_loss = self._compute_actor_bc_loss(bc_batch, backprop_encoder=bc_backprop_encoder)
        assert actor_loss_total.size() == bc_loss.size()

        ratio = 1
        if self.cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.cfg.act_method
                self.cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                curr_q = self.critic.q_value_for_policy(bc_obs["feat"], bc_obs["observation.state"], curr_action)
                ref_q = self.critic.q_value_for_policy(bc_obs["feat"], bc_obs["observation.state"], ref_action)

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss_total + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        # Conditionally update encoder along with actor if BC loss should backprop
        if bc_backprop_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient clipping
        metrics["train/actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.actor_grad_clip_norm
        ).item()

        if bc_backprop_encoder:
            metrics["train/encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self._encoder_parameters(), self.cfg.actor_grad_clip_norm
            ).item()

        if bc_backprop_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: QAgent | None = None,
    ):
        obs: dict[str, torch.Tensor] = batch["obs"]
        action: torch.Tensor = batch["action"]
        reward: torch.Tensor = batch[("next", "reward")]
        discount: torch.Tensor = batch["gamma"]
        next_nonterminal: torch.Tensor = batch["nonterminal"]
        next_obs: dict[str, torch.Tensor] = batch[("next", "obs")]

        # To not bootstrap on terminal states we zero out the discount factor for terminal next states
        effective_discount = discount * next_nonterminal

        obs["feat"] = self._encode(obs, augment=True)

        with torch.no_grad():
            prev_depth_patch_dropout = self.depth_patch_token_dropout_enabled
            self.depth_patch_token_dropout_enabled = False
            try:
                next_obs["feat"] = self._encode(next_obs, augment=True)
            finally:
                self.depth_patch_token_dropout_enabled = prev_depth_patch_dropout

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()
        chosen_horizon = batch.get("chosen_horizon", None) if self.uses_adaptive_horizons else None
        if chosen_horizon is not None:
            executed_horizon = batch.get("executed_horizon", None)
            for horizon in self.adaptive_horizons:
                metrics[f"data/chosen_horizon_{horizon}"] = (chosen_horizon == horizon).float().mean().item()
                if executed_horizon is not None:
                    metrics[f"data/executed_horizon_{horizon}"] = (executed_horizon == horizon).float().mean().item()

        # Extract importance sampling weights if available (for prioritized experience replay)
        importance_weights = batch.get("_weight", None)

        critic_metric = self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            discount=effective_discount,
            next_obs=next_obs,
            stddev=stddev,
            importance_weights=importance_weights,
        )
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            if self.depth_patch_token_scale_raw is not None:
                self.depth_patch_update_step += 1
            return metrics

        # NOTE: actor loss does not backprop into the encoder
        obs["feat"] = obs["feat"].detach()

        if bc_batch is None:
            actor_metric = self.update_actor(obs, stddev)
        else:
            assert ref_agent is not None
            actor_metric = self.update_actor_rft(obs, stddev, bc_batch, ref_agent)

        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)
        if self.depth_patch_token_scale_raw is not None:
            metrics["train/depth_patch_token_scale"] = self._depth_patch_token_scale().detach().item()
            metrics["train/depth_patch_token_scale_multiplier"] = self._depth_patch_token_scale_multiplier()
            self.depth_patch_update_step += 1

        return metrics

    def step_lr_schedulers(self):
        """Step the learning rate schedulers for warmup."""
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
