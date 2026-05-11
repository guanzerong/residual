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
from resfit.rl_finetuning.off_policy.networks.encoder import DepthAnythingV2TokenEncoder, VitEncoder
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.critic import Critic


class GROOTFeatureAdapter(nn.Module):
    """Small trainable adapter that makes GR00T hidden tokens TD3-friendly."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        use_layer_norm: bool,
        dropout: float,
        scale_init: float,
        scale_learnable: bool,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"GR00T adapter dimensions must be positive, got {input_dim=} {output_dim=}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"GR00T adapter dropout must be in [0, 1), got {dropout}.")

        self.input_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        if output_dim == input_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU())
        self.output_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        scale = torch.tensor(float(scale_init), dtype=torch.float32)
        if scale_learnable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(tokens)
        x = self.proj(x)
        x = self.output_norm(x)
        x = self.dropout(x)
        return x * self.scale


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
        adaptive_horizons: tuple[int, ...] | None = None,
        adaptive_horizon_entropy_reg: float = 0.0,
        adaptive_horizon_value_ce_coef: float = 0.0,
        adaptive_horizon_value_temperature: float = 0.01,
        adaptive_horizon_length_penalty: float = 0.0,
        adaptive_horizon_residual_penalty: float = 0.0,
        adaptive_horizon_conditioned_actions: bool = False,
        macro_action_horizon: int | None = None,
        action_scaler_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
        state_standardizer_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        world_to_camera_transform: torch.Tensor | None = None,
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
        self.critic_action_dim = int(action_dim)
        self.base_action_dim = int(base_action_dim if base_action_dim is not None else action_dim)
        self.adaptive_horizons = tuple(sorted(adaptive_horizons or ()))
        self.num_adaptive_horizons = len(self.adaptive_horizons)
        self.uses_adaptive_horizons = self.num_adaptive_horizons > 0
        self.adaptive_horizon_entropy_reg = float(adaptive_horizon_entropy_reg)
        self.adaptive_horizon_value_ce_coef = float(adaptive_horizon_value_ce_coef)
        self.adaptive_horizon_value_temperature = float(adaptive_horizon_value_temperature)
        self.adaptive_horizon_length_penalty = float(adaptive_horizon_length_penalty)
        self.adaptive_horizon_residual_penalty = float(adaptive_horizon_residual_penalty)
        self.max_action_horizon = max(self.adaptive_horizons) if self.adaptive_horizons else int(macro_action_horizon or 1)
        self.macro_action_horizon = self.max_action_horizon
        self.adaptive_horizon_conditioned_actions = bool(adaptive_horizon_conditioned_actions) and self.uses_adaptive_horizons
        self.primitive_action_dim = self.base_action_dim
        if self.uses_adaptive_horizons:
            if not residual_actor:
                raise ValueError("Adaptive macro horizons are only implemented for residual actors.")
            if self.base_action_dim % self.max_action_horizon != 0:
                raise ValueError(
                    "base_action_dim must be divisible by the maximum adaptive horizon. "
                    f"Got base_action_dim={self.base_action_dim}, max_horizon={self.max_action_horizon}."
                )
            self.primitive_action_dim = self.base_action_dim // self.max_action_horizon
            expected_critic_dim = self.base_action_dim + self.num_adaptive_horizons
            if self.critic_action_dim != expected_critic_dim:
                raise ValueError(
                    "Adaptive horizon critic action dimension mismatch. "
                    f"Expected {expected_critic_dim}, got {self.critic_action_dim}."
                )
        self.depth_cache_keys_by_camera = {
            cam_name: cam_name.replace("observation.images.", "observation.depth_cls.", 1) for cam_name in self.rl_cameras
        }
        self.macro_local_depth_cfg = getattr(self.cfg, "macro_local_depth_gating", None)
        self.uses_macro_local_depth_gating = bool(
            self.macro_local_depth_cfg is not None and self.macro_local_depth_cfg.enabled
        )
        self.macro_local_depth_camera_key = (
            self.macro_local_depth_cfg.camera_key if self.macro_local_depth_cfg is not None else None
        )
        self.obs_height = int(obs_shape[1])
        self.obs_width = int(obs_shape[2])

        if self.uses_macro_local_depth_gating:
            if not residual_actor:
                raise ValueError("Macro local depth gating is only implemented for residual actors.")
            if action_scaler_limits is None:
                raise ValueError("Macro local depth gating requires action_scaler_limits.")
            if state_standardizer_stats is None:
                raise ValueError("Macro local depth gating requires state_standardizer_stats.")
            if world_to_camera_transform is None:
                raise ValueError("Macro local depth gating requires a fixed world_to_camera_transform.")

            action_min, action_max = action_scaler_limits
            state_mean, state_std = state_standardizer_stats
            self.register_buffer("macro_local_action_min", action_min.float().clone())
            self.register_buffer("macro_local_action_max", action_max.float().clone())
            self.register_buffer("macro_local_state_mean", state_mean.float().clone())
            self.register_buffer("macro_local_state_std", state_std.float().clone())
            self.register_buffer("macro_local_world_to_camera", world_to_camera_transform.float().clone())
        else:
            self.register_buffer("macro_local_action_min", torch.empty(0), persistent=False)
            self.register_buffer("macro_local_action_max", torch.empty(0), persistent=False)
            self.register_buffer("macro_local_state_mean", torch.empty(0), persistent=False)
            self.register_buffer("macro_local_state_std", torch.empty(0), persistent=False)
            self.register_buffer("macro_local_world_to_camera", torch.empty(0), persistent=False)

        self.uses_groot_features = bool(getattr(self.cfg, "use_groot_features", False))
        self.groot_feature_key = str(getattr(self.cfg, "groot_feature_key", "observation.groot_features"))
        self.groot_token_count = int(getattr(self.cfg, "groot_token_count", 0))
        self.groot_token_dim = int(getattr(self.cfg, "groot_token_dim", 0))
        self.groot_feature_project_dim = int(getattr(self.cfg, "groot_feature_project_dim", 0))
        self.groot_feature_dropout = float(getattr(self.cfg, "groot_feature_dropout", 0.0))
        self.groot_feature_scale_init = float(getattr(self.cfg, "groot_feature_scale_init", 1.0))
        self.groot_feature_adapter: GROOTFeatureAdapter | None = None
        self.depth_anything_v2_encoder: DepthAnythingV2TokenEncoder | None = None
        self.depth_cls_projectors: nn.ModuleList | None = None
        depth_cfg = getattr(self.cfg, "depth_anything_v2_conditioning", None)
        self.uses_depth_anything_v2_conditioning = bool(depth_cfg is not None and depth_cfg.enabled)
        self.requires_depth_anything_v2 = self.uses_depth_anything_v2_conditioning or self.uses_macro_local_depth_gating
        self.num_depth_conditioned_layers = 0

        if self.uses_groot_features:
            if self.uses_depth_anything_v2_conditioning:
                raise ValueError("GR00T observation features and DepthAnythingV2 conditioning are mutually exclusive.")
            if self.groot_token_count <= 0 or self.groot_token_dim <= 0:
                raise ValueError(
                    "use_groot_features=True requires positive groot_token_count and groot_token_dim. "
                    f"Got count={self.groot_token_count}, dim={self.groot_token_dim}."
                )
            adapted_token_dim = self.groot_token_dim
            adapter_requested = (
                self.groot_feature_project_dim > 0
                or self.groot_feature_dropout > 0
                or self.groot_feature_scale_init != 1.0
            )
            if self.groot_feature_project_dim > 0:
                adapted_token_dim = self.groot_feature_project_dim
            if adapter_requested:
                self.groot_feature_adapter = GROOTFeatureAdapter(
                    input_dim=self.groot_token_dim,
                    output_dim=adapted_token_dim,
                    use_layer_norm=bool(getattr(self.cfg, "groot_feature_layer_norm", True)),
                    dropout=self.groot_feature_dropout,
                    scale_init=self.groot_feature_scale_init,
                    scale_learnable=bool(getattr(self.cfg, "groot_feature_scale_learnable", False)),
                )
            self.encoders = nn.ModuleList()
            repr_dim_single = self.groot_token_count * adapted_token_dim
            patch_repr_dim = adapted_token_dim
        else:
            # Build the per-camera encoders *after* `self.rl_cameras` is defined so
            # that the helper function can iterate over them.
            self.encoders: nn.ModuleList = self._build_encoders(obs_shape)

            # All encoders share the same architecture ⇒ repr / patch dim are identical.
            sample_encoder = self.encoders[0]
            repr_dim_single = int(sample_encoder.repr_dim)  # type: ignore[attr-defined]
            patch_repr_dim = int(sample_encoder.patch_repr_dim)  # type: ignore[attr-defined]

        if self.uses_depth_anything_v2_conditioning and self.uses_groot_features:
            raise ValueError(
                "GR00T observation features and DepthAnythingV2 CLS conditioning are mutually exclusive; "
                "enable macro_local_depth_gating for separate local depth geometry."
            )

        if self.uses_depth_anything_v2_conditioning:
            if self.cfg.enc_type != "vit":
                raise ValueError(
                    "DepthAnythingV2 CLS conditioning is currently only implemented for enc_type='vit'. "
                    f"Got enc_type={self.cfg.enc_type!r}."
                )
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

        if self.uses_macro_local_depth_gating:
            self.depth_anything_v2_encoder = DepthAnythingV2TokenEncoder(obs_shape, depth_cfg).to(self.cfg.device)
            print(common_utils.wrap_ruler("macro local depth gating"))
            print(
                f"enabled=True camera={self.macro_local_depth_camera_key} horizon={self.macro_action_horizon} "
                f"radius={self.macro_local_depth_cfg.neighborhood_radius}"
            )

        # Concatenate the patch dimension from every camera (dim=1) → overall
        # representation dimension scales linearly with #cameras.
        repr_dim = repr_dim_single if self.uses_groot_features else repr_dim_single * len(self.rl_cameras)
        print("encoder output dim: ", repr_dim)
        print("patch output dim: ", patch_repr_dim)
        if self.uses_groot_features:
            print(
                "Using precomputed GR00T observation features: "
                f"key={self.groot_feature_key} tokens={self.groot_token_count} dim={self.groot_token_dim}"
            )
            if self.groot_feature_adapter is not None:
                scale_kind = "learnable" if bool(getattr(self.cfg, "groot_feature_scale_learnable", False)) else "fixed"
                print(
                    "GR00T feature adapter enabled: "
                    f"out_dim={patch_repr_dim} layer_norm={bool(getattr(self.cfg, 'groot_feature_layer_norm', True))} "
                    f"dropout={self.groot_feature_dropout} scale={self.groot_feature_scale_init} ({scale_kind})"
                )

        assert len(prop_shape) == 1
        prop_dim = prop_shape[0] if cfg.use_prop else 0
        self.base_prop_dim = int(prop_dim)

        local_depth_feat_dim = 0
        local_depth_cls_dim = 0
        if self.uses_macro_local_depth_gating:
            if self.depth_anything_v2_encoder is None:
                raise RuntimeError("Macro local depth gating requires DepthAnythingV2TokenEncoder.")
            local_depth_feat_dim = int(self.depth_anything_v2_encoder.patch_repr_dim)
            local_depth_cls_dim = int(self.depth_anything_v2_encoder.patch_repr_dim)

        self.uses_macro_local_depth_critic_fusion = bool(
            self.uses_macro_local_depth_gating and self.macro_local_depth_cfg.fuse_to_critic
        )
        critic_prop_dim = prop_dim
        self.critic_local_depth_projector: nn.Module | None = None
        if self.uses_macro_local_depth_critic_fusion:
            local_depth_critic_in_dim = self.macro_action_horizon * local_depth_feat_dim
            if self.macro_local_depth_cfg.critic_use_global_depth_cls:
                local_depth_critic_in_dim += local_depth_cls_dim
            self.critic_local_depth_projector = self._build_local_depth_critic_projector(
                local_depth_critic_in_dim,
                int(self.macro_local_depth_cfg.critic_fusion_dim),
            )
            critic_prop_dim += int(self.macro_local_depth_cfg.critic_fusion_dim)

        # create critics & actor
        self.critic = Critic(
            repr_dim=repr_dim,
            patch_repr_dim=patch_repr_dim,
            prop_dim=critic_prop_dim,
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
            horizon_conditioned_actions=self.adaptive_horizon_conditioned_actions,
            macro_local_depth_gating_cfg=self.macro_local_depth_cfg,
            macro_action_horizon=self.macro_action_horizon,
            local_depth_feat_dim=local_depth_feat_dim,
            local_depth_cls_dim=local_depth_cls_dim,
        )

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target_local_depth_projector = (
            copy.deepcopy(self.critic_local_depth_projector)
            if self.critic_local_depth_projector is not None
            else None
        )
        self.actor_target = copy.deepcopy(self.actor)

        print(common_utils.wrap_ruler("encoder weights"))
        if self.uses_groot_features:
            print("Local image encoder disabled; QAgent consumes GR00T hidden tokens from replay/env observations.")
        else:
            print(self.encoders)
            common_utils.count_parameters(self.encoders)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)
        if self.critic_local_depth_projector is not None:
            print(common_utils.wrap_ruler("critic local depth adapter"))
            print(self.critic_local_depth_projector)
            common_utils.count_parameters(self.critic_local_depth_projector)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        # Freeze encoder parameters if requested
        if getattr(self.cfg, "freeze_encoder", False):
            for param in self.encoders.parameters():
                param.requires_grad = False
            if self.groot_feature_adapter is not None:
                for param in self.groot_feature_adapter.parameters():
                    param.requires_grad = False
            if self.depth_cls_projectors is not None:
                for param in self.depth_cls_projectors.parameters():
                    param.requires_grad = False
            if self.depth_anything_v2_encoder is not None:
                for param in self.depth_anything_v2_encoder.parameters():
                    param.requires_grad = False
            print("🧊 Encoder parameters frozen - no gradient updates will be performed")

        # Create optimizers (PyTorch will ignore frozen parameters)
        encoder_params = self._encoder_parameters()
        self.encoder_opt = torch.optim.AdamW(encoder_params, lr=self.cfg.critic_lr) if encoder_params else None
        self.critic_opt = torch.optim.AdamW(self._critic_parameters(), lr=self.cfg.critic_lr)
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
            if self.encoder_opt is not None:
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

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: common_utils.MultiCounter | None = None

        self.critic_target.train(False)
        if self.critic_target_local_depth_projector is not None:
            self.critic_target_local_depth_projector.train(False)
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
        if self.groot_feature_adapter is not None:
            self.groot_feature_adapter.train(training)
        if self.depth_cls_projectors is not None:
            self.depth_cls_projectors.train(training)
        if self.depth_anything_v2_encoder is not None:
            self.depth_anything_v2_encoder.train(training)
        if self.critic_local_depth_projector is not None:
            self.critic_local_depth_projector.train(training)
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        if self.critic_target_local_depth_projector is not None:
            assert not self.critic_target_local_depth_projector.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    def _encoder_parameters(self):
        params = list(self.encoders.parameters())
        if self.groot_feature_adapter is not None:
            params.extend(self.groot_feature_adapter.parameters())
        if self.depth_cls_projectors is not None:
            params.extend(self.depth_cls_projectors.parameters())
        if self.depth_anything_v2_encoder is not None:
            params.extend(self.depth_anything_v2_encoder.parameters())
        return params

    def _critic_parameters(self):
        params = list(self.critic.parameters())
        if self.critic_local_depth_projector is not None:
            params.extend(self.critic_local_depth_projector.parameters())
        return params

    def _build_local_depth_critic_projector(self, in_dim: int, out_dim: int) -> nn.Module:
        projector = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        init_scale = float(self.macro_local_depth_cfg.critic_fusion_init_scale)
        final_linear = projector[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("Expected the critic local depth projector to end with nn.Linear.")
        if init_scale == 0.0:
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
        else:
            nn.init.normal_(final_linear.weight, std=init_scale)
            nn.init.zeros_(final_linear.bias)
        return projector

    def _critic_prop(self, obs: dict[str, torch.Tensor], *, use_target: bool) -> torch.Tensor:
        if not self.uses_macro_local_depth_critic_fusion:
            return obs["observation.state"]

        self._augment_obs_with_macro_local_depth(obs)
        local_feat = obs.get("macro_local_depth_feat")
        if local_feat is None:
            raise KeyError("Critic local depth fusion expected `macro_local_depth_feat` in the observation.")
        if local_feat.dim() != 3:
            raise ValueError(f"Expected macro_local_depth_feat to be [B, H, C], got {tuple(local_feat.shape)}")

        depth_parts = [local_feat.flatten(1)]
        if self.macro_local_depth_cfg.critic_use_global_depth_cls:
            global_cls = obs.get("macro_local_depth_cls")
            if global_cls is None:
                raise KeyError("Critic local depth fusion expected `macro_local_depth_cls` in the observation.")
            depth_parts.append(global_cls.float())

        depth_context = torch.cat(depth_parts, dim=-1).float()
        if self.macro_local_depth_cfg.critic_detach_depth:
            depth_context = depth_context.detach()

        projector = self.critic_target_local_depth_projector if use_target else self.critic_local_depth_projector
        if projector is None:
            raise RuntimeError("Critic local depth fusion is enabled but no projector was created.")
        depth_prop = projector(depth_context)
        if self.base_prop_dim == 0:
            return depth_prop
        return torch.cat((obs["observation.state"], depth_prop), dim=-1)

    def _log_groot_feature_stats(self, metrics: dict[str, float], feat: torch.Tensor) -> None:
        if not self.uses_groot_features:
            return
        with torch.no_grad():
            feat_detached = feat.detach()
            metrics["train/groot_feature_abs_mean"] = feat_detached.abs().mean().item()
            metrics["train/groot_feature_rms"] = torch.sqrt(torch.mean(feat_detached.square())).item()
            if self.groot_feature_adapter is not None:
                scale = self.groot_feature_adapter.scale.detach()
                metrics["train/groot_feature_adapter_scale"] = scale.mean().item()

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
        self._augment_obs_with_macro_local_depth(obs)
        return actor.forward(obs, stddev)

    @staticmethod
    def _sample_dist(action_dist, *, eval_mode: bool, clip: float | None) -> torch.Tensor:
        if eval_mode:
            return action_dist.mean
        return action_dist.sample(clip=clip)

    @staticmethod
    def _sample_action_from_policy_output(policy_output, *, eval_mode: bool, clip: float | None) -> torch.Tensor:
        return QAgent._sample_dist(policy_output.action_dist, eval_mode=eval_mode, clip=clip)

    @staticmethod
    def _sample_horizon_actions_from_policy_output(
        policy_output, *, eval_mode: bool, clip: float | None
    ) -> torch.Tensor | None:
        if policy_output.horizon_action_dist is None:
            return None
        return QAgent._sample_dist(policy_output.horizon_action_dist, eval_mode=eval_mode, clip=clip)

    def _horizon_probs_from_logits(self, horizon_logits: torch.Tensor | None) -> torch.Tensor | None:
        if horizon_logits is None:
            return None
        return torch.softmax(horizon_logits, dim=-1)

    def _sample_horizon_onehot(
        self,
        horizon_logits: torch.Tensor,
        *,
        eval_mode: bool,
        horizon_epsilon: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(horizon_logits, dim=-1)
        if eval_mode:
            horizon_idx = torch.argmax(probs, dim=-1)
        else:
            horizon_idx = torch.distributions.Categorical(probs=probs).sample()
            horizon_epsilon = float(horizon_epsilon)
            if horizon_epsilon > 0:
                random_mask = torch.rand_like(horizon_idx.float()) < min(horizon_epsilon, 1.0)
                random_idx = torch.randint(self.num_adaptive_horizons, horizon_idx.shape, device=horizon_idx.device)
                horizon_idx = torch.where(random_mask, random_idx, horizon_idx)
        horizon_onehot = torch.nn.functional.one_hot(horizon_idx, num_classes=self.num_adaptive_horizons).to(
            probs.dtype
        )
        return horizon_onehot, probs

    def _combine_with_base_action(self, obs: dict[str, torch.Tensor], residual_action: torch.Tensor) -> torch.Tensor:
        if self.residual_actor:
            return torch.clamp(obs["observation.base_action"] + residual_action, -1.0, 1.0)
        return residual_action

    def _combine_horizon_actions_with_base_action(
        self,
        obs: dict[str, torch.Tensor],
        residual_actions_by_horizon: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_actor:
            base_action = obs["observation.base_action"].unsqueeze(1)
            return torch.clamp(base_action + residual_actions_by_horizon, -1.0, 1.0)
        return residual_actions_by_horizon

    @staticmethod
    def _select_horizon_candidate(candidate_actions: torch.Tensor, horizon_onehot: torch.Tensor) -> torch.Tensor:
        horizon_index = torch.argmax(horizon_onehot, dim=-1)
        batch_index = torch.arange(candidate_actions.shape[0], device=candidate_actions.device)
        return candidate_actions[batch_index, horizon_index]

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

    def _build_adaptive_candidate_critic_actions(self, combined_actions_by_horizon: torch.Tensor) -> torch.Tensor:
        batch_size, num_horizons, _ = combined_actions_by_horizon.shape
        if num_horizons != self.num_adaptive_horizons:
            raise ValueError(
                "Expected one action candidate per adaptive horizon. "
                f"Got {num_horizons}, expected {self.num_adaptive_horizons}."
            )

        combined_chunk = combined_actions_by_horizon.view(
            batch_size,
            num_horizons,
            self.max_action_horizon,
            self.primitive_action_dim,
        )
        candidate_actions = []
        for horizon_idx, horizon in enumerate(self.adaptive_horizons):
            executed_chunk = torch.zeros_like(combined_chunk[:, horizon_idx])
            executed_chunk[:, :horizon] = combined_chunk[:, horizon_idx, :horizon]
            horizon_onehot = torch.zeros(
                batch_size,
                self.num_adaptive_horizons,
                device=combined_actions_by_horizon.device,
                dtype=combined_actions_by_horizon.dtype,
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

    def _build_critic_actions_from_horizon_actions(
        self,
        obs: dict[str, torch.Tensor],
        residual_actions_by_horizon: torch.Tensor,
        *,
        horizon_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.uses_adaptive_horizons:
            raise RuntimeError("horizon-conditioned critic actions require adaptive horizons.")

        combined_actions = self._combine_horizon_actions_with_base_action(obs, residual_actions_by_horizon)
        candidate_actions = self._build_adaptive_candidate_critic_actions(combined_actions)
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

    def _adaptive_candidate_l2_penalty(
        self,
        residual_actions_by_horizon: torch.Tensor,
        horizon_probs: torch.Tensor,
    ) -> torch.Tensor:
        residual_chunk = residual_actions_by_horizon.view(
            residual_actions_by_horizon.shape[0],
            self.num_adaptive_horizons,
            self.max_action_horizon,
            self.primitive_action_dim,
        )
        per_step_sq = torch.sum(residual_chunk**2, dim=-1)
        horizon_penalties = torch.stack(
            [per_step_sq[:, horizon_idx, :horizon].sum(dim=-1) for horizon_idx, horizon in enumerate(self.adaptive_horizons)],
            dim=-1,
        )
        expected_penalty = torch.sum(horizon_probs * horizon_penalties, dim=-1)
        return self.cfg.actor.action_l2_reg_weight * expected_penalty.mean()

    def _score_adaptive_horizons(
        self,
        q_by_horizon: torch.Tensor,
        residual_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.uses_adaptive_horizons:
            return q_by_horizon, {}

        scores = q_by_horizon
        aux: dict[str, torch.Tensor] = {}
        horizon_values = torch.as_tensor(
            self.adaptive_horizons,
            device=q_by_horizon.device,
            dtype=q_by_horizon.dtype,
        ).view(1, -1)

        if self.adaptive_horizon_length_penalty > 0:
            length_cost = horizon_values.expand_as(q_by_horizon)
            scores = scores - (self.adaptive_horizon_length_penalty * length_cost)
            aux["horizon_length_cost"] = length_cost

        if self.adaptive_horizon_residual_penalty > 0:
            if residual_action.dim() == 3:
                residual_chunk = residual_action.view(
                    residual_action.shape[0],
                    self.num_adaptive_horizons,
                    self.max_action_horizon,
                    self.primitive_action_dim,
                )
                residual_cost = []
                for horizon_idx, horizon in enumerate(self.adaptive_horizons):
                    per_step_sq = torch.mean(residual_chunk[:, horizon_idx, :horizon] ** 2, dim=-1)
                    prefix_rms = torch.sqrt(per_step_sq.mean(dim=-1).clamp_min(1e-12))
                    residual_cost.append(prefix_rms * float(horizon))
                residual_cost = torch.stack(residual_cost, dim=-1)
            else:
                residual_chunk = residual_action.view(
                    residual_action.shape[0],
                    self.max_action_horizon,
                    self.primitive_action_dim,
                )
                per_step_sq = torch.mean(residual_chunk**2, dim=-1)
                prefix_sq_mean = torch.cumsum(per_step_sq, dim=-1) / torch.arange(
                    1,
                    self.max_action_horizon + 1,
                    device=residual_action.device,
                    dtype=residual_action.dtype,
                ).view(1, -1)
                prefix_rms = torch.sqrt(prefix_sq_mean.clamp_min(1e-12))
                residual_cost = torch.stack(
                    [prefix_rms[:, horizon - 1] * float(horizon) for horizon in self.adaptive_horizons],
                    dim=-1,
                )
            scores = scores - (self.adaptive_horizon_residual_penalty * residual_cost)
            aux["horizon_residual_cost"] = residual_cost

        return scores, aux

    def _adaptive_horizon_value_ce(
        self,
        horizon_logits: torch.Tensor,
        horizon_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = max(self.adaptive_horizon_value_temperature, 1e-6)
        with torch.no_grad():
            centered_scores = horizon_scores.detach() - horizon_scores.detach().mean(dim=-1, keepdim=True)
            target_probs = torch.softmax(centered_scores / temperature, dim=-1)
        log_probs = torch.log_softmax(horizon_logits, dim=-1)
        ce_loss = -(target_probs * log_probs).sum(dim=-1).mean()
        return ce_loss, target_probs

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

        if augment:
            data = self.aug(data)
        return data

    def _prepare_observation_images(self, obs: dict[str, torch.Tensor], *, augment: bool) -> dict[str, torch.Tensor]:
        return {cam_name: self._prepare_image_batch(obs[cam_name], augment=augment) for cam_name in self.rl_cameras}

    def get_depth_cache_keys(self) -> list[str]:
        return [self.depth_cache_keys_by_camera[cam_name] for cam_name in self.rl_cameras]

    def _recover_raw_state(self, standardized_state: torch.Tensor) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating:
            return standardized_state
        state_mean = self.macro_local_state_mean.to(standardized_state.device)
        state_std = self.macro_local_state_std.to(standardized_state.device)
        return standardized_state.float() * state_std + state_mean

    def _extract_current_eef_pos(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating:
            raise RuntimeError("_extract_current_eef_pos should only be used when macro local depth gating is enabled.")
        raw_state = self._recover_raw_state(obs["observation.state"])
        eef_indices = torch.as_tensor(
            self.macro_local_depth_cfg.eef_pos_indices,
            device=raw_state.device,
            dtype=torch.long,
        )
        return raw_state.index_select(dim=-1, index=eef_indices)

    def _unscale_base_action_chunk(self, base_action_flat: torch.Tensor) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating:
            raise RuntimeError(
                "_unscale_base_action_chunk should only be used when macro local depth gating is enabled."
            )
        if base_action_flat.shape[-1] != self.macro_action_horizon * self.primitive_action_dim:
            raise ValueError(
                "Macro local depth gating expected a flattened macro base action. "
                f"Got shape {tuple(base_action_flat.shape)} with horizon={self.macro_action_horizon} "
                f"and primitive_action_dim={self.primitive_action_dim}."
            )

        scaled_chunk = base_action_flat.float().view(-1, self.macro_action_horizon, self.primitive_action_dim)
        action_min = self.macro_local_action_min.to(scaled_chunk.device).view(1, 1, self.primitive_action_dim)
        action_max = self.macro_local_action_max.to(scaled_chunk.device).view(1, 1, self.primitive_action_dim)
        scaled_chunk = torch.clamp(scaled_chunk, -1.0, 1.0)
        return action_min + (scaled_chunk + 1.0) * (action_max - action_min) / 2.0

    def _rollout_future_eef_positions(
        self,
        curr_eef_pos: torch.Tensor,
        base_action_flat: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating:
            raise RuntimeError(
                "_rollout_future_eef_positions should only be used when macro local depth gating is enabled."
            )

        action_chunk = self._unscale_base_action_chunk(base_action_flat)
        action_pos_indices = torch.as_tensor(
            self.macro_local_depth_cfg.action_pos_indices,
            device=action_chunk.device,
            dtype=torch.long,
        )
        pos_deltas = action_chunk.index_select(dim=-1, index=action_pos_indices)
        return curr_eef_pos.unsqueeze(1) + pos_deltas.cumsum(dim=1)

    def _project_world_points_to_pixels(self, points_world: torch.Tensor) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating:
            raise RuntimeError(
                "_project_world_points_to_pixels should only be used when macro local depth gating is enabled."
            )
        if points_world.shape[-1] != 3:
            raise ValueError(f"Expected 3D world points, got shape {tuple(points_world.shape)}")

        transform = self.macro_local_world_to_camera.to(points_world.device)
        ones = torch.ones(*points_world.shape[:-1], 1, device=points_world.device, dtype=points_world.dtype)
        points_h = torch.cat((points_world.float(), ones), dim=-1)
        projected = points_h @ transform.t()
        depth = projected[..., 2:3]
        safe_depth = torch.where(depth > 1e-6, depth, torch.full_like(depth, 1e-6))
        pixel_xy = projected[..., :2] / safe_depth
        pixel_x = pixel_xy[..., 0].round().clamp(0, self.obs_width - 1)
        pixel_y = pixel_xy[..., 1].round().clamp(0, self.obs_height - 1)
        return torch.stack((pixel_y, pixel_x), dim=-1).long()

    def _pool_macro_local_depth_features(
        self,
        patch_tokens: torch.Tensor,
        pixels_rc: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating:
            raise RuntimeError(
                "_pool_macro_local_depth_features should only be used when macro local depth gating is enabled."
            )
        if self.depth_anything_v2_encoder is None:
            raise RuntimeError("Macro local depth gating requires DepthAnythingV2TokenEncoder.")
        if patch_tokens.dim() != 3:
            raise ValueError(f"Expected patch_tokens to be [B, N, C], got {tuple(patch_tokens.shape)}")
        if pixels_rc.shape[0] != patch_tokens.shape[0]:
            raise ValueError(
                "Batch size mismatch between patch tokens and projected pixels: "
                f"{tuple(patch_tokens.shape)} vs {tuple(pixels_rc.shape)}"
            )
        if pixels_rc.shape[1] != self.macro_action_horizon or pixels_rc.shape[-1] != 2:
            raise ValueError(
                "Expected projected pixels to be [B, H, 2] for macro local depth gating. "
                f"Got {tuple(pixels_rc.shape)}."
            )

        resize_to = int(self.depth_anything_v2_encoder.cfg.resize_to)
        depth_height = resize_to if resize_to > 0 else self.obs_height
        depth_width = resize_to if resize_to > 0 else self.obs_width
        patch_size = int(self.depth_anything_v2_encoder.patch_size)
        grid_h = depth_height // patch_size
        grid_w = depth_width // patch_size
        if grid_h * grid_w != patch_tokens.shape[1]:
            square_side = int(round(patch_tokens.shape[1] ** 0.5))
            if square_side * square_side != patch_tokens.shape[1]:
                raise ValueError(
                    "Could not infer a valid patch grid for macro local depth gating. "
                    f"resize=({depth_height}, {depth_width}), patch_size={patch_size}, "
                    f"num_tokens={patch_tokens.shape[1]}."
                )
            grid_h = square_side
            grid_w = square_side

        pixel_rows = pixels_rc[..., 0].float() * (depth_height / self.obs_height)
        pixel_cols = pixels_rc[..., 1].float() * (depth_width / self.obs_width)
        patch_rows = torch.div(pixel_rows, patch_size, rounding_mode="floor").long().clamp(0, grid_h - 1)
        patch_cols = torch.div(pixel_cols, patch_size, rounding_mode="floor").long().clamp(0, grid_w - 1)

        radius = int(self.macro_local_depth_cfg.neighborhood_radius)
        offsets = torch.arange(-radius, radius + 1, device=patch_tokens.device, dtype=torch.long)
        offset_rows, offset_cols = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_rows = offset_rows.reshape(1, 1, -1)
        offset_cols = offset_cols.reshape(1, 1, -1)

        neighbor_rows = (patch_rows.unsqueeze(-1) + offset_rows).clamp(0, grid_h - 1)
        neighbor_cols = (patch_cols.unsqueeze(-1) + offset_cols).clamp(0, grid_w - 1)
        linear_indices = neighbor_rows * grid_w + neighbor_cols

        batch_size, _, feat_dim = patch_tokens.shape
        gather_index = linear_indices.view(batch_size, -1, 1).expand(-1, -1, feat_dim)
        gathered = patch_tokens.gather(dim=1, index=gather_index)
        gathered = gathered.view(batch_size, self.macro_action_horizon, -1, feat_dim)
        return gathered.mean(dim=2)

    def _augment_obs_with_macro_local_depth(self, obs: dict[str, torch.Tensor]) -> None:
        if not self.uses_macro_local_depth_gating:
            return

        has_feat = "macro_local_depth_feat" in obs
        has_cls = (not self.macro_local_depth_cfg.use_global_depth_cls) or ("macro_local_depth_cls" in obs)
        if has_feat and has_cls:
            return

        if self.depth_anything_v2_encoder is None:
            raise RuntimeError("Macro local depth gating requires DepthAnythingV2TokenEncoder.")
        if self.macro_local_depth_camera_key not in obs:
            raise KeyError(
                "Macro local depth gating could not find the configured camera in the observation: "
                f"{self.macro_local_depth_camera_key!r}. Ensure this camera is available during acting and stored "
                "in replay if training should use it."
            )

        encoder_device = self.depth_anything_v2_encoder.pixel_mean.device
        camera_obs = obs[self.macro_local_depth_camera_key]
        if camera_obs.dim() == 3:
            camera_obs = camera_obs.unsqueeze(0)
        prepared_camera = self._prepare_image_batch(camera_obs, augment=False).to(encoder_device)

        with torch.no_grad():
            patch_tokens, cls_token = self.depth_anything_v2_encoder.forward_patches_and_cls(
                prepared_camera,
                flatten_patches=False,
            )
            current_eef_pos = self._extract_current_eef_pos(obs).to(encoder_device)
            future_eef_pos = self._rollout_future_eef_positions(
                current_eef_pos,
                obs["observation.base_action"].to(encoder_device),
            )
            projected_pixels = self._project_world_points_to_pixels(future_eef_pos)
            local_depth_feat = self._pool_macro_local_depth_features(patch_tokens, projected_pixels)

        target_device = obs["observation.state"].device
        obs["macro_local_depth_feat"] = local_depth_feat.to(target_device)
        if self.macro_local_depth_cfg.use_global_depth_cls:
            obs["macro_local_depth_cls"] = cls_token.to(target_device)

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

    def _encode_prepared_group(
        self,
        prepared: dict[str, torch.Tensor],
        *,
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

            feat_cam = self.encoders[cam_idx].forward(
                prepared[cam_name],
                flatten=False,
                layer_prefix_tokens=layer_prefix_tokens,
            )
            feats.append(feat_cam)

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
            encoded_groups.append(self._encode_prepared_group(prepared, depth_cls_by_cam=cls_by_cam))
        return encoded_groups

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        r"""This function encodes the observation into feature tensor.

        Images may be stored in the replay buffers as uint8 to save GPU memory.  In
        that case we convert them to float32 in \[0,1] before feeding them to the
        encoders.  If the image is already a float tensor (offline dataset or
        direct env observations during evaluation) we assume it is properly
        normalised.
        """
        if self.uses_groot_features:
            if self.groot_feature_key not in obs:
                raise KeyError(
                    f"Missing {self.groot_feature_key!r}; use_groot_features=True requires replay/env observations "
                    "to include precomputed GR00T hidden tokens."
                )
            feat = obs[self.groot_feature_key]
            if not isinstance(feat, torch.Tensor):
                feat = torch.as_tensor(feat, device=self.cfg.device)
            feat = feat.to(device=self.cfg.device, dtype=torch.float32)
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            if feat.dim() != 3:
                raise ValueError(f"Expected GR00T features with shape [B, T, D], got {tuple(feat.shape)}")
            if feat.shape[-2] != self.groot_token_count or feat.shape[-1] != self.groot_token_dim:
                raise ValueError(
                    "GR00T feature shape mismatch. "
                    f"Expected [*, {self.groot_token_count}, {self.groot_token_dim}], got {tuple(feat.shape)}"
                )
            if self.groot_feature_adapter is not None:
                feat = self.groot_feature_adapter(feat)
            return feat
        return self._encode_observation_groups([obs], augment=augment)[0]

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(
        self,
        obs: dict[str, torch.Tensor],
        *,
        eval_mode=False,
        stddev=0.0,
        cpu=True,
        horizon_epsilon: float = 0.0,
    ) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        # Make a shallow copy of the observation dict
        obs = copy.copy(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        assert "feat" not in obs
        obs["feat"] = self._encode(obs, augment=False)

        policy_output = self._forward_actor_policy(obs, stddev, use_target=False)
        if self.uses_adaptive_horizons:
            if policy_output.horizon_logits is None:
                raise RuntimeError("Adaptive horizon policy is enabled but the actor did not return horizon logits.")
            horizon_onehot, _ = self._sample_horizon_onehot(
                policy_output.horizon_logits,
                eval_mode=eval_mode,
                horizon_epsilon=horizon_epsilon,
            )
            horizon_actions = self._sample_horizon_actions_from_policy_output(
                policy_output,
                eval_mode=eval_mode,
                clip=None,
            )
            if horizon_actions is not None:
                residual_action = self._select_horizon_candidate(horizon_actions, horizon_onehot)
            else:
                residual_action = self._sample_action_from_policy_output(policy_output, eval_mode=eval_mode, clip=None)
            action = self._build_env_action(residual_action, horizon_onehot)
        else:
            residual_action = self._sample_action_from_policy_output(policy_output, eval_mode=eval_mode, clip=None)
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

            if self.uses_adaptive_horizons:
                if next_policy_output.horizon_logits is None:
                    raise RuntimeError("Adaptive horizon policy is enabled but the actor target did not return logits.")
                next_horizon_probs = self._horizon_probs_from_logits(next_policy_output.horizon_logits)
                assert next_horizon_probs is not None
                next_horizon_actions = self._sample_horizon_actions_from_policy_output(
                    next_policy_output,
                    eval_mode=not self.cfg.target_action_noise,
                    clip=self.cfg.stddev_clip,
                )
                if next_horizon_actions is not None:
                    next_critic_actions = self._build_critic_actions_from_horizon_actions(next_obs, next_horizon_actions)
                else:
                    next_residual_action = self._sample_action_from_policy_output(
                        next_policy_output,
                        eval_mode=not self.cfg.target_action_noise,
                        clip=self.cfg.stddev_clip,
                    )
                    next_critic_actions = self._build_critic_actions(next_obs, next_residual_action)
                next_q = self._evaluate_critic_actions(
                    self.critic_target,
                    next_obs["feat"],
                    self._critic_prop(next_obs, use_target=True),
                    next_critic_actions,
                    for_policy=False,
                )
                target_q_min = torch.sum(next_horizon_probs * next_q, dim=-1)
            else:
                next_residual_action = self._sample_action_from_policy_output(
                    next_policy_output,
                    eval_mode=not self.cfg.target_action_noise,
                    clip=self.cfg.stddev_clip,
                )
                next_critic_actions = self._build_critic_actions(next_obs, next_residual_action)
                next_action = next_critic_actions
                target_all = self.critic_target.q_value(
                    next_obs["feat"], self._critic_prop(next_obs, use_target=True), next_critic_actions
                )
                target_q_min = target_all.squeeze(-1)  # [B]
            target_q = (reward + (discount * target_q_min)).detach()

        if self.cfg.clip_q_target_to_reward_range:
            target_q = torch.clamp(target_q, min=0, max=1)  # Sparse rewards are in {0, 1}

        td_errors = None

        if self.critic.loss_cfg.type == "hl_gauss":
            # Compute logits for current Q heads and average HL-Gauss loss across heads
            q_per_head, logits_per_head = self.critic(obs["feat"], self._critic_prop(obs, use_target=False), action, return_logits=True)
            K = logits_per_head.shape[0]
            losses = [self.critic.hl_loss(logits_per_head[i], target_q) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        elif self.critic.loss_cfg.type == "c51":
            if self.uses_adaptive_horizons:
                raise NotImplementedError("Adaptive horizon residual TD3 currently supports only scalar-value critics.")
            # Compute logits for current Q heads and C51 distributional loss
            q_per_head, logits_per_head = self.critic(obs["feat"], self._critic_prop(obs, use_target=False), action, return_logits=True)

            # Get next state distribution for C51 target computation
            with torch.no_grad():
                _, next_logits = self.critic_target(
                    next_obs["feat"], self._critic_prop(next_obs, use_target=True), next_action, return_logits=True
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
            q_all = self.critic(obs["feat"], self._critic_prop(obs, use_target=False), action).squeeze(-1)  # [K,B]
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
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        # Gradient clipping
        encoder_params = self._encoder_parameters()
        if encoder_params:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder_params, self.cfg.critic_grad_clip_norm)
        else:
            encoder_grad_norm = torch.tensor(0.0, device=critic_loss.device)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self._critic_parameters(), self.cfg.critic_grad_clip_norm)

        # Store gradient norms for logging
        metrics["train/encoder_grad_norm"] = encoder_grad_norm.item()
        metrics["train/critic_grad_norm"] = critic_grad_norm.item()

        if self.encoder_opt is not None:
            self.encoder_opt.step()
        self.critic_opt.step()

        return metrics

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        assert "feat" in obs, "safety check"

        policy_output = self._forward_actor_policy(obs, 0.0, use_target=False)

        if self.uses_adaptive_horizons:
            if policy_output.horizon_logits is None:
                raise RuntimeError("Adaptive horizon policy is enabled but the actor did not return logits.")
            horizon_probs = self._horizon_probs_from_logits(policy_output.horizon_logits)
            assert horizon_probs is not None
            horizon_actions = self._sample_horizon_actions_from_policy_output(
                policy_output,
                eval_mode=True,
                clip=self.cfg.stddev_clip,
            )
            if horizon_actions is not None:
                critic_actions = self._build_critic_actions_from_horizon_actions(obs, horizon_actions)
                score_actions = horizon_actions
            else:
                action_pred = self._sample_action_from_policy_output(
                    policy_output,
                    eval_mode=True,
                    clip=self.cfg.stddev_clip,
                )
                critic_actions = self._build_critic_actions(obs, action_pred)
                score_actions = action_pred
            q_by_horizon = self._evaluate_critic_actions(
                self.critic,
                obs["feat"],
                self._critic_prop(obs, use_target=False),
                critic_actions,
                for_policy=True,
            )
            horizon_scores, horizon_score_aux = self._score_adaptive_horizons(q_by_horizon, score_actions)
            expected_raw_q = torch.sum(horizon_probs * q_by_horizon, dim=-1)
            expected_q = torch.sum(horizon_probs * horizon_scores, dim=-1)
            actor_loss_base = -expected_q.mean()
            if horizon_actions is not None:
                action_l2_penalty = self._adaptive_candidate_l2_penalty(horizon_actions, horizon_probs)
            else:
                action_l2_penalty = self._adaptive_prefix_l2_penalty(action_pred, horizon_probs)
            entropy = -(horizon_probs * torch.log(horizon_probs.clamp_min(1e-8))).sum(dim=-1).mean()
            actor_loss_total = actor_loss_base + action_l2_penalty - (self.adaptive_horizon_entropy_reg * entropy)
            alpha_l2_penalty = torch.zeros_like(actor_loss_base)
            if self.uses_macro_local_depth_gating and self.actor.last_macro_step_alpha is not None:
                alpha_reg_weight = float(self.macro_local_depth_cfg.alpha_l2_reg_weight)
                if alpha_reg_weight > 0.0:
                    alpha_l2_penalty = alpha_reg_weight * torch.mean((self.actor.last_macro_step_alpha - 1.0) ** 2)
                    actor_loss_total = actor_loss_total + alpha_l2_penalty
            horizon_value_ce = None
            horizon_value_target_probs = None
            if self.adaptive_horizon_value_ce_coef > 0:
                horizon_value_ce, horizon_value_target_probs = self._adaptive_horizon_value_ce(
                    policy_output.horizon_logits,
                    horizon_scores,
                )
                actor_loss_total = actor_loss_total + (self.adaptive_horizon_value_ce_coef * horizon_value_ce)

            greedy_horizon_onehot = torch.nn.functional.one_hot(
                torch.argmax(horizon_probs, dim=-1), num_classes=self.num_adaptive_horizons
            ).to(horizon_probs.dtype)
            if horizon_actions is not None:
                action_pred = self._select_horizon_candidate(horizon_actions, greedy_horizon_onehot)
                combined_action = self._build_critic_actions_from_horizon_actions(
                    obs,
                    horizon_actions,
                    horizon_onehot=greedy_horizon_onehot,
                )
            else:
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
                    "horizon_scores": horizon_scores,
                    "expected_raw_q": expected_raw_q,
                    "horizon_entropy": entropy,
                    "horizon_value_ce": horizon_value_ce,
                    "horizon_value_target_probs": horizon_value_target_probs,
                    "alpha_l2_penalty": alpha_l2_penalty,
                    **horizon_score_aux,
                },
            )

        action_pred = self._sample_action_from_policy_output(policy_output, eval_mode=True, clip=self.cfg.stddev_clip)
        action_l2_penalty = self.cfg.actor.action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))
        combined_action = self._build_critic_actions(obs, action_pred)
        q = self.critic.q_value_for_policy(obs["feat"], self._critic_prop(obs, use_target=False), combined_action)
        actor_loss_base = -q.mean()
        alpha_l2_penalty = torch.zeros_like(actor_loss_base)
        if self.uses_macro_local_depth_gating and self.actor.last_macro_step_alpha is not None:
            alpha_reg_weight = float(self.macro_local_depth_cfg.alpha_l2_reg_weight)
            if alpha_reg_weight > 0.0:
                alpha_l2_penalty = alpha_reg_weight * torch.mean((self.actor.last_macro_step_alpha - 1.0) ** 2)
        actor_loss_total = actor_loss_base + action_l2_penalty + alpha_l2_penalty

        return actor_loss_total, actor_loss_base, combined_action, action_pred, action_l2_penalty, {
            "alpha_l2_penalty": alpha_l2_penalty
        }

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
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()
        if self.uses_adaptive_horizons:
            horizon_probs = actor_aux["horizon_probs"].detach()
            q_by_horizon = actor_aux["q_by_horizon"].detach()
            horizon_scores = actor_aux["horizon_scores"].detach()
            metrics["train/horizon_entropy"] = actor_aux["horizon_entropy"].item()
            metrics["train/actor_expected_raw_q"] = actor_aux["expected_raw_q"].mean().item()
            if actor_aux["horizon_value_ce"] is not None:
                metrics["train/horizon_value_ce"] = actor_aux["horizon_value_ce"].item()
            if "horizon_length_cost" in actor_aux:
                metrics["train/horizon_length_cost"] = actor_aux["horizon_length_cost"].mean().item()
            if "horizon_residual_cost" in actor_aux:
                metrics["train/horizon_residual_cost"] = actor_aux["horizon_residual_cost"].mean().item()
            for horizon_idx, horizon in enumerate(self.adaptive_horizons):
                metrics[f"train/horizon_prob_{horizon}"] = horizon_probs[:, horizon_idx].mean().item()
                metrics[f"train/actor_q_h{horizon}"] = q_by_horizon[:, horizon_idx].mean().item()
                metrics[f"train/actor_score_h{horizon}"] = horizon_scores[:, horizon_idx].mean().item()
                if actor_aux["horizon_value_target_probs"] is not None:
                    target_probs = actor_aux["horizon_value_target_probs"].detach()
                    metrics[f"train/horizon_value_target_prob_{horizon}"] = target_probs[:, horizon_idx].mean().item()

        alpha_l2_penalty = actor_aux.get("alpha_l2_penalty")
        if alpha_l2_penalty is not None and alpha_l2_penalty.item() > 0.0:
            metrics["train/macro_step_alpha_l2_penalty"] = alpha_l2_penalty.item()
        if self.actor.last_macro_step_alpha is not None:
            alpha = self.actor.last_macro_step_alpha.detach()
            metrics["train/macro_step_alpha_mean"] = alpha.mean().item()
            metrics["train/macro_step_alpha_min"] = alpha.min().item()
            metrics["train/macro_step_alpha_max"] = alpha.max().item()
        if getattr(self.actor, "local_token_summary_scale", None) is not None:
            metrics["train/local_token_summary_scale"] = self.actor.local_token_summary_scale.detach().item()

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
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()
        if self.uses_adaptive_horizons:
            horizon_probs = actor_aux["horizon_probs"].detach()
            q_by_horizon = actor_aux["q_by_horizon"].detach()
            horizon_scores = actor_aux["horizon_scores"].detach()
            metrics["train/horizon_entropy"] = actor_aux["horizon_entropy"].item()
            metrics["train/actor_expected_raw_q"] = actor_aux["expected_raw_q"].mean().item()
            if actor_aux["horizon_value_ce"] is not None:
                metrics["train/horizon_value_ce"] = actor_aux["horizon_value_ce"].item()
            if "horizon_length_cost" in actor_aux:
                metrics["train/horizon_length_cost"] = actor_aux["horizon_length_cost"].mean().item()
            if "horizon_residual_cost" in actor_aux:
                metrics["train/horizon_residual_cost"] = actor_aux["horizon_residual_cost"].mean().item()
            for horizon_idx, horizon in enumerate(self.adaptive_horizons):
                metrics[f"train/horizon_prob_{horizon}"] = horizon_probs[:, horizon_idx].mean().item()
                metrics[f"train/actor_q_h{horizon}"] = q_by_horizon[:, horizon_idx].mean().item()
                metrics[f"train/actor_score_h{horizon}"] = horizon_scores[:, horizon_idx].mean().item()
                if actor_aux["horizon_value_target_probs"] is not None:
                    target_probs = actor_aux["horizon_value_target_probs"].detach()
                    metrics[f"train/horizon_value_target_prob_{horizon}"] = target_probs[:, horizon_idx].mean().item()

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

                curr_q = self.critic.q_value_for_policy(
                    bc_obs["feat"], self._critic_prop(bc_obs, use_target=False), curr_action
                )
                ref_q = self.critic.q_value_for_policy(
                    bc_obs["feat"], self._critic_prop(bc_obs, use_target=False), ref_action
                )

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss_total + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        # Conditionally update encoder along with actor if BC loss should backprop
        if bc_backprop_encoder and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient clipping
        metrics["train/actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.actor_grad_clip_norm
        ).item()

        if bc_backprop_encoder:
            encoder_params = self._encoder_parameters()
            if encoder_params:
                metrics["train/encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    encoder_params, self.cfg.actor_grad_clip_norm
                ).item()
            else:
                metrics["train/encoder_grad_norm"] = 0.0

        if bc_backprop_encoder and self.encoder_opt is not None:
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
            next_obs["feat"] = self._encode(next_obs, augment=True)

        metrics = {}
        self._log_groot_feature_stats(metrics, obs["feat"])
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
        if self.critic_local_depth_projector is not None and self.critic_target_local_depth_projector is not None:
            utils.soft_update_params(
                self.critic_local_depth_projector,
                self.critic_target_local_depth_projector,
                self.cfg.critic_target_tau,
            )
        metrics.update(critic_metric)

        if not update_actor:
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

        return metrics

    def step_lr_schedulers(self):
        """Step the learning rate schedulers for warmup."""
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
