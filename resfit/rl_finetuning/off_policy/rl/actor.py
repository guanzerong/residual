# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import NamedTuple

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig, MacroLocalDepthGatingConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout, use_layer_norm=True):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def build_mlp_trunk(in_dim, hidden_dim, num_layer, layer_norm, dropout, use_layer_norm=True):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    if not layers:
        return nn.Identity(), in_dim
    return nn.Sequential(*layers), dims[-1]


class ActorOutput(NamedTuple):
    action_dist: utils.TruncatedNormal
    horizon_logits: torch.Tensor | None
    horizon_action_dist: utils.TruncatedNormal | None = None


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, use_layer_norm=True):
        super().__init__()

        # if fuse_patch:
        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim

        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        layers = [nn.Linear(proj_in_dim, proj_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(proj_dim))
        layers.append(nn.ReLU(inplace=True))

        self.input_proj = nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z  # noqa: RET504


class TokenSpatialEmb(nn.Module):
    """Context-conditioned pooling for token/patch feature maps."""

    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, use_layer_norm=True):
        super().__init__()

        proj_in_dim = patch_dim + prop_dim

        self.num_patch = num_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        layers = [nn.Linear(proj_in_dim, proj_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(proj_dim))
        layers.append(nn.ReLU(inplace=True))

        self.input_proj = nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.zeros(1, num_patch, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        if feat.size(1) != self.num_patch or feat.size(-1) != self.patch_dim:
            raise ValueError(
                "TokenSpatialEmb expected features shaped [B, num_patch, patch_dim]. "
                f"Expected [*, {self.num_patch}, {self.patch_dim}], got {tuple(feat.shape)}"
            )

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).expand(-1, feat.size(1), -1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z  # noqa: RET504


class MacroLocalTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        horizon: int,
        local_depth_feat_dim: int,
        local_depth_cls_dim: int,
        cfg: MacroLocalDepthGatingConfig,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.use_global_depth_cls = bool(cfg.use_global_depth_cls)
        self.token_dim = int(cfg.local_token_dim)
        self.local_token_proj = nn.Linear(local_depth_feat_dim, self.token_dim)
        self.step_pos_embed = nn.Parameter(torch.zeros(1, self.horizon, self.token_dim))
        self.global_token_proj = (
            nn.Linear(local_depth_cls_dim, self.token_dim) if self.use_global_depth_cls else None
        )

        if int(cfg.local_token_num_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=int(cfg.local_token_num_heads),
                dim_feedforward=int(cfg.local_token_ffn_dim),
                dropout=float(cfg.local_token_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(cfg.local_token_num_layers))
        else:
            self.encoder = None
        self.norm = nn.LayerNorm(self.token_dim)
        nn.init.normal_(self.step_pos_embed, std=0.02)

    def forward(
        self,
        local_feat: torch.Tensor,
        global_cls: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_feat.dim() != 3:
            raise ValueError(f"Expected local_feat to be [B, H, C], got {tuple(local_feat.shape)}")
        if local_feat.shape[1] != self.horizon:
            raise ValueError(f"Expected local_feat horizon {self.horizon}, got {local_feat.shape[1]}")

        step_tokens = self.local_token_proj(local_feat) + self.step_pos_embed[:, : local_feat.shape[1]]
        tokens = step_tokens

        if self.global_token_proj is not None:
            if global_cls is None:
                raise KeyError("Local token fusion expected `macro_local_depth_cls` in the observation.")
            if global_cls.dim() != 2:
                raise ValueError(f"Expected global_cls to be [B, C], got {tuple(global_cls.shape)}")
            global_token = self.global_token_proj(global_cls).unsqueeze(1)
            tokens = torch.cat((global_token, step_tokens), dim=1)

        if self.encoder is not None:
            tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        if self.global_token_proj is not None:
            return tokens[:, 1:], tokens[:, 0]
        return tokens, tokens.mean(dim=1)


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        patch_repr_dim,
        prop_dim,
        action_dim,
        cfg: ActorConfig,
        residual_actor: bool = False,
        horizon_choices: tuple[int, ...] | None = None,
        horizon_conditioned_actions: bool = False,
        macro_local_depth_gating_cfg: MacroLocalDepthGatingConfig | None = None,
        macro_action_horizon: int = 1,
        local_depth_feat_dim: int = 0,
        local_depth_cls_dim: int = 0,
    ):
        super().__init__()

        self.state_prop_dim = int(prop_dim)
        self.policy_context_dim = self.state_prop_dim + (int(action_dim) if residual_actor else 0)
        self.prop_dim = self.policy_context_dim
        self.residual_actor = residual_actor
        self.cfg = cfg
        self.horizon_choices = tuple(horizon_choices or ())
        self.horizon_conditioned_actions = bool(self.horizon_choices) and bool(horizon_conditioned_actions)
        self.action_dim = int(action_dim)
        self.action_scale = float(cfg.action_scale)
        self.macro_local_depth_gating_cfg = macro_local_depth_gating_cfg
        self.uses_macro_local_depth_gating = bool(
            macro_local_depth_gating_cfg is not None and macro_local_depth_gating_cfg.enabled
        )
        self.macro_action_horizon = int(macro_action_horizon)
        self.last_macro_step_alpha: torch.Tensor | None = None

        if self.horizon_conditioned_actions:
            max_horizon = max(self.horizon_choices)
            if self.action_dim % max_horizon != 0:
                raise ValueError(
                    "horizon-conditioned actions require action_dim to be divisible by the maximum horizon. "
                    f"Got action_dim={self.action_dim}, max_horizon={max_horizon}."
                )
            self.primitive_action_dim = self.action_dim // max_horizon
        else:
            self.primitive_action_dim = 0
        if self.uses_macro_local_depth_gating:
            if self.macro_action_horizon < 1:
                raise ValueError(f"macro_action_horizon must be >= 1, got {self.macro_action_horizon}")
            if not residual_actor:
                raise ValueError("Macro local depth gating is only implemented for residual actors.")
            if int(local_depth_feat_dim) <= 0:
                raise ValueError("Macro local depth gating requires a positive local_depth_feat_dim.")
            if not self.horizon_conditioned_actions:
                if self.action_dim != self.macro_action_horizon * self.primitive_action_dim:
                    raise ValueError(
                        "Macro local depth gating expects flattened macro actions. "
                        f"Got action_dim={self.action_dim}, horizon={self.macro_action_horizon}, "
                        f"primitive_action_dim={self.primitive_action_dim}."
                    )
            if self.primitive_action_dim <= 0:
                self.primitive_action_dim = self.action_dim // self.macro_action_horizon

        if cfg.spatial_emb > 0:
            assert cfg.spatial_emb > 1, "this is the dimension"
            spatial_emb_cls = TokenSpatialEmb if residual_actor else SpatialEmb
            self.compress = spatial_emb_cls(
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                prop_dim=self.policy_context_dim,
                proj_dim=cfg.spatial_emb,
                dropout=cfg.dropout,
                use_layer_norm=cfg.use_layer_norm,
            )
            policy_in_dim = cfg.spatial_emb
        else:
            layers = [nn.Linear(repr_dim, cfg.feature_dim)]
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(cfg.feature_dim))
            layers.extend([nn.Dropout(cfg.dropout), nn.ReLU()])

            self.compress = nn.Sequential(*layers)
            policy_in_dim = cfg.feature_dim

        if self.policy_context_dim > 0:
            policy_in_dim += self.policy_context_dim

        self.uses_local_token_fusion = bool(
            self.uses_macro_local_depth_gating and self.macro_local_depth_gating_cfg.use_local_token_fusion
        )
        if self.uses_macro_local_depth_gating:
            if self.uses_local_token_fusion:
                self.macro_local_token_encoder = MacroLocalTokenEncoder(
                    horizon=self.macro_action_horizon,
                    local_depth_feat_dim=local_depth_feat_dim,
                    local_depth_cls_dim=local_depth_cls_dim,
                    cfg=self.macro_local_depth_gating_cfg,
                )
                policy_in_dim += self.macro_local_token_encoder.token_dim
                self.local_token_summary_scale = nn.Parameter(
                    torch.tensor(float(self.macro_local_depth_gating_cfg.local_token_summary_init_scale))
                )
                gate_input_dim = int(self.macro_local_token_encoder.token_dim)
            else:
                self.macro_local_token_encoder = None
                self.local_token_summary_scale = None
                gate_input_dim = int(local_depth_feat_dim)
                if self.macro_local_depth_gating_cfg.use_global_depth_cls:
                    gate_input_dim += int(local_depth_cls_dim)
            if self.macro_local_depth_gating_cfg.use_base_action_step:
                gate_input_dim += self.primitive_action_dim

            gate_hidden_dim = int(self.macro_local_depth_gating_cfg.gate_hidden_dim)
            gate_layers: list[nn.Module] = [nn.Linear(gate_input_dim, gate_hidden_dim)]
            if cfg.use_layer_norm:
                gate_layers.append(nn.LayerNorm(gate_hidden_dim))
            gate_layers.extend([nn.ReLU(), nn.Linear(gate_hidden_dim, 1)])
            self.macro_step_gate = nn.Sequential(*gate_layers)
        else:
            self.macro_local_token_encoder = None
            self.local_token_summary_scale = None
            self.macro_step_gate = None

        self.policy_trunk, trunk_out_dim = build_mlp_trunk(
            policy_in_dim,
            cfg.hidden_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
        if self.horizon_conditioned_actions:
            self.action_head = None
            self.horizon_action_heads = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(trunk_out_dim, horizon * self.primitive_action_dim), nn.Tanh())
                    for horizon in self.horizon_choices
                ]
            )
        else:
            self.action_head = nn.Sequential(nn.Linear(trunk_out_dim, action_dim), nn.Tanh())
            self.horizon_action_heads = None
        self.horizon_head = nn.Linear(trunk_out_dim, len(self.horizon_choices)) if self.horizon_choices else None

        # Apply weight initialization
        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        """Apply weight initialization to all networks."""
        # Determine initialization distributions
        intermediate_init = cfg.actor_intermediate_layer_init_distribution
        if cfg.orth and intermediate_init == "default":
            intermediate_init = "orthogonal"

        # Initialize compression layers
        if cfg.orth:
            # Backward compatibility: use existing orthogonal initialization
            self.compress.apply(utils.orth_weight_init)
        else:
            # Use the specified distribution for compression layers
            utils.apply_initialization_to_network(self.compress, intermediate_init)

        utils.apply_initialization_to_network(self.policy_trunk, intermediate_init)
        if self.action_head is not None:
            utils.apply_initialization_to_network(self.action_head, intermediate_init, exclude_final_layer=True)
        if self.horizon_action_heads is not None:
            for action_head in self.horizon_action_heads:
                utils.apply_initialization_to_network(action_head, intermediate_init, exclude_final_layer=True)
        if self.horizon_head is not None:
            # Start adaptive horizon selection unbiased; ordinary random init can collapse early.
            nn.init.zeros_(self.horizon_head.weight)
            nn.init.zeros_(self.horizon_head.bias)
        if self.macro_step_gate is not None:
            utils.apply_initialization_to_network(self.macro_step_gate, intermediate_init, exclude_final_layer=True)
            final_linear = None
            for module in reversed(list(self.macro_step_gate.modules())):
                if isinstance(module, nn.Linear):
                    final_linear = module
                    break
            if final_linear is not None:
                nn.init.zeros_(final_linear.weight)
                nn.init.constant_(final_linear.bias, self.macro_local_depth_gating_cfg.init_logit_bias)
        if self.macro_local_token_encoder is not None:
            utils.apply_initialization_to_network(self.macro_local_token_encoder, intermediate_init)

        # Initialize final layer with specific configuration if provided
        if cfg.actor_last_layer_init_scale is not None:
            if self.action_head is not None:
                utils.initialize_layer_weights(
                    self.action_head[0],
                    cfg.actor_last_layer_init_distribution,
                    cfg.actor_last_layer_init_scale,
                )
            if self.horizon_action_heads is not None:
                for action_head in self.horizon_action_heads:
                    utils.initialize_layer_weights(
                        action_head[0],
                        cfg.actor_last_layer_init_distribution,
                        cfg.actor_last_layer_init_scale,
                    )

    def _encode_macro_local_tokens(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.uses_macro_local_depth_gating:
            return None, None

        local_feat = obs.get("macro_local_depth_feat")
        if local_feat is None:
            return None, None
        if local_feat.dim() != 3:
            raise ValueError(f"Expected macro_local_depth_feat to be [B, H, C], got {tuple(local_feat.shape)}")
        if local_feat.shape[1] != self.macro_action_horizon:
            raise ValueError(
                "Macro local depth features have a mismatched horizon. "
                f"Expected {self.macro_action_horizon}, got {local_feat.shape[1]}."
            )

        if self.macro_local_token_encoder is None:
            return local_feat, None
        global_cls = obs.get("macro_local_depth_cls")
        return self.macro_local_token_encoder(local_feat, global_cls)

    def _compute_macro_step_alpha(
        self,
        obs: dict[str, torch.Tensor],
        *,
        local_step_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.uses_macro_local_depth_gating or self.macro_step_gate is None:
            batch_size = obs["observation.base_action"].shape[0]
            return torch.ones(batch_size, self.macro_action_horizon, device=obs["observation.base_action"].device)

        if local_step_tokens is None:
            batch_size = obs["observation.base_action"].shape[0]
            return torch.ones(batch_size, self.macro_action_horizon, device=obs["observation.base_action"].device)

        gate_inputs = [local_step_tokens]
        if self.macro_local_token_encoder is None and self.macro_local_depth_gating_cfg.use_global_depth_cls:
            global_cls = obs.get("macro_local_depth_cls")
            if global_cls is None:
                raise KeyError("Macro local depth gating expected `macro_local_depth_cls` in the actor observation.")
            if global_cls.dim() != 2:
                raise ValueError(f"Expected macro_local_depth_cls to be [B, C], got {tuple(global_cls.shape)}")
            gate_inputs.append(global_cls.unsqueeze(1).expand(-1, self.macro_action_horizon, -1))

        if self.macro_local_depth_gating_cfg.use_base_action_step:
            base_action = obs["observation.base_action"]
            if base_action.shape[-1] != self.action_dim:
                raise ValueError(
                    "observation.base_action has a mismatched dimension for macro local depth gating. "
                    f"Expected last dim {self.action_dim}, got {base_action.shape[-1]}."
                )
            base_action = base_action.view(base_action.shape[0], self.macro_action_horizon, self.primitive_action_dim)
            gate_inputs.append(base_action)

        gate_input = torch.cat(gate_inputs, dim=-1)
        gate_logits = self.macro_step_gate(gate_input).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        alpha_min = float(self.macro_local_depth_gating_cfg.alpha_min)
        alpha_max = float(self.macro_local_depth_gating_cfg.alpha_max)
        return alpha_min + (alpha_max - alpha_min) * gate_prob

    def _apply_macro_step_alpha(
        self,
        action: torch.Tensor,
        alpha_per_step: torch.Tensor,
    ) -> torch.Tensor:
        alpha_per_dim = (
            alpha_per_step.unsqueeze(-1)
            .expand(-1, -1, self.primitive_action_dim)
            .reshape(alpha_per_step.shape[0], self.macro_action_horizon * self.primitive_action_dim)
        )
        if action.dim() == 3:
            alpha_per_dim = alpha_per_dim.unsqueeze(1)
        if action.shape[-1] != alpha_per_dim.shape[-1]:
            raise ValueError(
                "Macro local depth gating produced a mismatched alpha tensor. "
                f"Expected action last dim {action.shape[-1]}, got alpha last dim {alpha_per_dim.shape[-1]}."
            )
        return action * alpha_per_dim

    def _forward_horizon_conditioned_actions(self, trunk_out: torch.Tensor) -> torch.Tensor:
        if self.horizon_action_heads is None:
            raise RuntimeError("horizon-conditioned action heads are not enabled.")

        batch_size = trunk_out.shape[0]
        candidate_actions = []
        for horizon, action_head in zip(self.horizon_choices, self.horizon_action_heads, strict=True):
            prefix_action = action_head(trunk_out)
            padded_action = torch.zeros(
                batch_size,
                self.action_dim,
                device=prefix_action.device,
                dtype=prefix_action.dtype,
            )
            padded_action[:, : horizon * self.primitive_action_dim] = prefix_action
            candidate_actions.append(padded_action)
        return torch.stack(candidate_actions, dim=1)

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        local_step_tokens, local_token_summary = self._encode_macro_local_tokens(obs)

        context_inputs = []
        if self.state_prop_dim > 0:
            context_inputs.append(obs["observation.state"])
        if self.residual_actor:
            context_inputs.append(obs["observation.base_action"])
        context = torch.cat(context_inputs, dim=-1) if context_inputs else None

        if isinstance(self.compress, (SpatialEmb, TokenSpatialEmb)):
            if context is None:
                context = obs["feat"].new_zeros(obs["feat"].shape[0], 0)
            feat = self.compress.forward(obs["feat"], context)
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if local_token_summary is not None:
            if self.local_token_summary_scale is not None:
                local_token_summary = local_token_summary * self.local_token_summary_scale
            all_input.append(local_token_summary)
        if context is not None and self.policy_context_dim > 0:
            all_input.append(context)

        policy_input = torch.cat(all_input, dim=-1)
        trunk_out = self.policy_trunk(policy_input)

        horizon_action_dist = None
        std_for_dist: float | torch.Tensor = std
        if self.horizon_conditioned_actions:
            horizon_mu = self._forward_horizon_conditioned_actions(trunk_out)
            if self.uses_macro_local_depth_gating:
                alpha_per_step = self._compute_macro_step_alpha(obs, local_step_tokens=local_step_tokens)
                horizon_mu = self._apply_macro_step_alpha(horizon_mu, alpha_per_step)
                if self.macro_local_depth_gating_cfg.scale_std_with_alpha:
                    alpha_per_dim = self._apply_macro_step_alpha(
                        torch.ones_like(horizon_mu[:, -1]),
                        alpha_per_step,
                    )
                    if isinstance(std, torch.Tensor):
                        std_for_dist = std * alpha_per_dim
                    elif float(std) != 0.0:
                        std_for_dist = alpha_per_dim * float(std)
                self.last_macro_step_alpha = alpha_per_step
            else:
                self.last_macro_step_alpha = None
            mu = horizon_mu[:, -1]
            horizon_action_dist = utils.TruncatedNormal(horizon_mu * self.action_scale, std_for_dist)
        else:
            assert self.action_head is not None
            mu = self.action_head(trunk_out)
            if self.uses_macro_local_depth_gating:
                alpha_per_step = self._compute_macro_step_alpha(obs, local_step_tokens=local_step_tokens)
                mu = self._apply_macro_step_alpha(mu, alpha_per_step)
                if self.macro_local_depth_gating_cfg.scale_std_with_alpha:
                    alpha_per_dim = self._apply_macro_step_alpha(torch.ones_like(mu), alpha_per_step)
                    if isinstance(std, torch.Tensor):
                        std_for_dist = std * alpha_per_dim
                    elif float(std) != 0.0:
                        std_for_dist = alpha_per_dim * float(std)
                self.last_macro_step_alpha = alpha_per_step
            else:
                self.last_macro_step_alpha = None
        horizon_logits = self.horizon_head(trunk_out) if self.horizon_head is not None else None

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std_for_dist)

        return ActorOutput(
            action_dist=action_dist,
            horizon_logits=horizon_logits,
            horizon_action_dist=horizon_action_dist,
        )
