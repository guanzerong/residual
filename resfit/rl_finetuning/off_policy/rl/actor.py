# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import NamedTuple

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
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
    # For horizon-conditioned residuals this is the same tensor as the
    # distribution mean, exposed explicitly for diagnostics and selection.
    residual_candidates: torch.Tensor | None


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
        adaptive_residual_mode: str = "shared_prefix",
    ):
        super().__init__()

        self.prop_dim = prop_dim
        self.residual_actor = residual_actor
        self.cfg = cfg
        self.horizon_choices = tuple(horizon_choices or ())
        self.adaptive_residual_mode = str(adaptive_residual_mode)
        if self.adaptive_residual_mode not in {"shared_prefix", "horizon_conditioned"}:
            raise ValueError(
                "adaptive_residual_mode must be 'shared_prefix' or 'horizon_conditioned', "
                f"got {self.adaptive_residual_mode!r}"
            )
        self.horizon_conditioned = self.adaptive_residual_mode == "horizon_conditioned"
        if self.horizon_conditioned and not self.horizon_choices:
            raise ValueError("horizon_conditioned residuals require at least one horizon choice.")

        if residual_actor:
            # The residual actor takes the base action as input alongside the state
            self.prop_dim += action_dim

        if cfg.spatial_emb > 0:
            assert cfg.spatial_emb > 1, "this is the dimension"
            self.compress = SpatialEmb(
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                prop_dim=self.prop_dim,
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

        if self.prop_dim > 0:
            policy_in_dim += self.prop_dim

        self.policy_trunk, trunk_out_dim = build_mlp_trunk(
            policy_in_dim,
            cfg.hidden_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
        action_head_in_dim = trunk_out_dim
        if self.horizon_conditioned:
            duration_dim = int(cfg.duration_embedding_dim)
            if duration_dim <= 0:
                raise ValueError(f"duration_embedding_dim must be positive, got {duration_dim}")
            self.duration_embedding = nn.Embedding(len(self.horizon_choices), duration_dim)
            action_head_in_dim += duration_dim
        else:
            self.duration_embedding = None
        # One decoder is shared by every horizon. In conditioned mode only the
        # duration embedding changes across candidates.
        self.action_head = nn.Sequential(nn.Linear(action_head_in_dim, action_dim), nn.Tanh())
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
        utils.apply_initialization_to_network(self.action_head, intermediate_init, exclude_final_layer=True)
        if self.duration_embedding is not None:
            nn.init.normal_(self.duration_embedding.weight, mean=0.0, std=0.02)
        if self.horizon_head is not None:
            # Start adaptive selection unbiased so a random value-head
            # offset cannot collapse exploration onto one horizon before the
            # critic has learned a useful preference.
            nn.init.zeros_(self.horizon_head.weight)
            nn.init.zeros_(self.horizon_head.bias)

        # Initialize final layer with specific configuration if provided
        if cfg.actor_last_layer_init_scale is not None:
            utils.initialize_layer_weights(
                self.action_head[0],
                cfg.actor_last_layer_init_distribution,
                cfg.actor_last_layer_init_scale,
            )

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        if isinstance(self.compress, SpatialEmb):
            assert not self.residual_actor, "Not implemented"
            feat = self.compress.forward(obs["feat"], obs["observation.state"])
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if self.prop_dim > 0:
            prop = obs["observation.state"]
            all_input.append(prop)
            if self.residual_actor:
                # The residual actor takes the base action as input alongside the state
                all_input.append(obs["observation.base_action"])

        policy_input = torch.cat(all_input, dim=-1)
        trunk_out = self.policy_trunk(policy_input)

        residual_candidates: torch.Tensor | None = None
        if self.horizon_conditioned:
            assert self.duration_embedding is not None
            batch_size = trunk_out.shape[0]
            duration_indices = torch.arange(
                len(self.horizon_choices), device=trunk_out.device, dtype=torch.long
            )
            duration_features = self.duration_embedding(duration_indices)
            trunk_features = trunk_out.unsqueeze(1).expand(-1, len(self.horizon_choices), -1)
            duration_features = duration_features.unsqueeze(0).expand(batch_size, -1, -1)
            decoder_input = torch.cat([trunk_features, duration_features], dim=-1)
            mu = self.action_head(decoder_input)
            residual_candidates = mu * self.cfg.action_scale
        else:
            mu = self.action_head(trunk_out)
        horizon_logits = self.horizon_head(trunk_out) if self.horizon_head is not None else None

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = residual_candidates if residual_candidates is not None else mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = utils.TruncatedNormal(scaled_mu, std)

        return ActorOutput(
            action_dist=action_dist,
            horizon_logits=horizon_logits,
            residual_candidates=residual_candidates,
        )  # noqa: RET504
