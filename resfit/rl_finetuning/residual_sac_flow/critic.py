from __future__ import annotations

import torch
from torch import nn

from resfit.rl_finetuning.config.residual_sac_flow import ResidualSACFlowCriticConfig


def build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float, use_layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        state_dim: int,
        action_dim: int,
        cfg: ResidualSACFlowCriticConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.feature_proj = nn.Sequential(
            nn.Linear(repr_dim, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim) if cfg.use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        q_input_dim = cfg.feature_dim + state_dim + action_dim + action_dim
        self.q1 = build_mlp(q_input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.use_layer_norm)
        self.q2 = build_mlp(q_input_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.use_layer_norm)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        feat_tokens: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = feat_tokens.flatten(1)
        feat = self.feature_proj(feat)
        q_input = torch.cat([feat, state, base_action, action], dim=-1)
        return self.q1(q_input), self.q2(q_input)

    def min_q(
        self,
        feat_tokens: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.forward(feat_tokens, state, base_action, action)
        return torch.minimum(q1, q2)
