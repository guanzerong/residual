from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from resfit.rl_finetuning.config.residual_sac_flow import ResidualSACFlowActorConfig


def atanh_clamped(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_log_prob(pre_tanh_action: torch.Tensor, action: torch.Tensor, base_log_prob: torch.Tensor) -> torch.Tensor:
    log_det = torch.log1p(-action.pow(2) + 1e-6)
    return base_log_prob - log_det.sum(dim=-1, keepdim=True)


@dataclass
class PolicyOutput:
    final_action: torch.Tensor
    residual_action: torch.Tensor
    log_prob: torch.Tensor | None
    mean_final_action: torch.Tensor
    mean_residual_action: torch.Tensor
    mean_delta: torch.Tensor
    std: torch.Tensor


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalStepEmbedding requires an even embedding dimension")
        self.dim = dim

    def forward(self, step_fraction: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=step_fraction.device, dtype=step_fraction.dtype)
        exponent = exponent / max(half_dim - 1, 1)
        frequencies = torch.exp(-torch.log(torch.tensor(10_000.0, device=step_fraction.device)) * exponent)
        angles = step_fraction.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TransformerFlowResidualActor(nn.Module):
    """SAC-flow-inspired residual actor with exact tanh-squashed Gaussian likelihood.

    The actor is centered around the current base action in the pre-tanh space:

        u_base = atanh(base_action)
        u_final = u_base + delta
        final_action = tanh(u_final)

    The distribution over ``delta`` is produced by an iterative transformer
    refinement stack conditioned on image tokens, proprioception, and the base
    action itself.
    """

    def __init__(
        self,
        repr_dim: int,
        patch_repr_dim: int,
        state_dim: int,
        action_dim: int,
        cfg: ResidualSACFlowActorConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim

        self.obs_token_proj = nn.Linear(patch_repr_dim, cfg.d_model)
        self.state_token_proj = nn.Linear(state_dim, cfg.d_model)
        self.base_token_proj = nn.Linear(action_dim, cfg.d_model)
        self.action_token_proj = nn.Linear(action_dim, cfg.d_model)

        self.step_embedding = SinusoidalStepEmbedding(cfg.d_model)
        self.step_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)
        self.refine_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, action_dim),
        )
        self.mean_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, action_dim),
        )
        self.log_std_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, action_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Keep the actor close to the base policy at initialization.
        final_layers = [
            self.refine_head[-1],
            self.mean_head[-1],
            self.log_std_head[-1],
        ]
        for layer in final_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _build_memory_tokens(
        self,
        feat_tokens: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = self.obs_token_proj(feat_tokens)
        state_token = self.state_token_proj(state).unsqueeze(1)
        base_token = self.base_token_proj(base_action).unsqueeze(1)
        return torch.cat([obs_tokens, state_token, base_token], dim=1)

    def forward(
        self,
        feat_tokens: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
        *,
        deterministic: bool,
    ) -> PolicyOutput:
        memory = self._build_memory_tokens(feat_tokens, state, base_action)
        u_base = atanh_clamped(base_action, eps=self.cfg.action_eps)

        latent_query = u_base
        decoder_summary = None
        for step in range(self.cfg.denoising_steps):
            step_fraction = torch.full(
                (base_action.shape[0],),
                float(step) / max(self.cfg.denoising_steps - 1, 1),
                device=base_action.device,
                dtype=base_action.dtype,
            )
            step_emb = self.step_proj(self.step_embedding(step_fraction)).unsqueeze(1)
            query_token = self.action_token_proj(latent_query).unsqueeze(1) + step_emb
            decoder_out = self.decoder(query_token, memory)
            decoder_summary = decoder_out.squeeze(1)
            refine_delta = self.refine_head(decoder_summary)
            latent_query = latent_query + refine_delta / float(self.cfg.denoising_steps)

        assert decoder_summary is not None
        mean_delta = self.cfg.latent_delta_scale * torch.tanh(self.mean_head(decoder_summary))

        raw_log_std = torch.tanh(self.log_std_head(decoder_summary))
        log_std = self.cfg.min_log_std + 0.5 * (self.cfg.max_log_std - self.cfg.min_log_std) * (raw_log_std + 1.0)
        std = self.cfg.latent_delta_scale * torch.exp(log_std)
        std = torch.clamp(std, min=1e-5)

        base_dist = Normal(mean_delta, std)
        if deterministic:
            delta = mean_delta
            log_prob = None
        else:
            delta = base_dist.rsample()
            log_prob = base_dist.log_prob(delta).sum(dim=-1, keepdim=True)

        mean_pre_tanh = u_base + mean_delta
        mean_final_action = torch.tanh(mean_pre_tanh)
        pre_tanh_action = u_base + delta
        final_action = torch.tanh(pre_tanh_action)

        if log_prob is not None:
            log_prob = squash_log_prob(pre_tanh_action, final_action, log_prob)

        residual_action = final_action - base_action
        mean_residual_action = mean_final_action - base_action
        return PolicyOutput(
            final_action=final_action,
            residual_action=residual_action,
            log_prob=log_prob,
            mean_final_action=mean_final_action,
            mean_residual_action=mean_residual_action,
            mean_delta=mean_delta,
            std=std,
        )
