from __future__ import annotations

import copy

import torch
from torch import nn

from resfit.rl_finetuning.config.residual_sac_flow import ResidualSACFlowAgentConfig
from resfit.rl_finetuning.off_policy.common_utils import RandomShiftsAug
from resfit.rl_finetuning.off_policy.common_utils.utils import soft_update_params
from resfit.rl_finetuning.off_policy.networks.encoder import VitEncoder
from resfit.rl_finetuning.residual_sac_flow.actor import PolicyOutput, TransformerFlowResidualActor
from resfit.rl_finetuning.residual_sac_flow.critic import TwinQCritic


class ResidualSACFlowAgent(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        cfg: ResidualSACFlowAgentConfig,
    ):
        super().__init__()
        if isinstance(rl_cameras, str):
            rl_cameras = [rl_cameras]
        if not rl_cameras:
            raise ValueError("At least one RL camera is required")

        self.rl_cameras = rl_cameras
        self.cfg = cfg
        self.action_dim = action_dim

        self.encoders = nn.ModuleList([VitEncoder(obs_shape, cfg.vit).to(cfg.device) for _ in rl_cameras])
        sample_encoder = self.encoders[0]
        self.patch_repr_dim = int(sample_encoder.patch_repr_dim)
        self.repr_dim = int(sample_encoder.repr_dim) * len(rl_cameras)

        if len(prop_shape) != 1:
            raise ValueError(f"Expected 1D proprioceptive shape, got {prop_shape}")
        self.state_dim = prop_shape[0] if cfg.use_prop else 0

        self.actor = TransformerFlowResidualActor(
            repr_dim=self.repr_dim,
            patch_repr_dim=self.patch_repr_dim,
            state_dim=self.state_dim,
            action_dim=action_dim,
            cfg=cfg.actor,
        )
        self.critic = TwinQCritic(
            repr_dim=self.repr_dim,
            state_dim=self.state_dim,
            action_dim=action_dim,
            cfg=cfg.critic,
        )
        self.critic_target = copy.deepcopy(self.critic)

        if cfg.freeze_encoder:
            for param in self.encoders.parameters():
                param.requires_grad = False

        self.encoder_opt = torch.optim.AdamW(self.encoders.parameters(), lr=cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        init_temperature = torch.tensor(cfg.init_temperature, device=cfg.device, dtype=torch.float32)
        self.log_alpha = nn.Parameter(init_temperature.log())
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=cfg.alpha_lr)

        self.target_entropy = (
            float(cfg.target_entropy) if cfg.target_entropy is not None else -float(action_dim)
        )

        self.aug = RandomShiftsAug(pad=4)
        self.residual_actor = True

        self.critic_target.eval()
        self.to(cfg.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoders.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(False)
        return self

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        feats = []
        for cam_idx, cam_name in enumerate(self.rl_cameras):
            data = obs[cam_name]
            if data.dtype == torch.uint8:
                data = data.float().div_(255.0)
            else:
                data = data.float()
            if augment:
                data = self.aug(data)
            feats.append(self.encoders[cam_idx](data, flatten=False))
        return torch.cat(feats, dim=1)

    def _maybe_unsqueeze_(self, obs: dict[str, torch.Tensor]) -> bool:
        if obs[self.rl_cameras[0]].dim() != 3:
            return False
        for key, value in obs.items():
            obs[key] = value.unsqueeze(0)
        return True

    def _policy(
        self,
        obs: dict[str, torch.Tensor],
        *,
        deterministic: bool,
        augment: bool,
        detach_encoder: bool,
    ) -> PolicyOutput:
        if "feat" not in obs:
            obs["feat"] = self._encode(obs, augment=augment)
        feat = obs["feat"].detach() if detach_encoder else obs["feat"]
        return self.actor(
            feat_tokens=feat,
            state=obs["observation.state"],
            base_action=obs["observation.base_action"],
            deterministic=deterministic,
        )

    @torch.no_grad
    def act(
        self,
        obs: dict[str, torch.Tensor],
        *,
        eval_mode: bool = False,
        cpu: bool = True,
    ) -> torch.Tensor:
        obs = {k: v for k, v in obs.items()}
        unsqueezed = self._maybe_unsqueeze_(obs)
        output = self._policy(obs, deterministic=eval_mode, augment=False, detach_encoder=False)
        action = output.mean_residual_action if eval_mode else output.residual_action
        if unsqueezed:
            action = action.squeeze(0)
        action = action.detach()
        return action.cpu() if cpu else action

    @torch.no_grad
    def act_final(
        self,
        obs: dict[str, torch.Tensor],
        *,
        eval_mode: bool = False,
        cpu: bool = True,
    ) -> torch.Tensor:
        obs = {k: v for k, v in obs.items()}
        unsqueezed = self._maybe_unsqueeze_(obs)
        output = self._policy(obs, deterministic=eval_mode, augment=False, detach_encoder=False)
        action = output.mean_final_action if eval_mode else output.final_action
        if unsqueezed:
            action = action.squeeze(0)
        action = action.detach()
        return action.cpu() if cpu else action

    @torch.no_grad
    def predict_q(self, obs: dict[str, torch.Tensor], final_action: torch.Tensor) -> torch.Tensor:
        obs = {k: v for k, v in obs.items()}
        if "feat" not in obs:
            obs["feat"] = self._encode(obs, augment=False)
        q = self.critic.min_q(
            feat_tokens=obs["feat"],
            state=obs["observation.state"],
            base_action=obs["observation.base_action"],
            action=final_action,
        )
        return q

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        importance_weights: torch.Tensor | None = None,
        *,
        deterministic_backup: bool = False,
        use_entropy_backup: bool = True,
        clip_q_target_to_reward_range: bool | None = None,
    ) -> dict[str, float | torch.Tensor]:
        with torch.no_grad():
            next_policy = self._policy(
                next_obs,
                deterministic=deterministic_backup,
                augment=False,
                detach_encoder=True,
            )
            target_q = self.critic_target.min_q(
                feat_tokens=next_obs["feat"],
                state=next_obs["observation.state"],
                base_action=next_obs["observation.base_action"],
                action=next_policy.final_action,
            )
            target_v = target_q
            if use_entropy_backup and next_policy.log_prob is not None:
                target_v = target_v - self.alpha.detach() * next_policy.log_prob
            target = reward + discount * target_v
            clip_target = (
                self.cfg.clip_q_target_to_reward_range
                if clip_q_target_to_reward_range is None
                else clip_q_target_to_reward_range
            )
            if clip_target:
                target = torch.clamp(target, min=0.0, max=1.0)

        q1, q2 = self.critic(
            feat_tokens=obs["feat"],
            state=obs["observation.state"],
            base_action=obs["observation.base_action"],
            action=action,
        )
        td_error = 0.5 * ((q1 - target).abs() + (q2 - target).abs()).squeeze(-1)
        critic_loss_per_sample = 0.5 * ((q1 - target).pow(2) + (q2 - target).pow(2)).squeeze(-1)

        if importance_weights is not None:
            critic_loss = (critic_loss_per_sample * importance_weights).mean()
        else:
            critic_loss = critic_loss_per_sample.mean()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoders.parameters(), self.cfg.critic_grad_clip_norm)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)
        self.encoder_opt.step()
        self.critic_opt.step()

        return {
            "train/critic_loss": critic_loss.item(),
            "train/critic_q1": q1.mean().item(),
            "train/critic_q2": q2.mean().item(),
            "train/critic_target": target.mean().item(),
            "train/encoder_grad_norm": float(encoder_grad_norm),
            "train/critic_grad_norm": float(critic_grad_norm),
            "_td_errors": td_error.detach().cpu(),
        }

    def update_actor_and_alpha(self, obs: dict[str, torch.Tensor]) -> dict[str, float | torch.Tensor]:
        policy = self._policy(obs, deterministic=False, augment=False, detach_encoder=True)
        q = self.critic.min_q(
            feat_tokens=obs["feat"],
            state=obs["observation.state"],
            base_action=obs["observation.base_action"],
            action=policy.final_action,
        )
        residual_penalty = (policy.residual_action.pow(2).sum(dim=-1).mean()) * self.cfg.actor.residual_l2_weight
        actor_loss = (self.alpha.detach() * policy.log_prob - q).mean() + residual_penalty

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)
        self.actor_opt.step()

        alpha_loss_value = 0.0
        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (policy.log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = alpha_loss.item()

        return {
            "train/actor_loss": actor_loss.item(),
            "train/actor_q": q.mean().item(),
            "train/actor_entropy": (-policy.log_prob).mean().item(),
            "train/alpha": self.alpha.item(),
            "train/alpha_loss": alpha_loss_value,
            "train/residual_l2": residual_penalty.item(),
            "train/actor_grad_norm": float(actor_grad_norm),
            "_actions": policy.residual_action.detach().cpu(),
            "_combined_actions": policy.final_action.detach().cpu(),
        }

    def update(
        self,
        batch,
        update_actor: bool,
        *,
        deterministic_backup: bool = False,
        use_entropy_backup: bool = True,
        clip_q_target_to_reward_range: bool | None = None,
    ) -> dict[str, float | torch.Tensor]:
        obs: dict[str, torch.Tensor] = batch["obs"]
        next_obs: dict[str, torch.Tensor] = batch[("next", "obs")]
        reward: torch.Tensor = batch[("next", "reward")]
        action: torch.Tensor = batch["action"]
        effective_discount: torch.Tensor = batch["gamma"] * batch["nonterminal"]
        importance_weights = batch.get("_weight", None)

        obs["feat"] = self._encode(obs, augment=True)
        with torch.no_grad():
            next_obs["feat"] = self._encode(next_obs, augment=True)

        metrics: dict[str, float | torch.Tensor] = {
            "data/batch_reward": reward.mean().item(),
        }
        metrics.update(
            self.update_critic(
                obs=obs,
                action=action,
                reward=reward,
                discount=effective_discount,
                next_obs=next_obs,
                importance_weights=importance_weights,
                deterministic_backup=deterministic_backup,
                use_entropy_backup=use_entropy_backup,
                clip_q_target_to_reward_range=clip_q_target_to_reward_range,
            )
        )

        if update_actor:
            obs["feat"] = obs["feat"].detach()
            metrics.update(self.update_actor_and_alpha(obs))

        return metrics

    def soft_update_targets(self) -> None:
        soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
