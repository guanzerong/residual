# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper


class ResidualChunkVecEnvWrapper:
    """Vectorized env wrapper for chunk-level residual RL with a queued base plan.

    This wrapper assumes a single environment instance. The base policy plans a long
    action sequence once, while the residual policy acts at a coarser macro cadence.
    """

    def __init__(
        self,
        vec_env: VectorizedEnvWrapper,
        base_policy,
        action_scaler,
        state_standardizer,
        *,
        macro_horizon: int,
        base_plan_horizon: int,
        chunk_success_threshold: float = 0.5,
    ):
        if getattr(vec_env, "num_envs", 1) != 1:
            raise ValueError("ResidualChunkVecEnvWrapper currently supports exactly one environment.")
        if macro_horizon <= 0:
            raise ValueError("macro_horizon must be positive.")
        if base_plan_horizon < macro_horizon:
            raise ValueError("base_plan_horizon must be >= macro_horizon.")

        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.macro_horizon = macro_horizon
        self.base_plan_horizon = base_plan_horizon
        self.chunk_success_threshold = chunk_success_threshold
        self.num_envs = 1

        self.primitive_action_dim = vec_env.action_space.shape[-1]
        self.macro_action_dim = self.primitive_action_dim * self.macro_horizon
        self.image_keys = list(base_policy.config.image_features.keys())
        self._base_naction_plan: torch.Tensor | None = None

        self._setup_observation_space()

    def _setup_observation_space(self):
        orig_obs_space = self.vec_env.observation_space
        orig_action_space = self.vec_env.action_space

        macro_shape = list(orig_action_space.shape)
        macro_shape[-1] = self.macro_action_dim
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=tuple(macro_shape), dtype=np.float32)

        obs_spaces = dict(orig_obs_space.spaces)
        obs_spaces["observation.base_action"] = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=tuple(macro_shape),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _plan_base_nactions(self, raw_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.base_policy, 'select_action_chunk'):
                base_actions = self.base_policy.select_action_chunk(raw_obs, n_steps=self.base_plan_horizon)
                base_nactions = self.action_scaler.scale(base_actions)
            else:
                # Test fallback for fake policies that already emit normalized chunks.
                base_nactions = self.base_policy.select_action_chunk_normalized(raw_obs, n_steps=self.base_plan_horizon)
        if base_nactions.shape != (self.num_envs, self.base_plan_horizon, self.primitive_action_dim):
            raise ValueError(
                "Base policy returned an unexpected action chunk shape: "
                f"got {tuple(base_nactions.shape)}, expected "
                f"({self.num_envs}, {self.base_plan_horizon}, {self.primitive_action_dim})."
            )
        return base_nactions

    def _ensure_base_plan(self, raw_obs: dict[str, torch.Tensor]) -> None:
        if self._base_naction_plan is None or self._base_naction_plan.shape[1] < self.macro_horizon:
            self._base_naction_plan = self._plan_base_nactions(raw_obs)

    def _current_base_chunk(self) -> torch.Tensor:
        if self._base_naction_plan is None:
            raise RuntimeError("Base action plan is not initialized. Call reset() first.")
        return self._base_naction_plan[:, : self.macro_horizon, :]

    def _consume_executed_steps(self, executed_steps: int) -> None:
        if self._base_naction_plan is None:
            return
        self._base_naction_plan = self._base_naction_plan[:, executed_steps:, :]

    def _flatten_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(actions.shape[0], -1)

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_chunk_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_chunk_flat
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])
        return augmented_obs

    def _augment_final_obs(self, final_obs_dict: dict, device: torch.device) -> dict[str, torch.Tensor]:
        final_obs = {}
        for key, value in final_obs_dict.items():
            if isinstance(value, torch.Tensor):
                final_obs[key] = value.to(device)
            else:
                final_obs[key] = torch.as_tensor(value, device=device)
        final_obs["observation.base_action"] = torch.zeros(self.macro_action_dim, device=device, dtype=torch.float32)
        final_obs["observation.state"] = self.state_standardizer.standardize(final_obs["observation.state"])
        return final_obs

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        raw_obs, info = self.vec_env.reset(**kwargs)
        self.base_policy.reset()
        self._base_naction_plan = self._plan_base_nactions(raw_obs)
        base_chunk_flat = self._flatten_chunk(self._current_base_chunk())
        return self._augment_obs(raw_obs, base_chunk_flat), info

    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if residual_naction.dim() == 1:
            residual_naction = residual_naction.unsqueeze(0)
        residual_chunk = residual_naction.view(self.num_envs, self.macro_horizon, self.primitive_action_dim)

        base_chunk = self._current_base_chunk()
        combined_chunk = torch.clamp(base_chunk + residual_chunk, -1.0, 1.0)

        primitive_rewards: list[torch.Tensor] = []
        raw_obs = None
        last_info: dict = {}
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=residual_naction.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=residual_naction.device)
        final_obs_entry: dict[str, torch.Tensor] | None = None
        success_within_chunk = False
        executed_steps = 0

        for step_idx in range(self.macro_horizon):
            env_action = self.action_scaler.unscale(combined_chunk[:, step_idx, :])
            raw_obs, reward, terminated, truncated, step_info = self.vec_env.step(env_action)
            primitive_rewards.append(reward)
            last_info = dict(step_info)
            executed_steps += 1

            if bool((reward > self.chunk_success_threshold).any().item()):
                success_within_chunk = True

            step_done = terminated | truncated
            if step_done.any():
                if "final_obs" in step_info and step_info["final_obs"][0] is not None:
                    final_obs_entry = self._augment_final_obs(step_info["final_obs"][0], residual_naction.device)
                break

        if raw_obs is None:
            raise RuntimeError("Chunk step did not execute any primitive environment steps.")

        reward_chunk = torch.full(
            (self.num_envs,),
            1.0 if success_within_chunk else 0.0,
            dtype=torch.float32,
            device=residual_naction.device,
        )

        step_done = terminated | truncated
        if success_within_chunk and not step_done.any():
            final_obs_entry = self._augment_final_obs(
                {key: value[0].clone() for key, value in raw_obs.items()}, residual_naction.device
            )
            raw_obs, reset_info = self.vec_env.reset()
            last_info.update(reset_info)
            terminated = torch.ones_like(terminated)
            truncated = torch.zeros_like(truncated)
            step_done = terminated | truncated

        if step_done.any():
            if final_obs_entry is None:
                final_obs_entry = self._augment_final_obs(
                    {key: value[0].clone() for key, value in raw_obs.items()}, residual_naction.device
                )
            self.base_policy.reset()
            self._base_naction_plan = self._plan_base_nactions(raw_obs)
        else:
            self._consume_executed_steps(executed_steps)
            self._ensure_base_plan(raw_obs)

        next_base_chunk_flat = self._flatten_chunk(self._current_base_chunk())
        augmented_obs = self._augment_obs(raw_obs, next_base_chunk_flat)

        macro_info = dict(last_info)
        macro_info["scaled_action"] = self._flatten_chunk(combined_chunk)
        macro_info["primitive_rewards"] = torch.stack(primitive_rewards, dim=1)
        macro_info["executed_steps"] = torch.full(
            (self.num_envs,), executed_steps, dtype=torch.long, device=residual_naction.device
        )
        macro_info["success_within_chunk"] = torch.full(
            (self.num_envs,), success_within_chunk, dtype=torch.bool, device=residual_naction.device
        )
        if step_done.any():
            macro_info["final_obs"] = [final_obs_entry]

        return augmented_obs, reward_chunk, terminated, truncated, macro_info

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)
