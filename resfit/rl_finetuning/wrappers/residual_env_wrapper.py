# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Environment wrapper that includes a base policy, enabling residual RL training
to be done in a standard way without explicit base policy handling in the training loop.

This wrapper assumes:
- The environment is already vectorized (batched)
- All inputs/outputs are torch tensors
- The environment comes from create_vectorized_env
"""

import gymnasium as gym
import numpy as np
import torch

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.lerobot.policies.act.modeling_act import ACTPolicy


class BasePolicyVecEnvWrapper:
    """
    Wraps a vectorized environment with a base policy to enable standard RL training of residual policies.

    This wrapper:
    1. Takes raw observations from the vectorized environment
    2. Passes them through the base policy to get base actions
    3. Augments observations with base actions for the residual policy
    4. Combines base + residual actions before stepping the environment
    5. Returns augmented observations that include base actions

    Assumes the environment is already vectorized and works with torch tensors.
    """

    def __init__(
        self,
        vec_env: VectorizedEnvWrapper,
        base_policy: ACTPolicy,
        action_scaler,
        state_standardizer,
        macro_action_horizon: int = 1,
        macro_discount_gamma: float = 0.99,
        adaptive_horizons: tuple[int, ...] | None = None,
    ):
        """
        Args:
            vec_env: Vectorized environment from create_vectorized_env
            base_policy: Base policy (e.g., ACTPolicy) to augment with residual actions
            action_scaler: ActionScaler object for scaling/unscaling actions (REQUIRED)
            state_standardizer: StateStandardizer object for standardizing states (REQUIRED)
        """
        assert action_scaler is not None, "action_scaler is required for consistent normalization"
        assert state_standardizer is not None, "state_standardizer is required for consistent normalization"

        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.macro_action_horizon = int(macro_action_horizon)
        self.macro_discount_gamma = float(macro_discount_gamma)
        self.adaptive_horizons = tuple(sorted(adaptive_horizons or ()))
        self.num_adaptive_horizons = len(self.adaptive_horizons)

        if self.macro_action_horizon < 1:
            raise ValueError(f"macro_action_horizon must be >= 1, got {self.macro_action_horizon}")

        # Get action dimension from the environment
        self.primitive_action_dim = vec_env.action_space.shape[-1]
        self.macro_action_enabled = self.macro_action_horizon > 1
        self.base_action_dim = self.primitive_action_dim * self.macro_action_horizon
        self.adaptive_horizon_enabled = self.macro_action_enabled and self.num_adaptive_horizons > 0
        self.action_dim = self.base_action_dim + self.num_adaptive_horizons if self.adaptive_horizon_enabled else self.base_action_dim

        if self.macro_action_enabled:
            if vec_env.num_envs != 1:
                raise ValueError("Macro residual actions currently only support num_envs == 1.")
            if self.macro_action_horizon > base_policy.config.n_action_steps:
                raise ValueError(
                    "macro_action_horizon exceeds the base policy executable horizon. "
                    f"Requested {self.macro_action_horizon}, but ACT exposes {base_policy.config.n_action_steps} steps."
                )
        if self.adaptive_horizon_enabled:
            if self.adaptive_horizons[-1] != self.macro_action_horizon:
                raise ValueError(
                    "adaptive_horizons must end at macro_action_horizon. "
                    f"Got adaptive_horizons={self.adaptive_horizons}, macro_action_horizon={self.macro_action_horizon}."
                )
            if self.adaptive_horizons[0] < 1:
                raise ValueError(f"Adaptive horizons must be >= 1, got {self.adaptive_horizons}.")

        # Store image keys from base policy config
        self.image_keys = list(base_policy.config.image_features.keys())

        # Create modified observation space that includes base actions
        self._setup_observation_space()

    def _setup_observation_space(self):
        """Setup observation space to include base actions in the state."""

        # Get original observation space
        orig_obs_space = self.vec_env.observation_space

        # Copy the action space unless macro actions expand the residual dimension
        if self.macro_action_enabled:
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.vec_env.num_envs, self.action_dim),
                dtype=np.float32,
            )
        else:
            self.action_space = self.vec_env.action_space

        # Create new observation space with augmented state
        obs_spaces = {}
        for key, space in orig_obs_space.spaces.items():
            if key == "observation.state":
                # Augment state dimension with base actions
                orig_shape = list(space.shape)
                new_shape = orig_shape.copy()
                # new_shape[-1] += self.action_dim  # Not anymore
                obs_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(new_shape), dtype=space.dtype)
            else:
                # Keep other observations unchanged
                obs_spaces[key] = space

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset environment and base policy."""
        # Reset the underlying vectorized environment
        raw_obs, info = self.vec_env.reset(**kwargs)

        # Reset base policy
        self.base_policy.reset()

        # Get base action from the base policy
        base_naction = self._compute_base_naction(raw_obs)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Store for later use in step
        self._last_base_naction = base_naction

        return augmented_obs, info

    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment with residual action.

        Args:
            residual_action: The residual action from the residual policy

        Returns:
            augmented_obs: Observations augmented with base actions
            reward: Reward tensor
            terminated: Terminated tensor
            truncated: Truncated tensor
            info: Info dict
        """
        if self.macro_action_enabled:
            return self._step_macro(residual_naction)
        return self._step_single(residual_naction)

    def _step_single(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Combine base and residual actions
        # Residual action is already scaled inside the Actor class
        # To ensure that we can use the same exploration for all dimensions,
        # we use the normalized actions as the action space
        # The normalized base action is stored as [-1, 1] in the replay buffer
        # and the residual action is predicted as action_scale * [-1, 1]
        combined_naction = self._last_base_naction + residual_naction

        # Unscale back to original action space for environment execution
        env_action = self.action_scaler.unscale(combined_naction)

        # Step the underlying vectorized environment
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)

        # Store the scaled action for replay buffer (already computed above)
        info["scaled_action"] = combined_naction

        # Get next base action from the base policy
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)

        base_naction = self.action_scaler.scale(base_action)

        # Handle policy reset for terminated environments
        if terminated.any():
            reset_ids = torch.where(terminated)[0]
            self.base_policy.reset(env_ids=reset_ids)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Handle final_obs in info dict to ensure consistent shapes
        if "final_obs" in info:
            info = self._process_final_obs_in_info(info, combined_naction.device)

        # Store for next step
        self._last_base_naction = base_naction

        return augmented_obs, reward, terminated, truncated, info

    def _step_macro(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if residual_naction.dim() == 1:
            residual_naction = residual_naction.unsqueeze(0)

        batch_size = residual_naction.shape[0]
        chosen_horizon = self.macro_action_horizon
        horizon_onehot = None
        if self.adaptive_horizon_enabled:
            residual_chunk = residual_naction[:, : self.base_action_dim]
            horizon_logits = residual_naction[:, self.base_action_dim :]
            horizon_idx = torch.argmax(horizon_logits, dim=-1)
            horizon_onehot = torch.nn.functional.one_hot(horizon_idx, num_classes=self.num_adaptive_horizons).to(
                residual_naction.dtype
            )
            chosen_horizon = int(self.adaptive_horizons[int(horizon_idx[0].item())])
            residual_naction = residual_chunk

        combined_naction = self._last_base_naction + residual_naction
        combined_chunk = combined_naction.view(batch_size, self.macro_action_horizon, self.primitive_action_dim)
        env_action_chunk = self.action_scaler.unscale(combined_chunk)

        discounted_reward = torch.zeros(batch_size, device=combined_naction.device, dtype=torch.float32)
        undiscounted_reward = torch.zeros_like(discounted_reward)
        primitive_steps = torch.zeros(batch_size, device=combined_naction.device, dtype=torch.long)
        executed_chunk = torch.zeros_like(combined_chunk)

        terminated = torch.zeros(batch_size, device=combined_naction.device, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        macro_success = torch.zeros_like(terminated)

        raw_obs = None
        info: dict = {}

        for step_idx in range(chosen_horizon):
            env_action = env_action_chunk[:, step_idx]
            raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)

            discounted_reward += (self.macro_discount_gamma**step_idx) * reward
            undiscounted_reward += reward
            primitive_steps += 1
            executed_chunk[:, step_idx] = combined_chunk[:, step_idx]

            done = terminated | truncated
            if done.any():
                macro_success = macro_success | terminated
                reset_ids = torch.where(done)[0]
                self.base_policy.reset(env_ids=reset_ids)
                break

        if raw_obs is None:
            raise RuntimeError("Macro residual wrapper failed to execute any primitive environment step.")

        base_naction = self._compute_base_naction(raw_obs)
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        padded_action = executed_chunk.reshape(batch_size, -1)
        if horizon_onehot is not None:
            padded_action = torch.cat([padded_action, horizon_onehot], dim=-1)
        info["scaled_action"] = padded_action
        info["primitive_steps"] = primitive_steps
        info["macro_success"] = macro_success
        info["undiscounted_reward"] = undiscounted_reward
        if self.adaptive_horizon_enabled:
            info["chosen_horizon"] = torch.full(
                (batch_size,), chosen_horizon, device=combined_naction.device, dtype=torch.long
            )
            info["executed_horizon"] = primitive_steps.clone()
            info["gamma"] = torch.pow(
                torch.full((batch_size,), self.macro_discount_gamma, device=combined_naction.device, dtype=torch.float32),
                primitive_steps.float(),
            )
            info["nonterminal"] = ~(terminated | truncated)

        if "final_obs" in info:
            info = self._process_final_obs_in_info(info, combined_naction.device)

        self._last_base_naction = base_naction

        return augmented_obs, discounted_reward, terminated, truncated, info

    def _compute_base_naction(self, raw_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            if self.macro_action_enabled:
                base_action = self.base_policy.select_action_chunk(
                    raw_obs,
                    n_action_steps=self.macro_action_horizon,
                )
                return self.action_scaler.scale(base_action).reshape(base_action.shape[0], -1)

            base_action = self.base_policy.select_action(raw_obs)
            return self.action_scaler.scale(base_action)

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        # New way to do this is to just add the base action to the state under its own key
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])

        return augmented_obs

    def _process_final_obs_in_info(self, info: dict, device: torch.device) -> dict:
        """Pad final_obs state with zeros to match augmented observation format."""
        if "final_obs" not in info or info["final_obs"] is None:
            return info

        for final_obs_dict in info["final_obs"]:
            if final_obs_dict is not None and "observation.state" in final_obs_dict:
                # Pad with zeros (no action taken at terminal state)
                final_obs_dict["observation.base_action"] = torch.zeros(
                    self.base_action_dim, device=device, dtype=torch.float32
                )

        return info

    def render(self):
        """Pass through to underlying environment."""
        return self.vec_env.render()

    def close(self):
        """Close the environment."""
        return self.vec_env.close()

    # Pass through any other attributes/methods to the underlying environment
    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)
