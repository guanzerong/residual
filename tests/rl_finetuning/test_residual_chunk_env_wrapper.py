from __future__ import annotations

import gymnasium as gym
import torch

from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.wrappers.residual_chunk_env_wrapper import ResidualChunkVecEnvWrapper


class _FakeBasePolicy:
    def __init__(self, plans: list[torch.Tensor]):
        self.config = type("Config", (), {"image_features": {"observation.images.agentview": None}})()
        self._plans = plans
        self._plan_idx = 0
        self.reset_calls = 0

    def reset(self, env_ids=None):
        self.reset_calls += 1

    def select_action_chunk_normalized(self, batch, n_steps: int):
        plan = self._plans[self._plan_idx]
        self._plan_idx = min(self._plan_idx + 1, len(self._plans) - 1)
        return plan[:, :n_steps, :].clone()


class _FakeVecEnv:
    def __init__(self, *, success_at_step: int | None):
        self.num_envs = 1
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=float)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 3), dtype=float),
                "observation.images.agentview": gym.spaces.Box(low=0.0, high=1.0, shape=(1, 3, 4, 4), dtype=float),
            }
        )
        self.success_at_step = success_at_step
        self.step_count = 0
        self.metadata = {"horizon": 50}

    def _make_obs(self, value: float):
        return {
            "observation.state": torch.full((1, 3), value, dtype=torch.float32),
            "observation.images.agentview": torch.full((1, 3, 4, 4), value, dtype=torch.float32),
        }

    def reset(self, **kwargs):
        self.step_count = 0
        return self._make_obs(0.0), {}

    def step(self, action):
        self.step_count += 1
        reward = torch.tensor([1.0 if self.success_at_step == self.step_count else 0.0], dtype=torch.float32)
        terminated = torch.tensor([self.success_at_step == self.step_count], dtype=torch.bool)
        truncated = torch.tensor([False], dtype=torch.bool)
        obs = self._make_obs(float(self.step_count))
        info = {}
        if terminated.item():
            info["final_obs"] = [self._make_obs(99.0)]
            info["final_info"] = {"episode_steps": torch.tensor([self.step_count])}
            obs = self._make_obs(-1.0)
            self.step_count = 0
        return obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError

    def close(self):
        return None


def _build_scalers():
    action_scaler = ActionScaler(
        action_min=torch.tensor([-1.0, -1.0]),
        action_max=torch.tensor([1.0, 1.0]),
        action_scale=0.0,
        min_range_per_dim=0.1,
        device="cpu",
    )
    state_standardizer = StateStandardizer(
        state_mean=torch.zeros(3),
        state_std=torch.ones(3),
        min_std=0.1,
        device="cpu",
    )
    return action_scaler, state_standardizer


def test_chunk_wrapper_success_reward_and_reset_plan():
    plan_one = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
    plan_two = torch.tensor([[[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]], dtype=torch.float32)
    base_policy = _FakeBasePolicy([plan_one, plan_two])
    action_scaler, state_standardizer = _build_scalers()
    env = ResidualChunkVecEnvWrapper(
        vec_env=_FakeVecEnv(success_at_step=2),
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=2,
        base_plan_horizon=4,
    )

    obs, _ = env.reset()
    assert torch.allclose(obs["observation.base_action"], torch.tensor([[0.1, 0.2, 0.3, 0.4]]))

    next_obs, reward, terminated, truncated, info = env.step(torch.zeros((1, 4), dtype=torch.float32))

    assert reward.tolist() == [1.0]
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    assert info["executed_steps"].tolist() == [2]
    assert torch.allclose(info["scaled_action"], torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    assert torch.allclose(next_obs["observation.base_action"], torch.tensor([[0.9, 1.0, 1.1, 1.2]]))
    assert torch.allclose(info["final_obs"][0]["observation.base_action"], torch.zeros(4))


def test_chunk_wrapper_non_terminal_queue_rolls_forward():
    plan_one = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
    base_policy = _FakeBasePolicy([plan_one])
    action_scaler, state_standardizer = _build_scalers()
    env = ResidualChunkVecEnvWrapper(
        vec_env=_FakeVecEnv(success_at_step=None),
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=2,
        base_plan_horizon=4,
    )

    obs, _ = env.reset()
    assert torch.allclose(obs["observation.base_action"], torch.tensor([[0.1, 0.2, 0.3, 0.4]]))

    next_obs, reward, terminated, truncated, info = env.step(torch.zeros((1, 4), dtype=torch.float32))

    assert reward.tolist() == [0.0]
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [False]
    assert info["executed_steps"].tolist() == [2]
    assert torch.allclose(next_obs["observation.base_action"], torch.tensor([[0.5, 0.6, 0.7, 0.8]]))
