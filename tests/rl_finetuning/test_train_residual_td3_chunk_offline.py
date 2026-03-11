from __future__ import annotations

import torch

from resfit.rl_finetuning.scripts.train_residual_td3_chunk import populate_chunk_offline_buffer
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


class _RecordingRB:
    def __init__(self):
        self.items = []

    def add(self, td):
        self.items.append(td.clone())

    def __len__(self):
        return len(self.items)


class _FakeBasePolicy:
    def __init__(self, plan: torch.Tensor):
        self.plan = plan
        self.reset_calls = 0

    def reset(self, env_ids=None):
        self.reset_calls += 1

    def select_action_chunk(self, batch, n_steps: int):
        return self.plan[:, :n_steps, :].clone()


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


def _sample(step: int, *, done: bool) -> dict:
    return {
        "episode_index": torch.tensor(0, dtype=torch.long),
        "observation.state": torch.tensor([float(step), float(step + 1), float(step + 2)], dtype=torch.float32),
        "observation.images.agentview": torch.full((3, 4, 4), float(step), dtype=torch.float32),
        "action": torch.tensor([0.1 * (step + 1), -0.1 * (step + 1)], dtype=torch.float32),
        "next.done": torch.tensor(done, dtype=torch.bool),
    }


def test_populate_chunk_offline_buffer_builds_macro_transitions():
    action_scaler, state_standardizer = _build_scalers()
    rb = _RecordingRB()
    dataset = [
        _sample(0, done=False),
        _sample(1, done=False),
        _sample(2, done=True),
    ]
    base_plan = torch.tensor([[[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]]], dtype=torch.float32)
    base_policy = _FakeBasePolicy(base_plan)

    transitions = populate_chunk_offline_buffer(
        dataset=dataset,
        rb=rb,
        image_keys=["observation.images.agentview"],
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        macro_horizon=2,
        base_plan_horizon=4,
        num_episodes=1,
        use_base_policy_for_base_actions=True,
        base_policy=base_policy,
        device=torch.device("cpu"),
    )

    assert transitions == 2
    assert len(rb) == 2

    first, second = rb.items

    assert torch.allclose(
        first["obs"]["observation.base_action"].squeeze(0),
        torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32),
    )
    assert torch.allclose(
        first["action"].squeeze(0),
        torch.tensor([0.1, -0.1, 0.2, -0.2], dtype=torch.float32),
    )
    assert first["next"]["reward"].item() == 0.0
    assert first["next"]["done"].item() is False
    assert torch.allclose(
        first["next"]["obs"]["observation.base_action"].squeeze(0),
        torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float32),
    )

    assert torch.allclose(
        second["obs"]["observation.base_action"].squeeze(0),
        torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float32),
    )
    assert torch.allclose(
        second["action"].squeeze(0),
        torch.tensor([0.3, -0.3, 0.3, 0.2], dtype=torch.float32),
    )
    assert second["next"]["reward"].item() == 1.0
    assert second["next"]["done"].item() is True
    assert torch.allclose(
        second["next"]["obs"]["observation.base_action"].squeeze(0),
        torch.zeros(4, dtype=torch.float32),
    )
