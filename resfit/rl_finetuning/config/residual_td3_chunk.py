# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.residual_td3 import (
    BasePolicyConfig,
    OfflineDataConfig,
    ResidualTD3AlgoConfig,
    ResidualTD3BoxCleanConfig,
    ResidualTD3CanConfig,
    ResidualTD3CoffeeConfig,
    ResidualTD3DexmgConfig,
    ResidualTD3SquareConfig,
    ResidualTD3TwoArmCanSortConfig,
    WandBConfig,
)


def _clone(value):
    return copy.deepcopy(value)


@dataclass
class ResidualTD3ChunkAlgoConfig(ResidualTD3AlgoConfig):
    macro_horizon: int = 2
    base_plan_horizon: int = 20
    n_step: int = 5
    offline_fraction: float = 0.0
    chunk_success_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.macro_horizon <= 0:
            raise ValueError("macro_horizon must be positive.")
        if self.base_plan_horizon < self.macro_horizon:
            raise ValueError("base_plan_horizon must be >= macro_horizon.")
        if self.base_plan_horizon % self.macro_horizon != 0:
            raise ValueError("base_plan_horizon must be divisible by macro_horizon.")


@dataclass
class ResidualTD3ChunkDexmgConfig(ResidualTD3DexmgConfig):
    algo: ResidualTD3ChunkAlgoConfig = field(default_factory=ResidualTD3ChunkAlgoConfig)
    eval_num_envs: int = 1


@dataclass
class ResidualTD3ChunkCanConfig(ResidualTD3ChunkDexmgConfig):
    task: str = ResidualTD3CanConfig.task
    offline_data: OfflineDataConfig = field(default_factory=lambda: _clone(ResidualTD3CanConfig().offline_data))
    base_policy: BasePolicyConfig = field(default_factory=lambda: _clone(ResidualTD3CanConfig().base_policy))
    wandb: WandBConfig = field(default_factory=lambda: _clone(ResidualTD3CanConfig().wandb))


@dataclass
class ResidualTD3ChunkSquareConfig(ResidualTD3ChunkDexmgConfig):
    task: str = ResidualTD3SquareConfig.task
    offline_data: OfflineDataConfig = field(default_factory=lambda: _clone(ResidualTD3SquareConfig().offline_data))
    base_policy: BasePolicyConfig = field(default_factory=lambda: _clone(ResidualTD3SquareConfig().base_policy))
    wandb: WandBConfig = field(default_factory=lambda: _clone(ResidualTD3SquareConfig().wandb))


@dataclass
class ResidualTD3ChunkBoxCleanConfig(ResidualTD3ChunkDexmgConfig):
    task: str = ResidualTD3BoxCleanConfig.task
    rl_camera: list[str] = field(default_factory=lambda: _clone(ResidualTD3BoxCleanConfig().rl_camera))
    algo: ResidualTD3ChunkAlgoConfig = field(
        default_factory=lambda: ResidualTD3ChunkAlgoConfig(total_timesteps=ResidualTD3BoxCleanConfig().algo.total_timesteps)
    )
    offline_data: OfflineDataConfig = field(default_factory=lambda: _clone(ResidualTD3BoxCleanConfig().offline_data))
    base_policy: BasePolicyConfig = field(default_factory=lambda: _clone(ResidualTD3BoxCleanConfig().base_policy))
    wandb: WandBConfig = field(default_factory=lambda: _clone(ResidualTD3BoxCleanConfig().wandb))


@dataclass
class ResidualTD3ChunkCoffeeConfig(ResidualTD3ChunkBoxCleanConfig):
    task: str = ResidualTD3CoffeeConfig.task
    rl_camera: list[str] = field(default_factory=lambda: _clone(ResidualTD3CoffeeConfig().rl_camera))
    algo: ResidualTD3ChunkAlgoConfig = field(
        default_factory=lambda: ResidualTD3ChunkAlgoConfig(total_timesteps=ResidualTD3CoffeeConfig().algo.total_timesteps)
    )
    offline_data: OfflineDataConfig = field(default_factory=lambda: _clone(ResidualTD3CoffeeConfig().offline_data))
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="dexmimicgen-test/10ncsd0h",
            wt_type="best",
            wt_version="v2",
        )
    )
    wandb: WandBConfig = field(default_factory=lambda: _clone(ResidualTD3CoffeeConfig().wandb))


@dataclass
class ResidualTD3ChunkTwoArmCanSortConfig(ResidualTD3ChunkBoxCleanConfig):
    task: str = ResidualTD3TwoArmCanSortConfig.task
    rl_camera: list[str] = field(default_factory=lambda: _clone(ResidualTD3TwoArmCanSortConfig().rl_camera))
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: _clone(ResidualTD3TwoArmCanSortConfig().offline_data)
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: _clone(ResidualTD3TwoArmCanSortConfig().base_policy)
    )
    wandb: WandBConfig = field(default_factory=lambda: _clone(ResidualTD3TwoArmCanSortConfig().wandb))


cs = ConfigStore.instance()
cs.store(name="residual_td3_chunk_dexmg_config", node=ResidualTD3ChunkDexmgConfig)
cs.store(name="residual_td3_chunk_can_config", node=ResidualTD3ChunkCanConfig)
cs.store(name="residual_td3_chunk_square_config", node=ResidualTD3ChunkSquareConfig)
cs.store(name="residual_td3_chunk_box_clean_config", node=ResidualTD3ChunkBoxCleanConfig)
cs.store(name="residual_td3_chunk_coffee_config", node=ResidualTD3ChunkCoffeeConfig)
cs.store(name="residual_td3_chunk_two_arm_cansort_config", node=ResidualTD3ChunkTwoArmCanSortConfig)
