# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig, VitEncoderConfig
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
class LiberoEnvConfig:
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    num_steps_wait: int = 10
    env_resolution: int = 256
    max_steps: int | None = None


@dataclass
class LiberoEvalConfig:
    enabled: bool = False
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_episodes: int = 3
    interval_every_steps: int = 10_000
    eval_first: bool = True
    num_steps_wait: int = 10
    env_resolution: int = 256
    max_steps: int | None = None
    save_best_checkpoint: bool = True


@dataclass
class ResidualTD3ChunkDexmgConfig(ResidualTD3DexmgConfig):
    env_backend: str = "dexmg"
    algo: ResidualTD3ChunkAlgoConfig = field(default_factory=ResidualTD3ChunkAlgoConfig)
    eval_num_envs: int = 1
    libero_env: LiberoEnvConfig = field(default_factory=LiberoEnvConfig)
    libero_eval: LiberoEvalConfig = field(default_factory=LiberoEvalConfig)


@dataclass
class ResidualTD3ChunkCanConfig(ResidualTD3ChunkDexmgConfig):
    task: str = ResidualTD3CanConfig.task
    offline_data: OfflineDataConfig = field(default_factory=lambda: _clone(ResidualTD3CanConfig().offline_data))
    base_policy: BasePolicyConfig = field(default_factory=lambda: _clone(ResidualTD3CanConfig().base_policy))
    wandb: WandBConfig = field(default_factory=lambda: _clone(ResidualTD3CanConfig().wandb))


@dataclass
class ResidualTD3ChunkCanOpenPIConfig(ResidualTD3ChunkDexmgConfig):
    """Recommended preset for OpenPI pi05_robomimic_lcs on the robomimic Can task."""

    task: str = ResidualTD3CanConfig.task
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
            use_base_policy_for_base_actions=True,
        )
    )
    algo: ResidualTD3ChunkAlgoConfig = field(
        default_factory=lambda: ResidualTD3ChunkAlgoConfig(
            macro_horizon=2,
            base_plan_horizon=20,
            n_step=5,
            offline_fraction=0.5,
            chunk_success_threshold=0.5,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            source="openpi_ws",
            openpi_host="127.0.0.1",
            openpi_port=8000,
            openpi_chunk_size=20,
            task_prompt="pick up the coke can and place it on the correct place",
        )
    )
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="robomimic-can-residual-td3",
            group="openpi_pickplacecan",
        )
    )


@dataclass
class ResidualTD3ChunkCanOpenPINativeConfig(ResidualTD3ChunkDexmgConfig):
    """OpenPI-aligned Can preset: 224 images, 8-dim axis-angle state, mixed local dataset subset."""

    task: str = ResidualTD3CanConfig.task
    env_camera_size: int = 224
    env_state_encoding: str = "axis_angle"
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/data_all/gzr1/openpi/datasets/robomimic_ph__lift_can_square_lerobot",
            num_episodes=200,
            episode_start=200,
            dataset_schema="openpi",
            use_base_policy_for_base_actions=True,
        )
    )
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            vit=VitEncoderConfig(
                patch_size=16,
                stride=8,
            ),
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,
            ),
        )
    )
    algo: ResidualTD3ChunkAlgoConfig = field(
        default_factory=lambda: ResidualTD3ChunkAlgoConfig(
            macro_horizon=2,
            base_plan_horizon=20,
            n_step=5,
            offline_fraction=0.5,
            chunk_success_threshold=0.5,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            source="openpi_ws",
            openpi_host="127.0.0.1",
            openpi_port=8000,
            openpi_chunk_size=20,
            task_prompt="pick up the coke can and place it on the correct place",
        )
    )
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="robomimic-can-residual-td3",
            group="openpi_native_pickplacecan",
        )
    )


@dataclass
class ResidualTD3ChunkLiberoGoalOpenPIConfig(ResidualTD3ChunkDexmgConfig):
    """OpenPI-aligned LIBERO Goal preset using local LeRobot data and a 10-step base plan."""

    task: str = "LIBERO"
    env_backend: str = "libero"
    env_camera_size: int = 224
    env_state_encoding: str = "axis_angle"
    eval_num_episodes: int = 20
    save_video: bool = False
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/data_all/gzr1/libero_goal_lerobot",
            num_episodes=None,
            dataset_schema="openpi",
            use_base_policy_for_base_actions=True,
        )
    )
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            vit=VitEncoderConfig(
                patch_size=16,
                stride=8,
            ),
            actor=ActorConfig(
                action_scale=0.05,
                actor_last_layer_init_scale=0.0,
            ),
        )
    )
    algo: ResidualTD3ChunkAlgoConfig = field(
        default_factory=lambda: ResidualTD3ChunkAlgoConfig(
            macro_horizon=2,
            base_plan_horizon=10,
            n_step=5,
            offline_fraction=0.5,
            chunk_success_threshold=0.5,
        )
    )
    libero_env: LiberoEnvConfig = field(
        default_factory=lambda: LiberoEnvConfig(
            task_suite_name="libero_goal",
            task_id=0,
            num_steps_wait=10,
            env_resolution=256,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            source="openpi_ws",
            openpi_host="127.0.0.1",
            openpi_port=8000,
            openpi_chunk_size=10,
        )
    )
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="libero-goal-residual-td3",
            group="openpi_base10_goal",
        )
    )


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
cs.store(name="residual_td3_chunk_can_openpi_config", node=ResidualTD3ChunkCanOpenPIConfig)
cs.store(name="residual_td3_chunk_can_openpi_native_config", node=ResidualTD3ChunkCanOpenPINativeConfig)
cs.store(name="residual_td3_chunk_libero_goal_openpi_config", node=ResidualTD3ChunkLiberoGoalOpenPIConfig)
cs.store(name="residual_td3_chunk_square_config", node=ResidualTD3ChunkSquareConfig)
cs.store(name="residual_td3_chunk_box_clean_config", node=ResidualTD3ChunkBoxCleanConfig)
cs.store(name="residual_td3_chunk_coffee_config", node=ResidualTD3ChunkCoffeeConfig)
cs.store(name="residual_td3_chunk_two_arm_cansort_config", node=ResidualTD3ChunkTwoArmCanSortConfig)
