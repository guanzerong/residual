from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import VitEncoderConfig


@dataclass
class ResidualSACFlowActorConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    denoising_steps: int = 2
    dropout: float = 0.1
    latent_delta_scale: float = 0.2
    min_log_std: float = -5.0
    max_log_std: float = 0.0
    action_eps: float = 1e-4
    residual_l2_weight: float = 5e-3


@dataclass
class ResidualSACFlowCriticConfig:
    feature_dim: int = 256
    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.0
    use_layer_norm: bool = True


@dataclass
class ResidualSACFlowAgentConfig:
    device: str = "cuda"
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    critic_target_tau: float = 0.005
    init_temperature: float = 0.02
    target_entropy: float | None = None
    autotune_alpha: bool = True
    clip_q_target_to_reward_range: bool = False

    freeze_encoder: bool = False
    actor_grad_clip_norm: float = 1.0
    critic_grad_clip_norm: float = 1.0

    use_prop: bool = True
    enc_type: str = "vit"
    vit: VitEncoderConfig = field(default_factory=VitEncoderConfig)
    actor: ResidualSACFlowActorConfig = field(default_factory=ResidualSACFlowActorConfig)
    critic: ResidualSACFlowCriticConfig = field(default_factory=ResidualSACFlowCriticConfig)

    # Expands the dataset min/max range before mapping actions to [-1, 1].
    action_range_scale: float = 0.1


@dataclass
class ResidualSACFlowAlgoConfig:
    total_timesteps: int = 300_000
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    gamma: float = 0.995

    # Match the current TD3 Coffee training rhythm:
    # 4 gradient updates per environment step, with 1 actor update for every 4 updates.
    num_updates_per_iteration: int = 4
    actor_update_frequency: int = 4
    update_every_n_steps: int = 1
    target_update_frequency: int = 1

    offline_fraction: float = 0.5
    n_step: int = 5
    critic_warmup_steps: int = 10_000
    actor_learning_starts: int = 0
    critic_warmup_use_entropy_backup: bool = False
    critic_warmup_deterministic_backup: bool = True
    critic_warmup_clip_q_target: bool = True

    # Match TD3 warmup by default: collect online data around the base policy
    # instead of sampling uniformly over the full action box.
    random_action_noise_scale: float = 0.2
    use_base_policy_for_warmup: bool = True

    sampling_strategy: str = "uniform"
    prefetch_batches: int = 4
    priority_alpha: float = 0.6
    priority_beta: float = 0.4

    def __post_init__(self):
        if not 0.0 <= self.offline_fraction <= 1.0:
            raise ValueError(f"offline_fraction must be in [0, 1], got {self.offline_fraction}")
        if self.actor_update_frequency <= 0:
            raise ValueError("actor_update_frequency must be >= 1")
        if self.target_update_frequency <= 0:
            raise ValueError("target_update_frequency must be >= 1")


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 300
    use_base_policy_for_base_actions: bool = True
    min_action_range: float = 1e-1
    min_state_std: float = 1e-1


@dataclass
class WandBConfig:
    project: str = "residual-sac-flow"
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    name: str | None = None
    group: str | None = None


@dataclass
class BasePolicyConfig:
    wandb_id: str = "TODO"
    wt_type: str = "best"
    wt_version: str = "latest"


@dataclass
class ResidualSACFlowDexmgConfig:
    seed: int | None = None
    torch_deterministic: bool = False
    debug: bool = False

    task: str = "Can"
    num_envs: int = 1
    eval_num_envs: int = 8
    eval_num_episodes: int = 50
    headless: bool = True
    video_key: str = "observation.images.agentview"
    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]
    )

    algo: ResidualSACFlowAlgoConfig = field(default_factory=ResidualSACFlowAlgoConfig)
    agent: ResidualSACFlowAgentConfig = field(default_factory=ResidualSACFlowAgentConfig)
    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)
    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    log_freq: int = 100
    eval_interval_every_steps: int = 10_000
    checkpoint_interval: int = -1
    save_video: bool = True
    eval_first: bool = True


@dataclass
class ResidualSACFlowCanConfig(ResidualSACFlowDexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-can-bc/sdo8cku7",
        )
    )
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-residual-sac-flow"))


@dataclass
class ResidualSACFlowSquareConfig(ResidualSACFlowDexmgConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-square-bc/dzbkdpwp",
        )
    )
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-square-residual-sac-flow"))


@dataclass
class ResidualSACFlowBoxCleanConfig(ResidualSACFlowDexmgConfig):
    task: str = "TwoArmBoxCleanup"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-box-cleanup",
            num_episodes=1_000,
        )
    )
    algo: ResidualSACFlowAlgoConfig = field(
        default_factory=lambda: ResidualSACFlowAlgoConfig(
            total_timesteps=500_000,
        )
    )
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-box-clean-residual-sac-flow"))


@dataclass
class ResidualSACFlowCoffeeConfig(ResidualSACFlowDexmgConfig):
    task: str = "TwoArmCoffee"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-coffee",
            num_episodes=1_000,
        )
    )
    algo: ResidualSACFlowAlgoConfig = field(
        default_factory=lambda: ResidualSACFlowAlgoConfig(
            total_timesteps=500_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="dexmimicgen-test/10ncsd0h",
            wt_version="v2",
        )
    )
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-coffee-residual-sac-flow"))


@dataclass
class ResidualSACFlowTwoArmCanSortConfig(ResidualSACFlowDexmgConfig):
    task: str = "TwoArmCanSortRandom"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.frontview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )
    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-can-sort-random",
            num_episodes=1_000,
        )
    )
    algo: ResidualSACFlowAlgoConfig = field(
        default_factory=lambda: ResidualSACFlowAlgoConfig(
            total_timesteps=500_000,
        )
    )
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-cansort-residual-sac-flow"))


cs = ConfigStore.instance()
cs.store(name="residual_sac_flow_dexmg_config", node=ResidualSACFlowDexmgConfig)
cs.store(name="residual_sac_flow_can_config", node=ResidualSACFlowCanConfig)
cs.store(name="residual_sac_flow_square_config", node=ResidualSACFlowSquareConfig)
cs.store(name="residual_sac_flow_box_clean_config", node=ResidualSACFlowBoxCleanConfig)
cs.store(name="residual_sac_flow_coffee_config", node=ResidualSACFlowCoffeeConfig)
cs.store(name="residual_sac_flow_two_arm_cansort_config", node=ResidualSACFlowTwoArmCanSortConfig)
