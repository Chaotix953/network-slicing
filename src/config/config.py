# -*- coding: utf-8 -*-
"""
Configuration management using dataclasses.
Provides type-safe configuration with YAML serialization.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class SLAConfig:
    """SLA requirements for network slices."""
    urllc_max_delay_ms: float = 1.0
    urllc_max_loss: float = 1e-5
    mmtc_max_delay_ms: float = 1000.0
    mmtc_max_loss: float = 0.01


@dataclass
class TrafficConfig:
    """Traffic generation parameters."""
    urllc_lambda: float = 2000.0  # packets/s
    urllc_packet_size: int = 200  # bytes
    mmtc_lambda_on: float = 5000.0
    mmtc_lambda_off: float = 100.0
    mmtc_p_on: float = 0.1
    mmtc_p_off: float = 0.3
    mmtc_packet_size: int = 50


@dataclass
class NetworkConfig:
    """Network infrastructure configuration."""
    ryu_base_url: str = "http://localhost:8080"
    link_capacity_mbps: float = 100.0
    timeout_s: float = 2.0
    congestion_threshold: float = 0.85
    stats_interval_s: float = 5.0


@dataclass
class EnvConfig:
    """Environment configuration."""
    max_steps: int = 1000
    dt_s: float = 0.1  # timestep duration

    # Observation space bounds
    obs_high: float = 2.0

    # Reward weights
    w_urllc: float = 10.0
    w_mmtc: float = 2.0
    w_balance: float = 0.5

    # Penalty coefficients
    alpha_urllc: float = 50.0
    alpha_mmtc: float = 5.0
    alpha_smooth: float = 0.1
    alpha_fair: float = 0.0

    # SLA config
    sla: SLAConfig = field(default_factory=SLAConfig)

    # Traffic config
    traffic: TrafficConfig = field(default_factory=TrafficConfig)


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class SACConfig:
    """SAC algorithm hyperparameters."""
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"


@dataclass
class TD3Config:
    """TD3 algorithm hyperparameters."""
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5


@dataclass
class AgentConfig:
    """Agent configuration."""
    algorithm: str = "ppo"  # ppo, sac, td3, baseline_fixed, baseline_priority
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    td3: TD3Config = field(default_factory=TD3Config)

    # Baseline specific
    baseline_beta: float = 0.5  # for fixed baseline
    baseline_latency_threshold: float = 2.0  # for priority baseline


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_timesteps: int = 100000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    save_freq: int = 10000
    log_interval: int = 10
    seed: Optional[int] = None
    deterministic: bool = False

    # Directories
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "tensorboard"


@dataclass
class Config:
    """Main configuration container."""
    experiment_name: str = "default"
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Handle nested dataclasses
        if "env" in data:
            env_data = data["env"]
            if "sla" in env_data:
                env_data["sla"] = SLAConfig(**env_data["sla"])
            if "traffic" in env_data:
                env_data["traffic"] = TrafficConfig(**env_data["traffic"])
            data["env"] = EnvConfig(**env_data)

        if "agent" in data:
            agent_data = data["agent"]
            if "ppo" in agent_data:
                agent_data["ppo"] = PPOConfig(**agent_data["ppo"])
            if "sac" in agent_data:
                agent_data["sac"] = SACConfig(**agent_data["sac"])
            if "td3" in agent_data:
                agent_data["td3"] = TD3Config(**agent_data["td3"])
            data["agent"] = AgentConfig(**agent_data)

        if "training" in data:
            data["training"] = TrainingConfig(**data["training"])

        if "network" in data:
            data["network"] = NetworkConfig(**data["network"])

        return cls(**data)


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config.from_dict(data) if data else Config()


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
