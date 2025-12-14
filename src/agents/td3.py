# -*- coding: utf-8 -*-
"""TD3 agent implementation using Stable-Baselines3."""

from pathlib import Path
from typing import Tuple, Optional, Any, Union
import numpy as np

from .base_agent import BaseAgent

try:
    from stable_baselines3 import TD3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    TD3 = None


class TD3Agent(BaseAgent):
    """Twin Delayed Deep Deterministic Policy Gradient agent.

    Addresses overestimation bias in DDPG with twin critics and delayed updates.
    Best for: Continuous control, deterministic policies, stability.
    """

    def __init__(self, env, config=None, **kwargs):
        """Initialize TD3 agent.

        Args:
            env: Gymnasium environment
            config: TD3Config or AgentConfig with td3 field
            **kwargs: Override parameters passed to SB3 TD3
        """
        super().__init__(env, config)

        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )

        td3_params = self._get_td3_params(config, kwargs)
        self._model = TD3("MlpPolicy", env, **td3_params)

    def _get_td3_params(self, config, kwargs) -> dict:
        """Extract TD3 parameters from config."""
        params = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "verbose": 1,
        }

        if config is not None:
            td3_cfg = getattr(config, "td3", config)
            for key in params:
                if hasattr(td3_cfg, key):
                    params[key] = getattr(td3_cfg, key)

        params.update(kwargs)
        return params

    @property
    def model(self):
        """Access underlying SB3 model."""
        return self._model

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action given observation."""
        return self._model.predict(observation, deterministic=deterministic)

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10,
        progress_bar: bool = True,
    ) -> "TD3Agent":
        """Train the TD3 agent."""
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )
        self._is_trained = True
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save agent to disk."""
        self._model.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path], env=None) -> "TD3Agent":
        """Load agent from disk."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        agent = cls.__new__(cls)
        agent.env = env
        agent.config = None
        agent._model = TD3.load(str(path), env=env)
        agent._is_trained = True
        return agent
