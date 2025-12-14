# -*- coding: utf-8 -*-
"""SAC agent implementation using Stable-Baselines3."""

from pathlib import Path
from typing import Tuple, Optional, Any, Union
import numpy as np

from .base_agent import BaseAgent

try:
    from stable_baselines3 import SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    SAC = None


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent.

    Off-policy algorithm with entropy regularization for exploration.
    Best for: Sample efficiency, continuous control, exploration-exploitation balance.
    """

    def __init__(self, env, config=None, **kwargs):
        """Initialize SAC agent.

        Args:
            env: Gymnasium environment
            config: SACConfig or AgentConfig with sac field
            **kwargs: Override parameters passed to SB3 SAC
        """
        super().__init__(env, config)

        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )

        sac_params = self._get_sac_params(config, kwargs)
        self._model = SAC("MlpPolicy", env, **sac_params)

    def _get_sac_params(self, config, kwargs) -> dict:
        """Extract SAC parameters from config."""
        params = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "verbose": 1,
        }

        if config is not None:
            sac_cfg = getattr(config, "sac", config)
            for key in params:
                if hasattr(sac_cfg, key):
                    params[key] = getattr(sac_cfg, key)

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
    ) -> "SACAgent":
        """Train the SAC agent."""
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
    def load(cls, path: Union[str, Path], env=None) -> "SACAgent":
        """Load agent from disk."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        agent = cls.__new__(cls)
        agent.env = env
        agent.config = None
        agent._model = SAC.load(str(path), env=env)
        agent._is_trained = True
        return agent
