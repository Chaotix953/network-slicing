# -*- coding: utf-8 -*-
"""PPO agent implementation using Stable-Baselines3."""

from pathlib import Path
from typing import Tuple, Optional, Any, Union
import numpy as np

from .base_agent import BaseAgent

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent.

    Uses clipped surrogate objective for stable policy updates.
    Best for: Continuous control, sample efficiency not critical.
    """

    def __init__(self, env, config=None, **kwargs):
        """Initialize PPO agent.

        Args:
            env: Gymnasium environment
            config: PPOConfig or AgentConfig with ppo field
            **kwargs: Override parameters passed to SB3 PPO
        """
        super().__init__(env, config)

        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )

        # Extract PPO-specific config
        ppo_params = self._get_ppo_params(config, kwargs)

        self._model = PPO("MlpPolicy", env, **ppo_params)

    def _get_ppo_params(self, config, kwargs) -> dict:
        """Extract PPO parameters from config."""
        params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
        }

        # Override from config if available
        if config is not None:
            ppo_cfg = getattr(config, "ppo", config)
            for key in params:
                if hasattr(ppo_cfg, key):
                    params[key] = getattr(ppo_cfg, key)

        # Override from kwargs
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
    ) -> "PPOAgent":
        """Train the PPO agent."""
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
    def load(cls, path: Union[str, Path], env=None) -> "PPOAgent":
        """Load agent from disk."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        agent = cls.__new__(cls)
        agent.env = env
        agent.config = None
        agent._model = PPO.load(str(path), env=env)
        agent._is_trained = True
        return agent
