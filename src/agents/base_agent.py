# -*- coding: utf-8 -*-
"""
Base agent interface for all RL and baseline agents.
Provides a common API for training, evaluation, and inference.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, env, config=None, **kwargs):
        """Initialize agent.

        Args:
            env: Gymnasium environment
            config: Agent configuration (dataclass or dict)
            **kwargs: Additional parameters
        """
        self.env = env
        self.config = config
        self._is_trained = False

    @property
    def name(self) -> str:
        """Agent name for logging."""
        return self.__class__.__name__

    @property
    def is_trained(self) -> bool:
        """Whether the agent has been trained."""
        return self._is_trained

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action given observation.

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, optional_state)
        """
        pass

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10,
        progress_bar: bool = True,
    ) -> "BaseAgent":
        """Train the agent.

        Args:
            total_timesteps: Total number of training timesteps
            callback: Optional callback for training monitoring
            log_interval: Logging frequency
            progress_bar: Whether to display progress bar

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save agent to disk.

        Args:
            path: Save path (without extension)
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], env=None) -> "BaseAgent":
        """Load agent from disk.

        Args:
            path: Load path
            env: Optional environment (needed for some agents)

        Returns:
            Loaded agent
        """
        pass

    def evaluate(
        self,
        env=None,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate agent on environment.

        Args:
            env: Evaluation environment (uses training env if None)
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary with evaluation metrics
        """
        eval_env = env or self.env

        episode_rewards = []
        episode_lengths = []
        sla_violations_urllc = []
        sla_violations_mmtc = []

        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            viol_urllc = 0
            viol_mmtc = 0
            done = False

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                episode_reward += reward
                episode_length += 1

                # Count SLA violations
                if info.get("viol_d1", 0) > 0 or info.get("viol_p1", 0) > 0:
                    viol_urllc += 1
                if info.get("viol_d2", 0) > 0 or info.get("viol_p2", 0) > 0:
                    viol_mmtc += 1

                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            sla_violations_urllc.append(viol_urllc)
            sla_violations_mmtc.append(viol_mmtc)

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_viol_urllc": np.mean(sla_violations_urllc),
            "mean_viol_mmtc": np.mean(sla_violations_mmtc),
            "n_episodes": n_episodes,
        }

    def __repr__(self) -> str:
        return f"{self.name}(trained={self.is_trained})"
