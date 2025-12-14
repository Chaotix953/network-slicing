# -*- coding: utf-8 -*-
"""Training callbacks for monitoring and checkpointing."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time
import logging

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

    class BaseCallback:
        """Fallback base class when SB3 not available."""
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
            self.num_timesteps = 0

        def _on_step(self) -> bool:
            return True


logger = logging.getLogger(__name__)


class MetricsCallback(BaseCallback):
    """Callback to log training metrics."""

    def __init__(
        self,
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        # Track episode rewards
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])

        # Log periodically
        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            mean_reward = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
            logger.info(
                f"Step {self.num_timesteps}: "
                f"mean_reward={mean_reward:.2f}, "
                f"episodes={len(self.episode_rewards)}"
            )

        return True


class CheckpointCallback(BaseCallback):
    """Callback to save model checkpoints periodically."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Save checkpoint at specified frequency."""
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose > 0:
                logger.info(f"Checkpoint saved: {path}")
        return True


class EvalCallback(BaseCallback):
    """Callback for periodic evaluation during training."""

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        best_model_save_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.best_mean_reward = float("-inf")
        self.eval_results: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        """Run evaluation at specified frequency."""
        if self.n_calls % self.eval_freq == 0:
            results = self._evaluate()
            self.eval_results.append({
                "timestep": self.num_timesteps,
                **results,
            })

            mean_reward = results["mean_reward"]

            if self.verbose > 0:
                logger.info(
                    f"Eval at step {self.num_timesteps}: "
                    f"mean_reward={mean_reward:.2f} (+/- {results['std_reward']:.2f})"
                )

            # Save best model
            if mean_reward > self.best_mean_reward and self.best_model_save_path:
                self.best_mean_reward = mean_reward
                path = self.best_model_save_path / "best_model"
                self.model.save(str(path))
                logger.info(f"New best model saved: {mean_reward:.2f}")

        return True

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation episodes."""
        episode_rewards = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "mean_reward": sum(episode_rewards) / len(episode_rewards),
            "std_reward": (sum((r - sum(episode_rewards)/len(episode_rewards))**2 for r in episode_rewards) / len(episode_rewards)) ** 0.5,
            "mean_length": sum(episode_lengths) / len(episode_lengths),
        }


class EarlyStoppingCallback(BaseCallback):
    """Callback to stop training when no improvement."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float("-inf")
        self.no_improvement_count = 0

    def _on_step(self) -> bool:
        """Check for improvement."""
        # Get current reward from logger
        if hasattr(self, "logger") and self.logger:
            # This would need integration with reward tracking
            pass
        return True
