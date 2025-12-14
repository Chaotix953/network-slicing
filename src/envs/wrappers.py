# -*- coding: utf-8 -*-
"""Environment wrappers for preprocessing and normalization."""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to zero mean and unit variance."""

    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        self._update_stats(obs)
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean and variance."""
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.count


class ClipAction(gym.ActionWrapper):
    """Clip actions to action space bounds."""

    def __init__(self, env):
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid range."""
        return np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )


class RewardScaling(gym.RewardWrapper):
    """Scale rewards by a constant factor."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        """Scale reward."""
        return reward * self.scale


class EpisodeMonitor(gym.Wrapper):
    """Monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def reset(self, **kwargs):
        if self._current_length > 0:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
        self._current_reward = 0.0
        self._current_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_reward += reward
        self._current_length += 1

        info["episode_reward"] = self._current_reward
        info["episode_length"] = self._current_length

        if terminated or truncated:
            info["episode"] = {
                "r": self._current_reward,
                "l": self._current_length,
            }

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self):
        """Get episode statistics."""
        if not self.episode_rewards:
            return {}
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "n_episodes": len(self.episode_rewards),
        }
