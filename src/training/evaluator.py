# -*- coding: utf-8 -*-
"""Evaluation utilities for trained agents."""

from typing import Dict, List, Any, Optional
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for comparing agent performance."""

    def __init__(self, env, n_episodes: int = 10):
        """
        Initialize evaluator.

        Args:
            env: Evaluation environment
            n_episodes: Default number of evaluation episodes
        """
        self.env = env
        self.n_episodes = n_episodes
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        agent,
        agent_name: Optional[str] = None,
        n_episodes: Optional[int] = None,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate a single agent.

        Args:
            agent: Agent to evaluate (must have predict method)
            agent_name: Name for logging
            n_episodes: Number of episodes (uses default if None)
            deterministic: Use deterministic policy
            verbose: Print progress

        Returns:
            Dictionary with evaluation metrics
        """
        n_eps = n_episodes or self.n_episodes
        name = agent_name or getattr(agent, "name", "Agent")

        episode_rewards = []
        episode_lengths = []
        sla_violations_urllc = []
        sla_violations_mmtc = []
        all_infos = []

        if verbose:
            logger.info(f"Evaluating {name} for {n_eps} episodes...")

        for ep in range(n_eps):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            viol_urllc = 0
            viol_mmtc = 0
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)

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
            all_infos.append(info)

        # Compute statistics
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_viol_urllc": np.mean(sla_violations_urllc),
            "mean_viol_mmtc": np.mean(sla_violations_mmtc),
            "total_viol_urllc": np.sum(sla_violations_urllc),
            "total_viol_mmtc": np.sum(sla_violations_mmtc),
            "n_episodes": n_eps,
        }

        # Store results
        self.results[name] = results

        if verbose:
            logger.info(
                f"{name}: reward={results['mean_reward']:.2f} (+/- {results['std_reward']:.2f}), "
                f"viol_urllc={results['mean_viol_urllc']:.1f}, viol_mmtc={results['mean_viol_mmtc']:.1f}"
            )

        return results

    def compare(
        self,
        agents: Dict[str, Any],
        n_episodes: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple agents.

        Args:
            agents: Dictionary mapping agent names to agent objects
            n_episodes: Number of episodes per agent
            deterministic: Use deterministic policies

        Returns:
            Dictionary with results for each agent
        """
        results = {}

        for name, agent in agents.items():
            results[name] = self.evaluate(
                agent,
                agent_name=name,
                n_episodes=n_episodes,
                deterministic=deterministic,
            )

        return results

    def print_comparison(self) -> None:
        """Print formatted comparison table."""
        if not self.results:
            print("No results to display.")
            return

        print("\n" + "=" * 80)
        print("AGENT COMPARISON RESULTS")
        print("=" * 80)
        print(
            f"{'Agent':<25} {'Mean Reward':>12} {'Std':>8} "
            f"{'Viol URLLC':>12} {'Viol mMTC':>12}"
        )
        print("-" * 80)

        for name, res in sorted(
            self.results.items(),
            key=lambda x: x[1]["mean_reward"],
            reverse=True,
        ):
            print(
                f"{name:<25} {res['mean_reward']:>12.2f} {res['std_reward']:>8.2f} "
                f"{res['mean_viol_urllc']:>12.1f} {res['mean_viol_mmtc']:>12.1f}"
            )

        print("=" * 80)

        # Highlight best
        best = max(self.results.items(), key=lambda x: x[1]["mean_reward"])
        print(f"\nBest agent: {best[0]} (reward={best[1]['mean_reward']:.2f})")

    def save_results(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {path}")

    def load_results(self, path: str) -> None:
        """Load results from JSON file."""
        with open(path, "r") as f:
            self.results = json.load(f)
        logger.info(f"Results loaded from {path}")
