# -*- coding: utf-8 -*-
"""Plotting utilities for DRL training visualization."""

from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_training_curves(
    rewards: List[float],
    lengths: Optional[List[int]] = None,
    window: int = 100,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training reward curves with smoothing.

    Args:
        rewards: List of episode rewards
        lengths: Optional list of episode lengths
        window: Smoothing window size
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(2 if lengths else 1, 1, figsize=(10, 8))

    if not lengths:
        axes = [axes]

    # Rewards
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, label="Raw")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(rewards)), smoothed, label=f"Smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{title} - Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lengths
    if lengths:
        ax = axes[1]
        ax.plot(lengths, alpha=0.3, label="Raw")
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(lengths)), smoothed, label=f"Smoothed (w={window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length")
        ax.set_title(f"{title} - Episode Lengths")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "mean_reward",
    title: str = "Agent Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot bar comparison of agents.

    Args:
        results: Dictionary of agent results
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        return

    agents = list(results.keys())
    values = [results[a].get(metric, 0) for a in agents]

    # Get standard deviation if available
    stds = [results[a].get(f"std_{metric.replace('mean_', '')}", 0) for a in agents]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(agents, values, yerr=stds, capsize=5, alpha=0.8)

    # Color best bar
    best_idx = np.argmax(values)
    bars[best_idx].set_color("green")

    ax.set_xlabel("Agent")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate labels if many agents
    if len(agents) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_sla_violations(
    results: Dict[str, Dict[str, float]],
    title: str = "SLA Violations Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot SLA violations comparison.

    Args:
        results: Dictionary of agent results
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        return

    agents = list(results.keys())
    viol_urllc = [results[a].get("mean_viol_urllc", 0) for a in agents]
    viol_mmtc = [results[a].get("mean_viol_mmtc", 0) for a in agents]

    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, viol_urllc, width, label="URLLC Violations", color="red", alpha=0.7)
    ax.bar(x + width/2, viol_mmtc, width, label="mMTC Violations", color="blue", alpha=0.7)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Violations per Episode")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if len(agents) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_allocation_trajectory(
    allocations: List[float],
    rewards: Optional[List[float]] = None,
    title: str = "Bandwidth Allocation Trajectory",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot bandwidth allocation over time.

    Args:
        allocations: List of beta_urllc values
        rewards: Optional reward values
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Allocation
    ax1.plot(allocations, "b-", label="URLLC Allocation (beta)", linewidth=1.5)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50-50 baseline")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("URLLC Allocation (beta)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Rewards on secondary axis
    if rewards:
        ax2 = ax1.twinx()
        ax2.plot(rewards, "r-", alpha=0.5, label="Reward", linewidth=1)
        ax2.set_ylabel("Reward", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.legend(loc="upper right")

    ax1.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
