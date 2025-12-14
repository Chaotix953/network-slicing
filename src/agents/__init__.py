# -*- coding: utf-8 -*-
"""RL agents for network slicing."""

from .base_agent import BaseAgent
from .ppo import PPOAgent
from .sac import SACAgent
from .td3 import TD3Agent
from .baselines import BaselineFixedAgent, BaselinePriorityAgent

__all__ = [
    "BaseAgent",
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    "BaselineFixedAgent",
    "BaselinePriorityAgent",
]


def create_agent(algorithm: str, env, config=None, **kwargs):
    """Factory function to create agents.

    Args:
        algorithm: Agent type ('ppo', 'sac', 'td3', 'baseline_fixed', 'baseline_priority')
        env: Gymnasium environment
        config: AgentConfig dataclass
        **kwargs: Additional agent parameters

    Returns:
        BaseAgent instance
    """
    agents = {
        "ppo": PPOAgent,
        "sac": SACAgent,
        "td3": TD3Agent,
        "baseline_fixed": BaselineFixedAgent,
        "baseline_priority": BaselinePriorityAgent,
    }

    if algorithm not in agents:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(agents.keys())}")

    return agents[algorithm](env, config=config, **kwargs)
