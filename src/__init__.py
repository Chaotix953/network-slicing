# -*- coding: utf-8 -*-
"""
5G Network Slicing with Deep Reinforcement Learning

A DRL-based system for dynamic bandwidth allocation between URLLC and mMTC slices.

Modules:
    config: Configuration management
    envs: Gymnasium environments
    agents: RL and baseline agents
    infrastructure: Network interface layer (NIL)
    traffic: Traffic generation models
    training: Training pipeline and callbacks
    utils: Utility functions
"""

__version__ = "2.0.0"
__author__ = "Richy"

from .config import Config, load_config, save_config
from .envs import SlicingEnv
from .agents import create_agent, PPOAgent, SACAgent, TD3Agent
from .infrastructure import NILInterface, NetworkMetrics
from .training import Trainer, Evaluator

__all__ = [
    # Config
    "Config",
    "load_config",
    "save_config",
    # Environment
    "SlicingEnv",
    # Agents
    "create_agent",
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    # Infrastructure
    "NILInterface",
    "NetworkMetrics",
    # Training
    "Trainer",
    "Evaluator",
]
