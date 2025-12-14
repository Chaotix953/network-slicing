#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for 5G Network Slicing DRL.

Usage:
    python scripts/train.py --algorithm ppo --timesteps 100000
    python scripts/train.py --algorithm sac --simulation
    python scripts/train.py --compare-baselines
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # <--- CHANGED: Removed / "src"

# Imports must now include 'src.' prefix
from src.config import Config, load_config
from src.training import Trainer
from src.utils import set_seed, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DRL agents for 5G network slicing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Algorithm
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["ppo", "sac", "td3"],
        default="ppo",
        help="RL algorithm to train",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency (timesteps)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    # Environment
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode (no Ryu connection)",
    )
    parser.add_argument(
        "--ryu-url",
        type=str,
        default="http://localhost:8080",
        help="Ryu controller URL",
    )

    # Experiment
    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        default="experiment",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )

    # Actions
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Compare trained agent with baselines after training",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to model to load for evaluation",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging()

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Override config with CLI arguments
    config.experiment_name = args.experiment_name
    config.agent.algorithm = args.algorithm
    config.training.total_timesteps = args.timesteps
    config.training.eval_freq = args.eval_freq
    config.training.n_eval_episodes = args.eval_episodes
    config.network.ryu_base_url = args.ryu_url

    if args.seed:
        config.training.seed = args.seed

    print("=" * 70)
    print("5G NETWORK SLICING - DEEP REINFORCEMENT LEARNING")
    print("=" * 70)
    print(f"Algorithm: {config.agent.algorithm.upper()}")
    print(f"Timesteps: {config.training.total_timesteps:,}")
    print(f"Simulation mode: {args.simulation}")
    print("=" * 70)

    # Create trainer
    trainer = Trainer(config=config, experiment_name=args.experiment_name)

    # Setup environment
    trainer.setup_environment(simulation_mode=args.simulation)

    if args.eval_only:
        # Evaluation only mode
        if args.load_model:
            trainer.load_agent(args.load_model)
        else:
            print("Error: --load-model required for --eval-only mode")
            sys.exit(1)
    else:
        # Setup and train agent
        trainer.setup_agent()
        trainer.train()

    # Compare with baselines if requested
    if args.compare_baselines:
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        trainer.compare_with_baselines(n_episodes=args.eval_episodes)

    # Cleanup
    trainer.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {trainer.log_dir}")


if __name__ == "__main__":
    main()
