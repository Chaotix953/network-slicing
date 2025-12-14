# -*- coding: utf-8 -*-
"""
Unified training pipeline for DRL network slicing.

Provides a single entry point for training, evaluation, and comparison.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import logging
import time

import numpy as np

from ..config import Config, save_config
from ..envs import SlicingEnv
from ..agents import create_agent, BaseAgent
from ..infrastructure import NILInterface, NILConfig
from .callbacks import MetricsCallback, CheckpointCallback, EvalCallback
from .evaluator import Evaluator


logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified trainer for DRL network slicing experiments.

    Features:
    - Configurable training pipeline
    - Automatic logging and checkpointing
    - Support for multiple algorithms
    - Baseline comparison
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            experiment_name: Name for this experiment run
        """
        self.config = config or Config()

        # Set experiment name
        if experiment_name:
            self.config.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.config.experiment_name}_{timestamp}"

        # Setup directories
        self._setup_directories()

        # Setup logging
        self._setup_logging()

        # Initialize environment and agent
        self.env = None
        self.eval_env = None
        self.agent = None
        self.nil = None

        # Training state
        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []

    def _setup_directories(self) -> None:
        """Create output directories."""
        base = Path(self.config.training.log_dir)
        self.log_dir = base / self.run_name
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.tensorboard_dir = self.log_dir / "tensorboard"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_file = self.log_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    def setup_environment(self, simulation_mode: bool = False) -> SlicingEnv:
        """
        Setup training environment.

        Args:
            simulation_mode: If True, use simulated network instead of Ryu

        Returns:
            Configured SlicingEnv
        """
        logger.info("Setting up environment...")

        # Setup NIL if not simulation mode
        if not simulation_mode:
            nil_config = NILConfig(
                ryu_base_url=self.config.network.ryu_base_url,
                link_capacity_mbps=self.config.network.link_capacity_mbps,
                timeout_s=self.config.network.timeout_s,
            )
            self.nil = NILInterface(nil_config)
        else:
            self.nil = None
            logger.info("Running in simulation mode (no Ryu connection)")

        # Create environment
        self.env = SlicingEnv(config=self.config.env, nil=self.nil)
        self.eval_env = SlicingEnv(config=self.config.env, nil=self.nil)

        logger.info(f"Environment ready: obs_space={self.env.observation_space.shape}, "
                   f"action_space={self.env.action_space.shape}")

        return self.env

    def setup_agent(self, algorithm: Optional[str] = None, **kwargs) -> BaseAgent:
        """
        Setup RL agent.

        Args:
            algorithm: Override algorithm from config
            **kwargs: Additional agent parameters

        Returns:
            Configured agent
        """
        if self.env is None:
            raise RuntimeError("Environment not setup. Call setup_environment first.")

        algo = algorithm or self.config.agent.algorithm
        logger.info(f"Setting up {algo} agent...")

        self.agent = create_agent(
            algo,
            self.env,
            config=self.config.agent,
            tensorboard_log=str(self.tensorboard_dir),
            **kwargs,
        )

        logger.info(f"Agent ready: {self.agent}")
        return self.agent

    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_freq: Optional[int] = None,
        save_freq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run training.

        Args:
            total_timesteps: Override training length
            eval_freq: Evaluation frequency
            save_freq: Checkpoint frequency

        Returns:
            Training results dictionary
        """
        if self.agent is None:
            raise RuntimeError("Agent not setup. Call setup_agent first.")

        timesteps = total_timesteps or self.config.training.total_timesteps
        eval_f = eval_freq or self.config.training.eval_freq
        save_f = save_freq or self.config.training.save_freq

        logger.info(f"Starting training for {timesteps} timesteps...")
        logger.info(f"Config: eval_freq={eval_f}, save_freq={save_f}")

        # Save config
        save_config(self.config, str(self.log_dir / "config.yaml"))

        # Setup callbacks
        callbacks = self._create_callbacks(eval_f, save_f)

        # Train
        start_time = time.time()

        self.agent.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=self.config.training.log_interval,
            progress_bar=True,
        )

        training_time = time.time() - start_time
        self.is_trained = True

        # Save final model
        final_path = self.checkpoint_dir / "final_model"
        self.agent.save(str(final_path))
        logger.info(f"Final model saved to {final_path}")

        # Results
        results = {
            "algorithm": self.config.agent.algorithm,
            "total_timesteps": timesteps,
            "training_time_s": training_time,
            "final_model_path": str(final_path),
        }

        self.training_history.append(results)
        self._save_results(results)

        logger.info(f"Training completed in {training_time:.1f}s")
        return results

    def _create_callbacks(self, eval_freq: int, save_freq: int) -> List:
        """Create training callbacks."""
        callbacks = []

        # Metrics logging
        callbacks.append(MetricsCallback(log_freq=100))

        # Checkpointing
        callbacks.append(CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.checkpoint_dir),
            name_prefix="checkpoint",
        ))

        # Evaluation
        if self.eval_env:
            callbacks.append(EvalCallback(
                eval_env=self.eval_env,
                n_eval_episodes=self.config.training.n_eval_episodes,
                eval_freq=eval_freq,
                best_model_save_path=str(self.checkpoint_dir),
            ))

        return callbacks

    def evaluate(
        self,
        n_episodes: Optional[int] = None,
        agents: Optional[Dict[str, BaseAgent]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent(s).

        Args:
            n_episodes: Number of evaluation episodes
            agents: Additional agents to compare (uses trained agent if None)

        Returns:
            Evaluation results
        """
        n_eps = n_episodes or self.config.training.n_eval_episodes
        evaluator = Evaluator(self.eval_env or self.env, n_episodes=n_eps)

        if agents is None:
            if self.agent is None:
                raise RuntimeError("No agent to evaluate")
            agents = {self.config.agent.algorithm: self.agent}

        results = evaluator.compare(agents)
        evaluator.print_comparison()
        evaluator.save_results(str(self.log_dir / "evaluation_results.json"))

        return results

    def compare_with_baselines(
        self,
        n_episodes: int = 10,
        include_trained: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare trained agent with baselines.

        Args:
            n_episodes: Evaluation episodes per agent
            include_trained: Include trained RL agent

        Returns:
            Comparison results
        """
        from ..agents.baselines import BaselineFixedAgent, BaselinePriorityAgent

        agents = {}

        # Add baselines
        agents["Baseline-50-50"] = BaselineFixedAgent(beta_urllc=0.5)
        agents["Baseline-70-30"] = BaselineFixedAgent(beta_urllc=0.7)
        agents["Baseline-Priority"] = BaselinePriorityAgent()

        # Add trained agent
        if include_trained and self.agent is not None:
            agents[self.config.agent.algorithm] = self.agent

        return self.evaluate(n_episodes=n_episodes, agents=agents)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save training results."""
        path = self.log_dir / "training_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def load_agent(self, path: str) -> BaseAgent:
        """
        Load a trained agent.

        Args:
            path: Path to saved model

        Returns:
            Loaded agent
        """
        from ..agents import PPOAgent, SACAgent, TD3Agent

        # Determine algorithm from path or config
        algo = self.config.agent.algorithm

        agent_classes = {
            "ppo": PPOAgent,
            "sac": SACAgent,
            "td3": TD3Agent,
        }

        if algo not in agent_classes:
            raise ValueError(f"Unknown algorithm: {algo}")

        self.agent = agent_classes[algo].load(path, env=self.env)
        self.is_trained = True
        logger.info(f"Loaded agent from {path}")

        return self.agent

    def close(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()
        if self.eval_env:
            self.eval_env.close()
