# -*- coding: utf-8 -*-
"""Fixed allocation baseline agent."""

from pathlib import Path
from typing import Tuple, Optional, Any, Union
import json
import numpy as np

from ..base_agent import BaseAgent


class BaselineFixedAgent(BaseAgent):
    """Baseline agent with fixed bandwidth allocation.

    Always allocates a constant fraction to URLLC.
    Useful as a control baseline for RL experiments.
    """

    def __init__(self, env=None, config=None, beta_urllc: float = 0.5, **kwargs):
        """Initialize fixed allocation baseline.

        Args:
            env: Gymnasium environment (optional for baselines)
            config: AgentConfig with baseline_beta field
            beta_urllc: Fixed allocation for URLLC slice [0, 1]
        """
        super().__init__(env, config)

        # Get beta from config or parameter
        if config is not None and hasattr(config, "baseline_beta"):
            self._beta = float(config.baseline_beta)
        else:
            self._beta = float(beta_urllc)

        self._beta = np.clip(self._beta, 0.0, 1.0)
        self._is_trained = True  # No training needed

    @property
    def name(self) -> str:
        return f"BaselineFixed({self._beta:.2f})"

    @property
    def beta_urllc(self) -> float:
        """Current URLLC allocation."""
        return self._beta

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Return fixed allocation regardless of observation."""
        action = np.array([self._beta], dtype=np.float32)
        return action, None

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10,
        progress_bar: bool = True,
    ) -> "BaselineFixedAgent":
        """No-op for baseline (no learning)."""
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save agent configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{path}.json", "w") as f:
            json.dump({"beta_urllc": self._beta}, f)

    @classmethod
    def load(cls, path: Union[str, Path], env=None) -> "BaselineFixedAgent":
        """Load agent configuration."""
        with open(f"{path}.json", "r") as f:
            data = json.load(f)
        return cls(env=env, beta_urllc=data["beta_urllc"])
