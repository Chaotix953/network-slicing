# -*- coding: utf-8 -*-
"""Priority-based heuristic baseline agent."""

from pathlib import Path
from typing import Tuple, Optional, Any, Union
import json
import numpy as np

from ..base_agent import BaseAgent


class BaselinePriorityAgent(BaseAgent):
    """Baseline agent with URLLC priority heuristic.

    Dynamically adjusts allocation based on observed URLLC latency:
    - High latency: Increase URLLC allocation
    - Low latency: Decrease to give mMTC more bandwidth

    This is a simple control-theoretic approach without learning.
    """

    def __init__(
        self,
        env=None,
        config=None,
        latency_threshold: float = 2.0,
        min_urllc: float = 0.3,
        max_urllc: float = 0.9,
        step_up: float = 0.1,
        step_down: float = 0.05,
        **kwargs,
    ):
        """Initialize priority baseline.

        Args:
            env: Gymnasium environment (optional)
            config: AgentConfig with baseline_latency_threshold
            latency_threshold: Latency threshold (ms) for increasing URLLC allocation
            min_urllc: Minimum URLLC allocation
            max_urllc: Maximum URLLC allocation
            step_up: Allocation increase step
            step_down: Allocation decrease step
        """
        super().__init__(env, config)

        # Get params from config or arguments
        if config is not None and hasattr(config, "baseline_latency_threshold"):
            self._threshold = float(config.baseline_latency_threshold)
        else:
            self._threshold = float(latency_threshold)

        self._min_urllc = float(min_urllc)
        self._max_urllc = float(max_urllc)
        self._step_up = float(step_up)
        self._step_down = float(step_down)

        # Internal state
        self._current_beta = 0.5
        self._is_trained = True  # No training needed

    @property
    def name(self) -> str:
        return "BaselinePriority"

    @property
    def beta_urllc(self) -> float:
        """Current URLLC allocation."""
        return self._current_beta

    def reset(self) -> None:
        """Reset agent state."""
        self._current_beta = 0.5

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action based on URLLC latency heuristic.

        Observation format: [d1_norm, p1_norm, d2_norm, p2_norm, ...]
        d1_norm is normalized URLLC latency (d1 / d1_max)
        """
        # Extract normalized URLLC latency from observation
        d1_norm = observation[0] if len(observation) > 0 else 0.0

        # Denormalize: assume d1_max = 1.0ms (SLA), so actual latency = d1_norm * d1_max
        # But observation can exceed 1.0 if SLA is violated
        # For decision: d1_norm > 1.0 means SLA violation
        latency_ratio = float(d1_norm)

        # Heuristic control logic
        if latency_ratio > 1.0:  # SLA violated
            # Urgently increase URLLC allocation
            self._current_beta = min(self._current_beta + self._step_up * 2, self._max_urllc)
        elif latency_ratio > 0.8:  # Approaching SLA limit
            # Increase URLLC allocation
            self._current_beta = min(self._current_beta + self._step_up, self._max_urllc)
        elif latency_ratio < 0.3:  # URLLC has plenty of headroom
            # Decrease to give mMTC more bandwidth
            self._current_beta = max(self._current_beta - self._step_down, self._min_urllc)
        # Otherwise: maintain current allocation

        action = np.array([self._current_beta], dtype=np.float32)
        return action, None

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10,
        progress_bar: bool = True,
    ) -> "BaselinePriorityAgent":
        """No-op for baseline (no learning)."""
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save agent configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "threshold": self._threshold,
            "min_urllc": self._min_urllc,
            "max_urllc": self._max_urllc,
            "step_up": self._step_up,
            "step_down": self._step_down,
            "current_beta": self._current_beta,
        }
        with open(f"{path}.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path], env=None) -> "BaselinePriorityAgent":
        """Load agent configuration."""
        with open(f"{path}.json", "r") as f:
            data = json.load(f)
        agent = cls(
            env=env,
            latency_threshold=data["threshold"],
            min_urllc=data["min_urllc"],
            max_urllc=data["max_urllc"],
            step_up=data["step_up"],
            step_down=data["step_down"],
        )
        agent._current_beta = data["current_beta"]
        return agent
