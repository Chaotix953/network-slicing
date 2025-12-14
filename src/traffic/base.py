# -*- coding: utf-8 -*-
"""Base class for traffic generators."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseTrafficGenerator(ABC):
    """Abstract base class for traffic generation models."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize traffic generator.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def sample_arrivals(self, dt: float) -> int:
        """Sample number of packet arrivals in time interval.

        Args:
            dt: Time interval duration (seconds)

        Returns:
            Number of packets arriving in the interval
        """
        pass

    @abstractmethod
    def get_rate(self) -> float:
        """Get current arrival rate (packets/second)."""
        pass

    def reset(self) -> None:
        """Reset generator state."""
        pass

    def set_seed(self, seed: int) -> None:
        """Set random seed."""
        self._rng = np.random.default_rng(seed)
