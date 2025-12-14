# -*- coding: utf-8 -*-
"""URLLC traffic generator using homogeneous Poisson process."""

from typing import Optional
import numpy as np

from .base import BaseTrafficGenerator


class URLLCTraffic(BaseTrafficGenerator):
    """
    URLLC traffic generator based on homogeneous Poisson process.

    URLLC (Ultra-Reliable Low-Latency Communications) traffic is typically:
    - Small packets (control signals, sensor data)
    - Constant/predictable arrival rate
    - Strict latency requirements (< 1ms)

    Model: Poisson(lambda * dt) arrivals per timestep
    """

    def __init__(
        self,
        lambda_rate: float = 2000.0,
        packet_size_bytes: int = 200,
        seed: Optional[int] = None,
    ):
        """Initialize URLLC traffic generator.

        Args:
            lambda_rate: Mean arrival rate (packets/second)
            packet_size_bytes: Packet size in bytes
            seed: Random seed
        """
        super().__init__(seed)
        self._lambda = float(lambda_rate)
        self._packet_size = int(packet_size_bytes)

    @property
    def lambda_rate(self) -> float:
        """Mean arrival rate (packets/s)."""
        return self._lambda

    @property
    def packet_size(self) -> int:
        """Packet size (bytes)."""
        return self._packet_size

    def sample_arrivals(self, dt: float) -> int:
        """Sample Poisson arrivals in time interval dt."""
        expected = self._lambda * dt
        return int(self._rng.poisson(expected))

    def get_rate(self) -> float:
        """Get current arrival rate."""
        return self._lambda

    def get_bitrate_mbps(self) -> float:
        """Get expected bitrate in Mbps."""
        return (self._lambda * self._packet_size * 8) / 1e6

    def set_rate(self, lambda_rate: float) -> None:
        """Update arrival rate."""
        self._lambda = float(lambda_rate)
