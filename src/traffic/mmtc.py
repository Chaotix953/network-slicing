# -*- coding: utf-8 -*-
"""mMTC traffic generator using ON/OFF Markov chain with Poisson arrivals."""

from typing import Optional
import numpy as np

from .base import BaseTrafficGenerator


class MMTCTraffic(BaseTrafficGenerator):
    """
    mMTC traffic generator using ON/OFF Markov-modulated Poisson process.

    mMTC (massive Machine-Type Communications) traffic is typically:
    - Bursty (ON/OFF pattern from many devices)
    - Small packets (sensor readings)
    - Tolerant to delay (1-1000ms acceptable)

    Model:
    - Two states: ON (high rate) and OFF (low rate)
    - State transitions follow Markov chain
    - Given state, arrivals are Poisson
    """

    def __init__(
        self,
        lambda_on: float = 5000.0,
        lambda_off: float = 100.0,
        p_on: float = 0.1,
        p_off: float = 0.3,
        packet_size_bytes: int = 50,
        seed: Optional[int] = None,
    ):
        """Initialize mMTC traffic generator.

        Args:
            lambda_on: Arrival rate in ON state (packets/s)
            lambda_off: Arrival rate in OFF state (packets/s)
            p_on: Transition probability OFF -> ON
            p_off: Transition probability ON -> OFF
            packet_size_bytes: Packet size in bytes
            seed: Random seed
        """
        super().__init__(seed)
        self._lambda_on = float(lambda_on)
        self._lambda_off = float(lambda_off)
        self._p_on = float(p_on)
        self._p_off = float(p_off)
        self._packet_size = int(packet_size_bytes)

        # State: True = ON, False = OFF
        self._is_on = False

    @property
    def lambda_on(self) -> float:
        """Arrival rate in ON state."""
        return self._lambda_on

    @property
    def lambda_off(self) -> float:
        """Arrival rate in OFF state."""
        return self._lambda_off

    @property
    def packet_size(self) -> int:
        """Packet size (bytes)."""
        return self._packet_size

    @property
    def is_on(self) -> bool:
        """Current state (ON or OFF)."""
        return self._is_on

    def sample_arrivals(self, dt: float) -> int:
        """Sample arrivals with state transition."""
        # First, transition state
        self._update_state()

        # Then sample arrivals based on current state
        current_lambda = self._lambda_on if self._is_on else self._lambda_off
        expected = current_lambda * dt
        return int(self._rng.poisson(expected))

    def _update_state(self) -> None:
        """Update ON/OFF state based on transition probabilities."""
        if self._is_on:
            # ON -> OFF with probability p_off
            if self._rng.random() < self._p_off:
                self._is_on = False
        else:
            # OFF -> ON with probability p_on
            if self._rng.random() < self._p_on:
                self._is_on = True

    def get_rate(self) -> float:
        """Get current arrival rate based on state."""
        return self._lambda_on if self._is_on else self._lambda_off

    def get_expected_rate(self) -> float:
        """Get long-term average arrival rate.

        Based on stationary distribution of Markov chain:
        pi_on = p_on / (p_on + p_off)
        """
        pi_on = self._p_on / (self._p_on + self._p_off)
        return pi_on * self._lambda_on + (1 - pi_on) * self._lambda_off

    def get_bitrate_mbps(self) -> float:
        """Get current bitrate in Mbps."""
        return (self.get_rate() * self._packet_size * 8) / 1e6

    def reset(self) -> None:
        """Reset to OFF state."""
        self._is_on = False

    def set_state(self, is_on: bool) -> None:
        """Force state transition."""
        self._is_on = is_on
