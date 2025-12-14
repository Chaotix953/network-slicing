# -*- coding: utf-8 -*-
"""
Queueing theory models for latency and loss estimation.

Reference: Kleinrock, L. (1975). Queueing Systems, Volume 1: Theory
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class MM1Queue:
    """
    M/M/1 queue model for single-server queueing analysis.

    Assumptions:
    - Poisson arrivals (exponential inter-arrival times)
    - Exponential service times
    - Single server (FCFS)
    - Infinite queue capacity

    Parameters:
        lambda_rate: Arrival rate (packets/s or Mbps)
        mu_rate: Service rate (packets/s or Mbps)
    """
    lambda_rate: float  # Arrival rate
    mu_rate: float      # Service rate

    @property
    def utilization(self) -> float:
        """Server utilization rho = lambda/mu."""
        if self.mu_rate <= 0:
            return float("inf")
        return self.lambda_rate / self.mu_rate

    @property
    def is_stable(self) -> bool:
        """Queue is stable if rho < 1."""
        return self.utilization < 1.0

    def mean_delay(self) -> float:
        """
        Mean time in system (waiting + service).

        W = 1 / (mu - lambda) for stable queue
        """
        if not self.is_stable:
            return float("inf")
        return 1.0 / (self.mu_rate - self.lambda_rate)

    def mean_queue_length(self) -> float:
        """
        Mean number of packets in system.

        L = lambda * W = rho / (1 - rho)
        """
        rho = self.utilization
        if rho >= 1.0:
            return float("inf")
        return rho / (1.0 - rho)

    def mean_waiting_time(self) -> float:
        """
        Mean time waiting in queue (excluding service).

        Wq = W - 1/mu = rho / (mu - lambda)
        """
        if not self.is_stable:
            return float("inf")
        rho = self.utilization
        return rho / (self.mu_rate - self.lambda_rate)


def estimate_latency_mm1(
    arrival_rate_mbps: float,
    service_rate_mbps: float,
    packet_size_bytes: float = 200,
    max_latency_ms: float = 999.0,
    base_latency_ms: float = 0.0,
) -> Tuple[float, float]:
    """
    Estimate latency using M/M/1 model.

    Args:
        arrival_rate_mbps: Traffic arrival rate (Mbps)
        service_rate_mbps: Link/queue service rate (Mbps)
        packet_size_bytes: Average packet size
        max_latency_ms: Cap for unstable queues
        base_latency_ms: Fixed latency component (propagation, processing)

    Returns:
        Tuple of (latency_ms, utilization)
    """
    # Convert to packets/s
    if packet_size_bytes <= 0:
        return max_latency_ms, 1.0

    lambda_pkt = (arrival_rate_mbps * 1e6) / (packet_size_bytes * 8)
    mu_pkt = (service_rate_mbps * 1e6) / (packet_size_bytes * 8)

    # Create queue model
    queue = MM1Queue(lambda_rate=lambda_pkt, mu_rate=mu_pkt)

    if not queue.is_stable:
        return max_latency_ms, min(queue.utilization, 2.0)

    # M/M/1 delay in seconds
    delay_s = queue.mean_delay()

    # Convert to ms and add base latency
    latency_ms = delay_s * 1000 + base_latency_ms

    return min(latency_ms, max_latency_ms), queue.utilization


def estimate_loss(
    utilization: float,
    threshold_low: float = 0.85,
    threshold_high: float = 0.95,
    max_loss: float = 0.5,
) -> float:
    """
    Estimate packet loss rate based on utilization.

    Simple threshold-based model:
    - Below threshold_low: No loss
    - Between low and high: Linear ramp
    - Above high: Increased loss rate
    - Above 1.0: Severe loss

    Args:
        utilization: Link utilization (rho = lambda/mu)
        threshold_low: Start of loss zone
        threshold_high: High congestion threshold
        max_loss: Maximum loss rate

    Returns:
        Estimated packet loss rate [0, 1]
    """
    if utilization < threshold_low:
        return 0.0
    elif utilization < threshold_high:
        # Linear ramp from 0 to 1%
        frac = (utilization - threshold_low) / (threshold_high - threshold_low)
        return frac * 0.01
    elif utilization < 1.0:
        # Ramp from 1% to 10%
        frac = (utilization - threshold_high) / (1.0 - threshold_high)
        return 0.01 + frac * 0.09
    else:
        # Over-saturated: severe loss
        excess = utilization - 1.0
        return min(0.1 + excess * 0.4, max_loss)


def compute_effective_capacity(
    total_capacity_mbps: float,
    beta_urllc: float,
    overhead_factor: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute effective capacities for URLLC and mMTC slices.

    Args:
        total_capacity_mbps: Total link capacity
        beta_urllc: URLLC bandwidth fraction [0, 1]
        overhead_factor: Account for protocol overhead

    Returns:
        Tuple of (urllc_capacity_mbps, mmtc_capacity_mbps)
    """
    effective_total = total_capacity_mbps * overhead_factor
    urllc_cap = beta_urllc * effective_total
    mmtc_cap = (1.0 - beta_urllc) * effective_total
    return urllc_cap, mmtc_cap
