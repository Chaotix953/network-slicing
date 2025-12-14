# -*- coding: utf-8 -*-
"""
Network Interface Layer (NIL) - Optimized implementation.

Handles communication with Ryu SDN controller and metrics estimation.
Uses M/M/1 queueing model for latency estimation when real measurements unavailable.
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import requests

from .metrics import NetworkMetrics
from .queueing_models import estimate_latency_mm1, estimate_loss, compute_effective_capacity


logger = logging.getLogger(__name__)


@dataclass
class NILConfig:
    """NIL configuration parameters."""
    ryu_base_url: str = "http://localhost:8080"
    link_capacity_mbps: float = 100.0
    timeout_s: float = 2.0
    retry_count: int = 3
    retry_delay_s: float = 0.5

    # Packet sizes for M/M/1 estimation
    urllc_packet_size: int = 200
    mmtc_packet_size: int = 50

    # Congestion threshold for loss estimation
    congestion_threshold: float = 0.85

    # M/M/1 parameters
    max_latency_ms: float = 999.0
    base_latency_ms: float = 0.0


class NILInterface:
    """
    Network Interface Layer for Ryu SDN controller communication.

    Features:
    - REST API communication with Ryu controller
    - M/M/1 latency estimation when real metrics unavailable
    - Retry logic for network failures
    - Metrics caching to reduce API calls
    """

    def __init__(self, config: Optional[NILConfig] = None, **kwargs):
        """
        Initialize NIL interface.

        Args:
            config: NILConfig dataclass
            **kwargs: Override individual config parameters
        """
        if config is None:
            config = NILConfig()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._config = config
        self._base_url = config.ryu_base_url.rstrip("/")

        # State
        self._current_beta: float = 0.5
        self._last_metrics: Optional[NetworkMetrics] = None
        self._last_metrics_time: float = 0.0
        self._connected: bool = False

        # Try initial connection
        self._check_connection()

    def _check_connection(self) -> bool:
        """Check connection to Ryu controller."""
        try:
            response = self._request("GET", "/slicing/status")
            if response:
                self._connected = True
                logger.info(f"Connected to Ryu controller at {self._base_url}")
                return True
        except Exception as e:
            logger.warning(f"Cannot connect to Ryu: {e}")

        self._connected = False
        return False

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Make HTTP request to Ryu controller with retry logic.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            data: Request data for POST

        Returns:
            Response JSON or None on failure
        """
        url = f"{self._base_url}{endpoint}"

        for attempt in range(self._config.retry_count):
            try:
                if method == "GET":
                    response = requests.get(url, timeout=self._config.timeout_s)
                elif method == "POST":
                    response = requests.post(
                        url,
                        json=data,
                        timeout=self._config.timeout_s,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self._config.retry_count - 1:
                    time.sleep(self._config.retry_delay_s)

        return None

    @property
    def is_connected(self) -> bool:
        """Check if connected to Ryu."""
        return self._connected

    @property
    def current_allocation(self) -> float:
        """Current URLLC bandwidth allocation."""
        return self._current_beta

    def apply_action(self, beta_urllc: float) -> bool:
        """
        Apply bandwidth allocation to network.

        Args:
            beta_urllc: URLLC bandwidth fraction [0, 1]

        Returns:
            True if successful
        """
        beta_urllc = float(max(0.0, min(1.0, beta_urllc)))
        beta_mmtc = 1.0 - beta_urllc

        # Calculate actual bandwidth allocations
        urllc_mbps = beta_urllc * self._config.link_capacity_mbps
        mmtc_mbps = beta_mmtc * self._config.link_capacity_mbps

        # Send to Ryu
        data = {
            "URLLC": {
                "bandwidth_percent": beta_urllc * 100,
                "max_rate_mbps": urllc_mbps,
            },
            "mMTC": {
                "bandwidth_percent": beta_mmtc * 100,
                "max_rate_mbps": mmtc_mbps,
            },
        }

        response = self._request("POST", "/slicing/qos", data)

        if response:
            self._current_beta = beta_urllc
            logger.debug(f"Applied allocation: URLLC={beta_urllc:.2%}, mMTC={beta_mmtc:.2%}")
            return True

        logger.warning("Failed to apply allocation")
        return False

    def get_metrics(self) -> NetworkMetrics:
        """
        Get current network metrics.

        Fetches real metrics from Ryu if available, otherwise estimates using M/M/1.

        Returns:
            NetworkMetrics dataclass
        """
        # Try to get real metrics from Ryu
        raw_metrics = self._fetch_raw_metrics()

        if raw_metrics:
            return self._process_metrics(raw_metrics)
        else:
            return self._estimate_metrics()

    def _fetch_raw_metrics(self) -> Optional[Dict]:
        """Fetch raw metrics from Ryu API."""
        return self._request("GET", "/slicing/metrics")

    def _process_metrics(self, raw: Dict) -> NetworkMetrics:
        """
        Process raw metrics from Ryu and compute latency/loss.

        Args:
            raw: Raw metrics dictionary from Ryu

        Returns:
            Processed NetworkMetrics
        """
        slices = raw.get("slices", {})
        urllc = slices.get("URLLC", {})
        mmtc = slices.get("mMTC", {})

        # Get throughput (measured)
        thr_urllc = urllc.get("throughput_mbps", 0.0)
        thr_mmtc = mmtc.get("throughput_mbps", 0.0)

        # Compute effective capacities
        cap_urllc, cap_mmtc = compute_effective_capacity(
            self._config.link_capacity_mbps,
            self._current_beta,
        )

        # Estimate latency using M/M/1
        delay_urllc, rho_urllc = estimate_latency_mm1(
            arrival_rate_mbps=thr_urllc,
            service_rate_mbps=cap_urllc,
            packet_size_bytes=self._config.urllc_packet_size,
            max_latency_ms=self._config.max_latency_ms,
            base_latency_ms=self._config.base_latency_ms,
        )

        delay_mmtc, rho_mmtc = estimate_latency_mm1(
            arrival_rate_mbps=thr_mmtc,
            service_rate_mbps=cap_mmtc,
            packet_size_bytes=self._config.mmtc_packet_size,
            max_latency_ms=self._config.max_latency_ms,
            base_latency_ms=self._config.base_latency_ms,
        )

        # Estimate packet loss
        loss_urllc = estimate_loss(rho_urllc, self._config.congestion_threshold)
        loss_mmtc = estimate_loss(rho_mmtc, self._config.congestion_threshold)

        # Queue size estimation (from byte count)
        queue_urllc = urllc.get("byte_count", 0) / 1024  # KB
        queue_mmtc = mmtc.get("byte_count", 0) / 1024

        metrics = NetworkMetrics(
            delay_urllc=delay_urllc,
            loss_urllc=loss_urllc,
            throughput_urllc=thr_urllc,
            queue_urllc=queue_urllc,
            delay_mmtc=delay_mmtc,
            loss_mmtc=loss_mmtc,
            throughput_mmtc=thr_mmtc,
            queue_mmtc=queue_mmtc,
        )

        self._last_metrics = metrics
        self._last_metrics_time = time.time()

        return metrics

    def _estimate_metrics(self) -> NetworkMetrics:
        """
        Estimate metrics when Ryu is unavailable.

        Uses previous metrics or returns defaults.
        """
        if self._last_metrics is not None:
            # Return cached metrics with warning
            logger.warning("Using cached metrics (Ryu unavailable)")
            return self._last_metrics

        # Return default metrics
        logger.warning("Returning default metrics (no data available)")
        return NetworkMetrics(
            delay_urllc=0.1,
            loss_urllc=0.0,
            throughput_urllc=0.0,
            queue_urllc=0.0,
            delay_mmtc=1.0,
            loss_mmtc=0.0,
            throughput_mmtc=0.0,
            queue_mmtc=0.0,
        )

    def reset(self) -> None:
        """Reset NIL state to initial allocation."""
        self._current_beta = 0.5
        self.apply_action(0.5)
        self._last_metrics = None

    def get_status(self) -> Dict[str, Any]:
        """Get NIL status information."""
        return {
            "connected": self._connected,
            "base_url": self._base_url,
            "current_beta": self._current_beta,
            "last_metrics_age_s": time.time() - self._last_metrics_time if self._last_metrics else None,
        }
