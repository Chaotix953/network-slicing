# -*- coding: utf-8 -*-
"""Network metrics data structures."""

from dataclasses import dataclass, field
from typing import Dict, Any
import time


@dataclass
class SliceMetrics:
    """Metrics for a single network slice."""
    delay_ms: float = 0.0
    loss_rate: float = 0.0
    throughput_mbps: float = 0.0
    packet_count: int = 0
    byte_count: int = 0
    queue_size_kb: float = 0.0
    utilization: float = 0.0


@dataclass
class NetworkMetrics:
    """Combined metrics for URLLC and mMTC slices."""
    # URLLC metrics
    delay_urllc: float = 0.0      # Latency (ms)
    loss_urllc: float = 0.0       # Packet loss rate [0, 1]
    throughput_urllc: float = 0.0  # Throughput (Mbps)
    queue_urllc: float = 0.0      # Queue size (KB)

    # mMTC metrics
    delay_mmtc: float = 0.0
    loss_mmtc: float = 0.0
    throughput_mmtc: float = 0.0
    queue_mmtc: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urllc": {
                "delay_ms": self.delay_urllc,
                "loss_rate": self.loss_urllc,
                "throughput_mbps": self.throughput_urllc,
                "queue_kb": self.queue_urllc,
            },
            "mmtc": {
                "delay_ms": self.delay_mmtc,
                "loss_rate": self.loss_mmtc,
                "throughput_mbps": self.throughput_mmtc,
                "queue_kb": self.queue_mmtc,
            },
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkMetrics":
        """Create from dictionary."""
        urllc = data.get("urllc", {})
        mmtc = data.get("mmtc", {})
        return cls(
            delay_urllc=urllc.get("delay_ms", 0.0),
            loss_urllc=urllc.get("loss_rate", 0.0),
            throughput_urllc=urllc.get("throughput_mbps", 0.0),
            queue_urllc=urllc.get("queue_kb", 0.0),
            delay_mmtc=mmtc.get("delay_ms", 0.0),
            loss_mmtc=mmtc.get("loss_rate", 0.0),
            throughput_mmtc=mmtc.get("throughput_mbps", 0.0),
            queue_mmtc=mmtc.get("queue_kb", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )
