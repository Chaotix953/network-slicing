# -*- coding: utf-8 -*-
"""Traffic generation models for network slicing simulation."""

from .base import BaseTrafficGenerator
from .urllc import URLLCTraffic
from .mmtc import MMTCTraffic

__all__ = ["BaseTrafficGenerator", "URLLCTraffic", "MMTCTraffic"]
