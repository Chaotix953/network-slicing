# -*- coding: utf-8 -*-
"""Gymnasium environments for network slicing."""

from .slicing_env import SlicingEnv
from .wrappers import NormalizeObservation, ClipAction

__all__ = ["SlicingEnv", "NormalizeObservation", "ClipAction"]


def make_env(config=None, nil=None, **kwargs):
    """Factory function to create SlicingEnv with optional wrappers.

    Args:
        config: EnvConfig dataclass
        nil: NILInterface instance
        **kwargs: Additional environment parameters

    Returns:
        Configured SlicingEnv instance
    """
    return SlicingEnv(config=config, nil=nil, **kwargs)
