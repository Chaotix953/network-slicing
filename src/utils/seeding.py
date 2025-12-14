# -*- coding: utf-8 -*-
"""Random seed management for reproducibility."""

import random
from typing import Optional
import numpy as np

_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.

    Sets seed for:
    - Python random module
    - NumPy random
    - PyTorch (if available)

    Args:
        seed: Random seed value
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_seed() -> Optional[int]:
    """Get the current global seed."""
    return _GLOBAL_SEED


def generate_seed() -> int:
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)
