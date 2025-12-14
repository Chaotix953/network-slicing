# -*- coding: utf-8 -*-
"""Multi-layer perceptron architectures."""

from typing import List, Optional, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = [64, 64],
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> "MLP":
    """
    Factory function to create MLP network.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Hidden layer activation ('relu', 'tanh', 'elu')
        output_activation: Output activation (None for linear)

    Returns:
        MLP network
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural networks")

    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
    )


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """Multi-layer perceptron with configurable architecture."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = [64, 64],
            activation: str = "relu",
            output_activation: Optional[str] = None,
        ):
            super().__init__()

            # Activation functions
            activations = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "elu": nn.ELU,
                "leaky_relu": nn.LeakyReLU,
                "sigmoid": nn.Sigmoid,
            }

            if activation not in activations:
                raise ValueError(f"Unknown activation: {activation}")

            # Build layers
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activations[activation]())
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))

            if output_activation and output_activation in activations:
                layers.append(activations[output_activation]())

            self.network = nn.Sequential(*layers)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize network weights using orthogonal initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.network(x)

else:
    # Fallback when PyTorch not available
    class MLP:
        """Placeholder MLP class when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural networks")
