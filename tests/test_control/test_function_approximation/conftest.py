"""Shared fixtures for function approximation control tests."""

import numpy as np
import pytest
import torch

from samsara_rl.control.function_approximation.functions.manual.linear import LinearFunction
from samsara_rl.control.function_approximation.functions.neural_networks.linear import LinearNetwork


@pytest.fixture
def one_hot_encoding():
    """Feature map that produces a one-hot vector for each discrete state.

    With 16 features for 16 states this gives the linear approximator
    the same representational power as a tabular method, making it a
    useful sanity check for convergence.
    """

    def X(S: np.ndarray) -> np.ndarray:
        arr = np.zeros(16)
        arr[int(S)] = 1
        return arr

    return X


@pytest.fixture
def one_hot_encoding_torch():
    """Feature map that produces a one-hot tensor for each discrete state.

    Torch equivalent of one_hot_encoding for use with PyTorch networks.
    """

    def X(S: torch.Tensor) -> torch.Tensor:
        arr = torch.zeros(16, dtype=torch.float64)
        arr[int(S)] = 1
        return arr

    return X


@pytest.fixture
def linear_q(one_hot_encoding):
    """A linear function approximator with one-hot features (16 states, 4 actions)."""
    return LinearFunction(16, 4, one_hot_encoding)


@pytest.fixture
def linear_q_torch(one_hot_encoding_torch):
    """A PyTorch linear network with one-hot features (16 states, 4 actions)."""
    return LinearNetwork(16, 4, False, preprocess=one_hot_encoding_torch)
