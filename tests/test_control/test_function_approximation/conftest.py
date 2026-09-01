"""Shared fixtures for function approximation control tests."""

import numpy as np
import pytest
import torch

from samsara_rl.control.function_approximation.functions.manual.linear import (
    LinearFunction,
)
from samsara_rl.control.function_approximation.functions.neural_networks.fully_connected import (
    FullyConnected,
)
from samsara_rl.control.function_approximation.functions.neural_networks.linear import (
    LinearNetwork,
)


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
        if isinstance(S, int) or S.ndim == 0 or (S.ndim == 1 and S.shape[0] == 1):
            arr = torch.zeros(16, dtype=torch.float64)
            arr[int(S)] = 1
            return arr

        output = torch.zeros((S.shape[0], 16))
        S_tensor = torch.from_numpy(S).to(torch.int32) if isinstance(S, np.ndarray) else S.to(torch.int32)
        output.scatter_(1, S_tensor.unsqueeze(1), 1.0)
        return output.to(torch.float64)

    return X


@pytest.fixture
def linear_q(one_hot_encoding):
    """A linear function approximator with one-hot features (16 states, 4 actions)."""
    return LinearFunction(16, 4, one_hot_encoding)


@pytest.fixture
def linear_q_torch(one_hot_encoding_torch):
    """A PyTorch linear network with one-hot features (16 states, 4 actions)."""
    return LinearNetwork(16, 4, False, preprocess=one_hot_encoding_torch)


@pytest.fixture
def fully_connected_one_hot_network(one_hot_encoding_torch):
    """A PyTorch linear network with one-hot features (16 states, 4 actions)."""
    return FullyConnected(16, 32, 4, preprocess=one_hot_encoding_torch)
