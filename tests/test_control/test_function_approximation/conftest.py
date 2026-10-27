"""Shared fixtures for function approximation control tests."""

import numpy as np
import pytest
import torch

from samsara_rl.functions.manual.linear import (
    LinearFunction,
)
from samsara_rl.functions.neural_networks.fully_connected import (
    FullyConnected,
)
from samsara_rl.functions.neural_networks.linear import (
    LinearNetwork,
)
from samsara_rl.functions.neural_networks.world_model_interpreter import (
    WorldModelInterpreter,
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
        return torch.nn.functional.one_hot(torch.tensor(S).long(), num_classes=16).to(torch.float64)

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


@pytest.fixture
def grid_world_model_interpretor():
    class GridWorldModelInterpreter(WorldModelInterpreter):
        def __init__(self, grid_count):
            self.grid_count = grid_count

        def next_state_logits(self, output):
            return output[:, 0 : self.grid_count]

        def predicted_next_state(self, output):
            return torch.argmax(output[:, 0 : self.grid_count], dim=1)

        def predicted_reward(self, output):
            return output[:, self.grid_count]

        def predicted_terminal(self, output):
            return output[:, self.grid_count + 1]

        def parse(self, output):
            return (
                self.next_state_logits(output),
                self.predicted_reward(output),
                self.predicted_terminal(output),
            )

        def prepare_state_target(self, next_states):
            return next_states.long()

    return GridWorldModelInterpreter(16)
