"""Shared fixtures for function approximation control tests."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as f
from gymnasium import spaces

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
from samsara_rl.policy.policy import Policy


class OneHotWrapper(gym.ObservationWrapper):
    """Wraps a Discrete observation space into a one-hot Box."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        n = env.observation_space.n
        self.n = n
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float64)

    def observation(self, obs: int) -> np.ndarray:
        one_hot = np.zeros(self.n, dtype=np.float64)
        one_hot[obs] = 1.0
        return one_hot


@pytest.fixture
def grid_world_mdp_one_hot(grid_world_mdp):
    """GridWorld with one-hot encoded observations for neural network agents."""
    return OneHotWrapper(grid_world_mdp)


@pytest.fixture
def grid_world_stochastic_policy(random_policy):
    class GridWorldStochasticPolicy(Policy):
        def __init__(self, distribution):
            self.distribution = torch.from_numpy(distribution)

        def __call__(self, state):
            if state.ndim > 2:
                raise ValueError("State must have at most 2 dimensions")
            state_tensor = torch.as_tensor(state) if isinstance(state, np.ndarray) else state
            idx = torch.argmax(state_tensor, dim=-1)
            action_probability = self.distribution[idx]
            sample = torch.multinomial(action_probability, num_samples=1, replacement=True)
            return sample.squeeze(-1)

    return GridWorldStochasticPolicy(random_policy)


@pytest.fixture
def one_hot_encoding():
    """Feature map that produces a one-hot vector for each discrete state.

    With 16 features for 16 states this gives the linear approximator
    the same representational power as a tabular method, making it a
    useful sanity check for convergence.
    """

    def X(S: np.ndarray) -> np.ndarray:
        idx = int(S.item()) if isinstance(S, np.ndarray) else int(S)
        arr = np.zeros(16)
        arr[idx] = 1
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
def linear_q_torch():
    """A PyTorch linear network (16 inputs, 4 actions)."""
    return LinearNetwork(16, 4, False)


@pytest.fixture
def fully_connected_one_hot_network():
    """A fully connected network (16 inputs, 32 hidden, 4 actions)."""
    return FullyConnected(16, 32, 4)


@pytest.fixture
def grid_world_model_interpretor():
    class GridWorldModelInterpreter(WorldModelInterpreter):
        def __init__(self, grid_count):
            self.grid_count = grid_count

        def next_state_logits(self, output):
            return output[:, 0 : self.grid_count]

        def predicted_next_state(self, output):
            next_state = torch.argmax(output[:, 0 : self.grid_count], dim=1)
            next_state_one_hot = f.one_hot(next_state, num_classes=16)
            return next_state_one_hot.to(dtype=output.dtype, device=output.device)

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
            return torch.argmax(next_states, dim=1)

    return GridWorldModelInterpreter(16)


@pytest.fixture
def grid_world_trained_model():
    model = FullyConnected(20, 32, 18, dtype=torch.float32)
    model.load_state_dict(torch.load(Path(__file__).parent / "assets" / "grid_world_model.pt"))
    return model
