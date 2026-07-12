import numpy as np
import pytest

from samsara_rl.mdp.grid_world.grid_world_gym import GridWorldMDP
from samsara_rl.utils.policy.policy_utils import init_uniform_random


@pytest.fixture
def grid_world_mdp():
    return GridWorldMDP()


@pytest.fixture
def random_policy(grid_world_mdp):
    return init_uniform_random(grid_world_mdp)


@pytest.fixture
def optimal_gridworld_policy():
    return np.array([
        [
            1,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            1,
            0,
        ],
        [
            0,
            0,
            1,
            0,
        ],
        [
            0,
            1,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
        ],
        [
            0,
            1,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
        ],
        [
            0,
            1,
            0,
            0,
        ],
        [
            0,
            1,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            1,
        ],
        [
            0,
            0,
            0,
            1,
        ],
        [
            1,
            0,
            0,
            0,
        ],
    ])
