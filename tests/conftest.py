import numpy as np
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
import pytest


@pytest.fixture
def grid_world_mdp():
    return GridWorldMDP()


@pytest.fixture
def optimal_gridworld_policy():
    return np.array(
        [
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
        ]
    )
