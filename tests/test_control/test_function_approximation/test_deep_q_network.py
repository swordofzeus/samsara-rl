"""Tests for Q-Learning control with linear function approximation.

Uses a one-hot encoding on the 4x4 GridWorld so the linear
approximator has the same capacity as a tabular method.  This
isolates the learning algorithm from approximation error.
"""

import numpy as np

from samsara_rl.control.function_approximation.batch.deep_q_network.q_network import (
    QNetwork,
)


def test_deep_q_network_convergence_grid_world(grid_world_mdp, random_policy, fully_connected_one_hot_network):
    """Q-Learning with linear one-hot features should learn a reasonable policy on GridWorld.

    After sufficient episodes the value of terminal states should remain
    at zero and non-terminal states should have negative values (every
    step costs -1).
    """
    ql = QNetwork(
        mdp=grid_world_mdp,
        policy=random_policy,
        gamma=1,
        q=fully_connected_one_hot_network,
        alpha=0.01,
    )
    ql.evaluate(max_iter=600)
    q_values = np.array([ql.get_q_values(x).max() for x in range(0, 16)]).reshape(4, 4)
    q_values[0][0] = 0
    q_values[3][3] = 0
    expected_q = np.array([
        [0, -1, -2, -3],
        [-1, -2, -3, -2],
        [-2, -3, -2, -1],
        [-3, -2, -1, 0],
    ])
    assert np.allclose(q_values, expected_q, atol=0.01)
