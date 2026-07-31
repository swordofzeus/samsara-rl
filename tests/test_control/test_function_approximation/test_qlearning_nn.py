"""Tests for Q-Learning control with PyTorch linear function approximation.

Uses a one-hot encoding on the 4x4 GridWorld so the linear
network has the same capacity as a tabular method.
"""

import numpy as np
import torch

from samsara_rl.control.function_approximation.online.td.td import TemporalDifferenceGradient


def test_qlearning_nn_convergence_grid_world(grid_world_mdp, random_policy, linear_q_torch):
    """Q-Learning with PyTorch linear one-hot features should learn a reasonable policy on GridWorld.

    After sufficient episodes the value of terminal states should remain
    at zero and non-terminal states should have negative values (every
    step costs -1).
    """
    nn_agent = TemporalDifferenceGradient(
        grid_world_mdp, random_policy, q=linear_q_torch, gamma=0.999, alpha=0.01, auto_grad=True, _lambda=0
    )
    nn_agent.evaluate(max_iter=20000)

    with torch.no_grad():
        v = np.array([linear_q_torch(torch.tensor(s)).max().item() for s in range(16)]).reshape(4, 4)

    assert v[0, 0] == 0.0, "Terminal state (0,0) should be 0"
    assert v[3, 3] == 0.0, "Terminal state (3,3) should be 0"
    assert np.all(v[1:, :-1] < 0), "Non-terminal states should have negative values"
