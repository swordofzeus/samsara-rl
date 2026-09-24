"""Tests for Q-Learning control with linear function approximation.

Uses a one-hot encoding on the 4x4 GridWorld so the linear
approximator has the same capacity as a tabular method.  This
isolates the learning algorithm from approximation error.
"""

import numpy as np
import pytest

from samsara_rl.control.semi_gradient.qlearning import QLearningGradient


@pytest.fixture
def expected_v_qlearning_linear():
    return np.array([
        [0.0, -0.95696558, -1.61760125, -1.66031643],
        [-0.95848265, -1.59917356, -2.11973123, -1.70186251],
        [-1.63400502, -2.10210605, -1.999, -1.0],
        [-1.77546489, -1.89773042, -1.0, 0.0],
    ])


def test_qlearning_convergence_grid_world(grid_world_mdp, random_policy, linear_q, expected_v_qlearning_linear):
    """Q-Learning with linear one-hot features should learn a reasonable policy on GridWorld."""
    ql = QLearningGradient(
        mdp=grid_world_mdp,
        gamma=0.999,
        q=linear_q,
        alpha=0.01,
    )
    ql.evaluate(max_iter=20000)

    # Collect V(s) = max_a Q(s, a) for each state
    v = np.array([linear_q(s).max(axis=0) for s in range(16)]).reshape(4, 4)

    assert v[0, 0] == 0.0, "Terminal state (0,0) should be 0"
    assert v[3, 3] == 0.0, "Terminal state (3,3) should be 0"
    assert np.allclose(v, expected_v_qlearning_linear, atol=0.5), f"V values should be close to expected.\nGot:\n{v}"
