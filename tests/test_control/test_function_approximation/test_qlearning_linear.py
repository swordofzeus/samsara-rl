"""Tests for Q-Learning control with linear function approximation.

Uses a one-hot encoding on the 4x4 GridWorld so the linear
approximator has the same capacity as a tabular method.  This
isolates the learning algorithm from approximation error.
"""

import numpy as np

from samsara_rl.control.function_approximation.qlearning import QLearningGradient


def test_qlearning_convergence_grid_world(grid_world_mdp, random_policy, linear_q):
    """Q-Learning with linear one-hot features should learn a reasonable policy on GridWorld.

    After sufficient episodes the value of terminal states should remain
    at zero and non-terminal states should have negative values (every
    step costs -1).
    """
    ql = QLearningGradient(
        mdp=grid_world_mdp,
        policy=random_policy,
        gamma=0.999,
        q=linear_q,
        alpha=0.01,
    )
    ql.evaluate(max_iter=20000)

    # Collect V(s) = max_a Q(s, a) for each state
    v = np.array([linear_q(s).max(axis=0) for s in range(16)]).reshape(4, 4)

    assert v[0, 0] == 0.0, "Terminal state (0,0) should be 0"
    assert v[3, 3] == 0.0, "Terminal state (3,3) should be 0"
    assert np.all(v[1:, :-1] < 0), "Non-terminal states should have negative values"
