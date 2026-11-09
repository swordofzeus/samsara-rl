import numpy as np
import pytest

from samsara_rl.control.tabular.q_learning import QLearning


@pytest.fixture
def expected_q_learning():
    return np.array([
        [0.0, 0.0, 0.0, 0.0],
        [-1.16178548, -1.63146585, -0.99922645, -1.44064858],
        [-1.8628158, -1.9599681, -1.86549154, -1.88496543],
        [-1.88411187, -1.95007203, -1.87498107, -1.9416491],
        [-0.99922645, -1.3498799, -1.18343806, -1.6132606],
        [-1.86174113, -1.93421051, -1.8668552, -1.8731368],
        [-2.58228442, -2.58875874, -2.58511372, -2.59393169],
        [-1.89426763, -1.89042575, -2.00017618, -1.90897019],
        [-1.85186858, -1.87218035, -1.91750642, -1.91715146],
        [-2.53213991, -2.54515044, -2.53096376, -2.53324256],
        [-3.29384593, -1.9, -3.22788754, -1.9],
        [-2.37443782, -1.0, -2.62419404, -1.83341685],
        [-2.00139779, -1.9410375, -2.01161928, -1.97575321],
        [-1.9320099, -1.92383536, -2.02842634, -1.89990478],
        [-2.70881725, -1.89978171, -2.7076441, -1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])


def test_q_learning_convergence(grid_world_mdp, random_policy, expected_q_learning):
    """Q-Learning should converge close to expected Q values for the grid world."""
    q_learning = QLearning(grid_world_mdp, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995)
    q_learning.evaluate(max_iter=5000)
    q = q_learning.q

    assert np.all(q[0] == 0.0), "Terminal state 0 should have Q=0"
    assert np.all(q[15] == 0.0), "Terminal state 15 should have Q=0"
    assert np.allclose(q, expected_q_learning, atol=0.5), f"Q values should be close to expected.\nGot:\n{q}"
