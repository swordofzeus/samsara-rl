import numpy as np
import pytest

from samsara_rl.control.tabular.q_learning import QLearning


@pytest.fixture
def expected_q_learning():
    return np.array([
        [0.0, 0.0, 0.0, 0.0],
        [-1.32886115, -1.55201453, -0.99975725, -1.4493094],
        [-1.92830592, -1.91426908, -1.87769455, -1.96411915],
        [-1.89366536, -1.86211627, -1.93270583, -1.88569013],
        [-0.9995889, -1.90021508, -1.46984962, -1.37324602],
        [-1.87689589, -1.9139836, -1.8757137, -2.03782533],
        [-2.61157872, -2.60904899, -2.61553235, -2.61072938],
        [-1.96265381, -1.893302, -1.90104562, -1.96026649],
        [-1.8761802, -1.88137833, -1.93435892, -2.0911225],
        [-2.60218283, -2.60401729, -2.59927851, -2.60867997],
        [-3.32154758, -1.9, -3.31045959, -1.9],
        [-2.46817038, -1.0, -2.5015089, -1.82294605],
        [-1.92252259, -1.99354313, -1.87472062, -1.90207959],
        [-2.08527515, -2.08137019, -2.09224193, -1.89991499],
        [-2.70942496, -1.89907309, -2.7072102, -1.0],
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
