import numpy as np
import pytest

from samsara_rl.control.tabular.sarsa import Sarsa


@pytest.fixture
def expected_sarsa_q():
    return np.array([
        [0.0, 0.0, 0.0, 0.0],
        [-1.12161148, -1.61119986, -0.99975725, -1.50428425],
        [-1.9872437, -2.03445737, -1.8938402, -2.07185665],
        [-2.20111581, -2.14428063, -2.10039718, -2.1115225],
        [-0.99997609, -1.42356143, -1.50229099, -1.50522571],
        [-1.91047241, -2.13550991, -1.93444772, -2.15459325],
        [-2.66717315, -2.67054961, -2.66692061, -2.67754257],
        [-2.23689761, -1.97408658, -2.28458272, -2.43903287],
        [-1.89396683, -2.05313593, -1.94764709, -1.92572534],
        [-2.67894936, -2.69100669, -2.6748804, -2.67850048],
        [-3.38330339, -2.17830654, -3.39225751, -1.90005299],
        [-2.76221569, -1.0, -2.90308879, -1.9025019],
        [-2.1041876, -2.06835085, -2.09554833, -2.04244933],
        [-2.37778086, -2.24164874, -2.11934515, -1.99679402],
        [-2.92205726, -1.92014764, -2.75929611, -1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])


def test_sarsa_convergence(grid_world_mdp, expected_sarsa_q):
    """SARSA should converge close to expected Q values for the grid world."""
    sarsa = Sarsa(grid_world_mdp, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995)
    sarsa.evaluate(max_iter=5000)
    q = sarsa.q

    assert np.all(q[0] == 0.0), "Terminal state 0 should have Q=0"
    assert np.all(q[15] == 0.0), "Terminal state 15 should have Q=0"
    assert np.allclose(q, expected_sarsa_q, atol=0.5), f"Q values should be close to expected.\nGot:\n{q}"
