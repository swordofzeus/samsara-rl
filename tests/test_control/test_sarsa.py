import numpy as np
import pytest

from samsara_rl.control.tabular.sarsa import Sarsa
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.utils.policy.policy_utils import init_uniform_random


@pytest.fixture
def gw_mdp():
    return GridWorldMDP()


@pytest.fixture
def random_policy(gw_mdp):
    return init_uniform_random(gw_mdp)


@pytest.fixture
def expected_sarsa_q():
    return np.array([
        [0.0, 0.0, 0.0, 0.0],
        [-5.46, -6.67, -1.0, -7.16],
        [-7.28, -7.33, -5.61, -7.70],
        [-7.69, -7.35, -7.36, -7.75],
        [-1.0, -7.09, -5.44, -6.65],
        [-5.46, -7.29, -5.38, -7.27],
        [-7.24, -6.83, -6.69, -7.27],
        [-7.73, -5.80, -7.26, -7.24],
        [-5.37, -7.64, -7.20, -7.27],
        [-6.77, -7.13, -7.20, -6.77],
        [-7.31, -5.61, -7.27, -5.48],
        [-7.17, -1.0, -6.77, -5.90],
        [-7.19, -7.63, -7.66, -7.19],
        [-7.25, -7.22, -7.71, -5.45],
        [-6.86, -5.68, -7.25, -1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])


def test_sarsa_convergence(gw_mdp, random_policy, expected_sarsa_q):
    """SARSA should converge close to expected Q values for the grid world."""
    sarsa = Sarsa(gw_mdp, random_policy, gamma=0.9)
    sarsa.evaluate(max_iter=5000)
    q = sarsa.agent.q_table

    assert np.all(q[0] == 0.0), "Terminal state 0 should have Q=0"
    assert np.all(q[15] == 0.0), "Terminal state 15 should have Q=0"
    assert np.allclose(q, expected_sarsa_q, atol=1.0), f"Q values should be close to expected.\nGot:\n{q}"
