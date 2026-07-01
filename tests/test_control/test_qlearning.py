import numpy as np
import pytest
from samsara_rl.control.tabular.q_learning import QLearning

from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.utils.policy.policy_utils import init_uniform_random


@pytest.fixture
def gw_mdp():
    return GridWorldMDP()


@pytest.fixture
def random_policy(gw_mdp):
    return init_uniform_random(gw_mdp)


@pytest.fixture
def expected_q_learning():
    return np.array([ [ 0.,          0. ,         0. ,         0. ,       ],
          [-1.89960069, -2.70980586, -0.99991276, -2.70972016],
          [-2.71006554, -3.43898023, -1.90005578, -3.43899869],
          [-3.43899811, -2.70999939, -2.71007744, -3.43899811],
          [-0.99994045, -2.71002121, -1.89982355, -2.7097441 ],
          [-1.90006711, -3.43897039, -1.90002382, -3.43898471],
          [-2.71007532, -2.70999969, -2.71005331, -2.7099981 ],
          [-3.43900239, -1.89999922, -3.43898893, -2.70999976],
          [-1.90003245, -3.43899143, -2.7101133 , -3.4389753 ],
          [-2.71004955, -2.71000165, -2.71006164, -2.70999983],
          [-3.43899199, -1.89999976, -3.43899026, -1.89999969],
          [-2.7099993,  -0.9999998 , -2.71000001, -1.89999649],
          [-2.71008411, -3.43898765, -3.43898181, -2.71000242],
          [-3.43897853, -2.7100025 , -3.43899707, -1.89999912],
          [-2.70999966, -1.89999832, -2.71000161, -0.99999985],
          [ 0. ,         0. ,         0.  ,        0.        ]]

    )


def test_q_learning_convergence(gw_mdp, random_policy, expected_q_learning):
    """SARSA should converge close to expected Q values for the grid world."""
    q_learning = QLearning(gw_mdp, random_policy, gamma=0.9)
    q_learning.evaluate(max_iter=5000)
    q = q_learning.agent.q_table

    assert np.all(q[0] == 0.0), "Terminal state 0 should have Q=0"
    assert np.all(q[15] == 0.0), "Terminal state 15 should have Q=0"
    assert np.allclose(q, expected_q_learning, atol=1.0), (
        f"Q values should be close to expected.\nGot:\n{q}"
    )
