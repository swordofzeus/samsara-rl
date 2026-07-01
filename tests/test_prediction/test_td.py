import numpy as np
import pytest

from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.prediction.td import TemporalDifference
from samsara_rl.utils.policy.policy_utils import init_uniform_random


@pytest.fixture
def gw_mdp():
    return GridWorldMDP()


@pytest.fixture
def random_policy(gw_mdp):
    return init_uniform_random(gw_mdp)


@pytest.fixture
def expected_v_random_policy():
    return np.array([
        [0.0, -4.75, -6.81, -7.39],
        [-4.75, -6.23, -6.87, -6.81],
        [-6.81, -6.87, -6.23, -4.75],
        [-7.39, -6.81, -4.75, 0.0],
    ])


def test_post_visit_single_step(gw_mdp, random_policy):
    """post_visit with a single-step trajectory should be a no-op."""
    td = TemporalDifference(gw_mdp, random_policy)
    trajectory = np.array([[10.0, 1.0, -1.0]])
    td.post_visit(trajectory)
    assert np.all(td.q_table == 0), "Q-table should be unchanged after a single-step trajectory"


def test_post_visit_updates_q_table(gw_mdp, random_policy):
    """post_visit with a two-step trajectory should update the Q-table."""
    td = TemporalDifference(gw_mdp, random_policy)
    trajectory = np.array([[10.0, 1.0, -1.0], [14.0, 2.0, -1.0]])
    td.post_visit(trajectory)
    assert td.q_table[10, 1] != 0, "Q(10, 1) should be updated after post_visit"


def test_post_visit_eligibility_decay(gw_mdp, random_policy):
    """Eligibility traces should decay by lambda and set visited (S, A) to 1."""
    td = TemporalDifference(gw_mdp, random_policy, _lambda=0.4)
    trajectory = np.array([[10.0, 1.0, -1.0], [14.0, 2.0, -1.0]])
    td.post_visit(trajectory)
    assert td.elibility[10, 1] == 1.0, "Visited (S, A) eligibility should be 1"

    trajectory = np.array([[10.0, 1.0, -1.0], [14.0, 2.0, -1.0], [13.0, 3.0, -1.0]])
    td.post_visit(trajectory)
    assert td.elibility[14, 2] == 1.0, "Most recent (S, A) eligibility should be 1"
    assert np.isclose(td.elibility[10, 1], 0.4), "Previous (S, A) eligibility should decay by lambda"


def test_evaluate_convergence(gw_mdp, random_policy, expected_v_random_policy):
    """TD(lambda) should converge close to the true V^pi for a random policy."""
    td = TemporalDifference(gw_mdp, random_policy, alpha=0.01, gamma=0.9, _lambda=0.4)
    td.evaluate(max_iter=6000)
    v = td.q_table.mean(axis=1).reshape(4, 4)

    assert v[0, 0] == 0.0, "Terminal state (0,0) should be 0"
    assert v[3, 3] == 0.0, "Terminal state (3,3) should be 0"
    assert np.allclose(v, expected_v_random_policy, atol=0.5), f"V^pi should be close to expected values.\nGot:\n{v}"
