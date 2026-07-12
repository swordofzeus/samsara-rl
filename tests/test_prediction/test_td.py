import numpy as np
import pytest

from samsara_rl.prediction.td import TemporalDifference
from samsara_rl.utils.history import History


@pytest.fixture
def expected_v_random_policy():
    return np.array([
        [0.0, -4.75, -6.81, -7.39],
        [-4.75, -6.23, -6.87, -6.81],
        [-6.81, -6.87, -6.23, -4.75],
        [-7.39, -6.81, -4.75, 0.0],
    ])


@pytest.fixture
def two_step_history():
    """History with S=10, A=1, R=-1, S'=14, A'=2."""
    h = History(1, 1, 10)
    h.record(action=1, reward=-1.0, s_prime=14, a_prime=2)
    return h


@pytest.fixture
def three_step_history(two_step_history):
    """History with S=10 -> S=14 -> S=13."""
    h = History(1, 1, 10)
    h.record(action=1, reward=-1.0, s_prime=14, a_prime=2)
    h.record(action=2, reward=-1.0, s_prime=13, a_prime=3)
    return h


def test_post_visit_single_step(grid_world_mdp, random_policy):
    """post_visit with a single-step trajectory should be a no-op."""
    td = TemporalDifference(grid_world_mdp, random_policy)
    history = History(1, 1, 10)
    td.post_visit(history)
    assert np.all(td.q == 0), "Q-table should be unchanged after a single-step trajectory"


def test_post_visit_updates_q(grid_world_mdp, random_policy, two_step_history):
    """post_visit with a two-step trajectory should update the Q-table."""
    td = TemporalDifference(grid_world_mdp, random_policy)
    td.post_visit(two_step_history, False)
    assert td.q[10, 1] != 0, "Q(10, 1) should be updated after post_visit"


def test_post_visit_eligibility_decay(grid_world_mdp, random_policy, two_step_history, three_step_history):
    """Eligibility traces should decay by lambda and set visited (S, A) to 1."""
    td = TemporalDifference(grid_world_mdp, random_policy, _lambda=0.4)
    td.post_visit(two_step_history)
    assert td.eligibility[10, 1] == 1.0, "Visited (S, A) eligibility should be 1"

    td.post_visit(three_step_history, False)
    assert td.eligibility[14, 2] == 1.0, "Most recent (S, A) eligibility should be 1"
    assert np.isclose(td.eligibility[10, 1], 0.4), "Previous (S, A) eligibility should decay by lambda"


def test_evaluate_convergence(grid_world_mdp, random_policy, expected_v_random_policy):
    """TD(lambda) should converge close to the true V^pi for a random policy."""
    td = TemporalDifference(grid_world_mdp, random_policy, alpha=0.01, gamma=0.9, _lambda=0.4)
    td.evaluate(max_iter=6000)
    v = td.q.mean(axis=1).reshape(4, 4)

    assert v[0, 0] == 0.0, "Terminal state (0,0) should be 0"
    assert v[3, 3] == 0.0, "Terminal state (3,3) should be 0"
    assert np.allclose(v, expected_v_random_policy, atol=0.5), f"V^pi should be close to expected values.\nGot:\n{v}"
