import numpy as np
import pytest

from samsara_rl.prediction.monte_carlo import MonteCarloPrediction


@pytest.fixture
def expected_v_random_policy():
    return np.array([
        [0.0, -4.75, -6.81, -7.39],
        [-4.75, -6.23, -6.87, -6.81],
        [-6.81, -6.87, -6.23, -4.75],
        [-7.39, -6.81, -4.75, 0.0],
    ])


# def test_post_episode_updates_q(grid_world_mdp, random_policy):
#     """post_episode should update Q-table entries for visited (S, A) pairs."""
#     mc = MonteCarloPrediction(grid_world_mdp, random_policy, gamma=0.9)
#     trajectory = np.array([[10.0, 1.0, -1.0], [14.0, 2.0, -1.0]])
#     mc.post_episode(trajectory)
#     assert mc.q[10, 1] != 0, "Q(10, 1) should be updated"
#     assert mc.q[14, 2] != 0, "Q(14, 2) should be updated"


# def test_post_episode_does_not_update_unvisited(grid_world_mdp, random_policy):
#     """post_episode should not modify Q values for unvisited (S, A) pairs."""
#     mc = MonteCarloPrediction(grid_world_mdp, random_policy, gamma=0.9)
#     trajectory = np.array([[10.0, 1.0, -1.0], [14.0, 2.0, -1.0]])
#     mc.post_episode(trajectory)
#     assert mc.q[5, 0] == 0, "Unvisited Q(5, 0) should remain 0"
#     assert mc.q[10, 0] == 0, "Unvisited action Q(10, 0) should remain 0"


# def test_post_episode_handles_duplicate_visits(grid_world_mdp, random_policy):
#     """Duplicate (S, A) pairs should all contribute updates via np.add.at."""
#     mc = MonteCarloPrediction(grid_world_mdp, random_policy, gamma=0.9)
#     trajectory = np.array([
#         [10.0, 1.0, -1.0],
#         [14.0, 2.0, -1.0],
#         [10.0, 1.0, -1.0],
#         [14.0, 3.0, -1.0],
#     ])
#     mc.post_episode(trajectory)
#     q_with_dupes = mc.q[10, 1]

#     mc2 = MonteCarloPrediction(grid_world_mdp, random_policy, gamma=0.9)
#     trajectory_no_dupe = np.array([
#         [10.0, 1.0, -1.0],
#         [14.0, 2.0, -1.0],
#         [13.0, 1.0, -1.0],
#         [14.0, 3.0, -1.0],
#     ])
#     mc2.post_episode(trajectory_no_dupe)
#     q_without_dupes = mc2.q[10, 1]

#     assert q_with_dupes != q_without_dupes, "Duplicate visits should produce different Q updates than single visits"


# def test_discounted_returns_no_discount(grid_world_mdp, random_policy):
#     """With gamma=1, discounted returns should equal simple cumulative sums."""
#     mc = MonteCarloPrediction(grid_world_mdp, random_policy, gamma=1)
#     trajectory = np.array([
#         [10.0, 1.0, -1.0],
#         [14.0, 2.0, -1.0],
#         [13.0, 3.0, -1.0],
#     ])
#     returns = mc._discounted_cum_trajectory(trajectory)
#     assert np.isclose(returns[2], -1.0), "Last step return should be its own reward"
#     assert np.isclose(returns[1], -2.0), "Second step return should be sum of remaining"
#     assert np.isclose(returns[0], -3.0), "First step return should be total sum"


def test_evaluate_convergence(grid_world_mdp, random_policy, expected_v_random_policy):
    """MC prediction should converge close to the true V^pi for a random policy."""
    mc = MonteCarloPrediction(grid_world_mdp, random_policy, alpha=0.01, gamma=0.9)
    mc.evaluate(max_iter=10000)
    v = mc.q.mean(axis=1).reshape(4, 4)

    assert v[0, 0] == 0.0, "Terminal state (0,0) should be 0"
    assert v[3, 3] == 0.0, "Terminal state (3,3) should be 0"
    assert np.allclose(v, expected_v_random_policy, atol=0.8), f"V^pi should be close to expected values.\nGot:\n{v}"
