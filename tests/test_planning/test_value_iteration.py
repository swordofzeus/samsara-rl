import numpy as np

from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.planning.value_iteration import ValueIteration


def test_value_iteration(optimal_gridworld_policy):
    gw = GridWorldMDP()
    gw.init_transition_probabilities()
    vi = ValueIteration(gw)
    vi.find_optimal_policy()
    assert np.allclose(optimal_gridworld_policy, vi.policy)
