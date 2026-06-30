from samsara_rl.planning.policy_iteration import PolicyIteration
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
import numpy as np

def test_policy_iteration(optimal_gridworld_policy):
    gw = GridWorldMDP()
    gw.init_transition_probabilities()
    pi = PolicyIteration(gw)
    pi.find_optimal_policy()
    assert(np.allclose(optimal_gridworld_policy, pi.policy))
