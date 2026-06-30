from samsara_rl.prediction.monte_carlo import MonteCarloPrediction
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.utils.policy.policy_utils import init_uniform_random
import numpy as np

def test_monte_carlo_prediction():
    gw_mdp = GridWorldMDP()
    initial_policy = init_uniform_random(gw_mdp)
    mc = MonteCarloPrediction(gw_mdp,initial_policy).evaluate()
    #trajectory  = np.array( [ [10.,  1., -1.], [14.,  3., -1.] ] )
    #mc.update_q_values(trajectory)
