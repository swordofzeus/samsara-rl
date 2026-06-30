from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP


def test_foo():
    gw = GridWorldMDP()
    gw.init_transition_probabilities()
