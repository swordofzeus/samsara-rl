import numpy as np


def init_uniform_random(mdp):
    '''Return uniform random policy given an MDP
       of all actions being uniform'''
    return np.full(
            (mdp.STATE_COUNT, mdp.ACTION_COUNT),
            1.0 / mdp.ACTION_COUNT,
        )


def sample(policy, s):
    '''Returns a sample action based on an input policy'''
    action_distribution = policy[s]
    selected_action = np.random.choice(len(action_distribution),p=action_distribution)
    return selected_action
