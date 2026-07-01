import random
from samsara_rl.search.search import Search
from samsara_rl.utils.policy.policy_utils import sample


class EpsilonGreedy(Search):

    def __init__(self, epsilon, epsilon_decay):
        assert epsilon <= 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def step(self, policy, state, q_table):
        self.epsilon = self.epsilon * self.epsilon_decay
        return (
            q_table[state].argmax(axis=0)
            if random.uniform(0, 1) > self.epsilon
            else random.randint(0, q_table.shape[1] -1 )
        )

