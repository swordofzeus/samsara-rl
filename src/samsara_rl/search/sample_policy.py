from samsara_rl.search.search import Search
from samsara_rl.utils.policy.policy_utils import sample


class SamplePolicy(Search):

    def __init__(self):
        pass

    def step(self, policy, state, q_table):
        return (
            sample(policy, state)
        )
