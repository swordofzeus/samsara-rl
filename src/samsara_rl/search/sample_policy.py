import numpy as np

from samsara_rl.search.search import Search
from samsara_rl.utils.policy.policy_utils import sample


class SamplePolicy(Search):
    def __init__(self) -> None:
        pass

    def step(self, policy: np.ndarray, state: int, q_table: np.ndarray) -> int:
        return sample(policy, state)
