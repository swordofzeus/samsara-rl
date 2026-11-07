from typing import Any

import numpy as np

from samsara_rl.policy.policy import Policy
from samsara_rl.utils.policy.policy_utils import sample


class StochasticPolicy(Policy):
    def __init__(self, distribution: np.ndarray) -> None:
        self.distribution = distribution

    def action_distribution(self, state: int) -> np.ndarray:
        return np.asarray(self.distribution[state])

    def __call__(self, state: Any) -> int:
        return sample(self.distribution, state)
