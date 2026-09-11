from typing import Any

import numpy as np
from samsara_rl.policy.policy import Policy

from samsara_rl.utils.policy.policy_utils import sample


class StochasticPolicy(Policy):
    def __init__(self, policy: np.ndarray) -> None:
        self.policy = policy

    def step(self, state: Any) -> int:
        return sample(self.policy, state)
