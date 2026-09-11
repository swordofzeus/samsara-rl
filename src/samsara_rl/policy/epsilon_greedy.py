import random
from collections.abc import Callable
from typing import Any

import numpy as np

from samsara_rl.policy.policy import Policy


class EpsilonGreedy(Policy):
    def __init__(self, q_values: Callable[..., np.ndarray], epsilon: float, epsilon_decay: float) -> None:
        if epsilon > 1:
            raise ValueError(epsilon)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.05
        self.q_values = q_values

    def decay(self) -> None:
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def step(self, state: np.ndarray) -> int:
        action_values = self.q_values(state)
        exploit = random.uniform(0, 1) > self.epsilon
        sampled_action = action_values.argmax(axis=0) if exploit else random.randint(0, action_values.shape[0] - 1)
        return int(sampled_action)
