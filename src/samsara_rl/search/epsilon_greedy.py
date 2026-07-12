import random

import numpy as np

from samsara_rl.search.search import Search


class EpsilonGreedy(Search):
    def __init__(self, epsilon: float, epsilon_decay: float) -> None:
        if epsilon > 1:
            raise ValueError(epsilon)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def step(self, policy: np.ndarray, state, action_values: np.ndarray, log = False) -> int:
        self.epsilon = self.epsilon * self.epsilon_decay
        exploit = random.uniform(0, 1) > self.epsilon
        sampled_action = action_values.argmax(axis=0) if exploit else random.randint(0, action_values.shape[0] - 1)
        return (
            sampled_action
        )
