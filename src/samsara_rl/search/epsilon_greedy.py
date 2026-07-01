import random

import numpy as np

from samsara_rl.search.search import Search


class EpsilonGreedy(Search):
    def __init__(self, epsilon: float, epsilon_decay: float) -> None:
        if epsilon > 1:
            raise ValueError(epsilon)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def step(self, policy: np.ndarray, state: int, q_table: np.ndarray) -> int:
        self.epsilon = self.epsilon * self.epsilon_decay
        return (
            q_table[state].argmax(axis=0)
            if random.uniform(0, 1) > self.epsilon
            else random.randint(0, q_table.shape[1] - 1)
        )
