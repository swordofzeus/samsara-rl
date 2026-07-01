from typing import Any

import numpy as np

from samsara_rl.prediction.td import TemporalDifference
from samsara_rl.search.epsilon_greedy import EpsilonGreedy


def td_target(policy: np.ndarray, state: int, q_table: np.ndarray) -> float:
    result: float = q_table[state].max()
    return result


class QLearning:
    def __init__(self, mdp: Any, policy: np.ndarray, alpha: float = 0.01, gamma: float = 0.9):
        search = EpsilonGreedy(epsilon=0.99, epsilon_decay=0.98)
        self.agent = TemporalDifference(mdp, policy, alpha, gamma, search=search, td_target=td_target)

    def evaluate(self, max_iter: int = 5000) -> None:
        self.agent.evaluate(max_iter=max_iter)
