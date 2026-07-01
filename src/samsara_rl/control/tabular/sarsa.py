import numpy as np
from abc import ABC, abstractmethod
from samsara_rl.prediction.td import TemporalDifference
from typing import Any
from samsara_rl.utils.policy.policy_utils import sample

from samsara_rl.search.epsilon_greedy import EpsilonGreedy


def td_target(policy, state, q_table):
    return q_table[state, sample(policy, state)]


class Sarsa:
    def __init__(
        self, mdp: Any, policy: np.ndarray, alpha: float = 0.01, gamma: float = 0.9
    ):
        search = EpsilonGreedy(epsilon=0.99, epsilon_decay=0.98)
        self.agent = TemporalDifference(
            mdp, policy, alpha, gamma, search=search, td_target=td_target
        )

    def evaluate(self, max_iter=5000):
        self.agent.evaluate(max_iter=max_iter)
