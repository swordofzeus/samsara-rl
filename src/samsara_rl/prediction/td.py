from collections.abc import Callable
from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.search.search import Search
from samsara_rl.utils.history import History


def td_expectation(policy: np.ndarray, state: int, q: np.ndarray) -> float:
    result: float = q[state].dot(policy[state])
    return result


class TemporalDifference(Agent):
    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0.4,
        search: Search | None = None,
        td_target: Callable[[np.ndarray, int, np.ndarray], float] | None = None,
    ) -> None:
        super().__init__(mdp, policy, alpha, gamma)
        self.eligibility = np.zeros((self.mdp.observation_space.n, self.mdp.action_space.n))
        self._lambda = _lambda
        self.td_target = td_target if td_target else td_expectation
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))

    def get_q_values(self, state: Any) -> np.ndarray:
        result: np.ndarray = self.q[int(state)]
        return result

    def post_episode(self, history: History) -> None:
        self.eligibility = np.zeros((self.mdp.observation_space.n, self.mdp.action_space.n))

    def post_visit(self, history: History, terminal: bool = False) -> None:
        if history.curr_index < 1:
            return
        self.eligibility = self.eligibility * self._lambda
        visited_state = history.past_states()[-2].astype(int)
        el_action = history.past_actions()[-2].astype(int)
        self.eligibility[visited_state][el_action.astype(np.int8)] = 1

        R_prime = history.past_rewards()[-2]
        S_prime = history.past_states()[-1].astype(int)
        Q_expectation = self.td_target(self.policy, S_prime, self.q)

        S = history.past_states()[-2].astype(int)
        A = history.past_actions()[-2].astype(np.int8)
        Q = self.q[S][A]

        td_error = (R_prime + self.gamma * Q_expectation) - Q
        self.q = self.q + self.alpha * self.eligibility * td_error
