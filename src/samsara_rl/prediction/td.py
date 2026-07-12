from collections.abc import Callable
from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.search.search import Search


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
        self.elibility = np.zeros(
            (self.mdp.observation_space.n, self.mdp.action_space.n)
        )
        self._lambda = _lambda
        self.td_target = td_target if td_target else td_expectation
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))

    def get_q_values(self, state):
        return self.q[int(state)]

    def post_episode(self, trajectory: np.ndarray) -> None:
        self.elibility = np.zeros(
            (self.mdp.observation_space.n, self.mdp.action_space.n)
        )

    def post_visit(self, history: np.ndarray, terminal=False) -> None:

        if history.curr_index < 1:
            return
        self.elibility = self.elibility * self._lambda
        visited_state = history.past_states()[-2].astype(
            int
        )  # trajectory[0][-2].astype(np.int8)
        el_action = history.past_actions()[-2].astype(int)  # trajectory[1][-2]
        self.elibility[visited_state][el_action.astype(np.int8)] = 1

        R_prime = history.past_rewards()[-2]  # trajectory[2][-2]
        S_prime = history.past_states()[-1].astype(
            int
        )  # trajectory[0][-1].astype(np.int8)  # state we ended up in after A
        Q_expectation = self.td_target(self.policy, S_prime, self.q)

        S = history.past_states()[-2].astype(
            int
        )  # trajectory[0][-2].astype(np.int8)  # state before taking A
        A = history.past_actions()[-2].astype(np.int8)  # action we took to end up in S'
        Q = self.q[S][A]  # Estimate of how good S,A is before taking

        td_error = (R_prime + self.gamma * Q_expectation) - Q
        self.q = self.q + self.alpha * self.elibility * td_error
