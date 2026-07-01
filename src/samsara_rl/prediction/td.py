from typing import Any

import numpy as np
from samsara_rl.agent import Agent



def td_expectation(policy, state, q_table ):
    return q_table[state].dot(
        policy[state]
    )


class TemporalDifference(Agent):
    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0.4,
        search=None,
        td_target=None
    ) -> None:
        super().__init__(mdp, policy, alpha, gamma)
        self.elibility = np.zeros((mdp.STATE_COUNT, self.mdp.ACTION_COUNT))
        self._lambda = _lambda
        self.td_target = td_expectation if not td_target else td_target

    def post_episode(self, trajectory: np.ndarray) -> None:
        self.elibility = np.zeros((self.mdp.STATE_COUNT, self.mdp.ACTION_COUNT))


    def post_visit(self, trajectory: np.ndarray) -> None:

        if trajectory.shape[0] < 2:
            return
        self.elibility = self.elibility * self._lambda
        visited_state = trajectory[-2][0].astype(np.int8)
        el_action = trajectory[-2][1]
        self.elibility[visited_state][el_action.astype(np.int8)] = 1

        R_prime = trajectory[-2][2]
        S_prime = trajectory[-1][0].astype(np.int8)  # state we ended up in after A
        Q_expectation = self.td_target(self.policy, S_prime, self.q_table)

        S = trajectory[-2][0].astype(np.int8)  # state before taking A
        A = trajectory[-2][1].astype(np.int8)  # action we took to end up in S'
        Q = self.q_table[S][A]  # Estimate of how good S,A is before taking

        td_error = (R_prime + self.gamma * Q_expectation) - Q
        self.q_table = self.q_table + self.alpha * self.elibility * td_error
