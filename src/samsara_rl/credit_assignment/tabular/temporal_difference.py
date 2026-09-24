from typing import Any

import numpy as np

from samsara_rl.credit_assignment.credit_assignment import CreditAssignment
from samsara_rl.utils.memory.episode import Episode


class TemporalDifference(CreditAssignment):
    def __init__(
        self,
        q: Any,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0.4,
    ) -> None:
        super().__init__(alpha=alpha, gamma=gamma)
        self.eligibility = np.zeros(q.shape)
        self._lambda = _lambda
        self.q = q

    def reset(self, **kwargs: Any) -> None:
        self.eligibility = np.zeros(self.q.shape)

    def update(self, history: Episode, target: float) -> None:
        if history.curr_index < 1:
            return
        self.eligibility = self.eligibility * self._lambda
        visited_state = history.past_states()[-2].astype(int)
        el_action = history.past_actions()[-2].astype(int)
        self.eligibility[visited_state][el_action.astype(np.int8)] = 1

        R_prime = history.past_rewards()[-2]

        S = history.past_states()[-2].astype(int)
        A = history.past_actions()[-2].astype(np.int8)
        Q = self.q[S][A]

        td_error = (R_prime + self.gamma * target) - Q
        self.q += self.alpha * self.eligibility * td_error
