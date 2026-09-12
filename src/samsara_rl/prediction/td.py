from typing import Any

import numpy as np

from samsara_rl.credit_assignment.temporal_difference import TemporalDifference
from samsara_rl.policy.stochastic_policy import StochasticPolicy
from samsara_rl.utils.memory.episode import Episode


class TDPolicyEvaluation(TemporalDifference):
    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0.4,
    ) -> None:
        super().__init__(mdp, alpha=alpha, gamma=gamma, _lambda=_lambda)
        self.search = StochasticPolicy(policy)

    def td_target(self, history: Episode) -> float:
        state = history.past_states()[-1].astype(int)
        result: float = self.q[state].dot(self.search.policy[state])
        return result

    def select_action(self, state: Any) -> int:
        return self.search.step(state)
