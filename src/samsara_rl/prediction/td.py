from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.credit_assignment.tabular.temporal_difference import (
    TemporalDifference,
)
from samsara_rl.policy.stochastic_policy import StochasticPolicy
from samsara_rl.utils.memory.episode import Episode


class TDPolicyEvaluation(Agent):
    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0.4,
    ) -> None:
        super().__init__(mdp=mdp, alpha=alpha, gamma=gamma)
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))
        self.search = StochasticPolicy(policy)
        self.credit_assignment = TemporalDifference(q=self.q, alpha=alpha, gamma=gamma, _lambda=_lambda)

    def td_target(self, history: Episode) -> float:
        state = history.past_states()[-1].astype(int)
        result: float = self.q[state].dot(self.search.policy[state])
        return result

    def select_action(self, state: Any) -> int:
        return self.search.step(state)

    def post_visit(self, history: Episode, terminal: bool) -> None:
        target = self.td_target(history)
        self.credit_assignment.update(history, target)

    def post_episode(self, history: Episode) -> None:
        self.credit_assignment.reset()

    def get_q_values(self, state: int) -> np.ndarray:
        result: np.ndarray = self.q[state]
        return result
