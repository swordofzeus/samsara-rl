from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.credit_assignment.tabular.monte_carlo import MonteCarlo
from samsara_rl.policy.stochastic_policy import StochasticPolicy
from samsara_rl.utils.memory.episode import Episode


class MonteCarloPolicyEvaluation(Agent):
    """Monte Carlo policy evaluation.

    Generates episodes under a fixed stochastic policy and estimates
    Q(s, a) using the Monte Carlo credit assignment from the base class.

    Args:
        mdp: Gymnasium-compatible environment.
        policy: Stochastic policy array of shape ``(S, A)``.
        alpha: Learning rate for incremental Q updates.
        gamma: Discount factor.
    """

    def __init__(self, mdp: Any, policy: np.ndarray, alpha: float = 0.01, gamma: float = 1) -> None:
        super().__init__(mdp, alpha=alpha, gamma=gamma)
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))
        self.search = StochasticPolicy(policy)
        self.credit_assignment = MonteCarlo(self.q, alpha, gamma)

    def select_action(self, state: Any) -> int:
        return self.search.step(state)

    def post_episode(self, history: Episode) -> None:
        self.credit_assignment.terminal(history)

    def get_q_values(self, state: int) -> np.ndarray:
        result: np.ndarray = self.q[state]
        return result

    def post_visit(self, history: Episode, terminal: bool) -> None:
        return
