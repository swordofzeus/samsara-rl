from typing import Any

import numpy as np

from samsara_rl.credit_assignment.monte_carlo import MonteCarlo
from samsara_rl.policy.stochastic_policy import StochasticPolicy


class MonteCarloPolicyEvaluation(MonteCarlo):
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
        self.search = StochasticPolicy(policy)

    def select_action(self, state: Any) -> int:
        return self.search.step(state)
