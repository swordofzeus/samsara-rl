from typing import Any

import numpy as np

from samsara_rl.credit_assignment.credit_assignment import CreditAssignment
from samsara_rl.utils.bellman import discounted_cum_trajectory
from samsara_rl.utils.memory.episode import Episode, Transition


class MonteCarlo(CreditAssignment):
    """Every-visit Monte Carlo credit assignment for updating Q(s, a).

    Computes discounted returns in a fully vectorized manner
    without explicit loops over time steps. Updates happen at
    episode end via ``terminal``, not per-step.

    Args:
        q: Q-table array of shape ``(S, A)``.
        alpha: Learning rate for incremental Q updates.
        gamma: Discount factor.
    """

    def __init__(self, q: Any, alpha: float = 0.01, gamma: float = 1) -> None:
        super().__init__(alpha=alpha, gamma=gamma)
        self.q = q

    def observe(self, transition: Transition, target: float | None = None) -> None:
        """No-op. Monte Carlo waits until episode end."""
        pass

    def terminal(self, history: Episode) -> None:
        """Update Q-table from the complete episode trajectory.

        Uses advanced indexing to apply the constant-alpha MC update
        Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a)) for every visited
        (state, action) pair in the trajectory.

        Args:
            history: The complete episode history.
        """
        discounted_trajectory = discounted_cum_trajectory(self.gamma, history.past_rewards()[0:-1])
        s = history.past_states()[0:-1].astype(int)
        a = history.past_actions()[0:-1].astype(int)

        bellman_error = self.alpha * (discounted_trajectory - self.q[s, a])
        np.add.at(self.q, (s, a), bellman_error)
