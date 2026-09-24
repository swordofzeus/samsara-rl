from typing import Any

import numpy as np

from samsara_rl.credit_assignment.credit_assignment import CreditAssignment
from samsara_rl.utils.memory.episode import Episode, Transition


class TemporalDifference(CreditAssignment):
    """Tabular TD(lambda) credit assignment with eligibility traces.

    Updates Q values per-step via ``observe`` using the TD error
    and eligibility traces. Resets traces at episode end via ``terminal``.

    Args:
        q: Q-table array of shape ``(S, A)``.
        alpha: Learning rate.
        gamma: Discount factor.
        _lambda: Eligibility trace decay rate.
    """

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

    def observe(self, transition: Transition, target: float | None = None) -> None:
        """Update Q-table from a single transition using TD(lambda).

        Args:
            transition: The (S, A, R, S') transition.
            target: The scalar TD target (e.g. max_a Q(S', a) or Q(S', A')).
        """
        self.eligibility *= self._lambda
        S = int(transition.state)
        A = int(transition.action)
        self.eligibility[S][A] = 1

        if target is None:
            raise TypeError
        td_error = (transition.reward + self.gamma * target) - self.q[S][A]
        self.q += self.alpha * self.eligibility * td_error

    def terminal(self, history: Episode) -> None:
        """Reset eligibility traces at episode end."""
        self.eligibility = np.zeros(self.q.shape)
