from typing import Any

import numpy as np
import torch

from samsara_rl.credit_assignment.credit_assignment import CreditAssignment
from samsara_rl.utils.memory.episode import Episode, Transition


class TemporalDifferenceGradient(CreditAssignment):
    """Semi-gradient TD(lambda) credit assignment with function approximation.

    Maintains eligibility traces over the parameters of a function
    approximator and applies TD updates using the semi-gradient method.

    Args:
        alpha: Learning rate.
        gamma: Discount factor.
        q: Function approximator with ``parameters()`` and ``zero_grad()`` methods.
        _lambda: Eligibility trace decay rate.
        auto_grad: If True, use PyTorch autograd for gradient computation.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        gamma: float = 1,
        q: Any = None,
        _lambda: float = 0.2,
        auto_grad: bool = False,
    ) -> None:
        super().__init__(alpha, gamma)
        self.q = q
        self._lambda: float = _lambda
        self.auto_grad = auto_grad
        self.eligibility_traces: list[np.ndarray] = [np.zeros(p.shape) for p in self.q.parameters()]

    def observe(self, transition: Transition, target: float | None = None) -> None:
        """Semi-gradient TD(lambda) update from a single transition.

        Args:
            transition: The (S, A, R, S') transition.
            target: The scalar TD target (e.g. max_a Q(S', a) or Q(S', A')).
        """
        S = transition.state
        A = transition.action
        R = transition.reward
        Q_S = self.q(S, A)[A] if not self.auto_grad else self.q(S)[A]
        Q_S.backward() if self.auto_grad else self.q.backward(upstream=1)

        assert target is not None
        td_error = (R + self.gamma * target) - Q_S
        self.last_td_error = td_error
        self.update_eligibility_traces(A)
        self.step(td_error)
        self.q.zero_grad()

    def terminal(self, history: Episode) -> None:
        """Reset eligibility traces and gradients at episode end."""
        self.q.zero_grad()
        self.eligibility_traces = [np.zeros(p.shape) for p in self.q.parameters()]

    def update_eligibility_traces(self, A: int) -> None:
        """Decay traces by gamma * lambda and accumulate current gradient."""
        for index, param in enumerate(self.q.parameters()):
            curr_trace = self.eligibility_traces[index]
            curr_trace *= self._lambda * self.gamma
            gradient = param.grad.numpy() if self.auto_grad else param.grad
            curr_trace += gradient

    def step(self, td_error: Any) -> None:
        """Apply the TD update to parameters using eligibility traces.

        Args:
            td_error: The temporal difference error scalar.
        """
        with torch.no_grad():
            for index, param in enumerate(self.q.parameters()):
                curr_trace = self.eligibility_traces[index]
                trace: Any = torch.from_numpy(curr_trace) if isinstance(param, torch.Tensor) else curr_trace
                param.data += self.alpha * trace * td_error
