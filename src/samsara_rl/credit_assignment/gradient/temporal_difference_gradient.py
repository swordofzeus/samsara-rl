from typing import Any

import numpy as np
import torch

from samsara_rl.credit_assignment.credit_assignment import CreditAssignment


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

    def reset(self, **kwargs: Any) -> None:
        """Reset eligibility traces and gradients."""
        self.q.zero_grad()
        self.eligibility_traces = [np.zeros(p.shape) for p in self.q.parameters()]

    def update(self, target: float, history: Any, terminal: bool) -> None:
        """Semi-gradient TD(lambda) update after each step.

        Args:
            target: The scalar TD target (e.g. max_a Q(S', a) or Q(S', A')).
            history: The episode history recorded so far.
            terminal: Whether the episode has terminated.
        """
        if history.curr_index < 1:
            return

        S = history.past_states()[-2]
        A: int = int(history.past_actions()[-2])
        R: float = history.past_rewards()[-2]
        Q_S = self.q(S, A)[A] if not self.auto_grad else self.q(S)[A]
        Q_S.backward() if self.auto_grad else self.q.backward(upstream=1)

        td_error = (R + self.gamma * target) - Q_S
        self.last_td_error = td_error
        self.update_eligibility_traces(A)
        self.step(td_error)
        self.q.zero_grad()

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
