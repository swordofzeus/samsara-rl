from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from samsara_rl.agent import Agent
from samsara_rl.control.function_approximation.functions.manual.linear import (
    LinearFunction,
)
from samsara_rl.control.function_approximation.online.td.td_optimizer import (
    TDOptimizer,
)
from samsara_rl.search.epsilon_greedy import EpsilonGreedy
from samsara_rl.utils.memory.episode import Episode
from samsara_rl.utils.target import sarsa_target


class TemporalDifferenceGradient(Agent):
    """Semi-gradient TD(lambda) control with function approximation.

    Uses eligibility traces to assign credit to past state-action visits.
    The function approximator provides forward pass and local gradients;
    this class manages the TD error computation, eligibility traces, and
    parameter updates.

    Args:
        mdp: Gymnasium-compatible environment.
        policy: Stochastic policy array of shape (S, A).
        alpha: Learning rate.
        gamma: Discount factor.
        q: Function approximator (e.g. LinearFunction).
        feature_count: Number of features (unused, kept for compatibility).
        _lambda: Eligibility trace decay rate.
        target: TD target function.
    """

    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.001,
        gamma: float = 1,
        q: LinearFunction | None = None,
        feature_count: int = 0,
        _lambda: float = 0.2,
        target: Callable[..., float] = sarsa_target,
        epsilon: float = 1,
        epsilon_decay: float = 0.9999,
        auto_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mdp,
            policy,
            alpha,
            gamma,
            **kwargs,
        )
        self.q: LinearFunction = q  # type: ignore[assignment]
        self.td_target = target
        self._lambda: float = _lambda
        self.search = EpsilonGreedy(epsilon=epsilon, epsilon_decay=epsilon_decay)
        self.eligibility_traces: list[np.ndarray] = [np.zeros(p.shape) for p in self.q.parameters()]
        self.auto_grad = auto_grad
        self.td_optimizer = TDOptimizer(
            q,
            alpha,
            gamma,
            _lambda,
            self.eligibility_traces,
        )

    def get_q_values(self, curr_state: Any) -> np.ndarray:
        """Return Q values for all actions from the function approximator."""
        # input()
        return self.q(curr_state)

    def post_episode(self, history: Episode) -> None:
        """Reset gradients and eligibility traces at the end of each episode."""
        self.q.zero_grad()
        self.eligibility_traces = [np.zeros(p.shape) for p in self.q.parameters()]
        self.td_optimizer.eligibility_traces = self.eligibility_traces
        self.search.decay()

    def post_visit(self, history: Episode, terminal: bool = False) -> None:
        """Semi-gradient TD(lambda) update after each step."""
        if history.curr_index < 1:
            return

        S = history.past_states()[-2]
        A: int = int(history.past_actions()[-2])
        R: float = history.past_rewards()[-2]
        Q_S = self.q(S, A)[A] if not self.auto_grad else self.q(S)[A]
        Q_S.backward() if self.auto_grad else self.q.backward(upstream=1)

        with torch.no_grad():
            Q_SPrime = self.td_target(history, self.q, self.mdp, terminal)
        td_error = R + self.gamma * Q_SPrime - Q_S
        self.last_td_error = td_error
        self.update_el_traces(A)
        self.td_optimizer.step(A, td_error)
        self.q.zero_grad()

    def update_el_traces(self, A: int) -> None:
        for index, param in enumerate(self.q.parameters()):
            curr_trace = self.eligibility_traces[index]
            curr_trace *= self._lambda * self.gamma
            gradient = param.grad.numpy() if self.auto_grad else param.grad  # type: ignore[attr-defined]
            curr_trace += curr_trace + gradient
