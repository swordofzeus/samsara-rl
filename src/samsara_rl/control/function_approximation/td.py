from typing import Any

import numpy as np

from samsara_rl.search.epsilon_greedy import EpsilonGreedy
from samsara_rl.agent import Agent
from samsara_rl.control.function_approximation.functions.linear import LinearFunction
from samsara_rl.utils.target import sarsa_target, td_target

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
        target = sarsa_target
    ) -> None:
        super().__init__(mdp, policy, alpha, gamma)
        self.q: LinearFunction = q
        self.td_target = target
        self._lambda: float = _lambda
        self.search = EpsilonGreedy(epsilon=1, epsilon_decay=0.99995)
        self.eligibility_traces: list[np.ndarray] = [
            np.zeros(p.value.shape) for p in self.q.params
        ]

    def get_q_values(self, curr_state: int) -> np.ndarray:
        """Return Q values for all actions from the function approximator."""
        return self.q(curr_state)

    def post_episode(self, trajectory: np.ndarray) -> None:
        """Reset gradients and eligibility traces at the end of each episode."""
        self.q.zero_grad()
        self.eligibility_traces = [
            np.zeros(p.value.shape) for p in self.q.params
        ]

    def post_visit(self, history: np.ndarray, terminal=False) -> None:
        """Semi-gradient TD(lambda) update after each step."""
        if history.curr_index < 1:
            return

        S: float = history.past_states()[-2]
        A: int = int(history.past_actions()[-2])
        R: float = history.past_rewards()[-2]

        Q_S: float = self.q(S, A)[A]
        Q_SPrime: float = self.td_target(history, self.q, self.mdp, terminal)
        td_error: float = (R + self.gamma * Q_SPrime) - Q_S

        self.q.backward()
        for index, param in enumerate(self.q.params):
            curr_trace = self.eligibility_traces[index]
            curr_trace *= self._lambda * self.gamma
            curr_trace[..., A] = curr_trace[..., A] + param.grad[..., A]
            param.value[..., A] += self.alpha * param.grad[..., A] * td_error
        self.q.zero_grad()