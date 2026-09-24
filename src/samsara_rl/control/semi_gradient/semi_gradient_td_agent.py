"""Base class for TD control agents with function approximation."""

from abc import abstractmethod
from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.credit_assignment.gradient.temporal_difference_gradient import (
    TemporalDifferenceGradient,
)
from samsara_rl.policy.epsilon_greedy import EpsilonGreedy
from samsara_rl.utils.memory.episode import Episode


class SemiGradientTDAgent(Agent):
    """Base class for semi-gradient TD control with function approximation.

    Provides the shared structure for semi-gradient TD agents:
    Q function approximator, epsilon-greedy search, and credit
    assignment via ``TemporalDifferenceGradient``.

    Subclasses provide ``td_target`` to define the bootstrap target.
    """

    def __init__(
        self,
        mdp: Any,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        q: Any = None,
        auto_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(mdp, alpha=alpha, gamma=gamma)
        self.q = q
        self.auto_grad = auto_grad
        self.credit_assignment = TemporalDifferenceGradient(alpha, gamma, q, _lambda, auto_grad=auto_grad)
        self.search = EpsilonGreedy(self.get_q_values, epsilon=epsilon, epsilon_decay=epsilon_decay)

    @abstractmethod
    def td_target(self, history: Episode, terminal: bool) -> float:
        pass

    def get_q_values(self, curr_state: Any) -> np.ndarray:
        """Return Q values for all actions from the function approximator."""
        result: np.ndarray = self.q(curr_state)
        return result

    def post_visit(self, history: Episode, terminal: bool) -> None:
        transition = history.last_transition()
        if transition is None:
            return
        target = self.td_target(history, terminal)
        self.credit_assignment.observe(transition, target)

    def post_episode(self, history: Episode) -> None:
        """Reset gradients and eligibility traces at the end of each episode."""
        self.credit_assignment.terminal(history)
        self.search.decay()

    def select_action(self, state: Any) -> int:
        return self.search.step(state)
