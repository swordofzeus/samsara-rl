from abc import abstractmethod
from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.credit_assignment.tabular.temporal_difference import (
    TemporalDifference,
)
from samsara_rl.policy.epsilon_greedy import EpsilonGreedy
from samsara_rl.utils.memory.episode import Episode


class TabularAgent(Agent):
    """Base class for tabular TD control agents.

    Maintains a Q-table, epsilon-greedy exploration, and
    eligibility-trace-based credit assignment. Subclasses
    provide ``td_target`` to define the bootstrap target
    (e.g. max_a Q(S', a) for Q-Learning, Q(S', A') for SARSA).

    Args:
        mdp: Gymnasium-compatible environment.
        alpha: Learning rate for incremental Q updates.
        gamma: Discount factor applied to future rewards.
        epsilon: Initial exploration rate for epsilon-greedy.
        epsilon_decay: Multiplicative decay applied to epsilon each episode.
    """

    def __init__(
        self, mdp: Any, alpha: float = 0.01, gamma: float = 0.9, epsilon: float = 0.99, epsilon_decay: float = 0.98
    ) -> None:
        super().__init__(mdp, alpha, gamma)
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))
        self.search = EpsilonGreedy(self.get_q_values, epsilon=epsilon, epsilon_decay=epsilon_decay)
        self.credit_assignment = TemporalDifference(self.q, alpha, gamma, _lambda=0)

    def get_metrics(self, trajectory: Episode) -> dict[str, float]:
        """Return metrics including epsilon for logging."""
        metrics = super().get_metrics(trajectory)
        metrics["Epsilon"] = self.search.epsilon
        return metrics

    def post_episode(self, history: Episode) -> None:
        """Decay epsilon and reset eligibility traces after each episode."""
        self.search.decay()
        self.credit_assignment.reset()

    def post_visit(self, history: Episode, terminal: bool) -> None:
        """Compute the TD target and update Q via credit assignment."""
        target = self.td_target(history)
        self.credit_assignment.update(history, target)

    def select_action(self, state: Any) -> int:
        """Select an action using epsilon-greedy exploration."""
        return self.search.step(state)

    @abstractmethod
    def td_target(self, history: Episode) -> float:
        """Compute the bootstrap target for the TD update.

        Args:
            history: The episode history recorded so far.

        Returns:
            The scalar TD target value.
        """
        pass

    def get_q_values(self, state: int) -> np.ndarray:
        """Return Q values for all actions at the given state.

        Args:
            state: The discrete state index.

        Returns:
            Array of Q values for each action.
        """
        result: np.ndarray = self.q[state]
        return result
