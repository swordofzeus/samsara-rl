import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Planning(ABC):
    def __init__(self, mdp: Any, bellman_tolerance: float = 0.01) -> None:
        self.mdp = mdp
        self.bellman_tolerance = bellman_tolerance
        self.policy: np.ndarray = np.zeros((self.mdp.STATE_COUNT, self.mdp.ACTION_COUNT))
        self.values: np.ndarray = np.zeros(self.mdp.STATE_COUNT)

    @abstractmethod
    def find_optimal_policy(self) -> np.ndarray:
        pass

    def _bellman_error(self, value_history: list[np.ndarray]) -> float:
        """Return the sum of absolute value changes between the last two iterations."""
        if len(value_history) > 1:
            return float(np.absolute(value_history[-1] - value_history[-2]).sum())
        return math.inf

    def improve_policy(self) -> None:
        """
        Update the policy greedily with respect to the current state values.

        For each state, assigns probability 1.0 to the action with the highest
        expected next-state value (ties are broken by np.argmax, i.e. first index).

        Updates self.policy in place.
        """
        # Expected next-state value for each (s, a): shape (S, A)
        expected_next_value = np.dot(self.mdp.state_action_transition_matrix, self.values)

        best_actions = np.argmax(expected_next_value, axis=1)
        state_indices = np.arange(self.mdp.STATE_COUNT)

        self.policy = np.zeros((self.mdp.STATE_COUNT, self.mdp.ACTION_COUNT))
        self.policy[state_indices, best_actions] = 1.0
