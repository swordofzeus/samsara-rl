from typing import Any

import numpy as np

from samsara_rl.planning.planning import Planning


class ValueIteration(Planning):
    """
    Value Iteration for finite, discrete MDPs.

    Iteratively applies the Bellman optimality equation until state values
    converge, then extracts a greedy policy from the converged values.

    Follows the formulation in David Silver's RL Lecture 3 (Planning by DP).
    """

    def __init__(self, mdp: "Any", bellman_tolerance: float = 0.01) -> None:
        super().__init__(mdp, bellman_tolerance)

    def find_optimal_policy(self) -> np.ndarray:
        """
        Run value iteration to convergence, then extract a greedy policy.

        Returns:
            The optimal policy as a (STATE_COUNT, ACTION_COUNT) array.
        """
        self._evaluate_value_function()
        self.improve_policy()
        return self.policy

    def _evaluate_value_function(self) -> None:
        """
        Estimate optimal state values via iterative Bellman optimality updates.

        For each iteration, applies:
            V(s) = max_a Σ_s' T(s,a,s') [R(s,a,s') + V(s')]
        until the change in values falls below bellman_tolerance.

        Updates self.values in place.
        """
        value_history: list[np.ndarray] = []

        while self._bellman_error(value_history) > self.bellman_tolerance:
            # Expected next-state value for each (s, a): shape (S, A)
            expected_next_value = np.dot(self.mdp.state_action_transition_matrix, self.values)

            # Expected immediate reward for each (s, a): shape (S, A)
            expected_reward = (self.mdp.reward_matrix * self.mdp.state_action_transition_matrix).sum(axis=2)

            # Bellman optimality: take max over actions, shape (S,)
            new_values = np.max(expected_reward + expected_next_value, axis=1)

            self.values = new_values
            value_history.append(new_values.copy())
